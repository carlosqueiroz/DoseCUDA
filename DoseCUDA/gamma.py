"""
3D Gamma index analysis for dose comparison.

Provides gamma index computation with configurable criteria:
- Global or local dose difference normalization
- Configurable DTA (distance-to-agreement) and dose difference
- Dose threshold to exclude low-dose regions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import warnings


@dataclass
class GammaCriteria:
    """Configuration for gamma analysis criteria.

    Attributes
    ----------
    dta_mm : float
        Distance-to-agreement criterion in mm (default 3.0)
    dd_percent : float
        Dose difference criterion as percentage (default 3.0 for 3%)
    local : bool
        If True, use local dose normalization (percent of D_ref at each point).
        If False, use global normalization (percent of D_global).
        Default: False (global)
    dose_threshold_percent : float
        Only evaluate voxels where D_ref >= threshold × D_global.
        Default: 10.0 (10% threshold)
    global_dose : float, optional
        Reference dose for global normalization.
        If None, uses max(D_ref). Default: None
    max_gamma : float
        Cap gamma values at this maximum (for efficiency and statistics).
        Default: 2.0
    """
    dta_mm: float = 3.0
    dd_percent: float = 3.0
    local: bool = False
    dose_threshold_percent: float = 10.0
    global_dose: Optional[float] = None
    max_gamma: float = 2.0

    def label(self) -> str:
        """Return human-readable label like '3%/3mm'."""
        mode = "local" if self.local else "global"
        return f"{self.dd_percent}%/{self.dta_mm}mm ({mode})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dta_mm': self.dta_mm,
            'dd_percent': self.dd_percent,
            'local': self.local,
            'dose_threshold_percent': self.dose_threshold_percent,
            'global_dose': self.global_dose,
            'max_gamma': self.max_gamma
        }


@dataclass
class GammaResult:
    """Result container for gamma analysis.

    Attributes
    ----------
    pass_rate : float
        Fraction of evaluated voxels with gamma <= 1.0 (0 to 1)
    mean_gamma : float
        Mean gamma value for evaluated voxels
    gamma_p95 : float
        95th percentile of gamma values
    n_evaluated : int
        Number of voxels evaluated (above threshold)
    n_passed : int
        Number of voxels with gamma <= 1.0
    gamma_map : np.ndarray, optional
        Full 3D gamma map if requested
    criteria : dict
        Criteria used for this evaluation
    """
    pass_rate: float
    mean_gamma: float
    gamma_p95: float
    n_evaluated: int
    n_passed: int
    gamma_map: Optional[np.ndarray]
    criteria: Dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"GammaResult(pass_rate={self.pass_rate:.1%}, "
            f"mean={self.mean_gamma:.3f}, "
            f"p95={self.gamma_p95:.3f}, "
            f"n_eval={self.n_evaluated})"
        )


def _precompute_search_offsets(
    dta_mm: float,
    spacing_mm: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute voxel offsets within DTA search radius.

    Parameters
    ----------
    dta_mm : float
        Distance-to-agreement in mm
    spacing_mm : tuple
        Voxel spacing (sx, sy, sz) in mm

    Returns
    -------
    offsets : np.ndarray
        Array of shape (N, 3) with [di, dj, dk] offsets in voxels
    distances : np.ndarray
        Physical distance from center for each offset in mm
    """
    sx, sy, sz = spacing_mm

    # Maximum offset in each dimension
    max_di = int(np.ceil(dta_mm / sx))
    max_dj = int(np.ceil(dta_mm / sy))
    max_dk = int(np.ceil(dta_mm / sz))

    offsets = []
    distances = []

    for dk in range(-max_dk, max_dk + 1):
        for dj in range(-max_dj, max_dj + 1):
            for di in range(-max_di, max_di + 1):
                # Physical distance
                dist_mm = np.sqrt((di * sx)**2 + (dj * sy)**2 + (dk * sz)**2)
                if dist_mm <= dta_mm:
                    offsets.append([di, dj, dk])
                    distances.append(dist_mm)

    offsets = np.array(offsets, dtype=np.int32)
    distances = np.array(distances, dtype=np.float32)

    # Sort by distance (check closest first for early termination)
    sort_idx = np.argsort(distances)
    offsets = offsets[sort_idx]
    distances = distances[sort_idx]

    return offsets, distances


def compute_gamma_3d(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    criteria: Optional[GammaCriteria] = None,
    roi_mask: Optional[np.ndarray] = None,
    return_map: bool = False
) -> GammaResult:
    """
    Compute 3D gamma index between evaluated and reference dose.

    The gamma index is computed as:
        gamma(x) = min_{r in search_region} sqrt(
            (|r - x| / DTA)^2 + ((D_eval(r) - D_ref(x)) / delta_D)^2
        )

    where delta_D depends on global_dose (global mode) or D_ref(x) (local mode).

    Parameters
    ----------
    dose_eval : np.ndarray
        Evaluated dose array (DoseCUDA), shape (z, y, x) in Gy.
        MUST be resampled to the same grid as dose_ref before calling.
    dose_ref : np.ndarray
        Reference dose array (TPS), shape (z, y, x) in Gy
    spacing_mm : tuple
        Voxel spacing (sx, sy, sz) in mm.
        Note: This is (x, y, z) spacing, arrays are (z, y, x) shape.
    criteria : GammaCriteria, optional
        Gamma criteria configuration. Uses 3%/3mm global if None.
    roi_mask : np.ndarray, optional
        Boolean mask to restrict evaluation to specific region.
        Shape must match dose arrays.
    return_map : bool
        If True, return full 3D gamma map. Default False.

    Returns
    -------
    GammaResult
        Container with pass_rate, mean_gamma, gamma_p95, counts, and optional map

    Notes
    -----
    - Arrays must have the same shape (dose_eval resampled to dose_ref grid)
    - The search is performed from dose_eval to dose_ref grid
    - Offsets are precomputed and sorted by distance for efficiency
    - Early termination when gamma <= 1.0 is found
    """
    if criteria is None:
        criteria = GammaCriteria()

    # Validate inputs
    if dose_eval.shape != dose_ref.shape:
        raise ValueError(
            f"dose_eval shape {dose_eval.shape} != dose_ref shape {dose_ref.shape}. "
            "Arrays must be on the same grid."
        )

    if roi_mask is not None and roi_mask.shape != dose_ref.shape:
        raise ValueError(
            f"roi_mask shape {roi_mask.shape} != dose shape {dose_ref.shape}"
        )

    # Ensure float32 for efficiency
    dose_eval = dose_eval.astype(np.float32, copy=False)
    dose_ref = dose_ref.astype(np.float32, copy=False)

    # Compute global dose for normalization
    D_global = criteria.global_dose
    if D_global is None:
        D_global = float(np.max(dose_ref))

    if D_global <= 0:
        warnings.warn("Global dose is zero or negative. Returning empty result.")
        return GammaResult(
            pass_rate=0.0,
            mean_gamma=np.nan,
            gamma_p95=np.nan,
            n_evaluated=0,
            n_passed=0,
            gamma_map=None,
            criteria=criteria.to_dict()
        )

    # Compute dose threshold
    threshold = (criteria.dose_threshold_percent / 100.0) * D_global

    # Build evaluation mask
    eval_mask = dose_ref >= threshold
    if roi_mask is not None:
        eval_mask = eval_mask & roi_mask

    n_to_evaluate = int(np.sum(eval_mask))
    if n_to_evaluate == 0:
        warnings.warn("No voxels above threshold to evaluate.")
        return GammaResult(
            pass_rate=0.0,
            mean_gamma=np.nan,
            gamma_p95=np.nan,
            n_evaluated=0,
            n_passed=0,
            gamma_map=None,
            criteria=criteria.to_dict()
        )

    # Precompute search offsets
    offsets, offset_distances = _precompute_search_offsets(
        criteria.dta_mm, spacing_mm
    )

    # Precompute spatial term (distance / DTA)^2
    spatial_terms = (offset_distances / criteria.dta_mm) ** 2

    # Initialize gamma map
    nz, ny, nx = dose_ref.shape
    gamma_map = np.full((nz, ny, nx), np.inf, dtype=np.float32)

    # Dose difference criterion factor
    dd_factor = criteria.dd_percent / 100.0

    # Get indices to evaluate
    eval_indices = np.argwhere(eval_mask)

    # Main gamma computation loop
    for idx in eval_indices:
        k, j, i = idx  # z, y, x indices

        D_ref_local = dose_ref[k, j, i]

        # Compute delta_D criterion
        if criteria.local:
            delta_D = dd_factor * D_ref_local
        else:
            delta_D = dd_factor * D_global

        if delta_D <= 0:
            continue

        # Search over offsets
        min_gamma_sq = np.inf

        for n, (di, dj, dk) in enumerate(offsets):
            # Target position
            ii = i + di
            jj = j + dj
            kk = k + dk

            # Bounds check
            if ii < 0 or jj < 0 or kk < 0:
                continue
            if ii >= nx or jj >= ny or kk >= nz:
                continue

            # Get evaluated dose at offset position
            D_eval_at_offset = dose_eval[kk, jj, ii]

            # Compute dose term
            dose_diff = D_eval_at_offset - D_ref_local
            dose_term = (dose_diff / delta_D) ** 2

            # Compute gamma squared
            gamma_sq = spatial_terms[n] + dose_term

            if gamma_sq < min_gamma_sq:
                min_gamma_sq = gamma_sq

            # Early termination if gamma <= 1
            if min_gamma_sq <= 1.0:
                break

        # Store gamma value (capped at max_gamma)
        gamma_val = np.sqrt(min_gamma_sq)
        gamma_map[k, j, i] = min(gamma_val, criteria.max_gamma)

    # Compute statistics
    gamma_values = gamma_map[eval_mask]
    gamma_finite = gamma_values[np.isfinite(gamma_values)]

    if len(gamma_finite) == 0:
        return GammaResult(
            pass_rate=0.0,
            mean_gamma=np.nan,
            gamma_p95=np.nan,
            n_evaluated=n_to_evaluate,
            n_passed=0,
            gamma_map=gamma_map if return_map else None,
            criteria=criteria.to_dict()
        )

    n_passed = int(np.sum(gamma_finite <= 1.0))
    pass_rate = n_passed / len(gamma_finite)
    mean_gamma = float(np.mean(gamma_finite))
    gamma_p95 = float(np.percentile(gamma_finite, 95))

    # Update criteria dict with actual global_dose used
    criteria_dict = criteria.to_dict()
    criteria_dict['global_dose_used'] = D_global

    return GammaResult(
        pass_rate=pass_rate,
        mean_gamma=mean_gamma,
        gamma_p95=gamma_p95,
        n_evaluated=len(gamma_finite),
        n_passed=n_passed,
        gamma_map=gamma_map if return_map else None,
        criteria=criteria_dict
    )


def compute_gamma_multiple_criteria(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    criteria_list: Optional[list] = None,
    roi_mask: Optional[np.ndarray] = None
) -> Dict[str, GammaResult]:
    """
    Compute gamma for multiple criteria sets.

    Parameters
    ----------
    dose_eval : np.ndarray
        Evaluated dose array (already resampled to ref grid)
    dose_ref : np.ndarray
        Reference dose array
    spacing_mm : tuple
        Voxel spacing (sx, sy, sz) in mm
    criteria_list : list, optional
        List of GammaCriteria objects.
        Default: [3%/3mm global, 2%/2mm global]
    roi_mask : np.ndarray, optional
        Boolean mask to restrict evaluation

    Returns
    -------
    dict
        Dictionary mapping criteria label to GammaResult
    """
    if criteria_list is None:
        criteria_list = [
            GammaCriteria(dta_mm=3.0, dd_percent=3.0, local=False),
            GammaCriteria(dta_mm=2.0, dd_percent=2.0, local=False),
        ]

    results = {}
    for criteria in criteria_list:
        label = criteria.label()
        results[label] = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing_mm,
            criteria=criteria,
            roi_mask=roi_mask,
            return_map=False
        )

    return results


# Standard clinical criteria presets
GAMMA_3_3_GLOBAL = GammaCriteria(dta_mm=3.0, dd_percent=3.0, local=False)
GAMMA_2_2_GLOBAL = GammaCriteria(dta_mm=2.0, dd_percent=2.0, local=False)
GAMMA_3_3_LOCAL = GammaCriteria(dta_mm=3.0, dd_percent=3.0, local=True)
GAMMA_2_2_LOCAL = GammaCriteria(dta_mm=2.0, dd_percent=2.0, local=True)
