"""
3D Gamma index analysis for dose comparison.

Provides gamma index computation with configurable criteria:
- Global or local dose difference normalization
- Configurable DTA (distance-to-agreement) and dose difference
- Dose threshold to exclude low-dose regions

Supports GPU acceleration via CUDA when available, with automatic
fallback to CPU implementation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import warnings

# Try to import CUDA backend
_CUDA_AVAILABLE = False
try:
    from DoseCUDA.dose_kernels import gamma_3d_cuda as _gamma_3d_cuda
    from DoseCUDA.dose_kernels import gamma_cuda_available as _check_cuda
    _CUDA_AVAILABLE = _check_cuda()
except ImportError:
    _gamma_3d_cuda = None
    _check_cuda = None


def cuda_gamma_available() -> bool:
    """Check if CUDA gamma acceleration is available."""
    return _CUDA_AVAILABLE


def set_cuda_gamma_enabled(enabled: bool) -> None:
    """
    Enable or disable CUDA gamma acceleration.
    
    Parameters
    ----------
    enabled : bool
        If True, use CUDA when available. If False, force CPU implementation.
    """
    global _CUDA_AVAILABLE
    if enabled and _check_cuda is not None:
        _CUDA_AVAILABLE = _check_cuda()
    else:
        _CUDA_AVAILABLE = False


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
    sampling: float = 1.0  # sub-voxel sampling factor (1.0 = voxel)

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
            'max_gamma': self.max_gamma,
            'sampling': self.sampling
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
    spacing_mm: Tuple[float, float, float],
    max_gamma: float,
    sampling: float
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
    offsets_vox : np.ndarray
        Array of shape (N, 3) with fractional offsets in voxel units (x,y,z).
    spatial_terms : np.ndarray
        (dist / dta_mm)^2 for each offset (float32).
    """
    sx, sy, sz = spacing_mm
    search_mm = float(dta_mm * max_gamma)
    samp = float(sampling) if sampling > 0 else 1.0
    step_x = sx * samp
    step_y = sy * samp
    step_z = sz * samp

    max_ix = int(np.ceil(search_mm / step_x))
    max_iy = int(np.ceil(search_mm / step_y))
    max_iz = int(np.ceil(search_mm / step_z))

    offsets = []
    spatial_terms = []

    for kz in range(-max_iz, max_iz + 1):
        dz_mm = kz * step_z
        for jy in range(-max_iy, max_iy + 1):
            dy_mm = jy * step_y
            for ix in range(-max_ix, max_ix + 1):
                dx_mm = ix * step_x
                dist_mm = np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm)
                if dist_mm <= search_mm + 1e-6:
                    offsets.append([dx_mm / sx, dy_mm / sy, dz_mm / sz])
                    spatial_terms.append((dist_mm / dta_mm) ** 2 if dta_mm > 0 else 0.0)

    offsets = np.array(offsets, dtype=np.float32)
    spatial_terms = np.array(spatial_terms, dtype=np.float32)

    # Sort by distance (spatial_terms already proportional to dist^2)
    sort_idx = np.argsort(spatial_terms)
    offsets = offsets[sort_idx]
    spatial_terms = spatial_terms[sort_idx]

    return offsets, spatial_terms


def _compute_gamma_cuda(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    criteria: 'GammaCriteria',
    roi_mask: Optional[np.ndarray] = None,
    return_map: bool = False,
    gpu_id: int = 0
) -> 'GammaResult':
    """
    Compute gamma using CUDA acceleration.
    
    Internal function called by compute_gamma_3d when CUDA is available.
    """
    # Ensure contiguous float32 arrays
    dose_eval = np.ascontiguousarray(dose_eval, dtype=np.float32)
    dose_ref = np.ascontiguousarray(dose_ref, dtype=np.float32)
    
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
    
    roi_arg = None
    if roi_mask is not None:
        roi_arg = np.ascontiguousarray(roi_mask.astype(np.uint8, copy=False))

    # Call CUDA function
    try:
        result = _gamma_3d_cuda(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing=spacing_mm,  # C expects "spacing", not "spacing_mm"
            dta_mm=criteria.dta_mm,
            dd_percent=criteria.dd_percent,
            dose_threshold_percent=criteria.dose_threshold_percent,
            global_dose=D_global,
            local=criteria.local,
            max_gamma=criteria.max_gamma,
            sampling=criteria.sampling,
            return_map=return_map,
            roi_mask=roi_arg,
            gpu_id=gpu_id
        )
        
        # Extract results from CUDA return dict
        criteria_dict = criteria.to_dict()
        criteria_dict['global_dose_used'] = D_global
        
        return GammaResult(
            pass_rate=result['pass_rate'],
            mean_gamma=result['mean_gamma'],
            gamma_p95=result['gamma_p95'],
            n_evaluated=result['n_evaluated'],
            n_passed=result['n_passed'],
            gamma_map=result.get('gamma_map') if return_map else None,
            criteria=criteria_dict
        )
    except Exception as e:
        warnings.warn(f"CUDA gamma failed: {e}. Falling back to CPU.")
        return None  # Signal to use CPU fallback


def compute_gamma_3d(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    criteria: Optional[GammaCriteria] = None,
    roi_mask: Optional[np.ndarray] = None,
    return_map: bool = False,
    use_cuda: Optional[bool] = None,
    gpu_id: int = 0
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
    use_cuda : bool, optional
        If True, force CUDA. If False, force CPU. If None (default),
        use CUDA if available, else CPU.
    gpu_id : int
        GPU device ID to use for CUDA computation. Default 0.

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
    - CUDA acceleration provides ~100x speedup on large grids
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

    # Determine whether to use CUDA
    should_use_cuda = use_cuda if use_cuda is not None else _CUDA_AVAILABLE

    # Try CUDA first if enabled
    if should_use_cuda:
        result = _compute_gamma_cuda(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing_mm,
            criteria=criteria,
            roi_mask=roi_mask,
            return_map=return_map,
            gpu_id=gpu_id
        )
        if result is not None:
            return result
        # If CUDA failed, fall through to CPU

    # CPU implementation
    return _compute_gamma_cpu(
        dose_eval=dose_eval,
        dose_ref=dose_ref,
        spacing_mm=spacing_mm,
        criteria=criteria,
        roi_mask=roi_mask,
        return_map=return_map
    )


def _compute_gamma_cpu(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    criteria: GammaCriteria,
    roi_mask: Optional[np.ndarray] = None,
    return_map: bool = False
) -> GammaResult:
    """
    Optimized CPU implementation of gamma computation using NumPy vectorization.
    
    Uses batch processing with vectorized operations for much better performance
    than the naive per-voxel loop approach.
    """
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

    # Precompute search offsets (fractional voxels) and spatial terms
    offsets_vox, spatial_terms = _precompute_search_offsets(
        criteria.dta_mm, spacing_mm, criteria.max_gamma, criteria.sampling
    )

    # Initialize gamma map
    nz, ny, nx = dose_ref.shape
    gamma_map = np.full((nz, ny, nx), np.inf, dtype=np.float32)

    # Dose difference criterion factor
    dd_factor = criteria.dd_percent / 100.0

    # Get indices to evaluate
    eval_indices = np.argwhere(eval_mask)
    
    # For progress reporting
    total_voxels = len(eval_indices)
    
    # Use vectorized batch processing for better performance
    # Process in batches to balance memory usage and speed
    BATCH_SIZE = 10000
    n_passed_quick = 0
    
    for batch_start in range(0, total_voxels, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_voxels)
        batch_indices = eval_indices[batch_start:batch_end]
        
        # Extract batch coordinates
        batch_k = batch_indices[:, 0]  # z
        batch_j = batch_indices[:, 1]  # y  
        batch_i = batch_indices[:, 2]  # x
        
        # Get reference doses for batch
        D_ref_batch = dose_ref[batch_k, batch_j, batch_i]
        
        # Compute delta_D for batch
        if criteria.local:
            delta_D_batch = dd_factor * D_ref_batch
        else:
            delta_D_batch = np.full(len(batch_indices), dd_factor * D_global, dtype=np.float32)
        
        # Initialize minimum gamma squared for batch
        min_gamma_sq_batch = np.full(len(batch_indices), np.inf, dtype=np.float32)
        
        # Mask for voxels still needing evaluation (gamma > 1)
        active_mask = np.ones(len(batch_indices), dtype=bool)
        
        # Search over offsets (sorted by distance, closest first)
        for n, (off_x, off_y, off_z) in enumerate(offsets_vox):
            if not np.any(active_mask):
                break  # All voxels found gamma <= 1
            
            # Compute target positions for active voxels only
            active_idx = np.where(active_mask)[0]
            
            fx = batch_i[active_idx].astype(np.float32) + off_x
            fy = batch_j[active_idx].astype(np.float32) + off_y
            fz = batch_k[active_idx].astype(np.float32) + off_z

            # Bounds check (allow exact boundary)
            valid = (
                (fx >= 0) & (fx <= nx - 1) &
                (fy >= 0) & (fy <= ny - 1) &
                (fz >= 0) & (fz <= nz - 1)
            )

            if not np.any(valid):
                continue
                
            valid_active_idx = active_idx[valid]
            fx = fx[valid]
            fy = fy[valid]
            fz = fz[valid]

            # Trilinear interpolation of dose_eval at fractional coords
            x0 = np.floor(fx).astype(np.int32)
            y0 = np.floor(fy).astype(np.int32)
            z0 = np.floor(fz).astype(np.int32)

            x1 = np.minimum(x0 + 1, nx - 1)
            y1 = np.minimum(y0 + 1, ny - 1)
            z1 = np.minimum(z0 + 1, nz - 1)

            tx = fx - x0
            ty = fy - y0
            tz = fz - z0

            # Gather corners
            c000 = dose_eval[z0, y0, x0]
            c100 = dose_eval[z0, y0, x1]
            c010 = dose_eval[z0, y1, x0]
            c110 = dose_eval[z0, y1, x1]
            c001 = dose_eval[z1, y0, x0]
            c101 = dose_eval[z1, y0, x1]
            c011 = dose_eval[z1, y1, x0]
            c111 = dose_eval[z1, y1, x1]

            c00 = c000 * (1 - tx) + c100 * tx
            c01 = c001 * (1 - tx) + c101 * tx
            c10 = c010 * (1 - tx) + c110 * tx
            c11 = c011 * (1 - tx) + c111 * tx

            c0 = c00 * (1 - ty) + c10 * ty
            c1 = c01 * (1 - ty) + c11 * ty

            D_eval_at_offset = c0 * (1 - tz) + c1 * tz
            
            # Get evaluated dose at offset positions
            D_eval_at_offset = dose_eval[valid_kk, valid_jj, valid_ii]
            
            # Get reference dose for these voxels
            D_ref_valid = D_ref_batch[valid_active_idx]
            delta_D_valid = delta_D_batch[valid_active_idx]
            
            # Skip voxels with invalid delta_D
            valid_delta = delta_D_valid > 0
            if not np.any(valid_delta):
                continue
            
            # Compute dose term
            dose_diff = D_eval_at_offset[valid_delta] - D_ref_valid[valid_delta]
            dose_term = (dose_diff / delta_D_valid[valid_delta]) ** 2
            
            # Compute gamma squared
            gamma_sq = spatial_terms[n] + dose_term
            
            # Update minimum gamma for valid voxels
            update_idx = valid_active_idx[valid_delta]
            min_gamma_sq_batch[update_idx] = np.minimum(min_gamma_sq_batch[update_idx], gamma_sq)
            
            # Update active mask - deactivate voxels with gamma <= 1
            active_mask[update_idx] = min_gamma_sq_batch[update_idx] > 1.0
        
        # Store gamma values (capped at max_gamma)
        gamma_val_batch = np.sqrt(min_gamma_sq_batch)
        gamma_val_batch = np.minimum(gamma_val_batch, criteria.max_gamma)
        gamma_map[batch_k, batch_j, batch_i] = gamma_val_batch
        
        # Count quick passes
        n_passed_quick += np.sum(gamma_val_batch <= 1.0)

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
