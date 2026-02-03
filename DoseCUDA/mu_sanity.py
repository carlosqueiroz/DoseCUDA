"""
MU sanity check for secondary dose verification.

Provides informational MU check by comparing dose at isocenter
between calculated and reference dose distributions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import warnings


@dataclass
class MUSanityResult:
    """Result container for MU sanity check.

    Attributes
    ----------
    isocenter_mm : np.ndarray
        Isocenter position [x, y, z] in mm
    dose_calc_at_iso : float
        Calculated dose at isocenter in Gy
    dose_ref_at_iso : float
        Reference dose at isocenter in Gy
    total_mu : float
        Total plan MU (sum of all beams)
    gy_per_mu_calc : float
        Calculated Gy/MU at isocenter
    gy_per_mu_ref : float
        Reference Gy/MU at isocenter
    mu_equiv_ratio : float
        Ratio of Gy/MU (calc / ref). 1.0 means perfect agreement.
    status : str
        Status: 'INFO' (within tolerance), 'WARN', or 'FAIL'
    message : str
        Human-readable status message
    """
    isocenter_mm: np.ndarray
    dose_calc_at_iso: float
    dose_ref_at_iso: float
    total_mu: float
    gy_per_mu_calc: float
    gy_per_mu_ref: float
    mu_equiv_ratio: float
    status: str
    message: str

    def __repr__(self) -> str:
        return (
            f"MUSanityResult(ratio={self.mu_equiv_ratio:.4f}, "
            f"status={self.status})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'isocenter_mm': self.isocenter_mm.tolist(),
            'dose_calc_at_iso': self.dose_calc_at_iso,
            'dose_ref_at_iso': self.dose_ref_at_iso,
            'total_mu': self.total_mu,
            'gy_per_mu_calc': self.gy_per_mu_calc,
            'gy_per_mu_ref': self.gy_per_mu_ref,
            'mu_equiv_ratio': self.mu_equiv_ratio,
            'status': self.status,
            'message': self.message
        }


def sample_dose_at_point(
    dose: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    point_mm: np.ndarray
) -> float:
    """
    Sample dose at a physical point using trilinear interpolation.

    Parameters
    ----------
    dose : np.ndarray
        3D dose array, shape (z, y, x) in Gy
    origin : np.ndarray
        Grid origin [x, y, z] in mm
    spacing : np.ndarray
        Voxel spacing [x, y, z] in mm
    point_mm : np.ndarray
        Physical point [x, y, z] in mm

    Returns
    -------
    float
        Interpolated dose value in Gy.
        Returns NaN if point is outside grid bounds.
    """
    # Convert physical coordinates to continuous voxel indices
    # origin and spacing are [x, y, z], dose array is (z, y, x)
    idx_x = (point_mm[0] - origin[0]) / spacing[0]
    idx_y = (point_mm[1] - origin[1]) / spacing[1]
    idx_z = (point_mm[2] - origin[2]) / spacing[2]

    nz, ny, nx = dose.shape

    # Check bounds (with small margin for floating point)
    if idx_x < -0.5 or idx_y < -0.5 or idx_z < -0.5:
        return np.nan
    if idx_x > nx - 0.5 or idx_y > ny - 0.5 or idx_z > nz - 0.5:
        return np.nan

    # Trilinear interpolation
    # Get integer indices and fractions
    x0 = int(np.floor(idx_x))
    y0 = int(np.floor(idx_y))
    z0 = int(np.floor(idx_z))

    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    z1 = min(z0 + 1, nz - 1)

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    z0 = max(z0, 0)

    xd = idx_x - x0
    yd = idx_y - y0
    zd = idx_z - z0

    # Clamp fractions
    xd = max(0, min(1, xd))
    yd = max(0, min(1, yd))
    zd = max(0, min(1, zd))

    # Trilinear interpolation
    # dose array is (z, y, x)
    c000 = dose[z0, y0, x0]
    c001 = dose[z0, y0, x1]
    c010 = dose[z0, y1, x0]
    c011 = dose[z0, y1, x1]
    c100 = dose[z1, y0, x0]
    c101 = dose[z1, y0, x1]
    c110 = dose[z1, y1, x0]
    c111 = dose[z1, y1, x1]

    # Interpolate along x
    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd

    # Interpolate along y
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    # Interpolate along z
    c = c0 * (1 - zd) + c1 * zd

    return float(c)


def extract_isocenter_from_plan(plan) -> Tuple[np.ndarray, float]:
    """
    Extract isocenter and total MU from IMRTPlan.

    Parameters
    ----------
    plan : IMRTPlan
        Loaded plan object with beams

    Returns
    -------
    isocenter : np.ndarray
        Plan isocenter in mm [x, y, z]
    total_mu : float
        Total MU for all beams

    Raises
    ------
    ValueError
        If plan has no beams or isocenter cannot be determined
    """
    if not hasattr(plan, 'beams') or len(plan.beams) == 0:
        raise ValueError("Plan has no beams")

    # Get isocenter from first beam's first control point
    first_beam = plan.beams[0]
    if not hasattr(first_beam, 'control_points') or len(first_beam.control_points) == 0:
        raise ValueError("First beam has no control points")

    first_cp = first_beam.control_points[0]

    # Isocenter is typically stored as [x, y, z] in mm
    if hasattr(first_cp, 'isocenter'):
        isocenter = np.array(first_cp.isocenter, dtype=np.float64)
    elif hasattr(first_beam, 'isocenter'):
        isocenter = np.array(first_beam.isocenter, dtype=np.float64)
    else:
        raise ValueError("Cannot find isocenter in plan")

    # Sum total MU from all beams
    total_mu = 0.0
    for beam in plan.beams:
        if hasattr(beam, 'meterset'):
            total_mu += float(beam.meterset)
        elif hasattr(beam, 'beam_meterset'):
            total_mu += float(beam.beam_meterset)

    if total_mu <= 0:
        warnings.warn("Total MU is zero or negative. MU sanity check may be invalid.")

    return isocenter, total_mu


def compute_mu_sanity_check(
    dose_calc: np.ndarray,
    dose_ref: np.ndarray,
    grid_origin: np.ndarray,
    grid_spacing: np.ndarray,
    isocenter: np.ndarray,
    total_mu: float,
    tolerance: float = 0.05
) -> MUSanityResult:
    """
    Compute MU sanity check at plan isocenter.

    This is an INFORMATIONAL check comparing dose/MU at isocenter.
    It does not replace proper MU verification methods.

    Parameters
    ----------
    dose_calc : np.ndarray
        Calculated dose array, shape (z, y, x) in Gy
    dose_ref : np.ndarray
        Reference dose array, shape (z, y, x) in Gy
    grid_origin : np.ndarray
        Grid origin in mm [x, y, z]
    grid_spacing : np.ndarray
        Voxel spacing in mm [x, y, z]
    isocenter : np.ndarray
        Isocenter position in mm [x, y, z]
    total_mu : float
        Total plan MU (sum of all beams)
    tolerance : float
        Acceptable deviation from 1.0 for mu_equiv_ratio.
        Default 0.05 (5%)

    Returns
    -------
    MUSanityResult
        Container with doses, ratios, and status

    Notes
    -----
    Status levels:
    - INFO: |ratio - 1| <= tolerance
    - WARN: tolerance < |ratio - 1| <= 2 * tolerance
    - FAIL: |ratio - 1| > 2 * tolerance
    """
    # Sample doses at isocenter
    dose_calc_at_iso = sample_dose_at_point(
        dose_calc, grid_origin, grid_spacing, isocenter
    )
    dose_ref_at_iso = sample_dose_at_point(
        dose_ref, grid_origin, grid_spacing, isocenter
    )

    # Handle invalid samples
    if np.isnan(dose_calc_at_iso):
        warnings.warn(f"Isocenter {isocenter} outside calculated dose grid")
        return MUSanityResult(
            isocenter_mm=isocenter,
            dose_calc_at_iso=np.nan,
            dose_ref_at_iso=dose_ref_at_iso,
            total_mu=total_mu,
            gy_per_mu_calc=np.nan,
            gy_per_mu_ref=np.nan,
            mu_equiv_ratio=np.nan,
            status='FAIL',
            message="Isocenter outside calculated dose grid"
        )

    if np.isnan(dose_ref_at_iso):
        warnings.warn(f"Isocenter {isocenter} outside reference dose grid")
        return MUSanityResult(
            isocenter_mm=isocenter,
            dose_calc_at_iso=dose_calc_at_iso,
            dose_ref_at_iso=np.nan,
            total_mu=total_mu,
            gy_per_mu_calc=np.nan,
            gy_per_mu_ref=np.nan,
            mu_equiv_ratio=np.nan,
            status='FAIL',
            message="Isocenter outside reference dose grid"
        )

    # Compute Gy/MU
    if total_mu > 0:
        gy_per_mu_calc = dose_calc_at_iso / total_mu
        gy_per_mu_ref = dose_ref_at_iso / total_mu
    else:
        gy_per_mu_calc = np.nan
        gy_per_mu_ref = np.nan

    # Compute ratio
    if gy_per_mu_ref > 0 and gy_per_mu_calc > 0:
        mu_equiv_ratio = gy_per_mu_calc / gy_per_mu_ref
    elif dose_ref_at_iso > 0 and dose_calc_at_iso > 0:
        # Fallback: ratio of doses
        mu_equiv_ratio = dose_calc_at_iso / dose_ref_at_iso
    else:
        mu_equiv_ratio = np.nan

    # Determine status
    if np.isnan(mu_equiv_ratio):
        status = 'FAIL'
        message = "Cannot compute MU equivalent ratio (zero or invalid values)"
    else:
        deviation = abs(mu_equiv_ratio - 1.0)
        if deviation <= tolerance:
            status = 'INFO'
            message = f"MU ratio {mu_equiv_ratio:.4f} within {tolerance*100:.1f}% tolerance"
        elif deviation <= 2 * tolerance:
            status = 'WARN'
            message = f"MU ratio {mu_equiv_ratio:.4f} exceeds {tolerance*100:.1f}% but within {2*tolerance*100:.1f}%"
        else:
            status = 'FAIL'
            message = f"MU ratio {mu_equiv_ratio:.4f} exceeds {2*tolerance*100:.1f}% tolerance"

    return MUSanityResult(
        isocenter_mm=isocenter,
        dose_calc_at_iso=dose_calc_at_iso,
        dose_ref_at_iso=dose_ref_at_iso,
        total_mu=total_mu,
        gy_per_mu_calc=gy_per_mu_calc if not np.isnan(gy_per_mu_calc) else 0.0,
        gy_per_mu_ref=gy_per_mu_ref if not np.isnan(gy_per_mu_ref) else 0.0,
        mu_equiv_ratio=mu_equiv_ratio,
        status=status,
        message=message
    )


def compute_mu_sanity_from_plan(
    dose_calc: np.ndarray,
    dose_ref: np.ndarray,
    grid_origin: np.ndarray,
    grid_spacing: np.ndarray,
    plan,
    tolerance: float = 0.05
) -> MUSanityResult:
    """
    Convenience function to compute MU sanity check directly from plan.

    Parameters
    ----------
    dose_calc : np.ndarray
        Calculated dose array
    dose_ref : np.ndarray
        Reference dose array
    grid_origin : np.ndarray
        Grid origin in mm [x, y, z]
    grid_spacing : np.ndarray
        Voxel spacing in mm [x, y, z]
    plan : IMRTPlan
        Loaded plan object
    tolerance : float
        Acceptable deviation (default 0.05 = 5%)

    Returns
    -------
    MUSanityResult
        MU sanity check result
    """
    isocenter, total_mu = extract_isocenter_from_plan(plan)

    return compute_mu_sanity_check(
        dose_calc=dose_calc,
        dose_ref=dose_ref,
        grid_origin=grid_origin,
        grid_spacing=grid_spacing,
        isocenter=isocenter,
        total_mu=total_mu,
        tolerance=tolerance
    )
