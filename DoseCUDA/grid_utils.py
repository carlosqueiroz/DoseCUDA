"""
Grid geometry utilities for dose calculation and resampling.

Provides standardized representation of 3D grid geometry (GridInfo)
and resampling operations for masks and dose distributions.

Key functions:
- GridInfo: Standardized grid representation
- resample_ct_to_isotropic(): Resample CT to isotropic spacing (required for dose engine)
- resample_mask_nearest(): Resample ROI masks to new grid
- resample_dose_linear(): Resample dose distributions
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Union

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    sitk = None


class GridInfo:
    """
    Standardized representation of 3D grid geometry.
    
    Used to represent CT, dose, and mask grids with consistent interface.
    
    Attributes
    ----------
    origin : np.ndarray
        Origin in mm, shape (3,) - [x, y, z]
    spacing : np.ndarray
        Spacing in mm, shape (3,) - [x, y, z]
    size : np.ndarray
        Size in voxels, shape (3,) - [nx, ny, nz]
    direction : np.ndarray
        Direction cosine matrix, shape (3, 3). Identity for axial orientation.
    frame_of_reference_uid : str, optional
        DICOM FrameOfReferenceUID for validation
        
    Notes
    -----
    - Follows SimpleITK convention: origin/spacing/size are (x,y,z)
    - Arrays are stored as (z,y,x) following numpy convention
    """
    
    def __init__(
        self,
        origin: np.ndarray,
        spacing: np.ndarray,
        size: np.ndarray,
        direction: Optional[np.ndarray] = None,
        frame_of_reference_uid: Optional[str] = None
    ):
        """
        Initialize GridInfo.
        
        Parameters
        ----------
        origin : array-like
            Origin in mm [x, y, z]
        spacing : array-like
            Spacing in mm [x, y, z]
        size : array-like
            Size in voxels [nx, ny, nz]
        direction : array-like, optional
            Direction cosine matrix (3x3). Default is identity (axial).
        frame_of_reference_uid : str, optional
            DICOM FrameOfReferenceUID
        """
        self.origin = np.array(origin, dtype=np.float32)
        self.spacing = np.array(spacing, dtype=np.float32)
        self.size = np.array(size, dtype=np.int32)
        
        if direction is None:
            self.direction = np.eye(3, dtype=np.float32)
        else:
            self.direction = np.array(direction, dtype=np.float32).reshape(3, 3)
        
        self.frame_of_reference_uid = frame_of_reference_uid or ""
        
        # Validate
        if self.origin.shape != (3,):
            raise ValueError(f"origin deve ter shape (3,), recebeu {self.origin.shape}")
        if self.spacing.shape != (3,):
            raise ValueError(f"spacing deve ter shape (3,), recebeu {self.spacing.shape}")
        if self.size.shape != (3,):
            raise ValueError(f"size deve ter shape (3,), recebeu {self.size.shape}")
        if self.direction.shape != (3, 3):
            raise ValueError(f"direction deve ter shape (3,3), recebeu {self.direction.shape}")
        
        if np.any(self.spacing <= 0):
            raise ValueError(f"spacing deve ser > 0, recebeu {self.spacing}")
        if np.any(self.size <= 0):
            raise ValueError(f"size deve ser > 0, recebeu {self.size}")
    
    def is_oblique(self, tolerance: float = 0.01) -> bool:
        """
        Check if grid has oblique orientation.
        
        Parameters
        ----------
        tolerance : float
            Tolerance for off-diagonal elements
            
        Returns
        -------
        bool
            True if grid is oblique (non-axial)
        """
        off_diag = np.abs(self.direction - np.eye(3))
        np.fill_diagonal(off_diag, 0.0)
        return np.max(off_diag) > tolerance
    
    def matches(
        self,
        other: 'GridInfo',
        origin_tol: float = 0.1,
        spacing_tol: float = 0.01,
        direction_tol: float = 0.01
    ) -> bool:
        """
        Check if this grid matches another grid geometry.
        
        Parameters
        ----------
        other : GridInfo
            Other grid to compare
        origin_tol : float
            Tolerance for origin in mm
        spacing_tol : float
            Tolerance for spacing in mm
        direction_tol : float
            Tolerance for direction cosines
            
        Returns
        -------
        bool
            True if grids match within tolerances
        """
        origin_match = np.allclose(self.origin, other.origin, atol=origin_tol)
        spacing_match = np.allclose(self.spacing, other.spacing, atol=spacing_tol)
        size_match = np.array_equal(self.size, other.size)
        direction_match = np.allclose(self.direction, other.direction, atol=direction_tol)
        
        return origin_match and spacing_match and size_match and direction_match
    
    def get_physical_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get physical bounding box in mm.
        
        Returns
        -------
        min_point : np.ndarray
            Minimum corner [x, y, z] in mm
        max_point : np.ndarray
            Maximum corner [x, y, z] in mm
        """
        min_point = self.origin
        max_point = self.origin + (self.size - 1) * self.spacing
        return min_point, max_point
    
    def voxel_volume(self) -> float:
        """
        Calculate voxel volume in cc.
        
        Returns
        -------
        float
            Volume in cc (cubic centimeters)
        """
        return np.prod(self.spacing) / 1000.0  # mm³ to cc
    
    def __repr__(self) -> str:
        return (
            f"GridInfo(origin={self.origin}, spacing={self.spacing}, "
            f"size={self.size}, oblique={self.is_oblique()})"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, GridInfo):
            return False
        return self.matches(other)
    
    @classmethod
    def from_sitk_image(cls, image) -> 'GridInfo':
        """
        Create GridInfo from SimpleITK image.
        
        Parameters
        ----------
        image : sitk.Image
            SimpleITK image
            
        Returns
        -------
        GridInfo
        """
        if not SITK_AVAILABLE:
            raise ImportError("SimpleITK necessário para from_sitk_image")
        
        origin = np.array(image.GetOrigin(), dtype=np.float32)
        spacing = np.array(image.GetSpacing(), dtype=np.float32)
        size = np.array(image.GetSize(), dtype=np.int32)
        
        direction_flat = np.array(image.GetDirection())
        direction = direction_flat.reshape(3, 3).astype(np.float32)
        
        return cls(origin, spacing, size, direction)
    
    def to_sitk_reference_image(self):
        """
        Create empty SimpleITK image with this grid geometry.
        
        Useful as reference for resampling operations.
        
        Returns
        -------
        sitk.Image
            Empty image with correct geometry
        """
        if not SITK_AVAILABLE:
            raise ImportError("SimpleITK necessário para to_sitk_reference_image")
        
        # SimpleITK size is (x, y, z) - same as our convention
        ref_img = sitk.Image(
            [int(self.size[0]), int(self.size[1]), int(self.size[2])],
            sitk.sitkFloat32
        )
        ref_img.SetOrigin(self.origin.tolist())
        ref_img.SetSpacing(self.spacing.tolist())
        ref_img.SetDirection(self.direction.flatten().tolist())
        
        return ref_img


def resample_ct_to_isotropic(
    hu_array: np.ndarray,
    source_grid: GridInfo,
    target_spacing_mm: float = 2.5,
    default_hu: float = -1000.0
) -> Tuple[np.ndarray, GridInfo]:
    """
    Resample CT to isotropic spacing required by dose calculation engine.
    
    DoseCUDA engine currently requires isotropic voxels (cubic). This function
    resamples anisotropic CT (e.g., 1×1×3 mm) to isotropic (e.g., 2.5×2.5×2.5 mm).
    
    Parameters
    ----------
    hu_array : np.ndarray
        HU array from CT, shape (nz, ny, nx)
    source_grid : GridInfo
        Original CT grid geometry
    target_spacing_mm : float
        Target isotropic spacing in mm (default 2.5 mm for clinical secondary check)
    default_hu : float
        Default HU for voxels outside original volume (default -1000 = air)
        
    Returns
    -------
    hu_resampled : np.ndarray
        Resampled HU array with isotropic spacing
    target_grid : GridInfo
        New grid geometry with isotropic spacing
        
    Notes
    -----
    - Uses linear interpolation for HU values
    - Preserves origin and direction from source
    - New size is calculated to cover same physical volume
    - This is REQUIRED before calling computeIMRTPlan() or computeIMPTPlan()
    
    Examples
    --------
    >>> # CT with anisotropic spacing (1×1×3 mm)
    >>> grid = GridInfo(origin=[0,0,0], spacing=[1.0, 1.0, 3.0], size=[512, 512, 100])
    >>> hu_iso, grid_iso = resample_ct_to_isotropic(hu_array, grid, target_spacing_mm=2.5)
    >>> # Result: isotropic 2.5×2.5×2.5 mm, size ~[205, 205, 120]
    """
    if not SITK_AVAILABLE:
        raise ImportError(
            "SimpleITK necessário para reamostrar CT. "
            "Instale com: pip install SimpleITK"
        )
    
    # Check if already isotropic
    if np.allclose(source_grid.spacing, target_spacing_mm, atol=0.01):
        print(f"CT já é isotrópico ({source_grid.spacing[0]:.2f} mm). Nenhuma reamostragem necessária.")
        return hu_array.copy(), source_grid
    
    print(f"\nReamostrando CT:")
    print(f"  Spacing original: {source_grid.spacing}")
    print(f"  Spacing alvo: [{target_spacing_mm}, {target_spacing_mm}, {target_spacing_mm}] mm")
    
    # Create source image
    hu_img = sitk.GetImageFromArray(hu_array.astype(np.float32))
    hu_img.SetOrigin(source_grid.origin.tolist())
    hu_img.SetSpacing(source_grid.spacing.tolist())
    hu_img.SetDirection(source_grid.direction.flatten().tolist())
    
    # Calculate new size to cover same physical extent
    # Physical size in each dimension
    physical_size_x = (source_grid.size[0] - 1) * source_grid.spacing[0]
    physical_size_y = (source_grid.size[1] - 1) * source_grid.spacing[1]
    physical_size_z = (source_grid.size[2] - 1) * source_grid.spacing[2]
    
    # New number of voxels needed
    new_nx = int(np.ceil(physical_size_x / target_spacing_mm)) + 1
    new_ny = int(np.ceil(physical_size_y / target_spacing_mm)) + 1
    new_nz = int(np.ceil(physical_size_z / target_spacing_mm)) + 1
    
    new_size = np.array([new_nx, new_ny, new_nz], dtype=int)
    
    # Create target grid
    target_spacing = np.array([target_spacing_mm, target_spacing_mm, target_spacing_mm])
    target_grid = GridInfo(
        origin=source_grid.origin,
        spacing=target_spacing,
        size=new_size,
        direction=source_grid.direction,
        frame_of_reference_uid=source_grid.frame_of_reference_uid
    )
    
    # Resample with linear interpolation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_grid.to_sitk_reference_image())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_hu))
    
    hu_resampled_img = resampler.Execute(hu_img)
    hu_resampled = sitk.GetArrayFromImage(hu_resampled_img)
    
    # Validate
    volume_original = np.prod(source_grid.size) * source_grid.voxel_volume()
    volume_resampled = np.prod(target_grid.size) * target_grid.voxel_volume()
    volume_diff_percent = abs(volume_resampled - volume_original) / volume_original * 100
    
    print(f"  Size original: {source_grid.size}")
    print(f"  Size novo: {target_grid.size}")
    print(f"  Volume: {volume_original:.1f} cc → {volume_resampled:.1f} cc ({volume_diff_percent:+.1f}%)")
    
    if volume_diff_percent > 5.0:
        warnings.warn(
            f"Reamostragem alterou volume em {volume_diff_percent:.1f}%. "
            "Verifique se target_spacing é apropriado."
        )
    
    return hu_resampled, target_grid


def resample_mask_nearest(
    mask: np.ndarray,
    source_grid: GridInfo,
    target_grid: GridInfo
) -> np.ndarray:
    """
    Resample binary mask from source grid to target grid using nearest neighbor.
    
    Nearest neighbor interpolation is appropriate for binary masks to avoid
    partial volume artifacts at boundaries.
    
    Parameters
    ----------
    mask : np.ndarray
        Source mask, shape (nz, ny, nx), boolean or 0/1 integer
    source_grid : GridInfo
        Grid geometry of source mask
    target_grid : GridInfo
        Target grid geometry
        
    Returns
    -------
    mask_resampled : np.ndarray
        Resampled mask on target grid, shape (target_nz, target_ny, target_nx), bool
        
    Notes
    -----
    - Uses SimpleITK if available (more accurate with direction handling)
    - Falls back to manual nearest neighbor if SimpleITK not available
    - Preserves binary nature of mask (no fractional values)
    
    Examples
    --------
    >>> # Resample PTV mask from CT grid to dose grid
    >>> dose_grid_info = GridInfo.from_sitk_image(dose_img)
    >>> ct_grid_info = GridInfo.from_sitk_image(ct_img)
    >>> ptv_mask_dose = resample_mask_nearest(ptv_mask_ct, ct_grid_info, dose_grid_info)
    """
    # Convert to boolean
    mask_bool = mask.astype(bool)
    
    # Check if grids already match
    if source_grid.matches(target_grid):
        return mask_bool.copy()
    
    if SITK_AVAILABLE:
        # Use SimpleITK (more accurate, handles direction properly)
        mask_img = sitk.GetImageFromArray(mask_bool.astype(np.uint8))
        mask_img.SetOrigin(source_grid.origin.tolist())
        mask_img.SetSpacing(source_grid.spacing.tolist())
        mask_img.SetDirection(source_grid.direction.flatten().tolist())
        
        # Create reference image with target geometry
        ref_img = target_grid.to_sitk_reference_image()
        
        # Resample with nearest neighbor
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        
        mask_resampled_img = resampler.Execute(mask_img)
        mask_resampled = sitk.GetArrayFromImage(mask_resampled_img).astype(bool)
        
    else:
        # Fallback: manual nearest neighbor (assumes no oblique orientation)
        if source_grid.is_oblique() or target_grid.is_oblique():
            raise ValueError(
                "Reamostrar máscaras com orientação oblíqua requer SimpleITK. "
                "Instale com: pip install SimpleITK"
            )
        
        # Create coordinate grids for target
        z_target = np.arange(target_grid.size[2]) * target_grid.spacing[2] + target_grid.origin[2]
        y_target = np.arange(target_grid.size[1]) * target_grid.spacing[1] + target_grid.origin[1]
        x_target = np.arange(target_grid.size[0]) * target_grid.spacing[0] + target_grid.origin[0]
        
        # Convert to indices in source grid
        z_source_idx = np.round((z_target[:, None, None] - source_grid.origin[2]) / source_grid.spacing[2]).astype(int)
        y_source_idx = np.round((y_target[None, :, None] - source_grid.origin[1]) / source_grid.spacing[1]).astype(int)
        x_source_idx = np.round((x_target[None, None, :] - source_grid.origin[0]) / source_grid.spacing[0]).astype(int)
        
        # Clip to valid range
        z_source_idx = np.clip(z_source_idx, 0, source_grid.size[2] - 1)
        y_source_idx = np.clip(y_source_idx, 0, source_grid.size[1] - 1)
        x_source_idx = np.clip(x_source_idx, 0, source_grid.size[0] - 1)
        
        # Sample from source mask
        mask_resampled = mask_bool[z_source_idx, y_source_idx, x_source_idx]
    
    # Log volume change
    source_volume = np.sum(mask_bool) * source_grid.voxel_volume()
    target_volume = np.sum(mask_resampled) * target_grid.voxel_volume()
    volume_diff_percent = 100.0 * (target_volume - source_volume) / source_volume if source_volume > 0 else 0
    
    if abs(volume_diff_percent) > 5.0:
        warnings.warn(
            f"Reamostragem de máscara alterou volume em {volume_diff_percent:+.1f}% "
            f"({source_volume:.2f} cc → {target_volume:.2f} cc). "
            "Verifique se a geometria dos grids está correta."
        )
    
    return mask_resampled


def resample_dose_linear(
    dose: np.ndarray,
    source_grid: GridInfo,
    target_grid: GridInfo
) -> np.ndarray:
    """
    Resample dose distribution from source grid to target grid using linear interpolation.
    
    Linear (trilinear) interpolation is appropriate for dose distributions
    to preserve smoothness and gradients.
    
    Parameters
    ----------
    dose : np.ndarray
        Source dose, shape (nz, ny, nx) in Gy
    source_grid : GridInfo
        Grid geometry of source dose
    target_grid : GridInfo
        Target grid geometry
        
    Returns
    -------
    dose_resampled : np.ndarray
        Resampled dose on target grid, shape (target_nz, target_ny, target_nx) in Gy
        
    Notes
    -----
    - Requires SimpleITK for accurate interpolation with direction handling
    - Use this for resampling reference RTDOSE to calculated dose grid
    
    Examples
    --------
    >>> # Resample reference dose to calculated dose grid for comparison
    >>> calc_grid_info = GridInfo(calc_origin, calc_spacing, calc_size)
    >>> ref_grid_info = GridInfo(ref_origin, ref_spacing, ref_size)
    >>> ref_dose_resampled = resample_dose_linear(ref_dose, ref_grid_info, calc_grid_info)
    """
    if not SITK_AVAILABLE:
        raise ImportError(
            "SimpleITK necessário para reamostrar dose. "
            "Instale com: pip install SimpleITK"
        )
    
    # Check if grids already match
    if source_grid.matches(target_grid):
        return dose.copy()
    
    # Create SimpleITK image for source dose
    dose_img = sitk.GetImageFromArray(dose.astype(np.float32))
    dose_img.SetOrigin(source_grid.origin.tolist())
    dose_img.SetSpacing(source_grid.spacing.tolist())
    dose_img.SetDirection(source_grid.direction.flatten().tolist())
    
    # Create reference image with target geometry
    ref_img = target_grid.to_sitk_reference_image()
    
    # Resample with linear interpolation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    
    dose_resampled_img = resampler.Execute(dose_img)
    dose_resampled = sitk.GetArrayFromImage(dose_resampled_img)
    
    return dose_resampled


def validate_frame_of_reference(
    grid1: GridInfo,
    grid2: GridInfo,
    grid1_name: str = "Grid 1",
    grid2_name: str = "Grid 2",
    strict: bool = False
) -> bool:
    """
    Validate that two grids have matching FrameOfReferenceUID.
    
    Parameters
    ----------
    grid1, grid2 : GridInfo
        Grids to compare
    grid1_name, grid2_name : str
        Names for logging
    strict : bool
        If True, raise error on mismatch. If False, only warn.
        
    Returns
    -------
    bool
        True if FrameOfReferenceUIDs match (or both empty)
    """
    uid1 = grid1.frame_of_reference_uid
    uid2 = grid2.frame_of_reference_uid
    
    if not uid1 or not uid2:
        msg = (
            f"{grid1_name} e/ou {grid2_name} não têm FrameOfReferenceUID. "
            "Não é possível validar referência geométrica."
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
        return False
    
    if uid1 != uid2:
        msg = (
            f"FrameOfReferenceUID não coincide:\n"
            f"  {grid1_name}: {uid1}\n"
            f"  {grid2_name}: {uid2}"
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
        return False
    
    return True
