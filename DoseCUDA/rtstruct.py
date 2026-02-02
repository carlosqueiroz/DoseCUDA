"""
RTSTRUCT import and rasterization for clinical secondary dose calculation.

This module provides robust reading of DICOM RTSTRUCT files and rasterization
of ROI contours to 3D binary masks aligned to CT/dose grids.

Based on OpenTPS dicomIO.py for DICOM parsing best practices.
"""

import numpy as np
import pydicom as pyd
import warnings
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ContourSlice:
    """Single contour polygon on one CT slice."""
    points: np.ndarray  # Shape (N, 3) - XYZ coordinates in mm
    z_position: float   # Z coordinate of this slice
    referenced_sop_instance_uid: Optional[str] = None


@dataclass
class ROI:
    """Region of Interest with all its contour slices."""
    name: str
    roi_number: int
    display_color: Tuple[int, int, int]
    contour_slices: List[ContourSlice] = field(default_factory=list)
    referenced_frame_of_reference_uid: Optional[str] = None
    
    def get_bounding_box_mm(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of ROI in mm coordinates.
        
        Returns
        -------
        min_coords, max_coords : np.ndarray
            Arrays of shape (3,) with [x_min, y_min, z_min] and [x_max, y_max, z_max]
        """
        if len(self.contour_slices) == 0:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        
        all_points = np.vstack([cs.points for cs in self.contour_slices])
        return np.min(all_points, axis=0), np.max(all_points, axis=0)


@dataclass
class RTStruct:
    """DICOM RTSTRUCT dataset."""
    name: str
    series_instance_uid: str
    frame_of_reference_uid: Optional[str]
    rois: Dict[str, ROI] = field(default_factory=dict)  # name -> ROI
    
    def get_roi_names(self) -> List[str]:
        """Get list of all ROI names in this structure set."""
        return list(self.rois.keys())


def read_rtstruct(rtstruct_path: str) -> RTStruct:
    """
    Read DICOM RTSTRUCT file and parse ROI definitions and contours.
    
    Robust parsing following clinical best practices:
    - Maps StructureSetROISequence to ROIContourSequence via ROINumber
    - Handles missing ContourSequence gracefully
    - Validates polygon point counts
    - Extracts ReferencedSOPInstanceUID when available
    
    Parameters
    ----------
    rtstruct_path : str
        Path to RTSTRUCT DICOM file
        
    Returns
    -------
    rtstruct : RTStruct
        Parsed structure set with all ROIs
        
    Raises
    ------
    ValueError
        If RTSTRUCT file is invalid or missing required sequences
    """
    dcm = pyd.dcmread(rtstruct_path, force=True)
    
    # Validate required sequences
    if not hasattr(dcm, 'StructureSetROISequence'):
        raise ValueError(f"RTSTRUCT inválido: StructureSetROISequence ausente em {rtstruct_path}")
    
    if not hasattr(dcm, 'ROIContourSequence'):
        raise ValueError(f"RTSTRUCT inválido: ROIContourSequence ausente em {rtstruct_path}")
    
    # Get metadata
    name = dcm.SeriesDescription if hasattr(dcm, 'SeriesDescription') else dcm.SeriesInstanceUID
    series_uid = dcm.SeriesInstanceUID if hasattr(dcm, 'SeriesInstanceUID') else ""
    frame_of_ref_uid = dcm.FrameOfReferenceUID if hasattr(dcm, 'FrameOfReferenceUID') else None
    
    rtstruct = RTStruct(
        name=name,
        series_instance_uid=series_uid,
        frame_of_reference_uid=frame_of_ref_uid
    )
    
    # Build mapping: ROINumber -> ROI metadata
    roi_metadata = {}
    for struct_roi in dcm.StructureSetROISequence:
        roi_number = int(struct_roi.ROINumber)
        roi_name = struct_roi.ROIName
        ref_frame_uid = struct_roi.ReferencedFrameOfReferenceUID if hasattr(struct_roi, 'ReferencedFrameOfReferenceUID') else None
        
        roi_metadata[roi_number] = {
            'name': roi_name,
            'referenced_frame_of_reference_uid': ref_frame_uid
        }
    
    # Parse contours for each ROI
    for roi_contour in dcm.ROIContourSequence:
        ref_roi_number = int(roi_contour.ReferencedROINumber)
        
        if ref_roi_number not in roi_metadata:
            warnings.warn(
                f"ROIContour referencia ROINumber={ref_roi_number} que não existe em StructureSetROISequence. "
                "Pulando este contorno."
            )
            continue
        
        metadata = roi_metadata[ref_roi_number]
        
        # Get display color (default to white if not present)
        if hasattr(roi_contour, 'ROIDisplayColor'):
            color = tuple(int(c) for c in roi_contour.ROIDisplayColor)
        else:
            color = (255, 255, 255)
            warnings.warn(f"ROI '{metadata['name']}': ROIDisplayColor ausente, usando branco.")
        
        # Check if contours exist
        if not hasattr(roi_contour, 'ContourSequence'):
            warnings.warn(
                f"ROI '{metadata['name']}': ContourSequence ausente. "
                "Esta estrutura não possui contornos. Pulando."
            )
            continue
        
        # Create ROI object
        roi = ROI(
            name=metadata['name'],
            roi_number=ref_roi_number,
            display_color=color,
            referenced_frame_of_reference_uid=metadata['referenced_frame_of_reference_uid']
        )
        
        # Parse each contour slice
        for contour in roi_contour.ContourSequence:
            if not hasattr(contour, 'ContourData'):
                warnings.warn(f"ROI '{metadata['name']}': Contour sem ContourData. Pulando.")
                continue
            
            contour_data = contour.ContourData
            
            # ContourData is flat list: [x1,y1,z1, x2,y2,z2, ...]
            if len(contour_data) < 9:  # At least 3 points (3 coords each)
                warnings.warn(
                    f"ROI '{metadata['name']}': Contour com menos de 3 pontos ({len(contour_data)//3}). "
                    "Pulando este polígono."
                )
                continue
            
            if len(contour_data) % 3 != 0:
                warnings.warn(
                    f"ROI '{metadata['name']}': ContourData com tamanho não múltiplo de 3 ({len(contour_data)}). "
                    "Dados corrompidos? Pulando."
                )
                continue
            
            # Reshape to (N, 3)
            points = np.array(contour_data).reshape(-1, 3)
            
            # Get Z position (should be constant within tolerance)
            z_values = points[:, 2]
            z_mean = np.mean(z_values)
            z_std = np.std(z_values)
            
            if z_std > 0.1:  # Tolerance 0.1 mm
                warnings.warn(
                    f"ROI '{metadata['name']}': Contour com Z variável (std={z_std:.3f} mm). "
                    "Esperado contorno planar. Usando Z médio."
                )
            
            # Get referenced SOP instance (if available)
            ref_sop_uid = None
            if hasattr(contour, 'ContourImageSequence') and len(contour.ContourImageSequence) > 0:
                ref_sop_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            
            contour_slice = ContourSlice(
                points=points,
                z_position=z_mean,
                referenced_sop_instance_uid=ref_sop_uid
            )
            
            roi.contour_slices.append(contour_slice)
        
        # Add ROI to structure set if it has valid contours
        if len(roi.contour_slices) > 0:
            rtstruct.rois[roi.name] = roi
        else:
            warnings.warn(
                f"ROI '{metadata['name']}': Nenhum contorno válido encontrado. "
                "ROI não será incluído no structure set."
            )
    
    return rtstruct


def rasterize_roi_to_mask(
    roi: ROI,
    origin: np.ndarray,
    spacing: np.ndarray,
    size: np.ndarray,
    direction: Optional[np.ndarray] = None,
    ct_z_positions: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Rasterize ROI contours to binary mask on specified grid.
    
    Converts contour polygons from mm coordinates to voxel indices and fills
    each polygon slice-by-slice using point-in-polygon algorithm.
    
    Parameters
    ----------
    roi : ROI
        ROI with contour slices to rasterize
    origin : np.ndarray
        Grid origin in mm, shape (3,) - [x, y, z]
    spacing : np.ndarray
        Voxel spacing in mm, shape (3,) - [x, y, z]
    size : np.ndarray
        Grid size in voxels, shape (3,) - [z, y, x] (array ordering)
    direction : np.ndarray, optional
        Direction cosine matrix (3x3). If None or identity, assumes axial orientation.
        Non-axial orientations are not currently supported.
    ct_z_positions : np.ndarray, optional
        Actual Z positions of CT slices in mm (from ImagePositionPatient).
        If provided, uses nearest-neighbor matching instead of simple rounding.
        This is more robust for irregular slice spacing.
        
    Returns
    -------
    mask : np.ndarray
        Binary mask with shape (z, y, x), dtype bool
        
    Raises
    ------
    ValueError
        If direction matrix indicates oblique orientation (not supported)
        
    Notes
    -----
    - Uses matplotlib.path.Path for point-in-polygon testing (fast and robust)
    - Multiple contours on same slice are combined with OR
    - Bounding box optimization limits polygon fill to relevant region
    - LIMITATION: Inner contours (holes) are not supported. All contours are OR'd.
      For structures with holes, the volume will be overestimated.
    """
    # Validate direction (must be axial or None)
    if direction is not None:
        identity = np.eye(3)
        if np.max(np.abs(direction - identity)) > 0.01:
            raise ValueError(
                "Rasterização de ROI não suportada para CT oblíquo. "
                f"Direction matrix:\n{direction}\n"
                "Por favor, reoriente o CT para axial antes de rasterizar estruturas."
            )
    
    # Initialize mask
    mask = np.zeros(size, dtype=bool)
    
    if len(roi.contour_slices) == 0:
        warnings.warn(f"ROI '{roi.name}': Nenhum contorno para rasterizar.")
        return mask
    
    # Import matplotlib.path for polygon filling
    try:
        from matplotlib.path import Path
    except ImportError:
        raise ImportError(
            "matplotlib é necessário para rasterização de ROI. "
            "Instale com: pip install matplotlib"
        )
    
    # Process each contour slice
    for contour_slice in roi.contour_slices:
        points_mm = contour_slice.points  # (N, 3) in mm
        
        # Convert mm to voxel indices
        # Formula: index = (coord_mm - origin) / spacing
        # Note: array indexing is [z, y, x] but coordinates are [x, y, z]
        points_voxel = (points_mm - origin) / spacing
        
        # Extract x, y indices (in-plane) and z index (slice)
        x_indices = points_voxel[:, 0]
        y_indices = points_voxel[:, 1]
        z_indices = points_voxel[:, 2]
        
        # Determine which slice this contour belongs to
        # IMPROVED: Use nearest-neighbor matching if CT Z positions provided
        if ct_z_positions is not None:
            # Find nearest CT slice by actual Z position
            z_contour_mm = contour_slice.z_position
            z_diffs = np.abs(ct_z_positions - z_contour_mm)
            k = int(np.argmin(z_diffs))
            
            # Warn if contour is far from nearest slice
            min_diff = z_diffs[k]
            if min_diff > spacing[2] * 0.6:  # More than 60% of slice spacing
                warnings.warn(
                    f"ROI '{roi.name}': Contour em Z={z_contour_mm:.2f} mm "
                    f"está {min_diff:.2f} mm do slice mais próximo (k={k}, Z={ct_z_positions[k]:.2f} mm). "
                    "Isso pode indicar RTSTRUCT desalinhado do CT."
                )
        else:
            # Fallback: simple rounding (works for regular spacing)
            z_mean = np.mean(z_indices)
            k = int(np.round(z_mean))
        
        # Check if slice is within bounds
        if k < 0 or k >= size[0]:
            warnings.warn(
                f"ROI '{roi.name}': Contour em Z={contour_slice.z_position:.2f} mm "
                f"(slice índice {k}) está fora do grid (tamanho Z: {size[0]}). Pulando."
            )
            continue
        
        # Get bounding box in voxel coordinates
        x_min_voxel = np.floor(np.min(x_indices)).astype(int)
        x_max_voxel = np.ceil(np.max(x_indices)).astype(int)
        y_min_voxel = np.floor(np.min(y_indices)).astype(int)
        y_max_voxel = np.ceil(np.max(y_indices)).astype(int)
        
        # Clip to grid bounds
        x_min_voxel = max(0, x_min_voxel)
        x_max_voxel = min(size[2], x_max_voxel + 1)  # +1 because range is exclusive
        y_min_voxel = max(0, y_min_voxel)
        y_max_voxel = min(size[1], y_max_voxel + 1)
        
        if x_min_voxel >= x_max_voxel or y_min_voxel >= y_max_voxel:
            warnings.warn(
                f"ROI '{roi.name}': Contour em slice {k} tem bounding box vazio. Pulando."
            )
            continue
        
        # Create polygon path
        polygon_points = np.column_stack([x_indices, y_indices])
        path = Path(polygon_points)
        
        # Generate grid of pixel centers in bounding box
        x_coords = np.arange(x_min_voxel, x_max_voxel) + 0.5
        y_coords = np.arange(y_min_voxel, y_max_voxel) + 0.5
        
        # Create meshgrid
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Flatten for point-in-polygon test
        points_to_test = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Test which points are inside polygon
        inside = path.contains_points(points_to_test)
        
        # Reshape back to 2D
        inside_2d = inside.reshape(len(y_coords), len(x_coords))
        
        # Write to mask (OR operation if multiple contours on same slice)
        mask[k, y_min_voxel:y_max_voxel, x_min_voxel:x_max_voxel] |= inside_2d
    
    return mask


def validate_rtstruct_with_ct(
    rtstruct: RTStruct,
    ct_frame_of_reference_uid: str,
    strict: bool = False
) -> bool:
    """
    Validate that RTSTRUCT references the same frame of reference as CT.
    
    Parameters
    ----------
    rtstruct : RTStruct
        Structure set to validate
    ct_frame_of_reference_uid : str
        FrameOfReferenceUID from CT
    strict : bool
        If True, raises ValueError on mismatch. If False, only warns.
        
    Returns
    -------
    valid : bool
        True if FrameOfReferenceUIDs match (or if either is missing)
        
    Raises
    ------
    ValueError
        If strict=True and UIDs don't match
    """
    if not rtstruct.frame_of_reference_uid or not ct_frame_of_reference_uid:
        warnings.warn(
            "Não foi possível validar FrameOfReferenceUID: "
            f"RTSTRUCT='{rtstruct.frame_of_reference_uid}', CT='{ct_frame_of_reference_uid}'. "
            "Pelo menos um está ausente."
        )
        return True  # Can't validate, assume OK
    
    if rtstruct.frame_of_reference_uid != ct_frame_of_reference_uid:
        msg = (
            f"RTSTRUCT e CT têm FrameOfReferenceUID diferentes:\n"
            f"  RTSTRUCT: {rtstruct.frame_of_reference_uid}\n"
            f"  CT: {ct_frame_of_reference_uid}\n"
            "Isto pode indicar que as estruturas não correspondem a este CT. "
            "Prossiga com cautela."
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
            return False
    
    return True
