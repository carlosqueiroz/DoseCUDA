"""
DVH (Dose-Volume Histogram) calculation and dose metrics for secondary check.

Provides differential and cumulative DVH calculation, plus standard metrics
like Dmean, Dmax, D95, D98, V20, etc.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union

try:
    import pydicom as pyd
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    pyd = None

try:
    from .grid_utils import GridInfo, resample_dose_linear
    GRID_UTILS_AVAILABLE = True
except ImportError:
    GRID_UTILS_AVAILABLE = False
    GridInfo = None
    resample_dose_linear = None

try:
    from .grid_utils import GridInfo, resample_dose_linear
    GRID_UTILS_AVAILABLE = True
except ImportError:
    GRID_UTILS_AVAILABLE = False
    GridInfo = None
    resample_dose_linear = None


def compute_dvh(
    dose: np.ndarray,
    mask: np.ndarray,
    voxel_volume: float,
    bin_width: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute differential and cumulative dose-volume histogram.
    
    Parameters
    ----------
    dose : np.ndarray
        3D dose array in Gy, shape (z, y, x)
    mask : np.ndarray
        3D boolean mask, shape (z, y, x). True = voxel in structure
    voxel_volume : float
        Volume of each voxel in cc (= spacing_x * spacing_y * spacing_z / 1000)
    bin_width : float
        Width of dose bins in Gy
        
    Returns
    -------
    dose_bins : np.ndarray
        Dose bin centers in Gy
    differential_dvh : np.ndarray
        Volume (cc) in each dose bin
    cumulative_dvh : np.ndarray
        Volume (cc) receiving at least this dose
        
    Notes
    -----
    - Empty masks return empty arrays
    - Dose values are rounded to bin_width precision
    """
    if not np.any(mask):
        warnings.warn("Máscara vazia: não há voxels na estrutura. DVH vazio.")
        return np.array([]), np.array([]), np.array([])
    
    # Extract doses in masked region
    doses_in_roi = dose[mask]
    
    if len(doses_in_roi) == 0:
        warnings.warn("Nenhum voxel na máscara após aplicação. DVH vazio.")
        return np.array([]), np.array([]), np.array([])
    
    # Determine dose range
    dose_min = np.min(doses_in_roi)
    dose_max = np.max(doses_in_roi)
    
    # Create bins
    n_bins = int(np.ceil((dose_max - dose_min) / bin_width)) + 1
    bins = np.linspace(dose_min, dose_max + bin_width, n_bins + 1)
    
    # Compute histogram
    counts, bin_edges = np.histogram(doses_in_roi, bins=bins)
    
    # Convert counts to volumes
    differential_dvh = counts * voxel_volume
    
    # Compute cumulative DVH (volume receiving >= dose)
    cumulative_dvh = np.cumsum(differential_dvh[::-1])[::-1]
    
    # Bin centers
    dose_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    return dose_bins, differential_dvh, cumulative_dvh


def compute_metrics(
    dose: np.ndarray,
    mask: np.ndarray,
    spacing: np.ndarray,
    metrics_spec: Optional[Dict[str, Union[float, List[float]]]] = None
) -> Dict[str, float]:
    """
    Compute standard dose metrics for a structure.
    
    Parameters
    ----------
    dose : np.ndarray
        3D dose array in Gy, shape (z, y, x)
    mask : np.ndarray
        3D boolean mask, shape (z, y, x)
    spacing : np.ndarray
        Voxel spacing in mm, shape (3,) - [x, y, z] or [z, y, x]
    metrics_spec : dict, optional
        Specification of metrics to compute. If None, computes default set.
        Examples:
            {'D_percent': [2, 50, 95, 98],  # D2%, D50%, D95%, D98%
             'V_dose': [20, 30, 40]}         # V20Gy, V30Gy, V40Gy
        
    Returns
    -------
    metrics : dict
        Dictionary with metric names and values
        Always includes: 'Dmean', 'Dmax', 'Dmin', 'Volume_cc'
        Plus any requested D% or V values
        
    Notes
    -----
    - D_percent: dose received by X% of volume (sorted high to low)
      E.g., D2% = dose received by 2% of hottest volume
      E.g., D95% = dose covering 95% of volume (near-minimum dose)
    - V_dose: percentage of volume receiving >= specified dose
      E.g., V20 = percent of volume receiving >= 20 Gy
    """
    if not np.any(mask):
        warnings.warn("Máscara vazia: métricas não podem ser calculadas.")
        return {
            'Dmean': np.nan,
            'Dmax': np.nan,
            'Dmin': np.nan,
            'Volume_cc': 0.0
        }
    
    # Calculate voxel volume in cc
    if len(spacing) == 3:
        # Assume spacing is [x, y, z] or uniform
        voxel_volume = np.prod(spacing) / 1000.0  # mm³ to cc
    else:
        warnings.warn(f"Spacing com formato inesperado: {spacing}. Assumindo isotrópico.")
        voxel_volume = spacing[0]**3 / 1000.0
    
    # Extract doses in ROI
    doses_in_roi = dose[mask]
    n_voxels = len(doses_in_roi)
    total_volume_cc = n_voxels * voxel_volume
    
    # Basic metrics
    metrics = {
        'Dmean': float(np.mean(doses_in_roi)),
        'Dmax': float(np.max(doses_in_roi)),
        'Dmin': float(np.min(doses_in_roi)),
        'Volume_cc': float(total_volume_cc)
    }
    
    # Default metrics if none specified
    if metrics_spec is None:
        metrics_spec = {
            'D_percent': [2, 50, 95, 98],
            'V_dose': []
        }
    
    # Compute D_percent metrics (dose percentiles)
    if 'D_percent' in metrics_spec:
        # Sort doses in descending order
        doses_sorted = np.sort(doses_in_roi)[::-1]
        
        for percent in metrics_spec['D_percent']:
            if percent < 0 or percent > 100:
                warnings.warn(f"Percentil {percent}% inválido (deve estar em [0, 100]). Pulando.")
                continue
            
            # D_percent means "dose to X% of volume"
            # For D2%: dose received by hottest 2% (high dose)
            # For D98%: dose covering 98% of volume (near minimum)
            index = int(np.floor(percent / 100.0 * n_voxels))
            index = min(index, n_voxels - 1)  # Clamp to valid range
            
            metric_name = f'D{percent}%'
            metrics[metric_name] = float(doses_sorted[index])
    
    # Compute V_dose metrics (volume at dose)
    if 'V_dose' in metrics_spec:
        for dose_threshold in metrics_spec['V_dose']:
            # V_dose = percentage of volume receiving >= dose_threshold
            n_above = np.sum(doses_in_roi >= dose_threshold)
            percent_above = (n_above / n_voxels) * 100.0
            
            metric_name = f'V{dose_threshold}Gy'
            metrics[metric_name] = float(percent_above)
    
    return metrics


def compare_dvh_metrics(
    metrics_calculated: Dict[str, float],
    metrics_reference: Dict[str, float],
    tolerance_abs: float = 0.5,
    tolerance_rel: float = 0.03
) -> Dict[str, Dict[str, Union[float, bool]]]:
    """
    Compare calculated metrics against reference with tolerances.
    
    Parameters
    ----------
    metrics_calculated : dict
        Metrics from secondary calculation
    metrics_reference : dict
        Metrics from primary TPS
    tolerance_abs : float
        Absolute tolerance in Gy (for dose metrics)
    tolerance_rel : float
        Relative tolerance as fraction (for volume metrics and large doses)
        
    Returns
    -------
    comparison : dict
        For each metric, provides:
        - 'calculated': calculated value
        - 'reference': reference value
        - 'diff': calculated - reference
        - 'diff_percent': (calculated - reference) / reference * 100
        - 'pass': bool, True if within tolerance
        
    Notes
    -----
    Tolerance criteria:
    - Dose metrics: pass if |diff| < tolerance_abs OR |diff_rel| < tolerance_rel
    - Volume metrics: pass if |diff| < tolerance_rel * 100 (percent points)
    """
    comparison = {}
    
    for metric_name in metrics_calculated.keys():
        if metric_name not in metrics_reference:
            warnings.warn(f"Métrica '{metric_name}' não encontrada na referência. Pulando comparação.")
            continue
        
        calc_value = metrics_calculated[metric_name]
        ref_value = metrics_reference[metric_name]
        
        diff = calc_value - ref_value
        diff_percent = (diff / ref_value * 100.0) if ref_value != 0 else np.nan
        
        # Determine if passes tolerance
        if 'Volume_cc' in metric_name or metric_name.startswith('V'):
            # Volume metric: use relative tolerance
            passes = abs(diff_percent) < (tolerance_rel * 100)
        else:
            # Dose metric: use absolute OR relative
            passes = (abs(diff) < tolerance_abs) or (abs(diff_percent) < (tolerance_rel * 100))
        
        comparison[metric_name] = {
            'calculated': calc_value,
            'reference': ref_value,
            'diff': diff,
            'diff_percent': diff_percent,
            'pass': passes
        }
    
    return comparison


def generate_dvh_report(
    roi_name: str,
    metrics: Dict[str, float],
    comparison: Optional[Dict[str, Dict]] = None
) -> str:
    """
    Generate human-readable DVH metrics report.
    
    Parameters
    ----------
    roi_name : str
        Name of ROI
    metrics : dict
        Calculated metrics
    comparison : dict, optional
        Comparison results from compare_dvh_metrics()
        
    Returns
    -------
    report : str
        Formatted text report
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"DVH Metrics Report: {roi_name}")
    lines.append(f"{'='*60}")
    
    # Volume
    if 'Volume_cc' in metrics:
        lines.append(f"\nVolume: {metrics['Volume_cc']:.2f} cc")
    
    # Basic dose metrics
    lines.append(f"\nDose Statistics:")
    lines.append(f"  Dmean: {metrics.get('Dmean', np.nan):.2f} Gy")
    lines.append(f"  Dmax:  {metrics.get('Dmax', np.nan):.2f} Gy")
    lines.append(f"  Dmin:  {metrics.get('Dmin', np.nan):.2f} Gy")
    
    # D_percent metrics
    d_metrics = [k for k in metrics.keys() if k.startswith('D') and '%' in k]
    if d_metrics:
        lines.append(f"\nDose Coverage:")
        for metric_name in sorted(d_metrics):
            value = metrics[metric_name]
            lines.append(f"  {metric_name}: {value:.2f} Gy")
    
    # V_dose metrics
    v_metrics = [k for k in metrics.keys() if k.startswith('V') and 'Gy' in k]
    if v_metrics:
        lines.append(f"\nVolume at Dose:")
        for metric_name in sorted(v_metrics):
            value = metrics[metric_name]
            lines.append(f"  {metric_name}: {value:.1f}%")
    
    # Comparison if available
    if comparison:
        lines.append(f"\n{'='*60}")
        lines.append(f"Comparison vs Reference:")
        lines.append(f"{'='*60}")
        
        all_pass = all(comp['pass'] for comp in comparison.values())
        status = "PASS ✓" if all_pass else "FAIL ✗"
        lines.append(f"\nOverall: {status}\n")
        
        for metric_name, comp in sorted(comparison.items()):
            status_symbol = "✓" if comp['pass'] else "✗"
            lines.append(
                f"  {status_symbol} {metric_name:12s}: "
                f"Calc={comp['calculated']:7.2f}, "
                f"Ref={comp['reference']:7.2f}, "
                f"Diff={comp['diff']:+6.2f} ({comp['diff_percent']:+5.1f}%)"
            )
    
    lines.append(f"\n{'='*60}\n")
    
    return '\n'.join(lines)


def read_reference_rtdose(rtdose_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read reference RTDOSE from primary TPS for comparison.
    
    Parameters
    ----------
    rtdose_path : str
        Path to RTDOSE DICOM file
        
    Returns
    -------
    dose : np.ndarray
        Dose array in Gy, shape (z, y, x)
    origin : np.ndarray
        Origin in mm, shape (3,)
    spacing : np.ndarray
        Spacing in mm, shape (3,)
    frame_of_reference_uid : str
        FrameOfReferenceUID for validation
        
    Raises
    ------
    ImportError
        If pydicom is not available
    ValueError
        If RTDOSE file is invalid or missing required fields
        
    Notes
    -----
    - DoseGridScaling is applied to convert pixel values to Gy
    - GridFrameOffsetVector is used to determine Z spacing
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom é necessário para ler RTDOSE. "
            "Instale com: pip install pydicom"
        )
    
    # Read DICOM
    dcm = pyd.dcmread(rtdose_path, force=True)
    
    # Validate modality
    if not hasattr(dcm, 'Modality') or dcm.Modality != 'RTDOSE':
        raise ValueError(f"Arquivo não é RTDOSE (Modality={getattr(dcm, 'Modality', 'desconhecido')})")
    
    # Get dose grid scaling
    if not hasattr(dcm, 'DoseGridScaling'):
        raise ValueError("RTDOSE inválido: DoseGridScaling ausente")
    
    dose_scaling = float(dcm.DoseGridScaling)
    
    # Get pixel array and convert to Gy
    if not hasattr(dcm, 'pixel_array'):
        raise ValueError("RTDOSE inválido: pixel_array ausente")
    
    # pixel_array is typically uint16 or uint32
    dose = dcm.pixel_array.astype(np.float32) * dose_scaling
    
    # Ensure shape is (z, y, x)
    if dose.ndim != 3:
        raise ValueError(f"RTDOSE com dimensões inválidas: {dose.shape} (esperado 3D)")
    
    # Get origin (ImagePositionPatient)
    if not hasattr(dcm, 'ImagePositionPatient'):
        raise ValueError("RTDOSE inválido: ImagePositionPatient ausente")
    
    origin = np.array([
        float(dcm.ImagePositionPatient[0]),
        float(dcm.ImagePositionPatient[1]),
        float(dcm.ImagePositionPatient[2])
    ])
    
    # Get pixel spacing (x, y)
    if not hasattr(dcm, 'PixelSpacing'):
        raise ValueError("RTDOSE inválido: PixelSpacing ausente")
    
    pixel_spacing_x = float(dcm.PixelSpacing[0])
    pixel_spacing_y = float(dcm.PixelSpacing[1])
    
    # Get Z spacing from GridFrameOffsetVector
    if not hasattr(dcm, 'GridFrameOffsetVector'):
        raise ValueError("RTDOSE inválido: GridFrameOffsetVector ausente")
    
    grid_frame_offsets = dcm.GridFrameOffsetVector
    if len(grid_frame_offsets) < 2:
        # Single slice - assume spacing from SliceThickness if available
        if hasattr(dcm, 'SliceThickness'):
            slice_spacing = float(dcm.SliceThickness)
        else:
            warnings.warn("RTDOSE com slice único e sem SliceThickness. Usando spacing=1.0mm")
            slice_spacing = 1.0
    else:
        slice_spacing = float(grid_frame_offsets[1]) - float(grid_frame_offsets[0])
    
    spacing = np.array([pixel_spacing_x, pixel_spacing_y, slice_spacing])
    
    # Get FrameOfReferenceUID
    frame_of_reference_uid = dcm.FrameOfReferenceUID if hasattr(dcm, 'FrameOfReferenceUID') else None
    
    # Log information
    dose_type = dcm.DoseType if hasattr(dcm, 'DoseType') else 'UNKNOWN'
    dose_summation = dcm.DoseSummationType if hasattr(dcm, 'DoseSummationType') else 'UNKNOWN'
    
    print(f"RTDOSE carregado:")
    print(f"  DoseType: {dose_type}")
    print(f"  DoseSummationType: {dose_summation}")
    print(f"  Shape: {dose.shape}")
    print(f"  Origin: {origin}")
    print(f"  Spacing: {spacing}")
    print(f"  Dose range: {np.min(dose):.2f} - {np.max(dose):.2f} Gy")
    print(f"  FrameOfReferenceUID: {frame_of_reference_uid or '(ausente)'}")
    
    return dose, origin, spacing, frame_of_reference_uid


def interpolate_dose_to_grid(
    dose_ref: np.ndarray,
    origin_ref: np.ndarray,
    spacing_ref: np.ndarray,
    origin_calc: np.ndarray,
    spacing_calc: np.ndarray,
    size_calc: np.ndarray,
    direction_ref: Optional[np.ndarray] = None,
    direction_calc: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Interpolate reference dose to calculated dose grid.
    
    Uses trilinear interpolation to resample reference dose onto
    the calculated dose grid for voxel-by-voxel comparison.
    
    NOTE: This function is DEPRECATED. Use grid_utils.resample_dose_linear() instead.
    This wrapper is kept for backward compatibility.
    
    Parameters
    ----------
    dose_ref : np.ndarray
        Reference dose array, shape (z, y, x)
    origin_ref, spacing_ref : np.ndarray
        Reference dose geometry
    origin_calc, spacing_calc, size_calc : np.ndarray
        Calculated dose grid geometry
    direction_ref, direction_calc : np.ndarray, optional
        Direction cosine matrices (3x3)
        
    Returns
    -------
    dose_ref_interpolated : np.ndarray
        Reference dose interpolated to calculated grid, shape size_calc
        
    Notes
    -----
    Uses grid_utils.resample_dose_linear() if available, otherwise falls back
    to legacy scipy/SimpleITK implementation.
    """
    try:
        from scipy.ndimage import map_coordinates
        USE_SCIPY = True
    except ImportError:
        USE_SCIPY = False
        try:
            import SimpleITK as sitk
            USE_SITK = True
        except ImportError:
            USE_SITK = False
    
    if not USE_SCIPY and not USE_SITK:
        raise ImportError(
            "Interpolação requer scipy ou SimpleITK. "
            "Instale com: pip install scipy  OU  pip install SimpleITK"
        )
    
    if USE_SITK:
        # Use SimpleITK (more accurate, preserves exact geometry)
        import SimpleITK as sitk
        
        # Create reference dose image
        dose_ref_img = sitk.GetImageFromArray(dose_ref)
        dose_ref_img.SetOrigin(origin_ref)
        dose_ref_img.SetSpacing(spacing_ref)
        
        # Create target grid
        target_img = sitk.Image(
            [int(size_calc[2]), int(size_calc[1]), int(size_calc[0])],
            sitk.sitkFloat32
        )
        target_img.SetOrigin(origin_calc)
        target_img.SetSpacing(spacing_calc)
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(target_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        
        dose_ref_resampled = resampler.Execute(dose_ref_img)
        dose_ref_interpolated = sitk.GetArrayFromImage(dose_ref_resampled)
        
    else:
        # Use scipy (faster but manual coordinate mapping)
        # Create coordinate grids for calculated dose
        z_calc = np.arange(size_calc[0]) * spacing_calc[2] + origin_calc[2]
        y_calc = np.arange(size_calc[1]) * spacing_calc[1] + origin_calc[1]
        x_calc = np.arange(size_calc[2]) * spacing_calc[0] + origin_calc[0]
        
        # Convert to indices in reference grid
        z_ref_idx = (z_calc[:, None, None] - origin_ref[2]) / spacing_ref[2]
        y_ref_idx = (y_calc[None, :, None] - origin_ref[1]) / spacing_ref[1]
        x_ref_idx = (x_calc[None, None, :] - origin_ref[0]) / spacing_ref[0]
        
        # Stack coordinates
        coords = np.array([
            z_ref_idx * np.ones(size_calc),
            y_ref_idx * np.ones(size_calc),
            x_ref_idx * np.ones(size_calc)
        ])
        
        # Interpolate
        dose_ref_interpolated = map_coordinates(
            dose_ref,
            coords,
            order=1,  # Linear interpolation
            mode='constant',
            cval=0.0
        )
    
    return dose_ref_interpolated
