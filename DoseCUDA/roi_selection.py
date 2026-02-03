"""
ROI classification for secondary check pipeline.

Classifies ROIs from RTSTRUCT into targets (PTV/CTV/GTV/ITV) vs OARs
based on configurable naming patterns.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ROIClassificationConfig:
    """Configuration for ROI classification patterns.

    All patterns are matched case-insensitively using regex.
    """

    # Patterns for target volumes (PTV, CTV, GTV, ITV)
    target_patterns: List[str] = field(default_factory=lambda: [
        r'^PTV',           # PTV, PTV1, PTV_boost, PTV70, etc.
        r'^CTV',           # CTV, CTV1, CTV_Prostate
        r'^GTV',           # GTV, GTV_primary, GTV-T
        r'^ITV',           # ITV (internal target volume)
        r'_PTV$',          # Prostate_PTV
        r'_CTV$',          # Prostate_CTV
        r'_GTV$',          # Tumor_GTV
        r'_ITV$',          # Lung_ITV
        r'PTV_',           # PTV_High, PTV_Low
        r'CTV_',           # CTV_High, CTV_Low
        r'GTV_',           # GTV_Primary, GTV_Node
    ])

    # Patterns for structures to exclude (technical, derived, support)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        r'^BODY$',
        r'^EXTERNAL$',
        r'^OUTER\s*CONTOUR',
        r'^SKIN$',
        r'^COUCH',
        r'^TABLE',
        r'^BOLUS',
        r'^SUPPORT',
        r'^FIXATION',
        r'^IMMOBIL',
        r'^Z_',            # Eclipse convention for derived structures
        r'^z_',
        r'^AVOID',
        r'^HELP',
        r'^RING',
        r'^PRV',           # Planning risk volume (derived)
        r'^OPT_',          # Optimization structures
        r'^opt_',
        r'^DOSE',          # Dose level contours
        r'^ISO',           # Isodose contours
        r'^\d+%',          # Percentage contours like "95%"
        r'^_',             # Underscore prefix (often internal)
    ])

    # Optional explicit OAR patterns (if institution uses specific naming)
    oar_patterns: List[str] = field(default_factory=list)


@dataclass
class ClassifiedROIs:
    """Result of ROI classification."""
    targets: List[str]
    oars: List[str]
    excluded: List[str]

    def __repr__(self) -> str:
        return (
            f"ClassifiedROIs(\n"
            f"  targets={self.targets},\n"
            f"  oars={self.oars},\n"
            f"  excluded={self.excluded}\n"
            f")"
        )


def _matches_any_pattern(name: str, patterns: List[str]) -> bool:
    """Check if name matches any of the regex patterns (case-insensitive)."""
    for pattern in patterns:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    return False


def classify_rois(
    roi_names: List[str],
    config: Optional[ROIClassificationConfig] = None
) -> ClassifiedROIs:
    """
    Classify ROIs into targets, OARs, and excluded structures.

    Parameters
    ----------
    roi_names : list
        List of ROI names from RTSTRUCT
    config : ROIClassificationConfig, optional
        Classification configuration. Uses defaults if None.

    Returns
    -------
    ClassifiedROIs
        Categorized ROI names with targets, oars, and excluded lists

    Examples
    --------
    >>> names = ['PTV_70', 'PTV_56', 'Bladder', 'Rectum', 'BODY', 'Couch']
    >>> result = classify_rois(names)
    >>> result.targets
    ['PTV_70', 'PTV_56']
    >>> result.oars
    ['Bladder', 'Rectum']
    >>> result.excluded
    ['BODY', 'Couch']
    """
    if config is None:
        config = ROIClassificationConfig()

    targets = []
    oars = []
    excluded = []

    for name in roi_names:
        # Check exclusion first
        if _matches_any_pattern(name, config.exclude_patterns):
            excluded.append(name)
            continue

        # Check if target
        if _matches_any_pattern(name, config.target_patterns):
            targets.append(name)
            continue

        # Check explicit OAR patterns
        if config.oar_patterns and _matches_any_pattern(name, config.oar_patterns):
            oars.append(name)
            continue

        # Default: classify as OAR (anything not target or excluded)
        oars.append(name)

    return ClassifiedROIs(
        targets=targets,
        oars=oars,
        excluded=excluded
    )


def get_target_metrics_spec() -> Dict:
    """
    Get DVH metrics specification appropriate for targets.

    Returns metrics commonly used for target volume evaluation:
    - D2%: Near-maximum dose (dose to hottest 2% of volume)
    - D50%: Median dose
    - D95%: Coverage dose (dose covering 95% of volume)
    - D98%: Near-minimum dose (dose covering 98% of volume)

    Returns
    -------
    dict
        Metrics spec compatible with dvh.compute_metrics()
    """
    return {
        'D_percent': [2, 50, 95, 98],
        'V_dose': []  # V metrics less commonly used for targets
    }


def get_oar_metrics_spec(dose_levels: Optional[List[float]] = None) -> Dict:
    """
    Get DVH metrics specification appropriate for OARs.

    Returns metrics commonly used for OAR evaluation:
    - D2%: Near-maximum dose (hot spot)
    - Vx: Volume receiving >= x Gy

    Parameters
    ----------
    dose_levels : list, optional
        Dose thresholds for V metrics in Gy.
        Default: [10, 20, 30, 40, 50]

    Returns
    -------
    dict
        Metrics spec compatible with dvh.compute_metrics()
    """
    if dose_levels is None:
        dose_levels = [10, 20, 30, 40, 50]

    return {
        'D_percent': [2],  # D2% as near-max
        'V_dose': dose_levels
    }


def get_all_rois_metrics_spec() -> Dict:
    """
    Get comprehensive metrics spec suitable for any ROI type.

    Returns
    -------
    dict
        Metrics spec with both D% and V metrics
    """
    return {
        'D_percent': [2, 50, 95, 98],
        'V_dose': [10, 20, 30, 40, 50]
    }


def filter_rois_by_volume(
    roi_masks: Dict[str, 'np.ndarray'],
    voxel_volume_cc: float,
    min_volume_cc: float = 0.1
) -> List[str]:
    """
    Filter ROI names by minimum volume.

    Useful for excluding very small structures that may cause
    numerical issues in DVH calculation.

    Parameters
    ----------
    roi_masks : dict
        Dictionary mapping ROI name to boolean mask array
    voxel_volume_cc : float
        Volume of each voxel in cubic centimeters
    min_volume_cc : float
        Minimum volume threshold in cc (default 0.1 cc)

    Returns
    -------
    list
        ROI names with volume >= min_volume_cc
    """
    import numpy as np

    valid_rois = []
    for name, mask in roi_masks.items():
        n_voxels = np.sum(mask)
        volume_cc = n_voxels * voxel_volume_cc
        if volume_cc >= min_volume_cc:
            valid_rois.append(name)

    return valid_rois
