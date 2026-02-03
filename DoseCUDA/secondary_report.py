"""
Secondary check report generation for DoseCUDA.

Orchestrates gamma analysis, DVH comparison, MU sanity check,
and generates structured JSON and CSV reports.
"""

import json
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np

from .gamma import GammaCriteria, GammaResult, compute_gamma_3d
from .roi_selection import ClassifiedROIs, get_target_metrics_spec, get_oar_metrics_spec
from .mu_sanity import MUSanityResult, compute_mu_sanity_check, extract_isocenter_from_plan
from .dvh import compute_metrics, compare_dvh_metrics


# Try to import jsonschema for validation
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


@dataclass
class SecondaryCheckCriteria:
    """Configurable pass/fail criteria for secondary check.

    All default values represent typical clinical thresholds.
    These should be customized per institution.
    """

    # Gamma criteria
    gamma_3_3_pass_rate: float = 0.95    # 3%/3mm minimum pass rate
    gamma_2_2_pass_rate: float = 0.90    # 2%/2mm minimum pass rate
    gamma_dose_threshold: float = 10.0   # Percent of max dose
    gamma_global_mode: bool = True       # True = global, False = local

    # DVH criteria for targets (relative tolerances)
    target_d95_tolerance_rel: float = 0.03   # |delta D95| <= 3%
    target_dmean_tolerance_rel: float = 0.02  # |delta Dmean| <= 2%

    # DVH criteria for OARs (absolute tolerances in Gy)
    oar_dmean_tolerance_abs: float = 1.0     # |delta Dmean| <= 1 Gy
    oar_dmax_tolerance_abs: float = 2.0      # |delta Dmax| <= 2 Gy

    # MU sanity (informational, does not affect overall pass/fail)
    mu_equiv_tolerance: float = 0.05         # |ratio - 1| <= 5%

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SecondaryCheckResult:
    """Complete secondary check result for one case."""

    # Case identification
    patient_id: str
    plan_name: str
    plan_uid: str
    timestamp: str

    # Software info
    dosecuda_version: str
    schema_version: str = "1.0"

    # Criteria used
    criteria: SecondaryCheckCriteria = field(default_factory=SecondaryCheckCriteria)

    # Results
    gamma_results: Dict[str, Any] = field(default_factory=dict)
    dvh_results: Dict[str, Any] = field(default_factory=dict)
    mu_sanity: Optional[Dict[str, Any]] = None

    # Overall status
    overall_status: str = "UNKNOWN"  # PASS, FAIL, ERROR
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'schema_version': self.schema_version,
            'patient_id': self.patient_id,
            'plan_name': self.plan_name,
            'plan_uid': self.plan_uid,
            'timestamp': self.timestamp,
            'dosecuda_version': self.dosecuda_version,
            'criteria': self.criteria.to_dict(),
            'gamma_results': self.gamma_results,
            'dvh_results': self.dvh_results,
            'mu_sanity': self.mu_sanity,
            'overall_status': self.overall_status,
            'failure_reasons': self.failure_reasons
        }


def _get_version() -> str:
    """Get DoseCUDA version string."""
    try:
        from . import __version__
        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def evaluate_gamma(
    dose_calc: np.ndarray,
    dose_ref: np.ndarray,
    spacing_mm: tuple,
    criteria: SecondaryCheckCriteria
) -> Dict[str, Any]:
    """
    Evaluate gamma for standard criteria (3%/3mm and 2%/2mm).

    Returns dict mapping criteria label to result dict with status.
    """
    results = {}

    # 3%/3mm
    gamma_criteria_3_3 = GammaCriteria(
        dta_mm=3.0,
        dd_percent=3.0,
        local=not criteria.gamma_global_mode,
        dose_threshold_percent=criteria.gamma_dose_threshold
    )
    result_3_3 = compute_gamma_3d(
        dose_eval=dose_calc,
        dose_ref=dose_ref,
        spacing_mm=spacing_mm,
        criteria=gamma_criteria_3_3
    )

    status_3_3 = "PASS" if result_3_3.pass_rate >= criteria.gamma_3_3_pass_rate else "FAIL"
    results["3%/3mm"] = {
        'criteria_label': gamma_criteria_3_3.label(),
        'pass_rate': result_3_3.pass_rate,
        'mean_gamma': result_3_3.mean_gamma,
        'gamma_p95': result_3_3.gamma_p95,
        'n_evaluated': result_3_3.n_evaluated,
        'n_passed': result_3_3.n_passed,
        'status': status_3_3
    }

    # 2%/2mm
    gamma_criteria_2_2 = GammaCriteria(
        dta_mm=2.0,
        dd_percent=2.0,
        local=not criteria.gamma_global_mode,
        dose_threshold_percent=criteria.gamma_dose_threshold
    )
    result_2_2 = compute_gamma_3d(
        dose_eval=dose_calc,
        dose_ref=dose_ref,
        spacing_mm=spacing_mm,
        criteria=gamma_criteria_2_2
    )

    status_2_2 = "PASS" if result_2_2.pass_rate >= criteria.gamma_2_2_pass_rate else "FAIL"
    results["2%/2mm"] = {
        'criteria_label': gamma_criteria_2_2.label(),
        'pass_rate': result_2_2.pass_rate,
        'mean_gamma': result_2_2.mean_gamma,
        'gamma_p95': result_2_2.gamma_p95,
        'n_evaluated': result_2_2.n_evaluated,
        'n_passed': result_2_2.n_passed,
        'status': status_2_2
    }

    return results


def evaluate_dvh_for_roi(
    dose_calc: np.ndarray,
    dose_ref: np.ndarray,
    mask: np.ndarray,
    spacing: np.ndarray,
    roi_name: str,
    roi_type: str,
    criteria: SecondaryCheckCriteria
) -> Dict[str, Any]:
    """
    Evaluate DVH metrics for a single ROI.

    Returns dict with metrics comparison and status.
    """
    # Select metrics spec based on ROI type
    if roi_type == "TARGET":
        metrics_spec = get_target_metrics_spec()
    else:
        metrics_spec = get_oar_metrics_spec()

    # Compute metrics for both distributions
    metrics_calc = compute_metrics(dose_calc, mask, spacing, metrics_spec)
    metrics_ref = compute_metrics(dose_ref, mask, spacing, metrics_spec)

    # Compare metrics
    comparison = compare_dvh_metrics(metrics_calc, metrics_ref)

    # Build result with status for each metric
    metrics_result = {}
    overall_pass = True

    for metric_name, comp in comparison.items():
        calc_val = comp['calculated']
        ref_val = comp['reference']
        diff_abs = comp['diff']
        diff_rel = comp['diff_percent'] / 100.0 if comp['diff_percent'] is not None else None

        # Determine status based on tolerances
        if roi_type == "TARGET":
            # Use relative tolerance for targets
            if 'D95' in metric_name:
                tol = criteria.target_d95_tolerance_rel
                passes = abs(diff_rel) <= tol if diff_rel is not None else False
            elif 'Dmean' in metric_name or metric_name == 'Dmean':
                tol = criteria.target_dmean_tolerance_rel
                passes = abs(diff_rel) <= tol if diff_rel is not None else False
            else:
                # Other metrics: use standard comparison result
                passes = comp['pass']
        else:
            # Use absolute tolerance for OARs
            if 'Dmean' in metric_name or metric_name == 'Dmean':
                tol = criteria.oar_dmean_tolerance_abs
                passes = abs(diff_abs) <= tol
            elif 'Dmax' in metric_name or metric_name == 'Dmax':
                tol = criteria.oar_dmax_tolerance_abs
                passes = abs(diff_abs) <= tol
            else:
                # Other metrics: use standard comparison result
                passes = comp['pass']

        status = "PASS" if passes else "FAIL"
        if not passes:
            overall_pass = False

        metrics_result[metric_name] = {
            'calculated': calc_val,
            'reference': ref_val,
            'diff_abs': diff_abs,
            'diff_rel': diff_rel,
            'status': status
        }

    return {
        'roi_type': roi_type,
        'volume_cc': metrics_calc.get('Volume_cc', 0.0),
        'metrics': metrics_result,
        'overall_status': "PASS" if overall_pass else "FAIL"
    }


def evaluate_secondary_check(
    dose_calc: np.ndarray,
    dose_ref: np.ndarray,
    grid_origin: np.ndarray,
    grid_spacing: np.ndarray,
    rois: Dict[str, np.ndarray],
    roi_classification: ClassifiedROIs,
    plan,
    patient_id: str,
    plan_name: str,
    plan_uid: str,
    criteria: Optional[SecondaryCheckCriteria] = None,
    skip_mu_sanity: bool = False
) -> SecondaryCheckResult:
    """
    Run complete secondary check evaluation.

    Parameters
    ----------
    dose_calc : np.ndarray
        Calculated dose array (resampled to ref grid), shape (z, y, x) in Gy
    dose_ref : np.ndarray
        Reference TPS dose array, shape (z, y, x) in Gy
    grid_origin : np.ndarray
        Grid origin [x, y, z] in mm
    grid_spacing : np.ndarray
        Voxel spacing [x, y, z] in mm
    rois : dict
        Dictionary mapping ROI name to boolean mask array (on ref grid)
    roi_classification : ClassifiedROIs
        Classification of ROIs into targets/OARs
    plan : IMRTPlan
        Loaded plan object for MU extraction
    patient_id : str
        Patient identifier
    plan_name : str
        Plan name
    plan_uid : str
        Plan DICOM SOPInstanceUID
    criteria : SecondaryCheckCriteria, optional
        Pass/fail criteria. Uses defaults if None.
    skip_mu_sanity : bool
        If True, skip MU sanity check. Default False.

    Returns
    -------
    SecondaryCheckResult
        Complete evaluation result
    """
    if criteria is None:
        criteria = SecondaryCheckCriteria()

    timestamp = datetime.now().isoformat()
    failure_reasons = []

    # Initialize result
    result = SecondaryCheckResult(
        patient_id=patient_id,
        plan_name=plan_name,
        plan_uid=plan_uid,
        timestamp=timestamp,
        dosecuda_version=_get_version(),
        criteria=criteria
    )

    # --- Gamma Analysis ---
    try:
        spacing_tuple = tuple(grid_spacing)
        gamma_results = evaluate_gamma(dose_calc, dose_ref, spacing_tuple, criteria)
        result.gamma_results = gamma_results

        # Check gamma failures
        for label, gamma_res in gamma_results.items():
            if gamma_res['status'] == 'FAIL':
                threshold = criteria.gamma_3_3_pass_rate if '3%/3mm' in label else criteria.gamma_2_2_pass_rate
                failure_reasons.append(
                    f"Gamma {label} pass rate {gamma_res['pass_rate']:.1%} < {threshold:.0%}"
                )
    except Exception as e:
        warnings.warn(f"Gamma analysis failed: {e}")
        result.gamma_results = {}
        failure_reasons.append(f"Gamma analysis error: {str(e)}")

    # --- DVH Comparison ---
    try:
        dvh_results = {}

        # Process targets
        for roi_name in roi_classification.targets:
            if roi_name in rois:
                dvh_res = evaluate_dvh_for_roi(
                    dose_calc, dose_ref, rois[roi_name], grid_spacing,
                    roi_name, "TARGET", criteria
                )
                dvh_results[roi_name] = dvh_res
                if dvh_res['overall_status'] == 'FAIL':
                    failure_reasons.append(f"DVH comparison failed for target {roi_name}")

        # Process OARs
        for roi_name in roi_classification.oars:
            if roi_name in rois:
                dvh_res = evaluate_dvh_for_roi(
                    dose_calc, dose_ref, rois[roi_name], grid_spacing,
                    roi_name, "OAR", criteria
                )
                dvh_results[roi_name] = dvh_res
                if dvh_res['overall_status'] == 'FAIL':
                    failure_reasons.append(f"DVH comparison failed for OAR {roi_name}")

        result.dvh_results = dvh_results

    except Exception as e:
        warnings.warn(f"DVH comparison failed: {e}")
        result.dvh_results = {}
        failure_reasons.append(f"DVH comparison error: {str(e)}")

    # --- MU Sanity Check ---
    if not skip_mu_sanity:
        try:
            isocenter, total_mu = extract_isocenter_from_plan(plan)
            mu_result = compute_mu_sanity_check(
                dose_calc=dose_calc,
                dose_ref=dose_ref,
                grid_origin=grid_origin,
                grid_spacing=grid_spacing,
                isocenter=isocenter,
                total_mu=total_mu,
                tolerance=criteria.mu_equiv_tolerance
            )
            result.mu_sanity = mu_result.to_dict()

            # MU sanity is informational - does not affect overall status
            # But we can warn if FAIL
            if mu_result.status == 'FAIL':
                warnings.warn(f"MU sanity check failed: {mu_result.message}")

        except Exception as e:
            warnings.warn(f"MU sanity check failed: {e}")
            result.mu_sanity = None

    # --- Determine Overall Status ---
    if failure_reasons:
        result.overall_status = "FAIL"
        result.failure_reasons = failure_reasons
    else:
        result.overall_status = "PASS"
        result.failure_reasons = []

    return result


def generate_json_report(
    result: SecondaryCheckResult,
    output_path: str,
    indent: int = 2
) -> None:
    """
    Generate JSON report file.

    Parameters
    ----------
    result : SecondaryCheckResult
        Evaluation result
    output_path : str
        Path for JSON output file
    indent : int
        JSON indentation (default 2)
    """
    report_dict = result.to_dict()

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=indent)


def generate_csv_report(
    result: SecondaryCheckResult,
    output_path: str
) -> None:
    """
    Generate flat CSV report for spreadsheet import.

    Creates rows for:
    - Summary row with gamma results and overall status
    - One row per ROI with DVH metrics

    Parameters
    ----------
    result : SecondaryCheckResult
        Evaluation result
    output_path : str
        Path for CSV output file
    """
    rows = []

    # Summary row
    summary = {
        'patient_id': result.patient_id,
        'plan_name': result.plan_name,
        'timestamp': result.timestamp,
        'overall_status': result.overall_status,
        'row_type': 'SUMMARY',
        'roi_name': '',
        'roi_type': '',
    }

    # Add gamma results to summary
    for label, gamma_res in result.gamma_results.items():
        key_prefix = label.replace('%', 'pct').replace('/', '_')
        summary[f'gamma_{key_prefix}_pass_rate'] = gamma_res.get('pass_rate', '')
        summary[f'gamma_{key_prefix}_status'] = gamma_res.get('status', '')

    # Add MU sanity to summary
    if result.mu_sanity:
        summary['mu_ratio'] = result.mu_sanity.get('mu_equiv_ratio', '')
        summary['mu_status'] = result.mu_sanity.get('status', '')

    rows.append(summary)

    # ROI rows
    for roi_name, dvh_res in result.dvh_results.items():
        row = {
            'patient_id': result.patient_id,
            'plan_name': result.plan_name,
            'timestamp': result.timestamp,
            'overall_status': result.overall_status,
            'row_type': 'ROI',
            'roi_name': roi_name,
            'roi_type': dvh_res.get('roi_type', ''),
            'volume_cc': dvh_res.get('volume_cc', ''),
            'roi_status': dvh_res.get('overall_status', ''),
        }

        # Add metrics
        metrics = dvh_res.get('metrics', {})
        for metric_name, metric_data in metrics.items():
            safe_name = metric_name.replace('%', 'pct').replace(' ', '_')
            row[f'{safe_name}_calc'] = metric_data.get('calculated', '')
            row[f'{safe_name}_ref'] = metric_data.get('reference', '')
            row[f'{safe_name}_diff'] = metric_data.get('diff_abs', '')
            row[f'{safe_name}_status'] = metric_data.get('status', '')

        rows.append(row)

    # Determine all columns
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())

    # Sort columns for consistent output
    priority_cols = [
        'patient_id', 'plan_name', 'timestamp', 'overall_status',
        'row_type', 'roi_name', 'roi_type', 'volume_cc', 'roi_status'
    ]
    other_cols = sorted(all_columns - set(priority_cols))
    columns = [c for c in priority_cols if c in all_columns] + other_cols

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def validate_report_schema(json_data: dict) -> bool:
    """
    Validate JSON report against schema.

    Parameters
    ----------
    json_data : dict
        Report data to validate

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ImportError
        If jsonschema is not available
    jsonschema.ValidationError
        If validation fails
    """
    if not JSONSCHEMA_AVAILABLE:
        raise ImportError(
            "jsonschema is required for validation. "
            "Install with: pip install jsonschema"
        )

    # Load schema from package
    schema_path = Path(__file__).parent / 'report_schema.json'
    with open(schema_path) as f:
        schema = json.load(f)

    jsonschema.validate(json_data, schema)
    return True


def load_report(json_path: str) -> SecondaryCheckResult:
    """
    Load a secondary check report from JSON file.

    Parameters
    ----------
    json_path : str
        Path to JSON report file

    Returns
    -------
    SecondaryCheckResult
        Loaded result object
    """
    with open(json_path) as f:
        data = json.load(f)

    criteria = SecondaryCheckCriteria(**data.get('criteria', {}))

    return SecondaryCheckResult(
        patient_id=data['patient_id'],
        plan_name=data['plan_name'],
        plan_uid=data.get('plan_uid', ''),
        timestamp=data['timestamp'],
        dosecuda_version=data.get('dosecuda_version', 'unknown'),
        schema_version=data.get('schema_version', '1.0'),
        criteria=criteria,
        gamma_results=data.get('gamma_results', {}),
        dvh_results=data.get('dvh_results', {}),
        mu_sanity=data.get('mu_sanity'),
        overall_status=data['overall_status'],
        failure_reasons=data.get('failure_reasons', [])
    )
