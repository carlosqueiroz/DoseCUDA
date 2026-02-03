"""
Clinical Secondary Dose Check - Complete Workflow

This example demonstrates a complete clinical secondary dose check workflow:
1. Load patient CT DICOM
2. Load RTPLAN and RTSTRUCT
3. Calculate dose using DoseCUDA
4. Load reference RTDOSE from primary TPS
5. Compare DVH metrics
6. Generate clinical report

For real clinical use with actual patient data.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DoseCUDA.plan import DoseGrid
from DoseCUDA.plan_imrt import IMRTPlan, IMRTDoseGrid
from DoseCUDA import rtstruct, dvh


def clinical_secondary_check_imrt(
    ct_path: str,
    rtplan_path: str,
    rtstruct_path: str,
    rtdose_ref_path: str,
    roi_names: list,
    machine_name: str = "VarianTrueBeamHF",
    gpu_id: int = 0,
    tolerance_abs: float = 0.5,  # Gy
    tolerance_rel: float = 0.03,  # 3%
):
    """
    Complete clinical secondary check for IMRT plan.
    
    Parameters
    ----------
    ct_path : str
        Path to CT DICOM directory
    rtplan_path : str
        Path to RTPLAN DICOM file
    rtstruct_path : str
        Path to RTSTRUCT DICOM file
    rtdose_ref_path : str
        Path to reference RTDOSE from primary TPS
    roi_names : list
        List of ROI names to analyze (e.g., ['PTV', 'Bladder', 'Rectum'])
    machine_name : str
        Machine configuration name
    gpu_id : int
        GPU device ID for calculation
    tolerance_abs : float
        Absolute tolerance in Gy
    tolerance_rel : float
        Relative tolerance (fraction)
        
    Returns
    -------
    results : dict
        Dictionary with all results and pass/fail status
    """
    print("\n" + "="*70)
    print("CLINICAL SECONDARY DOSE CHECK - IMRT")
    print("="*70)
    
    results = {
        'plan_type': 'IMRT',
        'machine': machine_name,
        'overall_pass': True,
        'roi_results': {}
    }
    
    # ========================================================================
    # STEP 1: Load CT
    # ========================================================================
    print("\n[1/7] Loading CT DICOM...")
    print(f"  Path: {ct_path}")
    
    grid = DoseGrid()
    grid.loadCTDCM(ct_path)
    
    print(f"  ✓ CT loaded:")
    print(f"    Origin: {grid.origin}")
    print(f"    Spacing: {grid.spacing} mm")
    print(f"    Size: {grid.size} voxels")
    print(f"    FrameOfReferenceUID: {grid.FrameOfReferenceUID or '(not set)'}")
    
    # ========================================================================
    # STEP 2: Load RTSTRUCT
    # ========================================================================
    print(f"\n[2/7] Loading RTSTRUCT...")
    print(f"  Path: {rtstruct_path}")
    
    struct = rtstruct.read_rtstruct(rtstruct_path)
    
    print(f"  ✓ RTSTRUCT loaded:")
    print(f"    Name: {struct.name}")
    print(f"    ROIs found: {len(struct.rois)}")
    print(f"    ROI names: {', '.join(struct.get_roi_names())}")
    
    # Validate frame of reference
    rtstruct.validate_rtstruct_with_ct(struct, grid.FrameOfReferenceUID, strict=False)
    
    # ========================================================================
    # STEP 3: Rasterize ROIs
    # ========================================================================
    print(f"\n[3/7] Rasterizing ROIs...")
    
    masks = {}
    voxel_volume = np.prod(grid.spacing) / 1000.0  # mm³ to cc
    
    # Get CT Z positions for robust slice matching
    ct_z_positions = getattr(grid, '_ct_z_positions', None)
    
    for roi_name in roi_names:
        if roi_name not in struct.rois:
            print(f"  ⚠ ROI '{roi_name}' não encontrado no RTSTRUCT. Pulando.")
            continue
        
        print(f"  Rasterizando '{roi_name}'...")
        mask = rtstruct.rasterize_roi_to_mask(
            struct.rois[roi_name],
            grid.origin,
            grid.spacing,
            grid.size,
            grid.direction,
            ct_z_positions  # Pass CT Z positions for robust matching
        )
        
        volume_cc = np.sum(mask) * voxel_volume
        print(f"    ✓ Volume: {volume_cc:.2f} cc")
        
        masks[roi_name] = mask
    
    # ========================================================================
    # STEP 4: Load RTPLAN and Calculate Dose
    # ========================================================================
    print(f"\n[4/7] Loading RTPLAN and calculating dose...")
    print(f"  RTPLAN path: {rtplan_path}")
    
    plan = IMRTPlan(machine_name=machine_name)
    plan.readPlanDicom(rtplan_path)
    
    print(f"  ✓ RTPLAN loaded:")
    print(f"    Number of beams: {plan.n_beams}")
    print(f"    Number of fractions: {plan.n_fractions}")
    
    print(f"\n  Calculating dose on GPU {gpu_id}...")
    dose_grid = IMRTDoseGrid()
    dose_grid.loadCTDCM(ct_path)
    dose_grid.computeIMRTPlan(plan, gpu_id=gpu_id)
    
    print(f"  ✓ Dose calculated:")
    print(f"    Dose range: {np.min(dose_grid.dose):.2f} - {np.max(dose_grid.dose):.2f} Gy")
    
    # ========================================================================
    # STEP 5: Load Reference RTDOSE
    # ========================================================================
    print(f"\n[5/7] Loading reference RTDOSE from primary TPS...")
    print(f"  Path: {rtdose_ref_path}")
    
    dose_ref, origin_ref, spacing_ref, frame_ref_uid = dvh.read_reference_rtdose(rtdose_ref_path)
    
    # Validate frame of reference
    if frame_ref_uid and grid.FrameOfReferenceUID:
        if frame_ref_uid != grid.FrameOfReferenceUID:
            print(f"  ⚠ Warning: FrameOfReferenceUID mismatch:")
            print(f"    CT: {grid.FrameOfReferenceUID}")
            print(f"    RTDOSE: {frame_ref_uid}")
    
    # Interpolate reference dose to calculated grid if needed
    if not (np.allclose(origin_ref, dose_grid.origin, atol=0.1) and
            np.allclose(spacing_ref, dose_grid.spacing, atol=0.01) and
            dose_ref.shape == dose_grid.dose.shape):
        print(f"  Interpolating reference dose to calculated grid...")
        dose_ref_interp = dvh.interpolate_dose_to_grid(
            dose_ref, origin_ref, spacing_ref,
            dose_grid.origin, dose_grid.spacing, dose_grid.size
        )
        print(f"    ✓ Reference dose interpolated")
    else:
        dose_ref_interp = dose_ref
        print(f"  ✓ Grids já estão alinhados")
    
    # ========================================================================
    # STEP 6: Compute DVH and Metrics
    # ========================================================================
    print(f"\n[6/7] Computing DVH and metrics for each structure...")
    
    metrics_spec = {
        'D_percent': [2, 50, 95, 98],
        'V_dose': [20, 30, 40, 50]
    }
    
    for roi_name, mask in masks.items():
        print(f"\n  Analyzing '{roi_name}'...")
        
        # Compute metrics for calculated dose
        metrics_calc = dvh.compute_metrics(
            dose_grid.dose, mask, dose_grid.spacing, metrics_spec
        )
        
        # Compute metrics for reference dose
        metrics_ref = dvh.compute_metrics(
            dose_ref_interp, mask, dose_grid.spacing, metrics_spec
        )
        
        # Compare
        comparison = dvh.compare_dvh_metrics(
            metrics_calc, metrics_ref,
            tolerance_abs=tolerance_abs,
            tolerance_rel=tolerance_rel
        )
        
        # Check pass/fail
        roi_pass = all(comp['pass'] for comp in comparison.values())
        
        print(f"    Status: {'✓ PASS' if roi_pass else '✗ FAIL'}")
        print(f"    Dmean: Calc={metrics_calc['Dmean']:.2f} Gy, "
              f"Ref={metrics_ref['Dmean']:.2f} Gy, "
              f"Diff={comparison['Dmean']['diff']:+.2f} Gy")
        
        if not roi_pass:
            results['overall_pass'] = False
            print(f"    ⚠ Failed metrics:")
            for metric, comp in comparison.items():
                if not comp['pass']:
                    print(f"      - {metric}: {comp['diff']:+.2f} ({comp['diff_percent']:+.1f}%)")
        
        # Store results
        results['roi_results'][roi_name] = {
            'metrics_calculated': metrics_calc,
            'metrics_reference': metrics_ref,
            'comparison': comparison,
            'pass': roi_pass
        }
    
    # ========================================================================
    # STEP 7: Generate Report
    # ========================================================================
    print(f"\n[7/7] Generating clinical report...")
    
    report_lines = []
    report_lines.append("\n" + "="*70)
    report_lines.append("CLINICAL SECONDARY DOSE CHECK REPORT")
    report_lines.append("="*70)
    report_lines.append(f"\nPlan Type: {results['plan_type']}")
    report_lines.append(f"Machine: {results['machine']}")
    report_lines.append(f"Tolerance: ±{tolerance_abs} Gy (abs) or ±{tolerance_rel*100}% (rel)")
    report_lines.append(f"\nOverall Status: {'✓✓✓ PASS ✓✓✓' if results['overall_pass'] else '✗✗✗ FAIL ✗✗✗'}")
    report_lines.append("\n" + "-"*70)
    
    for roi_name, roi_result in results['roi_results'].items():
        roi_pass = roi_result['pass']
        metrics_calc = roi_result['metrics_calculated']
        comparison = roi_result['comparison']
        
        report_lines.append(f"\n{roi_name}: {'✓ PASS' if roi_pass else '✗ FAIL'}")
        report_lines.append(f"  Volume: {metrics_calc['Volume_cc']:.2f} cc")
        
        # Show key metrics
        for metric in ['Dmean', 'Dmax', 'D95%', 'D98%']:
            if metric in comparison:
                comp = comparison[metric]
                status = "✓" if comp['pass'] else "✗"
                report_lines.append(
                    f"  {status} {metric:6s}: Calc={comp['calculated']:7.2f} Gy, "
                    f"Ref={comp['reference']:7.2f} Gy, "
                    f"Diff={comp['diff']:+6.2f} Gy ({comp['diff_percent']:+5.1f}%)"
                )
    
    report_lines.append("\n" + "="*70)
    
    report = '\n'.join(report_lines)
    print(report)
    
    results['report'] = report
    
    return results


def main():
    """
    Main function with example usage.
    
    Adjust paths below to match your clinical data structure.
    """
    
    # =======================================================================
    # CONFIGURATION - Adjust these paths for your data
    # =======================================================================
    
    # Example data paths (adjust to your directory structure)
    base_path = Path("/data/patients/patient_001")
    
    ct_path = str(base_path / "CT")
    rtplan_path = str(base_path / "RTPLAN.dcm")
    rtstruct_path = str(base_path / "RTSTRUCT.dcm")
    rtdose_ref_path = str(base_path / "RTDOSE_reference.dcm")
    
    # ROIs to analyze
    roi_names = ['PTV', 'Bladder', 'Rectum', 'Femur_L', 'Femur_R']
    
    # Machine configuration
    machine_name = "VarianTrueBeamHF"
    
    # Tolerances (adjust based on institutional policy)
    tolerance_abs = 0.5  # ±0.5 Gy
    tolerance_rel = 0.03  # ±3%
    
    # GPU device
    gpu_id = 0
    
    # =======================================================================
    # RUN SECONDARY CHECK
    # =======================================================================
    
    print("\n" + "#"*70)
    print("# DoseCUDA Clinical Secondary Check")
    print("#"*70)
    print(f"\nPatient data: {base_path}")
    print(f"Machine: {machine_name}")
    print(f"Tolerance: ±{tolerance_abs} Gy or ±{tolerance_rel*100}%")
    
    # Check if files exist
    if not Path(ct_path).exists():
        print(f"\n⚠ ERROR: CT directory not found: {ct_path}")
        print("\nFor testing, you can use the example with synthetic data:")
        print("  python tests/example_patient_pipeline.py")
        print("\nFor real clinical use, provide paths to actual DICOM files.")
        return
    
    try:
        results = clinical_secondary_check_imrt(
            ct_path=ct_path,
            rtplan_path=rtplan_path,
            rtstruct_path=rtstruct_path,
            rtdose_ref_path=rtdose_ref_path,
            roi_names=roi_names,
            machine_name=machine_name,
            gpu_id=gpu_id,
            tolerance_abs=tolerance_abs,
            tolerance_rel=tolerance_rel
        )
        
        # Save report to file
        report_path = base_path / "secondary_check_report.txt"
        with open(report_path, 'w') as f:
            f.write(results['report'])
        
        print(f"\n✓ Report saved to: {report_path}")
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_pass'] else 1
        print(f"\nExit code: {exit_code} ({'PASS' if exit_code == 0 else 'FAIL'})")
        
        return results
        
    except Exception as e:
        print(f"\n✗ ERROR during secondary check:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
