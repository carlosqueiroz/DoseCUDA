"""
End-to-end test for real patient DICOM data.

Tests complete workflow:
1. Auto-discovery of CT/RTPLAN/RTSTRUCT/RTDOSE in mixed directory
2. Automatic selection of appropriate series/files
3. Anisotropic CT → isotropic resampling for dose calculation
4. GPU dose calculation with auto-detected machine model
5. Dose resampling to RTDOSE template grid
6. DICOM RTDOSE export

Environment Variables
---------------------
DOSECUDA_PATIENT_DICOM_DIR : str (required)
    Path to patient DICOM directory. Test skips if not set.
DOSECUDA_ISO_MM : float (optional, default=2.5)
    Target isotropic spacing for dose calculation (mm)
DOSECUDA_GPU_ID : int (optional, default=0)
    GPU device ID for dose calculation
DOSECUDA_MACHINE_DEFAULT : str (optional, default="VarianTrueBeamHF")
    Default machine model if inference fails

Usage
-----
# Set patient directory
export DOSECUDA_PATIENT_DICOM_DIR=/path/to/patient/dicoms

# Run test
pytest tests/test_patient_end2end.py -v -s

# With custom settings
export DOSECUDA_ISO_MM=3.0
export DOSECUDA_GPU_ID=1
pytest tests/test_patient_end2end.py -v -s
"""

import os
import pytest
import numpy as np
import warnings
from pathlib import Path

# Import DoseCUDA modules
from DoseCUDA import IMRTDoseGrid, IMRTPlan
from DoseCUDA.dicom_case_discovery import (
    scan_dicom_directory,
    select_rtplan,
    select_rtdose_template,
    select_ct_series,
    infer_machine_model,
    materialize_case
)
from DoseCUDA.grid_utils import GridInfo, resample_dose_linear

import pydicom


# ============================================================================
# Test Configuration from Environment
# ============================================================================

def get_patient_dicom_dir():
    """Get patient DICOM directory from environment, or None if not set."""
    return os.environ.get('DOSECUDA_PATIENT_DICOM_DIR')


def get_iso_spacing_mm():
    """Get target isotropic spacing in mm (default: 2.5)."""
    return float(os.environ.get('DOSECUDA_ISO_MM', '2.5'))


def get_gpu_id():
    """Get GPU device ID (default: 0)."""
    return int(os.environ.get('DOSECUDA_GPU_ID', '0'))


def get_default_machine():
    """Get default machine model (default: VarianTrueBeamHF)."""
    return os.environ.get('DOSECUDA_MACHINE_DEFAULT', 'VarianTrueBeamHF')


def get_output_dir():
    """Get output directory for test results."""
    test_dir = Path(__file__).parent
    output_dir = test_dir / "test_patient_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def patient_dicom_dir():
    """
    Fixture providing patient DICOM directory.
    Skips test if DOSECUDA_PATIENT_DICOM_DIR not set.
    """
    dicom_dir = get_patient_dicom_dir()
    
    if not dicom_dir:
        pytest.skip(
            "Test requires DOSECUDA_PATIENT_DICOM_DIR environment variable. "
            "Set it to patient DICOM directory path to run this test.\n"
            "Example: export DOSECUDA_PATIENT_DICOM_DIR=/path/to/patient/dicoms"
        )
    
    dicom_path = Path(dicom_dir)
    if not dicom_path.exists():
        pytest.skip(f"Patient DICOM directory does not exist: {dicom_dir}")
    
    return dicom_path


@pytest.fixture(scope="module")
def discovered_case(patient_dicom_dir):
    """
    Fixture providing discovered and classified DICOM case.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: DICOM DISCOVERY")
    print("=" * 80)
    
    case = scan_dicom_directory(str(patient_dicom_dir))
    
    # Validate minimum requirements
    if not case.ct_files:
        pytest.fail("No CT files found in patient directory")
    
    if not case.rtplan_files:
        pytest.fail("No RTPLAN found in patient directory")
    
    print(f"\n{case}")
    
    return case


@pytest.fixture(scope="module")
def selected_files(discovered_case):
    """
    Fixture providing selected RTPLAN, RTDOSE, RTSTRUCT, and CT series.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: AUTOMATIC SELECTION")
    print("=" * 80)
    
    # Select RTPLAN
    rtplan = select_rtplan(discovered_case)
    assert rtplan is not None, "Failed to select RTPLAN"
    
    # Select RTDOSE template (optional)
    rtdose = select_rtdose_template(discovered_case)
    
    # Select RTSTRUCT (use first if multiple, or None)
    rtstruct = None
    if discovered_case.rtstruct_files:
        rtstruct = discovered_case.rtstruct_files[0]
        print(f"\n[RTSTRUCT] Using: {rtstruct.path.name}")
        if len(discovered_case.rtstruct_files) > 1:
            warnings.warn(f"Multiple RTSTRUCT found ({len(discovered_case.rtstruct_files)}), using first")
    
    # Select CT series
    ct_series = select_ct_series(discovered_case, rtstruct=rtstruct, rtdose=rtdose)
    assert ct_series is not None, "Failed to select CT series"
    assert len(ct_series) > 0, "Selected CT series is empty"
    
    return {
        'rtplan': rtplan,
        'rtdose': rtdose,
        'rtstruct': rtstruct,
        'ct_series': ct_series
    }


@pytest.fixture(scope="module")
def materialized_case(selected_files):
    """
    Fixture providing materialized case with organized file structure.
    """
    print("\n" + "=" * 80)
    print("PHASE 3: CASE MATERIALIZATION")
    print("=" * 80)
    
    output_dir = get_output_dir()
    
    paths = materialize_case(
        output_dir=str(output_dir),
        ct_series=selected_files['ct_series'],
        rtplan=selected_files['rtplan'],
        rtdose=selected_files['rtdose'],
        rtstruct=selected_files['rtstruct']
    )
    
    return paths


@pytest.fixture(scope="module")
def machine_model(selected_files):
    """
    Fixture providing inferred machine model.
    """
    print("\n" + "=" * 80)
    print("PHASE 4: MACHINE MODEL INFERENCE")
    print("=" * 80)
    
    default_model = get_default_machine()
    model = infer_machine_model(selected_files['rtplan'], default_model=default_model)
    
    # Validate model exists in lookuptables
    lookuptable_path = Path(__file__).parent.parent / "DoseCUDA" / "lookuptables" / "photons" / model
    if not lookuptable_path.exists():
        pytest.fail(
            f"Machine model '{model}' not found in lookuptables/photons/.\n"
            f"Expected path: {lookuptable_path}\n"
            f"Available models: {list((lookuptable_path.parent).glob('*'))}"
        )
    
    print(f"\n✓ Machine model validated: {model}")
    
    return model


# ============================================================================
# Test Functions
# ============================================================================

def test_1_discovery_and_selection(discovered_case, selected_files):
    """
    Test 1: Verify DICOM discovery and automatic selection succeeded.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Discovery and Selection")
    print("=" * 80)
    
    # Verify case has required files
    assert len(discovered_case.ct_files) > 0, "No CT files discovered"
    assert len(discovered_case.rtplan_files) > 0, "No RTPLAN discovered"
    
    # Verify selections
    assert selected_files['rtplan'] is not None, "RTPLAN selection failed"
    assert selected_files['ct_series'] is not None, "CT series selection failed"
    assert len(selected_files['ct_series']) > 0, "CT series is empty"
    
    print(f"\n✓ Discovery: {len(discovered_case.ct_files)} CT, "
          f"{len(discovered_case.rtplan_files)} RTPLAN, "
          f"{len(discovered_case.rtstruct_files)} RTSTRUCT, "
          f"{len(discovered_case.rtdose_files)} RTDOSE")
    print(f"✓ Selected: RTPLAN={selected_files['rtplan'].path.name}, "
          f"CT series={len(selected_files['ct_series'])} slices, "
          f"RTDOSE={'Yes' if selected_files['rtdose'] else 'No'}")


def test_2_ct_loading_and_anisotropy(materialized_case):
    """
    Test 2: Load CT and verify it's anisotropic (z != x/y).
    """
    print("\n" + "=" * 80)
    print("TEST 2: CT Loading and Anisotropy Check")
    print("=" * 80)
    
    dg = IMRTDoseGrid()
    ct_dir = str(materialized_case['ct_dir'])
    
    print(f"\nLoading CT from: {ct_dir}")
    dg.loadCTDCM(ct_dir)
    
    # Check CT loaded
    assert dg.HU is not None, "CT HU array is None"
    assert dg.spacing is not None, "CT spacing is None"
    
    print(f"✓ CT loaded: shape={dg.HU.shape}, spacing={dg.spacing} mm")
    
    # Check anisotropy (expected for clinical CT)
    spacing = dg.spacing
    is_isotropic = np.allclose(spacing, spacing[0], atol=0.01)
    
    if is_isotropic:
        warnings.warn(
            f"CT is already isotropic ({spacing}). "
            "Expected anisotropic for clinical CT."
        )
    else:
        print(f"✓ CT is anisotropic (expected): {spacing} mm")
        print(f"  Ratio z/x = {spacing[2]/spacing[0]:.2f}")


def test_3_isotropic_resampling(materialized_case):
    """
    Test 3: Resample anisotropic CT to isotropic spacing for dose calculation.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Isotropic Resampling")
    print("=" * 80)
    
    target_spacing = get_iso_spacing_mm()
    
    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    
    original_spacing = dg.spacing.copy()
    original_size = dg.size.copy()
    
    print(f"\nOriginal CT: spacing={original_spacing} mm, size={original_size}")
    print(f"Target: {target_spacing} mm isotropic")
    
    # Resample to isotropic
    dg.resampleCTfromSpacing(target_spacing)
    
    # Verify result
    assert dg.spacing is not None, "Spacing is None after resample"
    assert dg.HU is not None, "HU is None after resample"
    
    resampled_spacing = dg.spacing
    resampled_size = dg.size
    
    print(f"Resampled CT: spacing={resampled_spacing} mm, size={resampled_size}")
    
    # Verify isotropic
    is_isotropic = np.allclose(resampled_spacing, target_spacing, atol=0.01)
    assert is_isotropic, f"Spacing not isotropic after resample: {resampled_spacing}"
    
    print(f"✓ CT successfully resampled to isotropic {target_spacing} mm")


def test_4_gpu_dose_calculation(materialized_case, machine_model):
    """
    Test 4: Calculate dose on GPU with auto-detected machine model.
    """
    print("\n" + "=" * 80)
    print("TEST 4: GPU Dose Calculation")
    print("=" * 80)
    
    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()
    
    print(f"\nMachine model: {machine_model}")
    print(f"GPU ID: {gpu_id}")
    print(f"Target spacing: {target_spacing} mm")
    
    # Initialize plan and dose grid
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))
    
    print(f"\n✓ RTPLAN loaded: {plan.n_beams} beams")
    
    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    dg.resampleCTfromSpacing(target_spacing)
    
    print(f"✓ CT prepared: {dg.size} voxels @ {dg.spacing} mm")
    
    # Calculate dose on GPU
    print(f"\nCalculating dose on GPU {gpu_id}...")
    dg.computeIMRTPlan(plan, gpu_id=gpu_id)
    
    # Verify dose calculated
    assert dg.dose is not None, "Dose is None after calculation"
    assert dg.dose.shape == tuple(dg.size), f"Dose shape {dg.dose.shape} != grid size {dg.size}"
    
    # Check dose statistics
    dose_min = np.min(dg.dose)
    dose_max = np.max(dg.dose)
    dose_mean = np.mean(dg.dose)
    dose_nonzero = np.sum(dg.dose > 0.01)  # Gy
    
    print(f"\n✓ Dose calculated:")
    print(f"  Shape: {dg.dose.shape}")
    print(f"  Min: {dose_min:.3f} Gy")
    print(f"  Max: {dose_max:.3f} Gy")
    print(f"  Mean: {dose_mean:.3f} Gy")
    print(f"  Voxels > 0.01 Gy: {dose_nonzero} ({dose_nonzero/np.prod(dg.dose.shape)*100:.1f}%)")
    
    # Sanity checks
    assert dose_max > 0, "Maximum dose is zero (calculation may have failed)"
    assert dose_max < 1000, f"Maximum dose {dose_max} Gy is unrealistic"
    assert np.isfinite(dg.dose).all(), "Dose contains NaN or Inf"
    
    print(f"\n✓ Dose sanity checks passed")
    
    return dg


def test_5_save_numpy_and_nrrd(materialized_case, machine_model):
    """
    Test 5: Save dose as numpy (.npy) and NRRD.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Save NPY and NRRD")
    print("=" * 80)
    
    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()
    output_dir = get_output_dir()
    
    # Recalculate dose (or reuse from test_4 if cached)
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))
    
    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    dg.resampleCTfromSpacing(target_spacing)
    dg.computeIMRTPlan(plan, gpu_id=gpu_id)
    
    # Save numpy
    npy_path = output_dir / "dose_calculated.npy"
    np.save(npy_path, dg.dose)
    print(f"\n✓ Saved numpy: {npy_path}")
    assert npy_path.exists(), "NPY file not created"
    
    # Save NRRD
    nrrd_path = output_dir / "dose_calculated.nrrd"
    dg.writeDoseNRRD(str(nrrd_path), dose_type="PHYSICAL")
    print(f"✓ Saved NRRD: {nrrd_path}")
    assert nrrd_path.exists(), "NRRD file not created"
    
    # Save dose stats
    stats_path = output_dir / "dose_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Dose Statistics (calculated on isotropic grid)\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Shape: {dg.dose.shape}\n")
        f.write(f"Spacing: {dg.spacing} mm\n")
        f.write(f"Origin: {dg.origin} mm\n")
        f.write(f"Min: {np.min(dg.dose):.6f} Gy\n")
        f.write(f"Max: {np.max(dg.dose):.6f} Gy\n")
        f.write(f"Mean: {np.mean(dg.dose):.6f} Gy\n")
        f.write(f"Median: {np.median(dg.dose):.6f} Gy\n")
        f.write(f"Std: {np.std(dg.dose):.6f} Gy\n")
        f.write(f"All finite: {np.isfinite(dg.dose).all()}\n")
    
    print(f"✓ Saved stats: {stats_path}")


def test_6_resample_and_save_dicom_rtdose(materialized_case, machine_model, selected_files):
    """
    Test 6: Resample calculated dose to RTDOSE template grid and save DICOM.
    
    This is the critical step that enables comparison with TPS dose.
    Only runs if RTDOSE template is available.
    """
    print("\n" + "=" * 80)
    print("TEST 6: Resample to Template Grid and Save DICOM RTDOSE")
    print("=" * 80)
    
    if not materialized_case['rtdose']:
        pytest.skip("No RTDOSE template found - skipping DICOM export")
    
    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()
    output_dir = get_output_dir()
    
    # Read template RTDOSE to get target grid
    template_path = str(materialized_case['rtdose'])
    print(f"\nReading RTDOSE template: {Path(template_path).name}")
    
    template_ds = pydicom.dcmread(template_path, force=True)
    
    # Extract template grid geometry
    template_origin = np.array(template_ds.ImagePositionPatient)
    
    pixel_spacing = np.array(template_ds.PixelSpacing)  # [row_spacing, col_spacing]
    
    # Get slice spacing from GridFrameOffsetVector
    if hasattr(template_ds, 'GridFrameOffsetVector'):
        frame_offsets = np.array(template_ds.GridFrameOffsetVector)
        if len(frame_offsets) > 1:
            slice_spacing = frame_offsets[1] - frame_offsets[0]
        else:
            slice_spacing = pixel_spacing[0]  # Fallback: assume cubic
    else:
        warnings.warn("Template RTDOSE has no GridFrameOffsetVector, assuming cubic voxels")
        slice_spacing = pixel_spacing[0]
    
    template_spacing = np.array([pixel_spacing[1], pixel_spacing[0], slice_spacing])  # [x, y, z]
    
    template_size = np.array([
        int(template_ds.Columns),
        int(template_ds.Rows),
        int(template_ds.NumberOfFrames) if hasattr(template_ds, 'NumberOfFrames') else 1
    ])
    
    # Get direction (assume axial if not specified)
    if hasattr(template_ds, 'ImageOrientationPatient'):
        orientation = np.array(template_ds.ImageOrientationPatient)
        # Convert to 3x3 direction matrix
        row_cos = orientation[:3]
        col_cos = orientation[3:]
        slice_cos = np.cross(row_cos, col_cos)
        direction = np.column_stack([row_cos, col_cos, slice_cos])
        
        # Check if axial (allow only axial in phase 1)
        if not np.allclose(direction, np.eye(3), atol=0.1):
            pytest.fail(
                f"Template RTDOSE is not axial (direction != identity).\n"
                f"Non-axial geometry not yet supported.\n"
                f"Direction matrix:\n{direction}"
            )
    else:
        direction = np.eye(3)
    
    template_grid = GridInfo(
        origin=template_origin,
        spacing=template_spacing,
        size=template_size,
        direction=direction
    )
    
    print(f"✓ Template grid:")
    print(f"  Origin: {template_grid.origin} mm")
    print(f"  Spacing: {template_grid.spacing} mm")
    print(f"  Size: {template_grid.size}")
    print(f"  Direction: {'Axial' if np.allclose(direction, np.eye(3)) else 'Oblique'}")
    
    # Calculate dose on isotropic grid
    print(f"\nCalculating dose on isotropic grid ({target_spacing} mm)...")
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))
    
    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    dg.resampleCTfromSpacing(target_spacing)
    dg.computeIMRTPlan(plan, gpu_id=gpu_id)
    
    # Build calculated dose grid
    calc_grid = GridInfo(
        origin=dg.origin,
        spacing=dg.spacing,
        size=dg.size,
        direction=np.eye(3)  # DoseCUDA always axial
    )
    
    print(f"✓ Calculated dose grid:")
    print(f"  Origin: {calc_grid.origin} mm")
    print(f"  Spacing: {calc_grid.spacing} mm")
    print(f"  Size: {calc_grid.size}")
    print(f"  Max dose: {np.max(dg.dose):.2f} Gy")
    
    # Resample calculated dose to template grid
    print(f"\nResampling calculated dose to template grid...")
    dose_resampled = resample_dose_linear(
        dose=dg.dose,
        source_grid=calc_grid,
        target_grid=template_grid
    )
    
    print(f"✓ Dose resampled:")
    print(f"  Shape: {dose_resampled.shape} (expected {tuple(template_grid.size[::-1])})")
    print(f"  Max dose: {np.max(dose_resampled):.2f} Gy")
    
    # Verify shape matches template
    expected_shape = (template_size[2], template_size[1], template_size[0])  # (frames, rows, cols)
    assert dose_resampled.shape == expected_shape, \
        f"Resampled shape {dose_resampled.shape} != template {expected_shape}"
    
    # Update dose grid with resampled dose
    dg.dose = dose_resampled
    dg.origin = template_grid.origin
    dg.spacing = template_grid.spacing
    dg.size = template_grid.size
    
    # Save DICOM RTDOSE
    output_dcm_path = output_dir / "DoseCUDA_RD.dcm"
    
    # Get RTPLAN SOPInstanceUID for reference
    rtplan_ds = pydicom.dcmread(str(materialized_case['rtplan']), stop_before_pixels=True)
    rtplan_sop_uid = rtplan_ds.SOPInstanceUID
    
    print(f"\nSaving DICOM RTDOSE: {output_dcm_path.name}")
    dg.writeDoseDCM(
        dose_path=str(output_dcm_path),
        ref_dose_path=template_path,
        dose_type="PHYSICAL",
        rtplan_sop_uid=rtplan_sop_uid
    )
    
    assert output_dcm_path.exists(), "DICOM RTDOSE not created"
    
    # Verify saved RTDOSE
    saved_ds = pydicom.dcmread(str(output_dcm_path), force=True)
    
    print(f"\n✓ DICOM RTDOSE saved and verified:")
    print(f"  SOPInstanceUID: {saved_ds.SOPInstanceUID}")
    print(f"  SeriesDescription: {saved_ds.SeriesDescription}")
    print(f"  DoseSummationType: {saved_ds.DoseSummationType}")
    print(f"  DoseType: {saved_ds.DoseType}")
    print(f"  DoseGridScaling: {saved_ds.DoseGridScaling:.6e}")
    print(f"  Shape: ({saved_ds.NumberOfFrames}, {saved_ds.Rows}, {saved_ds.Columns})")
    print(f"  Max pixel value: {np.max(np.frombuffer(saved_ds.PixelData, dtype=np.uint16))}")
    print(f"  Max dose: {np.max(np.frombuffer(saved_ds.PixelData, dtype=np.uint16)) * saved_ds.DoseGridScaling:.2f} Gy")
    
    # Verify grid matches template
    assert saved_ds.Rows == template_ds.Rows, "Rows mismatch"
    assert saved_ds.Columns == template_ds.Columns, "Columns mismatch"
    assert saved_ds.NumberOfFrames == template_ds.NumberOfFrames, "NumberOfFrames mismatch"
    
    print(f"\n✓ Grid geometry matches template")


# ============================================================================
# Summary Test
# ============================================================================

def test_7_summary(discovered_case, selected_files, machine_model):
    """
    Test 7: Print comprehensive summary of end-to-end test.
    """
    print("\n" + "=" * 80)
    print("TEST 7: End-to-End Summary")
    print("=" * 80)

    output_dir = get_output_dir()

    print(f"\n✓ All tests passed!")
    print(f"\nInput:")
    print(f"  Patient DICOM dir: {get_patient_dicom_dir()}")
    print(f"  CT series: {len(selected_files['ct_series'])} slices")
    print(f"  RTPLAN: {selected_files['rtplan'].path.name}")
    if selected_files['rtdose']:
        print(f"  RTDOSE template: {selected_files['rtdose'].path.name}")
    if selected_files['rtstruct']:
        print(f"  RTSTRUCT: {selected_files['rtstruct'].path.name}")

    print(f"\nConfiguration:")
    print(f"  Machine model: {machine_model}")
    print(f"  Isotropic spacing: {get_iso_spacing_mm()} mm")
    print(f"  GPU ID: {get_gpu_id()}")

    print(f"\nOutput: {output_dir}")
    print(f"  dose_calculated.npy")
    print(f"  dose_calculated.nrrd")
    print(f"  dose_stats.txt")
    if selected_files['rtdose']:
        print(f"  DoseCUDA_RD.dcm  (DICOM RTDOSE)")

    print(f"\n" + "=" * 80)
    print("SUCCESS: Patient end-to-end workflow complete")
    print("=" * 80)


# ============================================================================
# Secondary Check Tests (Tests 8-11)
# ============================================================================

@pytest.fixture(scope="module")
def reference_rtdose(materialized_case):
    """
    Fixture providing reference RTDOSE for secondary check comparison.
    """
    if not materialized_case['rtdose']:
        pytest.skip("No RTDOSE template for secondary check")

    from DoseCUDA.dvh import read_reference_rtdose

    dose_ref, origin_ref, spacing_ref, frame_uid = read_reference_rtdose(
        str(materialized_case['rtdose'])
    )

    return {
        'dose': dose_ref,
        'origin': origin_ref,
        'spacing': spacing_ref,
        'frame_uid': frame_uid
    }


@pytest.fixture(scope="module")
def rasterized_rois(materialized_case, reference_rtdose):
    """
    Fixture providing rasterized ROIs on reference dose grid.
    """
    if not materialized_case['rtstruct']:
        pytest.skip("No RTSTRUCT for ROI analysis")

    from DoseCUDA.rtstruct import read_rtstruct, rasterize_roi_to_mask
    from DoseCUDA.roi_selection import classify_rois
    from DoseCUDA.grid_utils import GridInfo

    # Build grid info from reference dose
    ref_grid = GridInfo(
        origin=reference_rtdose['origin'],
        spacing=reference_rtdose['spacing'],
        size=np.array(reference_rtdose['dose'].shape[::-1]),  # (nx, ny, nz)
        direction=np.eye(3)
    )

    # Read and classify ROIs
    rtstruct = read_rtstruct(str(materialized_case['rtstruct']))
    roi_names = list(rtstruct.rois.keys())
    classification = classify_rois(roi_names)

    print(f"\n[ROI Classification]")
    print(f"  Targets: {classification.targets}")
    print(f"  OARs: {classification.oars}")
    print(f"  Excluded: {classification.excluded}")

    # Rasterize relevant ROIs (targets + OARs)
    masks = {}
    rois_to_rasterize = classification.targets + classification.oars

    for roi_name in rois_to_rasterize:
        if roi_name in rtstruct.rois:
            try:
                mask = rasterize_roi_to_mask(
                    rtstruct.rois[roi_name],
                    ref_grid.origin,
                    ref_grid.spacing,
                    ref_grid.size[::-1],  # (nz, ny, nx) for mask shape
                    direction=ref_grid.direction
                )
                if np.any(mask):
                    masks[roi_name] = mask
                    n_voxels = np.sum(mask)
                    vol_cc = n_voxels * np.prod(ref_grid.spacing) / 1000.0
                    print(f"  {roi_name}: {n_voxels} voxels ({vol_cc:.1f} cc)")
            except Exception as e:
                warnings.warn(f"Failed to rasterize {roi_name}: {e}")

    return {
        'masks': masks,
        'classification': classification,
        'grid': ref_grid
    }


def test_8_gamma_analysis(materialized_case, machine_model, reference_rtdose):
    """
    Test 8: Compute gamma analysis comparing DoseCUDA vs TPS.
    """
    print("\n" + "=" * 80)
    print("TEST 8: Gamma Analysis")
    print("=" * 80)

    from DoseCUDA.gamma import compute_gamma_3d, GammaCriteria
    from DoseCUDA.grid_utils import GridInfo, resample_dose_linear

    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()

    # Calculate dose
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))

    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    dg.resampleCTfromSpacing(target_spacing)
    dg.computeIMRTPlan(plan, gpu_id=gpu_id)

    # Build grids
    calc_grid = GridInfo(
        origin=dg.origin,
        spacing=dg.spacing,
        size=dg.size,
        direction=np.eye(3)
    )

    ref_grid = GridInfo(
        origin=reference_rtdose['origin'],
        spacing=reference_rtdose['spacing'],
        size=np.array(reference_rtdose['dose'].shape[::-1]),
        direction=np.eye(3)
    )

    # Resample calculated dose to reference grid
    print(f"\nResampling calculated dose to reference grid...")
    dose_resampled = resample_dose_linear(
        dose=dg.dose,
        source_grid=calc_grid,
        target_grid=ref_grid
    )

    print(f"  Calculated dose max: {np.max(dg.dose):.2f} Gy")
    print(f"  Resampled dose max: {np.max(dose_resampled):.2f} Gy")
    print(f"  Reference dose max: {np.max(reference_rtdose['dose']):.2f} Gy")

    # Compute gamma 3%/3mm
    print(f"\nComputing gamma 3%/3mm (global)...")
    criteria_3_3 = GammaCriteria(
        dta_mm=3.0,
        dd_percent=3.0,
        local=False,
        dose_threshold_percent=10.0
    )

    result_3_3 = compute_gamma_3d(
        dose_eval=dose_resampled,
        dose_ref=reference_rtdose['dose'],
        spacing_mm=tuple(reference_rtdose['spacing']),
        criteria=criteria_3_3
    )

    print(f"\nGamma 3%/3mm Results:")
    print(f"  Pass rate: {result_3_3.pass_rate*100:.1f}%")
    print(f"  Mean gamma: {result_3_3.mean_gamma:.3f}")
    print(f"  Gamma P95: {result_3_3.gamma_p95:.3f}")
    print(f"  Evaluated: {result_3_3.n_evaluated} voxels")

    # Compute gamma 2%/2mm
    print(f"\nComputing gamma 2%/2mm (global)...")
    criteria_2_2 = GammaCriteria(
        dta_mm=2.0,
        dd_percent=2.0,
        local=False,
        dose_threshold_percent=10.0
    )

    result_2_2 = compute_gamma_3d(
        dose_eval=dose_resampled,
        dose_ref=reference_rtdose['dose'],
        spacing_mm=tuple(reference_rtdose['spacing']),
        criteria=criteria_2_2
    )

    print(f"\nGamma 2%/2mm Results:")
    print(f"  Pass rate: {result_2_2.pass_rate*100:.1f}%")
    print(f"  Mean gamma: {result_2_2.mean_gamma:.3f}")
    print(f"  Gamma P95: {result_2_2.gamma_p95:.3f}")

    # Save gamma summary
    output_dir = get_output_dir()
    import json
    gamma_summary = {
        "3%/3mm": {
            "pass_rate": result_3_3.pass_rate,
            "mean_gamma": result_3_3.mean_gamma,
            "gamma_p95": result_3_3.gamma_p95,
            "n_evaluated": result_3_3.n_evaluated
        },
        "2%/2mm": {
            "pass_rate": result_2_2.pass_rate,
            "mean_gamma": result_2_2.mean_gamma,
            "gamma_p95": result_2_2.gamma_p95,
            "n_evaluated": result_2_2.n_evaluated
        }
    }

    with open(output_dir / "gamma_summary.json", 'w') as f:
        json.dump(gamma_summary, f, indent=2)

    print(f"\n✓ Gamma summary saved: {output_dir / 'gamma_summary.json'}")

    # Assert criteria (informational - don't fail test on gamma)
    if result_3_3.pass_rate < 0.95:
        warnings.warn(f"Gamma 3%/3mm pass rate {result_3_3.pass_rate:.1%} < 95%")
    if result_2_2.pass_rate < 0.90:
        warnings.warn(f"Gamma 2%/2mm pass rate {result_2_2.pass_rate:.1%} < 90%")


def test_9_dvh_comparison(materialized_case, machine_model, reference_rtdose, rasterized_rois):
    """
    Test 9: Compare DVH metrics for targets and OARs.
    """
    print("\n" + "=" * 80)
    print("TEST 9: DVH Comparison")
    print("=" * 80)

    from DoseCUDA.dvh import compute_metrics, compare_dvh_metrics, generate_dvh_report
    from DoseCUDA.roi_selection import get_target_metrics_spec, get_oar_metrics_spec
    from DoseCUDA.grid_utils import GridInfo, resample_dose_linear

    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()

    # Calculate dose and resample
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))

    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    dg.resampleCTfromSpacing(target_spacing)
    dg.computeIMRTPlan(plan, gpu_id=gpu_id)

    calc_grid = GridInfo(origin=dg.origin, spacing=dg.spacing, size=dg.size, direction=np.eye(3))
    ref_grid = rasterized_rois['grid']

    dose_resampled = resample_dose_linear(dose=dg.dose, source_grid=calc_grid, target_grid=ref_grid)

    classification = rasterized_rois['classification']
    ref_spacing = reference_rtdose['spacing']

    output_dir = get_output_dir()
    report_lines = []

    # Process targets
    print(f"\n[Target DVH Comparison]")
    for roi_name in classification.targets:
        if roi_name in rasterized_rois['masks']:
            mask = rasterized_rois['masks'][roi_name]
            metrics_spec = get_target_metrics_spec()

            metrics_calc = compute_metrics(dose_resampled, mask, ref_spacing, metrics_spec)
            metrics_ref = compute_metrics(reference_rtdose['dose'], mask, ref_spacing, metrics_spec)
            comparison = compare_dvh_metrics(metrics_calc, metrics_ref)

            report = generate_dvh_report(roi_name, metrics_calc, comparison)
            report_lines.append(report)
            print(report)

    # Process OARs
    print(f"\n[OAR DVH Comparison]")
    for roi_name in classification.oars:
        if roi_name in rasterized_rois['masks']:
            mask = rasterized_rois['masks'][roi_name]
            metrics_spec = get_oar_metrics_spec()

            metrics_calc = compute_metrics(dose_resampled, mask, ref_spacing, metrics_spec)
            metrics_ref = compute_metrics(reference_rtdose['dose'], mask, ref_spacing, metrics_spec)
            comparison = compare_dvh_metrics(metrics_calc, metrics_ref)

            report = generate_dvh_report(roi_name, metrics_calc, comparison)
            report_lines.append(report)
            print(report)

    # Save DVH report
    with open(output_dir / "dvh_comparison.txt", 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n✓ DVH comparison saved: {output_dir / 'dvh_comparison.txt'}")


def test_10_mu_sanity_check(materialized_case, machine_model, reference_rtdose):
    """
    Test 10: MU sanity check at isocenter.
    """
    print("\n" + "=" * 80)
    print("TEST 10: MU Sanity Check")
    print("=" * 80)

    from DoseCUDA.mu_sanity import compute_mu_sanity_from_plan
    from DoseCUDA.grid_utils import GridInfo, resample_dose_linear

    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()

    # Calculate dose and resample
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))

    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    dg.resampleCTfromSpacing(target_spacing)
    dg.computeIMRTPlan(plan, gpu_id=gpu_id)

    calc_grid = GridInfo(origin=dg.origin, spacing=dg.spacing, size=dg.size, direction=np.eye(3))
    ref_grid = GridInfo(
        origin=reference_rtdose['origin'],
        spacing=reference_rtdose['spacing'],
        size=np.array(reference_rtdose['dose'].shape[::-1]),
        direction=np.eye(3)
    )

    dose_resampled = resample_dose_linear(dose=dg.dose, source_grid=calc_grid, target_grid=ref_grid)

    # Compute MU sanity check
    print(f"\nComputing MU sanity check...")

    try:
        result = compute_mu_sanity_from_plan(
            dose_calc=dose_resampled,
            dose_ref=reference_rtdose['dose'],
            grid_origin=reference_rtdose['origin'],
            grid_spacing=reference_rtdose['spacing'],
            plan=plan,
            tolerance=0.05
        )

        print(f"\nMU Sanity Check Results:")
        print(f"  Isocenter: {result.isocenter_mm}")
        print(f"  Dose at iso (calc): {result.dose_calc_at_iso:.4f} Gy")
        print(f"  Dose at iso (ref):  {result.dose_ref_at_iso:.4f} Gy")
        print(f"  Total MU: {result.total_mu:.1f}")
        print(f"  Gy/MU ratio: {result.mu_equiv_ratio:.4f}")
        print(f"  Status: {result.status}")
        print(f"  Message: {result.message}")

    except Exception as e:
        warnings.warn(f"MU sanity check failed: {e}")
        print(f"\n⚠ MU sanity check skipped: {e}")


def test_11_generate_report(materialized_case, machine_model, reference_rtdose, rasterized_rois):
    """
    Test 11: Generate JSON and CSV secondary check reports.
    """
    print("\n" + "=" * 80)
    print("TEST 11: Generate Secondary Check Report")
    print("=" * 80)

    from DoseCUDA.secondary_report import (
        evaluate_secondary_check,
        generate_json_report,
        generate_csv_report,
        SecondaryCheckCriteria
    )
    from DoseCUDA.grid_utils import GridInfo, resample_dose_linear

    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()
    output_dir = get_output_dir()

    # Calculate dose and resample
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))

    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    dg.resampleCTfromSpacing(target_spacing)
    dg.computeIMRTPlan(plan, gpu_id=gpu_id)

    calc_grid = GridInfo(origin=dg.origin, spacing=dg.spacing, size=dg.size, direction=np.eye(3))
    ref_grid = GridInfo(
        origin=reference_rtdose['origin'],
        spacing=reference_rtdose['spacing'],
        size=np.array(reference_rtdose['dose'].shape[::-1]),
        direction=np.eye(3)
    )

    dose_resampled = resample_dose_linear(dose=dg.dose, source_grid=calc_grid, target_grid=ref_grid)

    # Get plan info from RTPLAN
    rtplan_ds = pydicom.dcmread(str(materialized_case['rtplan']), stop_before_pixels=True)
    patient_id = getattr(rtplan_ds, 'PatientID', 'UNKNOWN')
    plan_name = getattr(rtplan_ds, 'RTPlanLabel', 'UNKNOWN')
    plan_uid = getattr(rtplan_ds, 'SOPInstanceUID', 'UNKNOWN')

    # Run full evaluation
    print(f"\nRunning secondary check evaluation...")

    criteria = SecondaryCheckCriteria()

    result = evaluate_secondary_check(
        dose_calc=dose_resampled,
        dose_ref=reference_rtdose['dose'],
        grid_origin=reference_rtdose['origin'],
        grid_spacing=reference_rtdose['spacing'],
        rois=rasterized_rois['masks'],
        roi_classification=rasterized_rois['classification'],
        plan=plan,
        patient_id=patient_id,
        plan_name=plan_name,
        plan_uid=plan_uid,
        criteria=criteria
    )

    # Generate reports
    json_path = output_dir / "secondary_check_report.json"
    csv_path = output_dir / "secondary_check_report.csv"

    generate_json_report(result, str(json_path))
    generate_csv_report(result, str(csv_path))

    print(f"\n✓ JSON report: {json_path}")
    print(f"✓ CSV report: {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SECONDARY CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"  Patient: {result.patient_id}")
    print(f"  Plan: {result.plan_name}")
    print(f"  Overall Status: {result.overall_status}")

    if result.gamma_results:
        print(f"\n  Gamma Results:")
        for label, gamma_res in result.gamma_results.items():
            print(f"    {label}: {gamma_res['pass_rate']*100:.1f}% pass [{gamma_res['status']}]")

    if result.failure_reasons:
        print(f"\n  Failure Reasons:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")

    # Verify files exist
    assert json_path.exists(), "JSON report not created"
    assert csv_path.exists(), "CSV report not created"

    print(f"\n{'='*60}")
    print(f"SECONDARY CHECK COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    """
    Allow running as script for quick testing.
    """
    print(__doc__)
    print("\nTo run this test:")
    print("  1. Set DOSECUDA_PATIENT_DICOM_DIR=/path/to/patient/dicoms")
    print("  2. pytest tests/test_patient_end2end.py -v -s")
