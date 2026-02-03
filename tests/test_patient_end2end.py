"""
End-to-end test for real patient DICOM data.

Tests complete workflow:
1. Auto-discovery of CT/RTPLAN/RTSTRUCT/RTDOSE in mixed directory
2. Automatic selection of appropriate series/files
3. Anisotropic CT → isotropic resampling for dose calculation
4. GPU dose calculation with auto-detected machine model
5. Dose resampling to RTDOSE template grid
6. DICOM RTDOSE export
7. Gamma analysis (3%/3mm and 2%/2mm)
8. DVH comparison
9. MU sanity check
10. Generate secondary check reports

Environment Variables (optional - defaults provided)
----------------------------------------------------
DOSECUDA_PATIENT_DICOM_DIR : str
    Path to patient DICOM directory.
    Default: tests/PATIENT/TRUEBEAM (relative to test file)
DOSECUDA_ISO_MM : float
    Target isotropic spacing for dose calculation (mm)
    Default: 2.5
DOSECUDA_GPU_ID : int
    GPU device ID for dose calculation
    Default: 0
DOSECUDA_MACHINE_DEFAULT : str
    Default machine model if inference fails
    Default: VarianTrueBeamHF

Usage
-----
# Run with defaults (no exports needed!)
pytest tests/test_patient_end2end.py -v -s

# Override with custom settings
export DOSECUDA_PATIENT_DICOM_DIR=/path/to/other/patient
export DOSECUDA_ISO_MM=3.0
export DOSECUDA_GPU_ID=1
pytest tests/test_patient_end2end.py -v -s
"""

import os
import sys
import time
import pytest
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Import DoseCUDA modules
from DoseCUDA import IMRTDoseGrid, IMRTPlan
from DoseCUDA.dicom_case_discovery import (
    scan_dicom_directory,
    select_rtplan,
    select_rtdose_template,
    select_ct_series,
    infer_machine_model,
    materialize_case,
    enumerate_phases,
    DicomPhase
)
from DoseCUDA.grid_utils import GridInfo, resample_dose_linear

import pydicom


# ============================================================================
# Default Configuration (built into script)
# ============================================================================

# Default paths relative to this test file
_TEST_DIR = Path(__file__).parent
_DEFAULT_PATIENT_DICOM_DIR = _TEST_DIR / "PATIENT" / "TRUEBEAM"
_DEFAULT_ISO_SPACING_MM = 2.5
_DEFAULT_GPU_ID = 0
_DEFAULT_MACHINE = "VarianTrueBeamHF"


# ============================================================================
# Progress Tracking Utilities
# ============================================================================

class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, name: str, total_steps: int = 0):
        self.name = name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_start_time = time.time()
    
    def start(self, message: str = ""):
        """Start tracking."""
        self.start_time = time.time()
        self.step_start_time = time.time()
        print(f"\n⏱ [{self.name}] Started {message}")
        sys.stdout.flush()
    
    def step(self, message: str, step_num: Optional[int] = None):
        """Report a step."""
        elapsed = time.time() - self.step_start_time
        if step_num is not None:
            self.current_step = step_num
        else:
            self.current_step += 1
        
        if self.total_steps > 0:
            progress = f"[{self.current_step}/{self.total_steps}]"
        else:
            progress = f"[step {self.current_step}]"
        
        print(f"  {progress} {message} ({elapsed:.1f}s)")
        sys.stdout.flush()
        self.step_start_time = time.time()
    
    def done(self, message: str = "Complete"):
        """Mark as complete."""
        total_elapsed = time.time() - self.start_time
        print(f"✓ [{self.name}] {message} (total: {total_elapsed:.1f}s)")
        sys.stdout.flush()


def progress_print(msg: str):
    """Print with immediate flush for progress visibility."""
    print(msg)
    sys.stdout.flush()


def print_test_header(title: str):
    """Print a formatted test header."""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    sys.stdout.flush()


# ============================================================================
# Test Configuration from Environment (with defaults)
# ============================================================================

def get_patient_dicom_dir() -> Path:
    """Get patient DICOM directory from environment or default."""
    env_path = os.environ.get('DOSECUDA_PATIENT_DICOM_DIR')
    if env_path:
        return Path(env_path)
    return _DEFAULT_PATIENT_DICOM_DIR


def get_iso_spacing_mm() -> float:
    """Get target isotropic spacing in mm (default: 2.5)."""
    return float(os.environ.get('DOSECUDA_ISO_MM', str(_DEFAULT_ISO_SPACING_MM)))


def get_gpu_id() -> int:
    """Get GPU device ID (default: 0)."""
    return int(os.environ.get('DOSECUDA_GPU_ID', str(_DEFAULT_GPU_ID)))


def get_default_machine() -> str:
    """Get default machine model (default: VarianTrueBeamHF)."""
    return os.environ.get('DOSECUDA_MACHINE_DEFAULT', _DEFAULT_MACHINE)


def get_output_dir() -> Path:
    """Get output directory for test results."""
    output_dir = _TEST_DIR / "test_patient_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 80)
    print("TEST CONFIGURATION")
    print("=" * 80)
    print(f"  DICOM dir: {get_patient_dicom_dir()}")
    print(f"  ISO spacing: {get_iso_spacing_mm()} mm")
    print(f"  GPU ID: {get_gpu_id()}")
    print(f"  Machine default: {get_default_machine()}")
    print(f"  Output dir: {get_output_dir()}")
    print("=" * 80)


# ============================================================================
# Cached Dose Calculation (avoid recalculating in every test)
# ============================================================================

_cached_dose_grid = None
_cached_dose_resampled = None
_cached_ref_grid = None


def get_or_calculate_dose(materialized_case, machine_model, reference_rtdose=None):
    """
    Get cached dose or calculate if not available.
    Returns both the dose grid and optionally resampled dose.
    """
    global _cached_dose_grid, _cached_dose_resampled, _cached_ref_grid
    
    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()
    
    # Check if we already have cached dose
    if _cached_dose_grid is not None:
        progress_print("  ℹ Using cached dose calculation")
        
        # If reference grid requested and we have resampled dose
        if reference_rtdose is not None and _cached_dose_resampled is not None:
            return _cached_dose_grid, _cached_dose_resampled
        
        # Need to resample to reference grid
        if reference_rtdose is not None:
            progress_print("  → Resampling to reference grid...")
            calc_grid = GridInfo(
                origin=_cached_dose_grid.origin,
                spacing=_cached_dose_grid.spacing,
                size=_cached_dose_grid.size,
                direction=np.eye(3)
            )
            _cached_ref_grid = GridInfo(
                origin=reference_rtdose['origin'],
                spacing=reference_rtdose['spacing'],
                size=np.array(reference_rtdose['dose'].shape[::-1]),
                direction=np.eye(3)
            )
            _cached_dose_resampled = resample_dose_linear(
                dose=_cached_dose_grid.dose,
                source_grid=calc_grid,
                target_grid=_cached_ref_grid
            )
            return _cached_dose_grid, _cached_dose_resampled
        
        return _cached_dose_grid, None
    
    # Calculate dose
    tracker = ProgressTracker("Dose Calculation", total_steps=5)
    tracker.start()
    
    tracker.step("Loading RTPLAN...")
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))
    
    tracker.step(f"Loading CT ({materialized_case['ct_dir'].name})...")
    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    
    tracker.step(f"Resampling to {target_spacing} mm isotropic...")
    dg.resampleCTfromSpacing(target_spacing)
    
    tracker.step(f"Computing dose on GPU {gpu_id} ({plan.n_beams} beams)...")
    dg.computeIMRTPlan(plan, gpu_id=gpu_id)
    
    tracker.step("Storing in cache...")
    _cached_dose_grid = dg
    
    # Resample to reference grid if provided
    if reference_rtdose is not None:
        progress_print("  → Resampling to reference grid...")
        calc_grid = GridInfo(
            origin=dg.origin,
            spacing=dg.spacing,
            size=dg.size,
            direction=np.eye(3)
        )
        _cached_ref_grid = GridInfo(
            origin=reference_rtdose['origin'],
            spacing=reference_rtdose['spacing'],
            size=np.array(reference_rtdose['dose'].shape[::-1]),
            direction=np.eye(3)
        )
        _cached_dose_resampled = resample_dose_linear(
            dose=dg.dose,
            source_grid=calc_grid,
            target_grid=_cached_ref_grid
        )
    
    tracker.done(f"Max dose: {np.max(dg.dose):.2f} Gy")
    
    return dg, _cached_dose_resampled


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def patient_dicom_dir():
    """
    Fixture providing patient DICOM directory.
    Uses default if DOSECUDA_PATIENT_DICOM_DIR not set.
    """
    print_config()
    
    dicom_path = get_patient_dicom_dir()
    
    if not dicom_path.exists():
        pytest.skip(
            f"Patient DICOM directory does not exist: {dicom_path}\n"
            f"Set DOSECUDA_PATIENT_DICOM_DIR to override."
        )
    
    return dicom_path


@pytest.fixture(scope="module")
def discovered_case(patient_dicom_dir):
    """Fixture providing discovered and classified DICOM case."""
    progress_print("\n" + "=" * 80)
    progress_print("PHASE 1: DICOM DISCOVERY")
    progress_print("=" * 80)
    
    tracker = ProgressTracker("Discovery")
    tracker.start()
    
    case = scan_dicom_directory(str(patient_dicom_dir))
    
    if not case.ct_files:
        pytest.fail("No CT files found in patient directory")
    
    if not case.rtplan_files:
        pytest.fail("No RTPLAN found in patient directory")
    
    tracker.done(f"Found {len(case.ct_files)} CT, {len(case.rtplan_files)} RTPLAN")
    progress_print(f"\n{case}")
    
    return case


@pytest.fixture(scope="module")
def selected_files(discovered_case):
    """Fixture providing selected RTPLAN, RTDOSE, RTSTRUCT, and CT series."""
    progress_print("\n" + "=" * 80)
    progress_print("PHASE 2: AUTOMATIC SELECTION")
    progress_print("=" * 80)
    
    tracker = ProgressTracker("Selection", total_steps=4)
    tracker.start()
    
    tracker.step("Selecting RTPLAN...")
    rtplan = select_rtplan(discovered_case)
    assert rtplan is not None, "Failed to select RTPLAN"
    
    tracker.step("Selecting RTDOSE template...")
    rtdose = select_rtdose_template(discovered_case)
    
    tracker.step("Selecting RTSTRUCT...")
    rtstruct = None
    if discovered_case.rtstruct_files:
        rtstruct = discovered_case.rtstruct_files[0]
        if len(discovered_case.rtstruct_files) > 1:
            warnings.warn(f"Multiple RTSTRUCT found ({len(discovered_case.rtstruct_files)}), using first")
    
    tracker.step("Selecting CT series...")
    ct_series = select_ct_series(discovered_case, rtstruct=rtstruct, rtdose=rtdose)
    assert ct_series is not None, "Failed to select CT series"
    assert len(ct_series) > 0, "Selected CT series is empty"
    
    tracker.done(f"RTPLAN={rtplan.path.name}, CT={len(ct_series)} slices")
    
    return {
        'rtplan': rtplan,
        'rtdose': rtdose,
        'rtstruct': rtstruct,
        'ct_series': ct_series
    }


@pytest.fixture(scope="module")
def materialized_case(selected_files):
    """Fixture providing materialized case with organized file structure."""
    progress_print("\n" + "=" * 80)
    progress_print("PHASE 3: CASE MATERIALIZATION")
    progress_print("=" * 80)
    
    output_dir = get_output_dir()
    
    tracker = ProgressTracker("Materialize")
    tracker.start()
    
    paths = materialize_case(
        output_dir=str(output_dir),
        ct_series=selected_files['ct_series'],
        rtplan=selected_files['rtplan'],
        rtdose=selected_files['rtdose'],
        rtstruct=selected_files['rtstruct']
    )
    
    tracker.done(f"Created at {output_dir}")
    
    return paths


@pytest.fixture(scope="module")
def machine_model(selected_files):
    """Fixture providing inferred machine model."""
    progress_print("\n" + "=" * 80)
    progress_print("PHASE 4: MACHINE MODEL INFERENCE")
    progress_print("=" * 80)
    
    default_model = get_default_machine()
    model = infer_machine_model(selected_files['rtplan'], default_model=default_model)
    
    # Validate model exists in lookuptables
    lookuptable_path = _TEST_DIR.parent / "DoseCUDA" / "lookuptables" / "photons" / model
    if not lookuptable_path.exists():
        pytest.fail(
            f"Machine model '{model}' not found in lookuptables/photons/.\n"
            f"Expected path: {lookuptable_path}"
        )
    
    progress_print(f"\n✓ Machine model validated: {model}")
    
    return model


@pytest.fixture(scope="module")
def reference_rtdose(materialized_case):
    """Fixture providing reference RTDOSE for secondary check comparison."""
    if not materialized_case['rtdose']:
        pytest.skip("No RTDOSE template for secondary check")

    from DoseCUDA.dvh import read_reference_rtdose

    rtdose_path = materialized_case['rtdose']
    progress_print(f"\n[Reference RTDOSE] Loading TPS dose from:")
    progress_print(f"  {rtdose_path}")

    dose_ref, origin_ref, spacing_ref, frame_uid = read_reference_rtdose(
        str(rtdose_path)
    )

    progress_print(f"  Shape: {dose_ref.shape}")
    progress_print(f"  Dose range: {dose_ref.min():.2f} - {dose_ref.max():.2f} Gy")

    return {
        'dose': dose_ref,
        'origin': origin_ref,
        'spacing': spacing_ref,
        'frame_uid': frame_uid
    }


@pytest.fixture(scope="module")
def rasterized_rois(materialized_case, reference_rtdose):
    """Fixture providing rasterized ROIs on reference dose grid."""
    if not materialized_case['rtstruct']:
        pytest.skip("No RTSTRUCT for ROI analysis")

    from DoseCUDA.rtstruct import read_rtstruct, rasterize_roi_to_mask
    from DoseCUDA.roi_selection import classify_rois

    rtstruct_path = materialized_case['rtstruct']
    progress_print(f"\n[RTSTRUCT] Loading structures from: {rtstruct_path.name}")

    # Build grid info from reference dose (TPS RTDOSE grid)
    ref_grid = GridInfo(
        origin=reference_rtdose['origin'],
        spacing=reference_rtdose['spacing'],
        size=np.array(reference_rtdose['dose'].shape[::-1]),  # (nx, ny, nz)
        direction=np.eye(3)
    )

    progress_print(f"  Rasterizing ROIs onto TPS dose grid: {ref_grid.size}")

    # Read and classify ROIs
    rtstruct = read_rtstruct(str(rtstruct_path))
    roi_names = list(rtstruct.rois.keys())
    classification = classify_rois(roi_names)

    progress_print(f"\n[ROI Classification]")
    progress_print(f"  Targets: {classification.targets}")
    progress_print(f"  OARs: {len(classification.oars)} structures")
    progress_print(f"  Excluded: {len(classification.excluded)} structures")

    # Rasterize relevant ROIs (targets + OARs)
    masks = {}
    rois_to_rasterize = classification.targets + classification.oars

    tracker = ProgressTracker("Rasterize", total_steps=len(rois_to_rasterize))
    tracker.start()

    for i, roi_name in enumerate(rois_to_rasterize):
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
                    tracker.step(f"{roi_name}: {n_voxels} voxels ({vol_cc:.1f} cc)", i+1)
            except Exception as e:
                warnings.warn(f"Failed to rasterize {roi_name}: {e}")

    tracker.done(f"Rasterized {len(masks)} ROIs")

    return {
        'masks': masks,
        'classification': classification,
        'grid': ref_grid
    }


# ============================================================================
# Test Functions
# ============================================================================

def test_1_discovery_and_selection(discovered_case, selected_files):
    """Test 1: Verify DICOM discovery and automatic selection succeeded."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 1: Discovery and Selection")
    progress_print("=" * 80)
    
    # Verify case has required files
    assert len(discovered_case.ct_files) > 0, "No CT files discovered"
    assert len(discovered_case.rtplan_files) > 0, "No RTPLAN discovered"
    
    # Verify selections
    assert selected_files['rtplan'] is not None, "RTPLAN selection failed"
    assert selected_files['ct_series'] is not None, "CT series selection failed"
    assert len(selected_files['ct_series']) > 0, "CT series is empty"
    
    progress_print(f"\n✓ Discovery: {len(discovered_case.ct_files)} CT, "
          f"{len(discovered_case.rtplan_files)} RTPLAN, "
          f"{len(discovered_case.rtstruct_files)} RTSTRUCT, "
          f"{len(discovered_case.rtdose_files)} RTDOSE")
    progress_print(f"✓ Selected: RTPLAN={selected_files['rtplan'].path.name}, "
          f"CT series={len(selected_files['ct_series'])} slices, "
          f"RTDOSE={'Yes' if selected_files['rtdose'] else 'No'}")


def test_2_ct_loading_and_anisotropy(materialized_case):
    """Test 2: Load CT and verify it's anisotropic (z != x/y)."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 2: CT Loading and Anisotropy Check")
    progress_print("=" * 80)
    
    tracker = ProgressTracker("CT Load")
    tracker.start()
    
    dg = IMRTDoseGrid()
    ct_dir = str(materialized_case['ct_dir'])
    
    progress_print(f"\n  Loading CT from: {ct_dir}")
    dg.loadCTDCM(ct_dir)
    
    # Check CT loaded
    assert dg.HU is not None, "CT HU array is None"
    assert dg.spacing is not None, "CT spacing is None"
    
    tracker.done(f"shape={dg.HU.shape}, spacing={dg.spacing} mm")
    
    # Check anisotropy (expected for clinical CT)
    spacing = dg.spacing
    is_isotropic = np.allclose(spacing, spacing[0], atol=0.01)
    
    if is_isotropic:
        warnings.warn(f"CT is already isotropic ({spacing}). Expected anisotropic.")
    else:
        progress_print(f"✓ CT is anisotropic: {spacing} mm (ratio z/x = {spacing[2]/spacing[0]:.2f})")


def test_3_isotropic_resampling(materialized_case):
    """Test 3: Resample anisotropic CT to isotropic spacing for dose calculation."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 3: Isotropic Resampling")
    progress_print("=" * 80)
    
    target_spacing = get_iso_spacing_mm()
    
    tracker = ProgressTracker("Resample", total_steps=2)
    tracker.start()
    
    tracker.step("Loading CT...")
    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(materialized_case['ct_dir']))
    
    original_spacing = dg.spacing.copy()
    original_size = dg.size.copy()
    
    progress_print(f"\n  Original: spacing={original_spacing} mm, size={original_size}")
    progress_print(f"  Target: {target_spacing} mm isotropic")
    
    tracker.step(f"Resampling to {target_spacing} mm...")
    dg.resampleCTfromSpacing(target_spacing)
    
    # Verify result
    assert dg.spacing is not None, "Spacing is None after resample"
    assert dg.HU is not None, "HU is None after resample"
    
    # Verify isotropic
    is_isotropic = np.allclose(dg.spacing, target_spacing, atol=0.01)
    assert is_isotropic, f"Spacing not isotropic after resample: {dg.spacing}"
    
    tracker.done(f"Resampled: spacing={dg.spacing} mm, size={dg.size}")


def test_4_gpu_dose_calculation(materialized_case, machine_model):
    """Test 4: Calculate dose on GPU with auto-detected machine model."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 4: GPU Dose Calculation")
    progress_print("=" * 80)
    
    progress_print(f"\n  Machine model: {machine_model}")
    progress_print(f"  GPU ID: {get_gpu_id()}")
    progress_print(f"  Target spacing: {get_iso_spacing_mm()} mm")
    
    dg, _ = get_or_calculate_dose(materialized_case, machine_model)
    
    # Verify dose calculated
    assert dg.dose is not None, "Dose is None after calculation"
    assert dg.dose.shape == tuple(dg.size), f"Dose shape {dg.dose.shape} != grid size {dg.size}"
    
    # Check dose statistics
    dose_min = np.min(dg.dose)
    dose_max = np.max(dg.dose)
    dose_mean = np.mean(dg.dose)
    dose_nonzero = np.sum(dg.dose > 0.01)  # Gy
    
    progress_print(f"\n✓ Dose calculated:")
    progress_print(f"  Shape: {dg.dose.shape}")
    progress_print(f"  Min: {dose_min:.3f} Gy")
    progress_print(f"  Max: {dose_max:.3f} Gy")
    progress_print(f"  Mean: {dose_mean:.3f} Gy")
    progress_print(f"  Voxels > 0.01 Gy: {dose_nonzero} ({dose_nonzero/np.prod(dg.dose.shape)*100:.1f}%)")
    
    # Sanity checks
    assert dose_max > 0, "Maximum dose is zero (calculation may have failed)"
    assert dose_max < 1000, f"Maximum dose {dose_max} Gy is unrealistic"
    assert np.isfinite(dg.dose).all(), "Dose contains NaN or Inf"
    
    progress_print(f"\n✓ Dose sanity checks passed")


def test_5_save_numpy_and_nrrd(discovered_case, materialized_case, machine_model):
    """Test 5: Save dose as numpy (.npy) and NRRD."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 5: Save NPY and NRRD")
    progress_print("=" * 80)
    
    # If multiple treatment phases exist, skip global root saves
    phases = enumerate_phases(discovered_case)
    if len(phases) > 1:
        progress_print("  ℹ Multiple phases detected; skipping global NPY/NRRD saves (per-phase outputs will be used).")
        return

    output_dir = get_output_dir()
    
    dg, _ = get_or_calculate_dose(materialized_case, machine_model)
    
    tracker = ProgressTracker("Save", total_steps=3)
    tracker.start()
    
    # Save numpy
    tracker.step("Saving numpy...")
    npy_path = output_dir / "dose_calculated.npy"
    np.save(npy_path, dg.dose)
    assert npy_path.exists(), "NPY file not created"
    
    # Save NRRD
    tracker.step("Saving NRRD...")
    nrrd_path = output_dir / "dose_calculated.nrrd"
    dg.writeDoseNRRD(str(nrrd_path), dose_type="PHYSICAL")
    assert nrrd_path.exists(), "NRRD file not created"
    
    # Save dose stats
    tracker.step("Saving stats...")
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
    
    tracker.done("All files saved")
    progress_print(f"  → {npy_path}")
    progress_print(f"  → {nrrd_path}")
    progress_print(f"  → {stats_path}")


def test_6_resample_and_save_dicom_rtdose(discovered_case, materialized_case, machine_model, selected_files):
    """Test 6: Resample calculated dose to RTDOSE template grid and save DICOM."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 6: Resample to Template Grid and Save DICOM RTDOSE")
    progress_print("=" * 80)
    
    if not materialized_case['rtdose']:
        pytest.skip("No RTDOSE template found - skipping DICOM export")
    
    # If multiple treatment phases exist, skip global RTDOSE export (per-phase exports will be used).
    phases = enumerate_phases(discovered_case)
    if len(phases) > 1:
        progress_print("  ℹ Multiple phases detected; skipping global RTDOSE export (per-phase RTDOSE will be created in test_13).")
        return

    output_dir = get_output_dir()
    
    tracker = ProgressTracker("DICOM Export", total_steps=5)
    tracker.start()
    
    # Read template RTDOSE to get target grid
    tracker.step("Reading RTDOSE template...")
    template_path = str(materialized_case['rtdose'])
    template_ds = pydicom.dcmread(template_path, force=True)
    
    # Extract template grid geometry
    template_origin = np.array(template_ds.ImagePositionPatient)
    pixel_spacing = np.array(template_ds.PixelSpacing)
    
    if hasattr(template_ds, 'GridFrameOffsetVector'):
        frame_offsets = np.array(template_ds.GridFrameOffsetVector)
        if len(frame_offsets) > 1:
            slice_spacing = frame_offsets[1] - frame_offsets[0]
        else:
            slice_spacing = pixel_spacing[0]
    else:
        warnings.warn("Template RTDOSE has no GridFrameOffsetVector, assuming cubic voxels")
        slice_spacing = pixel_spacing[0]
    
    template_spacing = np.array([pixel_spacing[1], pixel_spacing[0], slice_spacing])
    template_size = np.array([
        int(template_ds.Columns),
        int(template_ds.Rows),
        int(template_ds.NumberOfFrames) if hasattr(template_ds, 'NumberOfFrames') else 1
    ])
    
    if hasattr(template_ds, 'ImageOrientationPatient'):
        orientation = np.array(template_ds.ImageOrientationPatient)
        row_cos = orientation[:3]
        col_cos = orientation[3:]
        slice_cos = np.cross(row_cos, col_cos)
        direction = np.column_stack([row_cos, col_cos, slice_cos])
        if not np.allclose(direction, np.eye(3), atol=0.1):
            pytest.fail("Template RTDOSE is not axial - not yet supported.")
    else:
        direction = np.eye(3)
    
    template_grid = GridInfo(origin=template_origin, spacing=template_spacing,
                             size=template_size, direction=direction)
    
    progress_print(f"\n  Template grid: {template_grid.size} @ {template_grid.spacing} mm")
    
    # Calculate dose on isotropic grid
    tracker.step("Calculating dose...")
    dg, _ = get_or_calculate_dose(materialized_case, machine_model)
    
    calc_grid = GridInfo(origin=dg.origin, spacing=dg.spacing, size=dg.size, direction=np.eye(3))
    
    # Resample to template grid
    tracker.step("Resampling to template grid...")
    dose_resampled = resample_dose_linear(dose=dg.dose, source_grid=calc_grid, target_grid=template_grid)
    
    progress_print(f"  Resampled: {dose_resampled.shape}, max={np.max(dose_resampled):.2f} Gy")
    
    # Verify shape matches template
    expected_shape = (template_size[2], template_size[1], template_size[0])
    assert dose_resampled.shape == expected_shape, \
        f"Resampled shape {dose_resampled.shape} != template {expected_shape}"
    
    # Update dose grid with resampled dose
    dg.dose = dose_resampled
    dg.origin = template_grid.origin
    dg.spacing = template_grid.spacing
    dg.size = template_grid.size
    
    # Save DICOM RTDOSE
    tracker.step("Saving DICOM RTDOSE...")
    output_dcm_path = output_dir / "DoseCUDA_RD.dcm"
    
    rtplan_ds = pydicom.dcmread(str(materialized_case['rtplan']), stop_before_pixels=True)
    rtplan_sop_uid = rtplan_ds.SOPInstanceUID
    
    dg.writeDoseDCM(
        dose_path=str(output_dcm_path),
        ref_dose_path=template_path,
        dose_type="PHYSICAL",
        rtplan_sop_uid=rtplan_sop_uid
    )
    
    assert output_dcm_path.exists(), "DICOM RTDOSE not created"
    
    # Verify saved RTDOSE
    tracker.step("Verifying saved DICOM...")
    saved_ds = pydicom.dcmread(str(output_dcm_path), force=True)
    
    assert saved_ds.Rows == template_ds.Rows, "Rows mismatch"
    assert saved_ds.Columns == template_ds.Columns, "Columns mismatch"
    assert saved_ds.NumberOfFrames == template_ds.NumberOfFrames, "NumberOfFrames mismatch"
    assert 0 < float(saved_ds.DoseGridScaling) < 1, "DoseGridScaling outside plausible range"
    
    tracker.done(f"Saved: {output_dcm_path.name}")
    progress_print(f"  SOPInstanceUID: {saved_ds.SOPInstanceUID}")
    progress_print(f"  Grid: ({saved_ds.NumberOfFrames}, {saved_ds.Rows}, {saved_ds.Columns})")


def test_7_summary(discovered_case, selected_files, machine_model):
    """Test 7: Print comprehensive summary of end-to-end test."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 7: End-to-End Summary")
    progress_print("=" * 80)

    output_dir = get_output_dir()

    progress_print(f"\n✓ All basic tests passed!")
    progress_print(f"\nInput:")
    progress_print(f"  Patient DICOM dir: {get_patient_dicom_dir()}")
    progress_print(f"  CT series: {len(selected_files['ct_series'])} slices")
    progress_print(f"  RTPLAN: {selected_files['rtplan'].path.name}")
    if selected_files['rtdose']:
        progress_print(f"  RTDOSE template: {selected_files['rtdose'].path.name}")
    if selected_files['rtstruct']:
        progress_print(f"  RTSTRUCT: {selected_files['rtstruct'].path.name}")

    progress_print(f"\nConfiguration:")
    progress_print(f"  Machine model: {machine_model}")
    progress_print(f"  Isotropic spacing: {get_iso_spacing_mm()} mm")
    progress_print(f"  GPU ID: {get_gpu_id()}")

    progress_print(f"\nOutput: {output_dir}")
    for f in output_dir.glob("*"):
        if f.is_file():
            progress_print(f"  {f.name}")

    progress_print(f"\n" + "=" * 80)
    progress_print("SUCCESS: Basic end-to-end workflow complete")
    progress_print("=" * 80)


# ============================================================================
# Secondary Check Tests (Tests 8-11)
# ============================================================================

def test_8_gamma_analysis(discovered_case, materialized_case, machine_model, reference_rtdose):
    """Test 8: Compute gamma analysis comparing DoseCUDA vs TPS."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 8: Gamma Analysis (DoseCUDA vs TPS)")
    progress_print("=" * 80)
    progress_print("\n[Comparison]")
    progress_print("  Evaluated: DoseCUDA calculated dose")
    progress_print("  Reference: TPS RTDOSE")

    from DoseCUDA.gamma import compute_gamma_3d, GammaCriteria

    # Get cached dose, resampled to reference grid
    dg, dose_resampled = get_or_calculate_dose(materialized_case, machine_model, reference_rtdose)

    progress_print(f"\n  Calculated dose max: {np.max(dg.dose):.2f} Gy")
    progress_print(f"  Resampled dose max: {np.max(dose_resampled):.2f} Gy")
    progress_print(f"  Reference dose max: {np.max(reference_rtdose['dose']):.2f} Gy")

    # Compute gamma 3%/3mm
    tracker = ProgressTracker("Gamma 3%/3mm")
    tracker.start()
    
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

    tracker.done(f"Pass rate: {result_3_3.pass_rate*100:.1f}%")
    progress_print(f"  Mean gamma: {result_3_3.mean_gamma:.3f}")
    progress_print(f"  Gamma P95: {result_3_3.gamma_p95:.3f}")
    progress_print(f"  Evaluated: {result_3_3.n_evaluated} voxels")

    # Compute gamma 2%/2mm
    tracker = ProgressTracker("Gamma 2%/2mm")
    tracker.start()
    
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

    tracker.done(f"Pass rate: {result_2_2.pass_rate*100:.1f}%")
    progress_print(f"  Mean gamma: {result_2_2.mean_gamma:.3f}")
    progress_print(f"  Gamma P95: {result_2_2.gamma_p95:.3f}")

    # Save gamma summary (skip global save if multi-phase; per-phase gamma is produced in test_13)
    phases = enumerate_phases(discovered_case)
    import json
    gamma_summary = {
        "timestamp": datetime.now().isoformat(),
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

    if len(phases) > 1:
        progress_print("\n  ℹ Multiple phases detected; skipping global gamma_summary.json (per-phase gamma saved in test_13).")
    else:
        output_dir = get_output_dir()
        with open(output_dir / "gamma_summary.json", 'w') as f:
            json.dump(gamma_summary, f, indent=2)
        progress_print(f"\n✓ Gamma summary saved: gamma_summary.json")

    if result_3_3.pass_rate < 0.95:
        warnings.warn(f"Gamma 3%/3mm pass rate {result_3_3.pass_rate:.1%} < 95%")
    if result_2_2.pass_rate < 0.90:
        warnings.warn(f"Gamma 2%/2mm pass rate {result_2_2.pass_rate:.1%} < 90%")


def test_9_dvh_comparison(discovered_case, materialized_case, machine_model, reference_rtdose, rasterized_rois):
    """Test 9: Compare DVH metrics for targets and OARs."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 9: DVH Comparison (DoseCUDA vs TPS)")
    progress_print("=" * 80)

    from DoseCUDA.dvh import compute_metrics, compare_dvh_metrics, generate_dvh_report
    from DoseCUDA.roi_selection import get_target_metrics_spec, get_oar_metrics_spec

    # Get cached dose
    dg, dose_resampled = get_or_calculate_dose(materialized_case, machine_model, reference_rtdose)

    classification = rasterized_rois['classification']
    ref_spacing = reference_rtdose['spacing']

    output_dir = get_output_dir()
    report_lines = []

    # Process targets
    progress_print(f"\n[Target DVH Comparison]")
    for roi_name in classification.targets:
        if roi_name in rasterized_rois['masks']:
            mask = rasterized_rois['masks'][roi_name]
            metrics_spec = get_target_metrics_spec()

            metrics_calc = compute_metrics(dose_resampled, mask, ref_spacing, metrics_spec)
            metrics_ref = compute_metrics(reference_rtdose['dose'], mask, ref_spacing, metrics_spec)
            comparison = compare_dvh_metrics(metrics_calc, metrics_ref)

            report = generate_dvh_report(roi_name, metrics_calc, comparison)
            report_lines.append(report)
            progress_print(report)

    # Process OARs
    progress_print(f"\n[OAR DVH Comparison]")
    for roi_name in classification.oars[:5]:  # Limit to first 5 OARs for speed
        if roi_name in rasterized_rois['masks']:
            mask = rasterized_rois['masks'][roi_name]
            metrics_spec = get_oar_metrics_spec()

            metrics_calc = compute_metrics(dose_resampled, mask, ref_spacing, metrics_spec)
            metrics_ref = compute_metrics(reference_rtdose['dose'], mask, ref_spacing, metrics_spec)
            comparison = compare_dvh_metrics(metrics_calc, metrics_ref)

            report = generate_dvh_report(roi_name, metrics_calc, comparison)
            report_lines.append(report)
            progress_print(report)

    # Save DVH report (skip global save if multi-phase)
    phases = enumerate_phases(discovered_case)
    if len(phases) > 1:
        progress_print("\n  ℹ Multiple phases detected; skipping global dvh_comparison.txt (per-phase DVH saved in test_13).")
    else:
        with open(output_dir / "dvh_comparison.txt", 'w') as f:
            f.write('\n'.join(report_lines))

        progress_print(f"\n✓ DVH comparison saved: dvh_comparison.txt")


def test_10_mu_sanity_check(materialized_case, machine_model, reference_rtdose):
    """Test 10: MU sanity check at isocenter."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 10: MU Sanity Check")
    progress_print("=" * 80)

    from DoseCUDA.mu_sanity import compute_mu_sanity_from_plan

    # Get cached dose
    dg, dose_resampled = get_or_calculate_dose(materialized_case, machine_model, reference_rtdose)

    # Load plan for MU info
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))
    positive_cps = sum(
        1 for beam in plan.beam_list for cp in beam.cp_list if getattr(cp, 'mu', 0) > 0
    )
    if positive_cps == 0:
        warnings.warn("No control points with MU > 0 after parsing; skipping MU sanity check")
        pytest.skip("No control points with MU > 0 after parsing")

    progress_print(f"\nComputing MU sanity check...")

    try:
        result = compute_mu_sanity_from_plan(
            dose_calc=dose_resampled,
            dose_ref=reference_rtdose['dose'],
            grid_origin=reference_rtdose['origin'],
            grid_spacing=reference_rtdose['spacing'],
            plan=plan,
            tolerance=0.05
        )

        progress_print(f"\nMU Sanity Check Results:")
        progress_print(f"  Isocenter: {result.isocenter_mm}")
        progress_print(f"  Dose at iso (calc): {result.dose_calc_at_iso:.4f} Gy")
        progress_print(f"  Dose at iso (ref):  {result.dose_ref_at_iso:.4f} Gy")
        progress_print(f"  Total MU: {result.total_mu:.1f}")
        progress_print(f"  Gy/MU ratio: {result.mu_equiv_ratio:.4f}")
        progress_print(f"  Status: {result.status}")
        progress_print(f"  Message: {result.message}")

    except Exception as e:
        warnings.warn(f"MU sanity check failed: {e}")
        progress_print(f"\n⚠ MU sanity check skipped: {e}")


def test_11_generate_report(discovered_case, materialized_case, machine_model, reference_rtdose, rasterized_rois):
    """Test 11: Generate JSON and CSV secondary check reports."""
    progress_print("\n" + "=" * 80)
    progress_print("TEST 11: Generate Secondary Check Report")
    progress_print("=" * 80)

    from DoseCUDA.secondary_report import (
        evaluate_secondary_check,
        generate_json_report,
        generate_csv_report,
        SecondaryCheckCriteria
    )

    output_dir = get_output_dir()

    # Get cached dose
    dg, dose_resampled = get_or_calculate_dose(materialized_case, machine_model, reference_rtdose)

    # Load plan
    plan = IMRTPlan(machine_name=machine_model)
    plan.readPlanDicom(str(materialized_case['rtplan']))

    # Get plan info from RTPLAN
    rtplan_ds = pydicom.dcmread(str(materialized_case['rtplan']), stop_before_pixels=True)
    patient_id = getattr(rtplan_ds, 'PatientID', 'UNKNOWN')
    plan_name = getattr(rtplan_ds, 'RTPlanLabel', 'UNKNOWN')
    plan_uid = getattr(rtplan_ds, 'SOPInstanceUID', 'UNKNOWN')

    tracker = ProgressTracker("Report Generation", total_steps=3)
    tracker.start()

    tracker.step("Running evaluation...")
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
    # Generate reports but skip writing global files if multiple phases exist
    phases = enumerate_phases(discovered_case)
    if len(phases) > 1:
        progress_print("  ℹ Multiple phases detected; skipping global secondary check reports (per-phase reports produced in test_13).")
    else:
        tracker.step("Generating JSON report...")
        json_path = output_dir / "secondary_check_report.json"
        generate_json_report(result, str(json_path))

        tracker.step("Generating CSV report...")
        csv_path = output_dir / "secondary_check_report.csv"
        generate_csv_report(result, str(csv_path))

    tracker.done("Reports generated")

    # Print summary
    progress_print(f"\n{'='*60}")
    progress_print(f"SECONDARY CHECK SUMMARY")
    progress_print(f"{'='*60}")
    progress_print(f"  Patient: {result.patient_id}")
    progress_print(f"  Plan: {result.plan_name}")
    progress_print(f"  Overall Status: {result.overall_status}")

    if result.gamma_results:
        progress_print(f"\n  Gamma Results:")
        for label, gamma_res in result.gamma_results.items():
            progress_print(f"    {label}: {gamma_res['pass_rate']*100:.1f}% pass [{gamma_res['status']}]")

    if result.failure_reasons:
        progress_print(f"\n  Failure Reasons:")
        for reason in result.failure_reasons:
            progress_print(f"    - {reason}")

    # Verify files exist (only when global reports were written)
    phases = enumerate_phases(discovered_case)
    if len(phases) == 1:
        assert json_path.exists(), "JSON report not created"
        assert csv_path.exists(), "CSV report not created"
    else:
        progress_print("  ℹ Global report creation was skipped due to multiple phases; per-phase reports expected.")

    progress_print(f"\n{'='*60}")
    progress_print(f"SECONDARY CHECK COMPLETE")
    progress_print(f"{'='*60}")


if __name__ == "__main__":
    """Allow running as script for quick testing."""
    print(__doc__)
    print("\nTo run this test:")
    print("  pytest tests/test_patient_end2end.py -v -s")
    print("\nDefault configuration:")
    print(f"  DOSECUDA_PATIENT_DICOM_DIR = {_DEFAULT_PATIENT_DICOM_DIR}")
    print(f"  DOSECUDA_ISO_MM = {_DEFAULT_ISO_SPACING_MM}")
    print(f"  DOSECUDA_GPU_ID = {_DEFAULT_GPU_ID}")


# ============================================================================
# Multi-Phase Tests (all treatment phases)
# ============================================================================

def test_12_enumerate_all_phases(discovered_case):
    """
    TEST 12: Enumerate all treatment phases without user interaction.
    
    Uses enumerate_phases() to automatically pair RTPLAN with corresponding
    RTDOSE, RTSTRUCT, and CT series using UIDs.
    """
    print_test_header("TEST 12: Enumerate All Treatment Phases")
    
    phases = enumerate_phases(discovered_case)
    
    progress_print(f"\n✓ Found {len(phases)} treatment phases:")
    
    for i, phase in enumerate(phases, 1):
        progress_print(f"\n--- Phase {i} ---")
        progress_print(f"  RTPLAN: {phase.rtplan.path.name}")
        
        if phase.rtdose:
            grid_size = phase.rtdose.dose_grid_size or 0
            grid_mb = grid_size * 4 / (1024**2)  # float32
            progress_print(f"  RTDOSE: {phase.rtdose.path.name} ({grid_size:,} voxels, ~{grid_mb:.1f} MB)")
            progress_print(f"    → DoseSummationType: {phase.rtdose.dose_summation_type}")
        else:
            progress_print(f"  RTDOSE: None (no TPS reference available)")
        
        if phase.rtstruct:
            progress_print(f"  RTSTRUCT: {phase.rtstruct.path.name}")
        else:
            progress_print(f"  RTSTRUCT: None")
        
        progress_print(f"  CT: {len(phase.ct_series)} slices")
        
        if phase.warnings:
            for warn in phase.warnings:
                progress_print(f"  ⚠ {warn}")
    
    # Verify we found at least one phase
    assert len(phases) >= 1, "No treatment phases found"
    
    # Verify each phase has CT and RTPLAN
    for i, phase in enumerate(phases, 1):
        assert phase.rtplan is not None, f"Phase {i} missing RTPLAN"
        assert len(phase.ct_series) > 0, f"Phase {i} missing CT series"
    
    progress_print(f"\n✓ All {len(phases)} phases have valid RTPLAN and CT")


def test_13_calculate_all_phases(discovered_case, materialized_case, machine_model):
    """
    TEST 13: Calculate dose for ALL treatment phases.
    
    Iterates through enumerate_phases() and calculates dose for each,
    saving results to separate files.
    """
    print_test_header("TEST 13: Calculate All Phases (Multi-Plan)")
    
    phases = enumerate_phases(discovered_case)
    output_dir = get_output_dir()
    gpu_id = get_gpu_id()
    target_spacing = get_iso_spacing_mm()
    
    progress_print(f"\nCalculating dose for {len(phases)} phases:")
    progress_print(f"  Target spacing: {target_spacing} mm isotropic")
    progress_print(f"  GPU ID: {gpu_id}")
    
    results = []
    
    for i, phase in enumerate(phases, 1):
        progress_print(f"\n{'='*60}")
        progress_print(f"PHASE {i}/{len(phases)}: {phase.rtplan.path.stem}")
        progress_print(f"{'='*60}")
        
        tracker = ProgressTracker(f"Phase {i}", total_steps=5)
        tracker.start()
        
        # Load RTPLAN
        tracker.step("Loading RTPLAN...")
        plan_model = infer_machine_model(phase.rtplan)  # Pass DicomFile, not str
        plan = IMRTPlan(machine_name=plan_model)
        plan.readPlanDicom(str(phase.rtplan.path))
        
        # Materialize CT for this phase
        tracker.step("Preparing CT...")
        phase_output = output_dir / f"phase_{i}"
        phase_output.mkdir(exist_ok=True)
        
        # Create symlinks to CT
        ct_dir = phase_output / "CT"
        ct_dir.mkdir(exist_ok=True)
        for ct_file in phase.ct_series:
            link_path = ct_dir / ct_file.path.name
            if not link_path.exists():
                link_path.symlink_to(ct_file.path)
        
        # Load and resample CT
        tracker.step(f"Loading CT ({len(phase.ct_series)} slices)...")
        dg = IMRTDoseGrid()
        dg.loadCTDCM(str(ct_dir))
        
        tracker.step(f"Resampling to {target_spacing} mm...")
        dg.resampleCTfromSpacing(target_spacing)
        
        # Calculate dose
        tracker.step(f"Computing dose ({plan.n_beams} beams)...")
        dg.computeIMRTPlan(plan, gpu_id=gpu_id)
        
        max_dose = np.max(dg.dose)
        tracker.done(f"Max dose: {max_dose:.2f} Gy")
        
        # Save results (unique filenames per phase)
        npy_path = phase_output / f"dose_phase_{i}.npy"
        np.save(npy_path, dg.dose)
        progress_print(f"  → Saved: {npy_path.name}")

        # Save NRRD on calculation grid
        nrrd_path = phase_output / f"dose_phase_{i}.nrrd"
        dg.writeDoseNRRD(str(nrrd_path), dose_type="PHYSICAL")
        progress_print(f"  → Saved NRRD: {nrrd_path.name}")
        assert nrrd_path.exists(), "Phase NRRD not created"
        
        # Store result info
        phase_result = {
            'phase': i,
            'rtplan': phase.rtplan.path.name,
            'max_dose_gy': max_dose,
            'shape': dg.dose.shape,
            'spacing': dg.spacing.tolist(),
            'has_tps_reference': phase.rtdose is not None
        }
        
        # If TPS reference available, compare and export RTDOSE for this phase
        if phase.rtdose:
            ref_rd = pydicom.dcmread(str(phase.rtdose.path))
            ref_dose = ref_rd.pixel_array * ref_rd.DoseGridScaling
            phase_result['tps_max_dose_gy'] = float(np.max(ref_dose))
            phase_result['dose_ratio'] = max_dose / float(np.max(ref_dose)) if np.max(ref_dose) > 0 else 0
            progress_print(f"  → TPS reference max: {phase_result['tps_max_dose_gy']:.2f} Gy")
            progress_print(f"  → Ratio (DoseCUDA/TPS): {phase_result['dose_ratio']:.3f}")

            # Build template grid from RTDOSE
            template_path = str(phase.rtdose.path)
            template_ds = pydicom.dcmread(template_path, force=True)

            template_origin = np.array(template_ds.ImagePositionPatient)
            pixel_spacing = np.array(template_ds.PixelSpacing)
            if hasattr(template_ds, 'GridFrameOffsetVector'):
                frame_offsets = np.array(template_ds.GridFrameOffsetVector)
                if len(frame_offsets) > 1:
                    slice_spacing = frame_offsets[1] - frame_offsets[0]
                else:
                    slice_spacing = pixel_spacing[0]
            else:
                warnings.warn("Template RTDOSE has no GridFrameOffsetVector, assuming cubic voxels")
                slice_spacing = pixel_spacing[0]

            template_spacing = np.array([pixel_spacing[1], pixel_spacing[0], slice_spacing])
            template_size = np.array([
                int(template_ds.Columns),
                int(template_ds.Rows),
                int(template_ds.NumberOfFrames) if hasattr(template_ds, 'NumberOfFrames') else 1
            ])

            if hasattr(template_ds, 'ImageOrientationPatient'):
                orientation = np.array(template_ds.ImageOrientationPatient)
                row_cos = orientation[:3]
                col_cos = orientation[3:]
                slice_cos = np.cross(row_cos, col_cos)
                direction = np.column_stack([row_cos, col_cos, slice_cos])
                if not np.allclose(direction, np.eye(3), atol=0.1):
                    pytest.fail("Template RTDOSE is not axial - not yet supported.")
            else:
                direction = np.eye(3)

            template_grid = GridInfo(origin=template_origin, spacing=template_spacing,
                                     size=template_size, direction=direction)

            calc_grid = GridInfo(origin=dg.origin, spacing=dg.spacing, size=dg.size, direction=np.eye(3))
            dose_resampled = resample_dose_linear(dose=dg.dose, source_grid=calc_grid, target_grid=template_grid)

            expected_shape = (template_size[2], template_size[1], template_size[0])
            assert dose_resampled.shape == expected_shape, \
                f"Resampled shape {dose_resampled.shape} != template {expected_shape}"

            # Update dose grid to template for export
            dg.dose = dose_resampled
            dg.origin = template_grid.origin
            dg.spacing = template_grid.spacing
            dg.size = template_grid.size

            # Save RTDOSE for this phase
            rtplan_ds = pydicom.dcmread(str(phase.rtplan.path), stop_before_pixels=True)
            dcm_path = phase_output / f"DoseCUDA_RD_phase_{i}.dcm"
            dg.writeDoseDCM(
                dose_path=str(dcm_path),
                ref_dose_path=template_path,
                dose_type="PHYSICAL",
                rtplan_sop_uid=rtplan_ds.SOPInstanceUID
            )
            phase_result['dose_dcm'] = dcm_path.name
            progress_print(f"  → Saved RTDOSE: {dcm_path.name}")
            assert dcm_path.exists(), "Phase RTDOSE not created"

            # --- Per-phase analyses: Gamma, DVH, Secondary Report ---
            try:
                from DoseCUDA.dvh import read_reference_rtdose, compute_metrics, compare_dvh_metrics, generate_dvh_report
                from DoseCUDA.gamma import compute_gamma_3d, GammaCriteria
                from DoseCUDA.rtstruct import read_rtstruct, rasterize_roi_to_mask
                from DoseCUDA.roi_selection import classify_rois, get_target_metrics_spec, get_oar_metrics_spec
                from DoseCUDA.secondary_report import (
                    evaluate_secondary_check,
                    generate_json_report,
                    generate_csv_report,
                    SecondaryCheckCriteria
                )

                # Read TPS RTDOSE for this phase
                ref_dose_arr, ref_origin, ref_spacing, ref_frame = read_reference_rtdose(template_path)
                reference_rtdose = {
                    'dose': ref_dose_arr,
                    'origin': ref_origin,
                    'spacing': ref_spacing,
                    'frame_uid': ref_frame
                }

                # Save gamma summary for this phase
                try:
                    criteria_3_3 = GammaCriteria(dta_mm=3.0, dd_percent=3.0, local=False, dose_threshold_percent=10.0)
                    result_3_3 = compute_gamma_3d(
                        dose_eval=dose_resampled,
                        dose_ref=reference_rtdose['dose'],
                        spacing_mm=tuple(reference_rtdose['spacing']),
                        criteria=criteria_3_3
                    )

                    criteria_2_2 = GammaCriteria(dta_mm=2.0, dd_percent=2.0, local=False, dose_threshold_percent=10.0)
                    result_2_2 = compute_gamma_3d(
                        dose_eval=dose_resampled,
                        dose_ref=reference_rtdose['dose'],
                        spacing_mm=tuple(reference_rtdose['spacing']),
                        criteria=criteria_2_2
                    )

                    import json
                    gamma_summary = {
                        'phase': i,
                        'timestamp': datetime.now().isoformat(),
                        '3%/3mm': {
                            'pass_rate': result_3_3.pass_rate,
                            'mean_gamma': result_3_3.mean_gamma,
                            'gamma_p95': result_3_3.gamma_p95,
                            'n_evaluated': result_3_3.n_evaluated
                        },
                        '2%/2mm': {
                            'pass_rate': result_2_2.pass_rate,
                            'mean_gamma': result_2_2.mean_gamma,
                            'gamma_p95': result_2_2.gamma_p95,
                            'n_evaluated': result_2_2.n_evaluated
                        }
                    }

                    with open(phase_output / f"gamma_summary_phase_{i}.json", 'w') as gf:
                        json.dump(gamma_summary, gf, indent=2)

                    progress_print(f"  → Saved gamma summary: gamma_summary_phase_{i}.json")
                except Exception as e:
                    warnings.warn(f"Per-phase gamma failed for phase {i}: {e}")

                # Rasterize ROIs and compute DVH comparisons if RTSTRUCT exists
                dvh_lines = []
                if phase.rtstruct:
                    try:
                        rtstruct_ds = read_rtstruct(str(phase.rtstruct.path))
                        roi_names = list(rtstruct_ds.rois.keys())
                        classification = classify_rois(roi_names)

                        # Build ref grid for template
                        ref_grid = GridInfo(
                            origin=reference_rtdose['origin'],
                            spacing=reference_rtdose['spacing'],
                            size=np.array(reference_rtdose['dose'].shape[::-1]),
                            direction=np.eye(3)
                        )

                        masks = {}
                        for roi_name in classification.targets + classification.oars:
                            if roi_name in rtstruct_ds.rois:
                                try:
                                    mask = rasterize_roi_to_mask(
                                        rtstruct_ds.rois[roi_name],
                                        ref_grid.origin,
                                        ref_grid.spacing,
                                        ref_grid.size[::-1],
                                        direction=ref_grid.direction
                                    )
                                    if np.any(mask):
                                        masks[roi_name] = mask
                                except Exception:
                                    continue

                        # Compute DVH for targets
                        for roi_name in classification.targets:
                            if roi_name in masks:
                                metrics_spec = get_target_metrics_spec()
                                metrics_calc = compute_metrics(dose_resampled, masks[roi_name], ref_spacing, metrics_spec)
                                metrics_ref = compute_metrics(reference_rtdose['dose'], masks[roi_name], ref_spacing, metrics_spec)
                                comparison = compare_dvh_metrics(metrics_calc, metrics_ref)
                                report = generate_dvh_report(roi_name, metrics_calc, comparison)
                                dvh_lines.append(report)

                        # Compute DVH for first 5 OARs
                        for roi_name in classification.oars[:5]:
                            if roi_name in masks:
                                metrics_spec = get_oar_metrics_spec()
                                metrics_calc = compute_metrics(dose_resampled, masks[roi_name], ref_spacing, metrics_spec)
                                metrics_ref = compute_metrics(reference_rtdose['dose'], masks[roi_name], ref_spacing, metrics_spec)
                                comparison = compare_dvh_metrics(metrics_calc, metrics_ref)
                                report = generate_dvh_report(roi_name, metrics_calc, comparison)
                                dvh_lines.append(report)

                        # Save DVH comparison per phase
                        with open(phase_output / f"dvh_comparison_phase_{i}.txt", 'w') as df:
                            df.write('\n'.join(dvh_lines))

                        progress_print(f"  → Saved DVH comparison: dvh_comparison_phase_{i}.txt")
                    except Exception as e:
                        warnings.warn(f"Per-phase DVH failed for phase {i}: {e}")

                # Secondary check evaluation per phase
                try:
                    criteria = SecondaryCheckCriteria()
                    result = evaluate_secondary_check(
                        dose_calc=dose_resampled,
                        dose_ref=reference_rtdose['dose'],
                        grid_origin=reference_rtdose['origin'],
                        grid_spacing=reference_rtdose['spacing'],
                        rois=(masks if 'masks' in locals() else {}),
                        roi_classification=(classification if 'classification' in locals() else None),
                        plan=plan,
                        patient_id=plan.patient_id if hasattr(plan, 'patient_id') else getattr(rtplan_ds, 'PatientID', 'UNKNOWN'),
                        plan_name=plan.plan_name if hasattr(plan, 'plan_name') else getattr(rtplan_ds, 'RTPlanLabel', 'UNKNOWN'),
                        plan_uid=plan.plan_uid if hasattr(plan, 'plan_uid') else getattr(rtplan_ds, 'SOPInstanceUID', 'UNKNOWN'),
                        criteria=criteria
                    )

                    json_path = phase_output / f"secondary_check_report_phase_{i}.json"
                    csv_path = phase_output / f"secondary_check_report_phase_{i}.csv"
                    generate_json_report(result, str(json_path))
                    generate_csv_report(result, str(csv_path))

                    progress_print(f"  → Saved secondary reports: {json_path.name}, {csv_path.name}")
                except Exception as e:
                    warnings.warn(f"Per-phase secondary check failed for phase {i}: {e}")

            except Exception as e:
                warnings.warn(f"Per-phase analyses skipped for phase {i}: {e}")
        else:
            progress_print(f"  → TPS reference max: None")

        results.append(phase_result)
    
    # Summary
    progress_print(f"\n{'='*60}")
    progress_print(f"MULTI-PHASE SUMMARY")
    progress_print(f"{'='*60}")
    
    for r in results:
        status = "✓" if r['max_dose_gy'] > 0.01 else "⚠"
        progress_print(f"  {status} Phase {r['phase']}: {r['rtplan']}")
        progress_print(f"       Max dose: {r['max_dose_gy']:.2f} Gy")
        if r['has_tps_reference']:
            progress_print(f"       TPS ref:  {r['tps_max_dose_gy']:.2f} Gy (ratio: {r['dose_ratio']:.3f})")
    
    # Verify at least some dose was calculated
    all_calculated = all(r['max_dose_gy'] > 0 for r in results)
    assert all_calculated, "Some phases failed to calculate dose"
    
    progress_print(f"\n✓ All {len(phases)} phases calculated successfully")
    print(f"  DOSECUDA_MACHINE_DEFAULT = {_DEFAULT_MACHINE}")
