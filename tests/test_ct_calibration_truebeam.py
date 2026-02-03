"""
Test CT Calibration with Real Patient Data (TRUEBEAM)
Tests end-to-end integration of CTCalibration with real DICOM CT data.
"""

import pytest
import numpy as np
import os
from pathlib import Path
import tempfile

from DoseCUDA import IMRTDoseGrid, IMRTPlan
from DoseCUDA.ct_calibration import CTCalibration
from DoseCUDA.dicom_case_discovery import (
    scan_dicom_directory,
    select_ct_series,
    select_rtplan,
    materialize_case
)


# Path to TRUEBEAM test patient
TRUEBEAM_DICOM_DIR = Path(__file__).parent / "PATIENT" / "TRUEBEAM"


@pytest.fixture(scope="module")
def discovered_case():
    """Fixture: scan DICOM directory."""
    if not TRUEBEAM_DICOM_DIR.exists():
        pytest.skip("TRUEBEAM patient data não encontrado")
    
    catalog = scan_dicom_directory(str(TRUEBEAM_DICOM_DIR))
    print(f"\n✓ Descobriu: {len(catalog.ct_files)} CT files")
    return catalog


@pytest.fixture(scope="module")
def materialized_case(discovered_case):
    """Fixture: materialize case with organized directory structure."""
    # Select files
    rtplan = select_rtplan(discovered_case)
    ct_series = select_ct_series(discovered_case)
    
    assert rtplan is not None, "RTPLAN não encontrado"
    assert ct_series is not None, "CT series não encontrado"
    
    # Materialize to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = materialize_case(
            output_dir=tmpdir,
            ct_series=ct_series,
            rtplan=rtplan
        )
        yield paths


def test_ct_calibration_truebeam_loading(materialized_case):
    """Test loading CT from TRUEBEAM patient and converting HU to density."""
    
    # Load CT into dose grid
    dose_grid = IMRTDoseGrid()
    dose_grid.loadCTDCM(str(materialized_case['ct_dir']))
    
    print(f"\n✓ CT carregado: {dose_grid.size} voxels")
    print(f"  HU range: [{np.min(dose_grid.HU):.0f}, {np.max(dose_grid.HU):.0f}]")
    print(f"  Spacing: {dose_grid.spacing} mm")
    
    # Create custom calibration
    hu_points = [-1024, -950, -120, -90, 60, 240, 930, 1060, 1560]
    density_points = [0.0012, 0.0012, 0.95, 0.98, 1.05, 1.15, 1.53, 1.69, 2.30]
    calibration = CTCalibration('TRUEBEAM_Custom', hu_points, density_points)
    
    # Convert HU to density using new calibration
    density = dose_grid.DensityFromHU('VarianTrueBeamHF', ct_calibration=calibration)
    
    print(f"✓ Densidade calculada com calibração customizada")
    print(f"  Density range: [{np.min(density):.4f}, {np.max(density):.4f}] g/cc")
    
    # Validation checks
    assert density.shape == dose_grid.HU.shape
    assert np.all(density >= 0.001), "Densidade negativa detectada"
    assert np.all(density <= 3.0), "Densidade irrealista (>3 g/cc)"
    
    # Check typical tissue values
    water_hu = 0
    water_density = calibration.convert([water_hu])[0]
    assert 0.98 <= water_density <= 1.02, f"Densidade água incorreta: {water_density}"
    
    bone_hu = 1000
    bone_density = calibration.convert([bone_hu])[0]
    assert 1.5 <= bone_density <= 2.0, f"Densidade osso incorreta: {bone_density}"


def test_ct_calibration_legacy_vs_new(materialized_case):
    """Compare legacy CSV loading vs new CTCalibration object."""
    
    # Load CT
    dose_grid1 = IMRTDoseGrid()
    dose_grid1.loadCTDCM(str(materialized_case['ct_dir']))
    
    dose_grid2 = IMRTDoseGrid()
    dose_grid2.loadCTDCM(str(materialized_case['ct_dir']))
    
    # Method 1: Legacy (loads CSV internally)
    density_legacy = dose_grid1.DensityFromHU('VarianTrueBeamHF')
    
    # Method 2: New (explicit CTCalibration object from same CSV)
    import pkg_resources
    csv_path = pkg_resources.resource_filename(
        'DoseCUDA', 
        os.path.join("lookuptables", "photons", "VarianTrueBeamHF", "HU_Density.csv")
    )
    calibration = CTCalibration.from_csv(csv_path, name='VarianTrueBeamHF_CSV')
    density_new = dose_grid2.DensityFromHU('VarianTrueBeamHF', ct_calibration=calibration)
    
    # Results should be identical
    np.testing.assert_array_almost_equal(density_legacy, density_new, decimal=5)
    
    print(f"\n✓ Legacy vs New: 100% equivalente")
    print(f"  Max diff: {np.max(np.abs(density_legacy - density_new)):.10f}")


def test_ct_calibration_extrapolation_warnings(materialized_case):
    """Test that extrapolation warnings are triggered for out-of-range HU."""
    
    # Load CT
    dose_grid = IMRTDoseGrid()
    dose_grid.loadCTDCM(str(materialized_case['ct_dir']))
    
    # Create calibration with narrow HU range (will force extrapolation)
    hu_narrow = [-500, 0, 500]
    density_narrow = [0.5, 1.0, 1.5]
    calibration_narrow = CTCalibration('Narrow', hu_narrow, density_narrow)
    
    # This should trigger warnings for HU outside [-500, 500]
    with pytest.warns(UserWarning, match="voxels com HU"):
        density = dose_grid.DensityFromHU('VarianTrueBeamHF', ct_calibration=calibration_narrow)
    
    # But density should still be computed (clamped or extrapolated)
    assert density.shape == dose_grid.HU.shape
    assert np.all(np.isfinite(density))
    
    print(f"\n✓ Avisos de extrapolação funcionando corretamente")


@pytest.mark.skipif(
    not TRUEBEAM_DICOM_DIR.exists(),
    reason="TRUEBEAM patient data não encontrado"
)
def test_ct_calibration_stored_in_grid():
    """Test that CTCalibration object is stored in dose grid."""
    
    dose_grid = IMRTDoseGrid()
    
    # Before conversion
    assert dose_grid.ct_calibration is None
    
    # Create synthetic HU data
    dose_grid.HU = np.random.randint(-1000, 1000, size=(10, 10, 10))
    
    # Convert with explicit calibration
    calibration = CTCalibration('Test', [-1000, 0, 1000], [0.001, 1.0, 2.0])
    density = dose_grid.DensityFromHU('VarianTrueBeamHF', ct_calibration=calibration)
    
    # After conversion
    assert dose_grid.ct_calibration is not None
    assert dose_grid.ct_calibration.name == 'Test'
    assert isinstance(dose_grid.ct_calibration, CTCalibration)
    
    print(f"\n✓ CTCalibration armazenado no grid: {dose_grid.ct_calibration}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
