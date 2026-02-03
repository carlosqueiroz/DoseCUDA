"""
Test CT Calibration Module
Tests validation, interpolation, extrapolation, and CSV I/O.
"""

import pytest
import numpy as np
import os
import tempfile
from DoseCUDA.ct_calibration import CTCalibration, CTCalibrationManager, get_default_calibration


def test_ct_calibration_basic():
    """Test basic CTCalibration creation and conversion."""
    hu = [-1024, -100, 0, 100, 1000]
    density = [0.001, 0.95, 1.0, 1.05, 1.8]
    
    cal = CTCalibration('Test', hu, density)
    
    assert cal.name == 'Test'
    assert len(cal.hu_points) == 5
    assert len(cal.density_points) == 5
    
    # Test interpolation
    result = cal.convert([0, 100])
    np.testing.assert_array_almost_equal(result, [1.0, 1.05], decimal=5)


def test_ct_calibration_validation():
    """Test validation catches errors."""
    
    # Non-monotonic HU
    with pytest.raises(ValueError, match="estritamente crescente"):
        CTCalibration('Bad', [0, 100, 50], [1.0, 1.1, 1.05])
    
    # Duplicate HU (will be caught by monotonicity check, not separate duplicate check)
    with pytest.raises(ValueError, match="estritamente crescente"):
        CTCalibration('Bad', [0, 100, 100], [1.0, 1.1, 1.2])
    
    # Mismatched lengths
    with pytest.raises(ValueError, match="mesmo tamanho"):
        CTCalibration('Bad', [0, 100], [1.0, 1.1, 1.2])
    
    # Too few points
    with pytest.raises(ValueError, match="mínimo 2 pontos"):
        CTCalibration('Bad', [0], [1.0])
    
    # Negative density
    with pytest.raises(ValueError, match="densidade negativa"):
        CTCalibration('Bad', [0, 100], [1.0, -0.5])


def test_ct_calibration_extrapolation():
    """Test extrapolation modes."""
    hu = [-100, 0, 100]
    density = [0.95, 1.0, 1.05]
    
    # Test clamp (default)
    cal_clamp = CTCalibration('Clamp', hu, density, 
                              extrapolate_low='clamp', extrapolate_high='clamp')
    result = cal_clamp.convert([-200, 200])
    assert result[0] == pytest.approx(0.95, rel=1e-5)  # Clamped to min
    assert result[1] == pytest.approx(1.05, rel=1e-5)  # Clamped to max
    
    # Test air extrapolation
    cal_air = CTCalibration('Air', hu, density, extrapolate_low='air')
    result = cal_air.convert([-2000])
    assert result[0] == pytest.approx(0.0012, rel=1e-5)  # Air density
    
    # Test linear extrapolation
    cal_linear = CTCalibration('Linear', hu, density, 
                               extrapolate_low='linear', extrapolate_high='linear')
    result_low = cal_linear.convert([-200])
    result_high = cal_linear.convert([200])
    # Should extrapolate linearly but clamp to reasonable values
    assert result_low[0] >= 0.0012  # At least air density
    assert result_high[0] <= 5.0  # Max reasonable density


def test_ct_calibration_array_conversion():
    """Test conversion with array inputs."""
    hu = [-100, 0, 100]
    density = [0.95, 1.0, 1.05]
    cal = CTCalibration('Test', hu, density)
    
    # 1D array
    hu_input = np.array([-100, 0, 100])
    result = cal.convert(hu_input)
    np.testing.assert_array_almost_equal(result, [0.95, 1.0, 1.05], decimal=5)
    
    # 2D array (CT slice)
    hu_2d = np.array([[-100, 0], [50, 100]])
    result_2d = cal.convert(hu_2d)
    assert result_2d.shape == (2, 2)
    assert result_2d[0, 0] == pytest.approx(0.95, rel=1e-5)
    assert result_2d[0, 1] == pytest.approx(1.0, rel=1e-5)
    
    # 3D array (CT volume)
    hu_3d = np.ones((10, 10, 10)) * 50
    result_3d = cal.convert(hu_3d)
    assert result_3d.shape == (10, 10, 10)
    assert np.all(result_3d == pytest.approx(1.025, rel=1e-5))


def test_ct_calibration_csv_io():
    """Test CSV save/load."""
    hu = [-1024, -100, 0, 100, 1000]
    density = [0.001, 0.95, 1.0, 1.05, 1.8]
    
    cal_original = CTCalibration('TestCSV', hu, density)
    
    # Save to temporary CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'test_calibration.csv')
        cal_original.to_csv(csv_path)
        
        # Load from CSV
        cal_loaded = CTCalibration.from_csv(csv_path)
        
        assert cal_loaded.name == 'test_calibration'
        np.testing.assert_array_equal(cal_loaded.hu_points, cal_original.hu_points)
        np.testing.assert_array_equal(cal_loaded.density_points, cal_original.density_points)
        
        # Test conversion consistency
        test_hu = np.array([-50, 0, 50, 500])
        result_original = cal_original.convert(test_hu)
        result_loaded = cal_loaded.convert(test_hu)
        np.testing.assert_array_almost_equal(result_original, result_loaded, decimal=5)


def test_ct_calibration_manager():
    """Test CTCalibrationManager."""
    manager = CTCalibrationManager()
    
    # Add calibrations
    cal1 = CTCalibration('Scanner1', [-100, 0, 100], [0.95, 1.0, 1.05])
    cal2 = CTCalibration('Scanner2', [-200, 0, 200], [0.90, 1.0, 1.10])
    
    manager.add_calibration(cal1)
    manager.add_calibration(cal2)
    
    assert len(manager.list_calibrations()) == 2
    assert 'Scanner1' in manager.list_calibrations()
    assert 'Scanner2' in manager.list_calibrations()
    
    # Get calibration
    retrieved = manager.get_calibration('Scanner1')
    assert retrieved.name == 'Scanner1'
    
    # Test error on missing calibration
    with pytest.raises(KeyError, match="não encontrada"):
        manager.get_calibration('NonExistent')


def test_default_calibrations():
    """Test default calibration curves."""
    
    # Generic calibration
    cal_generic = get_default_calibration('generic')
    assert cal_generic.name == 'Generic_9pt'
    assert len(cal_generic.hu_points) == 9
    
    # Test conversion
    result = cal_generic.convert([0])
    assert result[0] == pytest.approx(1.0, abs=0.05)  # Water should be ~1.0
    
    # Philips calibration
    cal_philips = get_default_calibration('philips')
    assert cal_philips.name == 'Philips_Extended'
    assert len(cal_philips.hu_points) > 9
    
    # Invalid scanner type
    with pytest.raises(ValueError, match="não suportado"):
        get_default_calibration('invalid')


def test_ct_calibration_repr():
    """Test string representation."""
    cal = CTCalibration('TestRepr', [-100, 100], [0.95, 1.05])
    repr_str = repr(cal)
    
    assert 'TestRepr' in repr_str
    assert 'n_points=2' in repr_str
    assert 'HU_range' in repr_str
    assert 'density_range' in repr_str


def test_ct_calibration_edge_cases():
    """Test edge cases and corner scenarios."""
    hu = [-1000, 0, 1000]
    density = [0.001, 1.0, 2.0]
    cal = CTCalibration('Edge', hu, density)
    
    # Exact boundary values
    result = cal.convert([-1000, 0, 1000])
    np.testing.assert_array_almost_equal(result, [0.001, 1.0, 2.0], decimal=5)
    
    # Single value conversion
    result_single = cal.convert(-500)
    assert isinstance(result_single, np.ndarray)
    assert result_single.shape == ()
    
    # Empty array
    result_empty = cal.convert([])
    assert len(result_empty) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
