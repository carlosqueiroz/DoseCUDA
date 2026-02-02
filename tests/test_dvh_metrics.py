"""
Unit tests for DVH computation and dose metrics.

Tests differential/cumulative DVH calculation and standard dose metrics
like Dmean, Dmax, D95, D98, V20, etc.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DoseCUDA.dvh import compute_dvh, compute_metrics, compare_dvh_metrics, generate_dvh_report


def test_dvh_uniform_dose():
    """Test DVH with uniform dose (all voxels same dose)."""
    # Create uniform dose
    dose = np.ones((10, 10, 10)) * 50.0  # 50 Gy everywhere
    
    # Create mask covering half the volume
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[:5, :, :] = True  # 500 voxels
    
    voxel_volume = 0.008  # 2mm x 2mm x 2mm = 8 mm³ = 0.008 cc
    
    dose_bins, diff_dvh, cum_dvh = compute_dvh(dose, mask, voxel_volume, bin_width=1.0)
    
    # All dose should be in one bin around 50 Gy
    assert len(dose_bins) > 0, "DVH should not be empty"
    
    # All volume should be at 50 Gy
    total_volume = np.sum(diff_dvh)
    expected_volume = 500 * voxel_volume  # 500 voxels * 0.008 cc
    
    assert abs(total_volume - expected_volume) < 0.01, \
        f"Total volume {total_volume} != expected {expected_volume}"
    
    # Cumulative DVH at lowest dose should equal total volume
    assert abs(cum_dvh[0] - expected_volume) < 0.01, \
        f"Cumulative volume {cum_dvh[0]} != expected {expected_volume}"


def test_dvh_dose_ramp():
    """Test DVH with dose ramp (linearly increasing dose)."""
    # Create dose ramp: dose increases along X axis
    dose = np.zeros((10, 10, 10))
    for i in range(10):
        dose[:, :, i] = i * 10.0  # 0, 10, 20, ..., 90 Gy
    
    # Full volume mask
    mask = np.ones((10, 10, 10), dtype=bool)
    
    voxel_volume = 1.0  # 10mm x 10mm x 10mm = 1000 mm³ = 1 cc
    
    dose_bins, diff_dvh, cum_dvh = compute_dvh(dose, mask, voxel_volume, bin_width=5.0)
    
    # Check total volume
    total_volume = np.sum(diff_dvh)
    expected_volume = 1000 * voxel_volume  # 1000 voxels * 1 cc
    
    assert abs(total_volume - expected_volume) < 1.0, \
        f"Total volume {total_volume} != expected {expected_volume}"
    
    # Cumulative DVH should be monotonically decreasing
    assert np.all(np.diff(cum_dvh) <= 0), "Cumulative DVH should be decreasing"
    
    # Check that cumulative DVH at lowest dose equals total volume
    assert abs(cum_dvh[0] - expected_volume) < 1.0


def test_metrics_basic():
    """Test basic dose metrics computation."""
    # Create simple dose distribution
    dose = np.array([[[10.0, 20.0, 30.0],
                      [40.0, 50.0, 60.0],
                      [70.0, 80.0, 90.0]]])  # Shape (1, 3, 3)
    
    # Mask covering all voxels
    mask = np.ones((1, 3, 3), dtype=bool)
    
    spacing = np.array([10.0, 10.0, 10.0])  # 10mm spacing
    
    metrics = compute_metrics(dose, mask, spacing)
    
    # Check Dmean: should be 50 Gy (mean of 10-90)
    assert abs(metrics['Dmean'] - 50.0) < 0.1, f"Dmean {metrics['Dmean']} != 50.0"
    
    # Check Dmax: should be 90 Gy
    assert abs(metrics['Dmax'] - 90.0) < 0.1, f"Dmax {metrics['Dmax']} != 90.0"
    
    # Check Dmin: should be 10 Gy
    assert abs(metrics['Dmin'] - 10.0) < 0.1, f"Dmin {metrics['Dmin']} != 10.0"
    
    # Check Volume: 9 voxels * (10mm)³ = 9 cc
    expected_volume = 9 * (10**3) / 1000  # Convert mm³ to cc
    assert abs(metrics['Volume_cc'] - expected_volume) < 0.1, \
        f"Volume {metrics['Volume_cc']} != {expected_volume}"


def test_metrics_percentiles():
    """Test D_percent metrics (dose percentiles)."""
    # Create dose with known distribution
    # 100 voxels: 10 at 90Gy, 20 at 50Gy, 70 at 10Gy
    dose = np.zeros((10, 10, 1))
    dose[:1, :, 0] = 90.0  # 10 voxels at high dose
    dose[1:3, :, 0] = 50.0  # 20 voxels at mid dose
    dose[3:, :, 0] = 10.0  # 70 voxels at low dose
    
    mask = np.ones((10, 10, 1), dtype=bool)
    spacing = np.array([1.0, 1.0, 1.0])
    
    metrics_spec = {'D_percent': [2, 10, 30, 50, 90]}
    metrics = compute_metrics(dose, mask, spacing, metrics_spec)
    
    # D2%: dose to hottest 2% (2 voxels) -> should be 90 Gy
    assert metrics['D2%'] > 85.0, f"D2% {metrics['D2%']} should be ~90 Gy"
    
    # D10%: dose to hottest 10% (10 voxels) -> should be 90 Gy
    assert metrics['D10%'] > 85.0, f"D10% {metrics['D10%']} should be ~90 Gy"
    
    # D30%: dose to 30% of volume (30 voxels) -> should be 50 Gy
    assert 45.0 < metrics['D30%'] < 55.0, f"D30% {metrics['D30%']} should be ~50 Gy"
    
    # D90%: dose covering 90% (90 voxels) -> should be 10 Gy
    assert metrics['D90%'] < 15.0, f"D90% {metrics['D90%']} should be ~10 Gy"


def test_metrics_volume_at_dose():
    """Test V_dose metrics (volume at dose threshold)."""
    # Create dose: 50 voxels at 30 Gy, 50 voxels at 10 Gy
    dose = np.zeros((10, 10, 1))
    dose[:5, :, 0] = 30.0  # 50 voxels
    dose[5:, :, 0] = 10.0  # 50 voxels
    
    mask = np.ones((10, 10, 1), dtype=bool)
    spacing = np.array([1.0, 1.0, 1.0])
    
    metrics_spec = {'V_dose': [20, 25, 35]}
    metrics = compute_metrics(dose, mask, spacing, metrics_spec)
    
    # V20Gy: volume receiving >= 20 Gy -> 50% (50 voxels at 30 Gy)
    assert abs(metrics['V20Gy'] - 50.0) < 1.0, f"V20Gy {metrics['V20Gy']} should be ~50%"
    
    # V25Gy: volume receiving >= 25 Gy -> 50% (50 voxels at 30 Gy)
    assert abs(metrics['V25Gy'] - 50.0) < 1.0, f"V25Gy {metrics['V25Gy']} should be ~50%"
    
    # V35Gy: volume receiving >= 35 Gy -> 0% (no voxels above 35)
    assert metrics['V35Gy'] < 1.0, f"V35Gy {metrics['V35Gy']} should be ~0%"


def test_metrics_empty_mask():
    """Test that empty mask returns NaN metrics gracefully."""
    dose = np.ones((10, 10, 10)) * 50.0
    mask = np.zeros((10, 10, 10), dtype=bool)  # Empty mask
    spacing = np.array([1.0, 1.0, 1.0])
    
    metrics = compute_metrics(dose, mask, spacing)
    
    # Should return NaN for dose metrics
    assert np.isnan(metrics['Dmean']), "Dmean should be NaN for empty mask"
    assert np.isnan(metrics['Dmax']), "Dmax should be NaN for empty mask"
    assert np.isnan(metrics['Dmin']), "Dmin should be NaN for empty mask"
    
    # Volume should be 0
    assert metrics['Volume_cc'] == 0.0, "Volume should be 0 for empty mask"


def test_compare_dvh_metrics():
    """Test comparison of calculated vs reference metrics."""
    calculated = {
        'Dmean': 50.0,
        'Dmax': 60.0,
        'D95%': 48.0,
        'V20Gy': 95.0,
        'Volume_cc': 100.0
    }
    
    reference = {
        'Dmean': 50.5,  # Diff: -0.5 Gy (-1%)
        'Dmax': 61.0,   # Diff: -1.0 Gy (-1.6%)
        'D95%': 47.5,   # Diff: +0.5 Gy (+1%)
        'V20Gy': 94.0,  # Diff: +1% point
        'Volume_cc': 102.0  # Diff: -2 cc (-2%)
    }
    
    # Tolerances: 0.5 Gy absolute, 3% relative
    comparison = compare_dvh_metrics(calculated, reference, tolerance_abs=0.5, tolerance_rel=0.03)
    
    # Dmean: -0.5 Gy is within 0.5 Gy absolute -> PASS
    assert comparison['Dmean']['pass'], "Dmean should pass"
    
    # Dmax: -1.0 Gy is NOT within 0.5 Gy, but -1.6% is within 3% -> PASS
    assert comparison['Dmax']['pass'], "Dmax should pass (within relative tolerance)"
    
    # D95%: +0.5 Gy is within 0.5 Gy absolute -> PASS
    assert comparison['D95%']['pass'], "D95% should pass"
    
    # V20Gy: +1% is within 3% -> PASS
    assert comparison['V20Gy']['pass'], "V20Gy should pass"
    
    # Volume_cc: -2% is within 3% -> PASS
    assert comparison['Volume_cc']['pass'], "Volume_cc should pass"


def test_compare_dvh_metrics_failure():
    """Test that comparison correctly identifies failures."""
    calculated = {
        'Dmean': 50.0,
        'D95%': 45.0
    }
    
    reference = {
        'Dmean': 55.0,  # Diff: -5 Gy (-9%) -> FAIL (outside both tolerances)
        'D95%': 50.0    # Diff: -5 Gy (-10%) -> FAIL
    }
    
    comparison = compare_dvh_metrics(calculated, reference, tolerance_abs=0.5, tolerance_rel=0.03)
    
    # Both should fail
    assert not comparison['Dmean']['pass'], "Dmean should fail (large difference)"
    assert not comparison['D95%']['pass'], "D95% should fail (large difference)"
    
    # Check diff values
    assert abs(comparison['Dmean']['diff'] - (-5.0)) < 0.01
    assert abs(comparison['Dmean']['diff_percent'] - (-9.09)) < 0.1


def test_generate_dvh_report():
    """Test DVH report generation."""
    metrics = {
        'Volume_cc': 150.5,
        'Dmean': 48.5,
        'Dmax': 62.3,
        'Dmin': 35.2,
        'D2%': 61.5,
        'D95%': 45.2,
        'D98%': 42.1,
        'V20Gy': 98.5,
        'V40Gy': 85.2
    }
    
    report = generate_dvh_report("PTV", metrics)
    
    # Check that report contains key information
    assert "PTV" in report
    assert "150.5" in report  # Volume
    assert "48.5" in report   # Dmean
    assert "62.3" in report   # Dmax
    assert "D95%" in report
    assert "V20Gy" in report
    
    # With comparison
    reference = {k: v * 1.01 for k, v in metrics.items()}  # 1% higher
    comparison = compare_dvh_metrics(metrics, reference, tolerance_abs=1.0, tolerance_rel=0.05)
    
    report_with_comparison = generate_dvh_report("PTV", metrics, comparison)
    
    assert "Comparison" in report_with_comparison
    assert "PASS" in report_with_comparison or "FAIL" in report_with_comparison


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
