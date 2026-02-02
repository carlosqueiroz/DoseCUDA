"""
Test GridInfo and resampling utilities.

Validates:
- GridInfo creation and properties
- Grid matching
- Mask resampling (nearest neighbor)
- Dose resampling (linear)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DoseCUDA.grid_utils import (
    GridInfo, 
    resample_mask_nearest,
    resample_dose_linear,
    validate_frame_of_reference
)


def test_gridinfo_creation():
    """Test GridInfo creation and properties."""
    print("\n[1/7] Testing GridInfo creation...")
    
    origin = np.array([0.0, 0.0, 0.0])
    spacing = np.array([2.0, 2.0, 3.0])
    size = np.array([100, 100, 50])
    
    grid = GridInfo(origin, spacing, size)
    
    assert np.array_equal(grid.origin, origin)
    assert np.array_equal(grid.spacing, spacing)
    assert np.array_equal(grid.size, size)
    assert grid.is_oblique() == False
    
    voxel_vol = grid.voxel_volume()
    expected_vol = 2.0 * 2.0 * 3.0 / 1000.0  # 0.012 cc
    assert abs(voxel_vol - expected_vol) < 1e-6
    
    print(f"  ✓ GridInfo created: {grid}")
    print(f"  ✓ Voxel volume: {voxel_vol:.6f} cc")


def test_gridinfo_oblique_detection():
    """Test oblique orientation detection."""
    print("\n[2/7] Testing oblique detection...")
    
    # Axial (not oblique)
    grid_axial = GridInfo(
        origin=[0, 0, 0],
        spacing=[1, 1, 1],
        size=[10, 10, 10],
        direction=np.eye(3)
    )
    assert grid_axial.is_oblique() == False
    print("  ✓ Axial grid correctly identified")
    
    # Oblique (rotated)
    angle = np.pi / 4
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    grid_oblique = GridInfo(
        origin=[0, 0, 0],
        spacing=[1, 1, 1],
        size=[10, 10, 10],
        direction=rotation
    )
    assert grid_oblique.is_oblique() == True
    print("  ✓ Oblique grid correctly identified")


def test_gridinfo_matching():
    """Test grid matching with tolerances."""
    print("\n[3/7] Testing grid matching...")
    
    grid1 = GridInfo(
        origin=[0.0, 0.0, 0.0],
        spacing=[2.0, 2.0, 3.0],
        size=[100, 100, 50]
    )
    
    grid2 = GridInfo(
        origin=[0.05, 0.05, 0.05],  # Slightly different origin (within tolerance)
        spacing=[2.0, 2.0, 3.0],
        size=[100, 100, 50]
    )
    
    grid3 = GridInfo(
        origin=[0.0, 0.0, 0.0],
        spacing=[2.5, 2.5, 3.0],  # Different spacing
        size=[100, 100, 50]
    )
    
    assert grid1.matches(grid2, origin_tol=0.1) == True
    print("  ✓ Grids with close origins match")
    
    assert grid1.matches(grid3) == False
    print("  ✓ Grids with different spacing don't match")


def test_resample_mask_same_grid():
    """Test mask resampling when grids already match."""
    print("\n[4/7] Testing mask resampling (same grid)...")
    
    grid = GridInfo(
        origin=[0, 0, 0],
        spacing=[2, 2, 2],
        size=[50, 50, 50]
    )
    
    # Create simple cube mask
    mask = np.zeros((50, 50, 50), dtype=bool)
    mask[10:20, 10:20, 10:20] = True
    
    mask_resampled = resample_mask_nearest(mask, grid, grid)
    
    # Should be identical
    assert np.array_equal(mask, mask_resampled)
    print("  ✓ Mask resampling on same grid returns identical mask")


def test_resample_mask_different_grid():
    """Test mask resampling to different grid."""
    print("\n[5/7] Testing mask resampling (different grid)...")
    
    # Source: 2mm spacing
    grid_fine = GridInfo(
        origin=[0, 0, 0],
        spacing=[2, 2, 2],
        size=[50, 50, 50]
    )
    
    # Target: 4mm spacing (coarser)
    grid_coarse = GridInfo(
        origin=[0, 0, 0],
        spacing=[4, 4, 4],
        size=[25, 25, 25]
    )
    
    # Create cube mask on fine grid
    mask_fine = np.zeros((50, 50, 50), dtype=bool)
    mask_fine[10:30, 10:30, 10:30] = True  # 40mm cube
    
    volume_fine = np.sum(mask_fine) * grid_fine.voxel_volume()
    
    # Resample to coarse grid
    mask_coarse = resample_mask_nearest(mask_fine, grid_fine, grid_coarse)
    
    volume_coarse = np.sum(mask_coarse) * grid_coarse.voxel_volume()
    
    # Volumes should be similar (within 10% due to discretization)
    volume_diff_percent = abs(volume_coarse - volume_fine) / volume_fine * 100
    
    print(f"  Volume on fine grid: {volume_fine:.2f} cc")
    print(f"  Volume on coarse grid: {volume_coarse:.2f} cc")
    print(f"  Difference: {volume_diff_percent:.1f}%")
    
    assert volume_diff_percent < 10.0
    print("  ✓ Mask resampled with acceptable volume preservation")


def test_resample_dose():
    """Test dose resampling with linear interpolation."""
    print("\n[6/7] Testing dose resampling (linear)...")
    
    try:
        import SimpleITK as sitk
    except ImportError:
        print("  ⚠ SimpleITK not available, skipping dose resampling test")
        return
    
    # Source: fine grid with linear dose gradient
    grid_fine = GridInfo(
        origin=[0, 0, 0],
        spacing=[2, 2, 2],
        size=[50, 50, 50]
    )
    
    # Create dose with linear gradient in Z
    z_coords = np.arange(50) * 2.0  # 0, 2, 4, ..., 98 mm
    dose_fine = np.zeros((50, 50, 50), dtype=np.float32)
    for i in range(50):
        dose_fine[i, :, :] = z_coords[i] * 0.5  # 0 to 49 Gy
    
    # Target: coarser grid
    grid_coarse = GridInfo(
        origin=[0, 0, 0],
        spacing=[4, 4, 4],
        size=[25, 25, 25]
    )
    
    # Resample
    dose_coarse = resample_dose_linear(dose_fine, grid_fine, grid_coarse)
    
    # Check that gradient is preserved
    # At z=20mm (slice 10 on fine, slice 5 on coarse)
    expected_dose_at_20mm = 20.0 * 0.5  # 10 Gy
    
    dose_at_20mm_fine = dose_fine[10, 25, 25]
    dose_at_20mm_coarse = dose_coarse[5, 12, 12]
    
    print(f"  Dose at z=20mm on fine grid: {dose_at_20mm_fine:.2f} Gy")
    print(f"  Dose at z=20mm on coarse grid: {dose_at_20mm_coarse:.2f} Gy")
    print(f"  Expected: {expected_dose_at_20mm:.2f} Gy")
    
    assert abs(dose_at_20mm_fine - expected_dose_at_20mm) < 0.1
    assert abs(dose_at_20mm_coarse - expected_dose_at_20mm) < 0.5
    
    print("  ✓ Dose gradient preserved after resampling")


def test_frame_of_reference_validation():
    """Test FrameOfReferenceUID validation."""
    print("\n[7/7] Testing FrameOfReferenceUID validation...")
    
    grid1 = GridInfo(
        origin=[0, 0, 0],
        spacing=[1, 1, 1],
        size=[10, 10, 10],
        frame_of_reference_uid="1.2.3.4.5"
    )
    
    grid2 = GridInfo(
        origin=[0, 0, 0],
        spacing=[1, 1, 1],
        size=[10, 10, 10],
        frame_of_reference_uid="1.2.3.4.5"
    )
    
    grid3 = GridInfo(
        origin=[0, 0, 0],
        spacing=[1, 1, 1],
        size=[10, 10, 10],
        frame_of_reference_uid="9.8.7.6.5"
    )
    
    # Same UID should pass
    result = validate_frame_of_reference(grid1, grid2, strict=False)
    assert result == True
    print("  ✓ Matching FrameOfReferenceUID validated")
    
    # Different UID should warn (not raise in non-strict mode)
    result = validate_frame_of_reference(grid1, grid3, strict=False)
    assert result == False
    print("  ✓ Mismatched FrameOfReferenceUID detected (warning mode)")
    
    # Different UID should raise in strict mode
    try:
        validate_frame_of_reference(grid1, grid3, strict=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ Mismatched FrameOfReferenceUID raises error (strict mode)")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING GRIDINFO AND RESAMPLING UTILITIES")
    print("="*70)
    
    try:
        test_gridinfo_creation()
        test_gridinfo_oblique_detection()
        test_gridinfo_matching()
        test_resample_mask_same_grid()
        test_resample_mask_different_grid()
        test_resample_dose()
        test_frame_of_reference_validation()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nGridInfo and resampling utilities are working correctly.")
        print("Ready for Task 3 (RTDOSE + gamma analysis).")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
