"""
Test CT resampling to isotropic spacing.

Validates resample_ct_to_isotropic() function required by dose engine.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DoseCUDA.grid_utils import GridInfo, resample_ct_to_isotropic


def test_resample_anisotropic_to_isotropic():
    """Test resampling from anisotropic (1×1×3) to isotropic (2.5×2.5×2.5)."""
    print("\n[TEST 1/3] Resample Anisotropic CT to Isotropic")
    print("=" * 60)
    
    # Create anisotropic CT (typical clinical: 1×1×3 mm)
    source_spacing = np.array([1.0, 1.0, 3.0])
    source_size = np.array([200, 200, 100])  # 200×200×300 mm volume
    source_origin = np.array([-100.0, -100.0, -150.0])
    
    source_grid = GridInfo(
        origin=source_origin,
        spacing=source_spacing,
        size=source_size
    )
    
    # Create simple HU pattern (water sphere in center)
    hu_array = np.full((100, 200, 200), -1000, dtype=np.float32)  # Air
    
    # Add water sphere (r=50mm ~ 20 voxels in xy, 17 slices in z)
    center = [50, 100, 100]  # z, y, x indices
    for z in range(100):
        for y in range(200):
            for x in range(200):
                z_mm = z * source_spacing[2] + source_origin[2] - 0  # Distance from z=0
                y_mm = y * source_spacing[1] + source_origin[1] - 0
                x_mm = x * source_spacing[0] + source_origin[0] - 0
                
                dist = np.sqrt(x_mm**2 + y_mm**2 + z_mm**2)
                if dist < 50.0:  # 50mm radius
                    hu_array[z, y, x] = 0.0  # Water
    
    print(f"Source CT:")
    print(f"  Spacing: {source_grid.spacing} mm")
    print(f"  Size: {source_grid.size} voxels")
    print(f"  Volume: {np.prod(source_grid.size) * source_grid.voxel_volume():.1f} cc")
    print(f"  Anisotropic: {source_grid.is_oblique() == False} (expected False)")
    
    # Resample to isotropic 2.5 mm
    target_spacing = 2.5
    hu_resampled, target_grid = resample_ct_to_isotropic(
        hu_array, source_grid, target_spacing_mm=target_spacing
    )
    
    print(f"\nResampled CT:")
    print(f"  Spacing: {target_grid.spacing} mm")
    print(f"  Size: {target_grid.size} voxels")
    print(f"  Shape: {hu_resampled.shape}")
    print(f"  Volume: {np.prod(target_grid.size) * target_grid.voxel_volume():.1f} cc")
    
    # Validate
    assert np.allclose(target_grid.spacing, [target_spacing] * 3, atol=0.001)
    
    # Shape is (nz, ny, nx) - note GridInfo.size is (nx, ny, nz)
    expected_shape = (target_grid.size[2], target_grid.size[1], target_grid.size[0])
    assert hu_resampled.shape == expected_shape, f"Shape mismatch: {hu_resampled.shape} vs {expected_shape}"
    
    # Check that water sphere is still present (HU near 0 in center)
    center_idx = tuple(target_grid.size // 2)
    center_hu = hu_resampled[center_idx[0], center_idx[1], center_idx[2]]
    
    print(f"\nValidation:")
    print(f"  Spacing is isotropic: {np.allclose(target_grid.spacing, target_spacing)}")
    print(f"  HU at center: {center_hu:.1f} (expected ~0 for water)")
    
    assert abs(center_hu - 0.0) < 200, f"Center HU should be near 0, got {center_hu}"
    
    print(f"\n✓ Resample anisotropic to isotropic PASSED")
    
    return True


def test_resample_already_isotropic():
    """Test that already-isotropic CT is returned unchanged."""
    print("\n\n[TEST 2/3] Already Isotropic CT")
    print("=" * 60)
    
    # Create isotropic CT (2.5×2.5×2.5)
    source_spacing = np.array([2.5, 2.5, 2.5])
    source_size = np.array([80, 80, 80])
    source_origin = np.array([-100.0, -100.0, -100.0])
    
    source_grid = GridInfo(
        origin=source_origin,
        spacing=source_spacing,
        size=source_size
    )
    
    hu_array = np.random.randn(80, 80, 80).astype(np.float32) * 100 - 500
    
    print(f"Source CT already isotropic ({source_spacing[0]} mm)")
    
    # "Resample" to same spacing
    hu_resampled, target_grid = resample_ct_to_isotropic(
        hu_array, source_grid, target_spacing_mm=2.5
    )
    
    # Should be identical (or nearly so)
    assert np.array_equal(hu_array, hu_resampled), "Should return copy of original"
    assert source_grid.matches(target_grid), "Grids should match"
    
    print(f"\n✓ Already isotropic CT handled correctly")
    
    return True


def test_resample_preserves_physical_extent():
    """Test that resampling preserves physical extent of volume."""
    print("\n\n[TEST 3/3] Physical Extent Preservation")
    print("=" * 60)
    
    # Anisotropic CT
    source_spacing = np.array([0.8, 0.8, 2.5])
    source_size = np.array([150, 256, 256])  # 120×204.8×204.8 mm
    source_origin = np.array([0.0, 0.0, 0.0])
    
    source_grid = GridInfo(origin=source_origin, spacing=source_spacing, size=source_size)
    
    hu_array = np.zeros((150, 256, 256), dtype=np.float32)
    
    # Get physical bounds
    min_bound, max_bound = source_grid.get_physical_bounds()
    
    print(f"Source physical extent:")
    print(f"  Min: {min_bound}")
    print(f"  Max: {max_bound}")
    print(f"  Size: {max_bound - min_bound}")
    
    # Resample to 2.0 mm isotropic
    hu_resampled, target_grid = resample_ct_to_isotropic(
        hu_array, source_grid, target_spacing_mm=2.0
    )
    
    min_bound_new, max_bound_new = target_grid.get_physical_bounds()
    
    print(f"\nTarget physical extent:")
    print(f"  Min: {min_bound_new}")
    print(f"  Max: {max_bound_new}")
    print(f"  Size: {max_bound_new - min_bound_new}")
    
    # Physical extent should be similar (within 1 voxel)
    extent_diff = np.abs((max_bound_new - min_bound_new) - (max_bound - min_bound))
    
    print(f"\nExtent difference: {extent_diff} mm")
    
    assert np.all(extent_diff < 3.0), f"Physical extent changed too much: {extent_diff}"
    
    print(f"\n✓ Physical extent preserved within tolerance")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING CT RESAMPLING TO ISOTROPIC")
    print("="*70)
    print("\nThese tests validate resample_ct_to_isotropic():")
    print("- Required by dose engine (only supports isotropic voxels)")
    print("- Resamples anisotropic CT (e.g., 1×1×3) to isotropic (e.g., 2.5×2.5×2.5)")
    print("- Preserves physical extent and HU values")
    
    try:
        test_resample_anisotropic_to_isotropic()
        test_resample_already_isotropic()
        test_resample_preserves_physical_extent()
        
        print("\n" + "="*70)
        print("✅ ALL RESAMPLING TESTS PASSED")
        print("="*70)
        print("\nKey points validated:")
        print("  ✓ Anisotropic → Isotropic resampling works")
        print("  ✓ Already-isotropic CT handled correctly")
        print("  ✓ Physical extent preserved")
        print("  ✓ HU values interpolated correctly")
        print("\n🚀 CT resampling ready for dose calculation workflow")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
