"""
Example: Complete patient pipeline for secondary dose calculation.

Demonstrates:
1. Load CT DICOM with validation
2. Read RTSTRUCT and rasterize ROIs to masks
3. Calculate dose (using existing phantom for demo)
4. Compute DVH and metrics
5. Compare with reference (if available)

This example shows how to use the clinical secondary check pipeline.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DoseCUDA.plan import DoseGrid
from DoseCUDA import rtstruct, dvh


def example_ct_loading():
    """Example 1: Load CT with validation."""
    print("\n" + "="*60)
    print("Example 1: Loading CT DICOM with validation")
    print("="*60)
    
    # Create synthetic CT (in real use, provide path to CT directory)
    grid = DoseGrid()
    grid.createCubePhantom(size=[100, 100, 100], spacing=3.0)
    
    print(f"CT loaded:")
    print(f"  Origin: {grid.origin}")
    print(f"  Spacing: {grid.spacing}")
    print(f"  Size: {grid.size}")
    print(f"  Direction:\n{grid.direction}")
    print(f"  FrameOfReferenceUID: {grid.FrameOfReferenceUID or '(not set)'}")
    
    return grid


def example_rtstruct_rasterization(grid):
    """Example 2: Create synthetic ROI and rasterize to mask."""
    print("\n" + "="*60)
    print("Example 2: Creating and rasterizing ROI")
    print("="*60)
    
    # Create synthetic ROI (sphere at center)
    # In real use: struct = rtstruct.read_rtstruct("path/to/RTSTRUCT.dcm")
    
    roi = rtstruct.ROI(
        name="PTV_Synthetic",
        roi_number=1,
        display_color=(255, 0, 0)
    )
    
    # Create spherical contours
    center = np.array([0.0, 0.0, 0.0])  # mm
    radius = 30.0  # mm
    
    # Generate contours slice by slice
    z_start = -40.0
    z_end = 40.0
    z_step = 3.0
    
    for z in np.arange(z_start, z_end, z_step):
        # Circle at this z
        if abs(z) < radius:
            r_at_z = np.sqrt(radius**2 - z**2)
            
            # Generate circle points
            n_points = 36
            angles = np.linspace(0, 2*np.pi, n_points + 1)
            
            points = np.zeros((n_points + 1, 3))
            points[:, 0] = center[0] + r_at_z * np.cos(angles)
            points[:, 1] = center[1] + r_at_z * np.sin(angles)
            points[:, 2] = z
            
            contour_slice = rtstruct.ContourSlice(
                points=points,
                z_position=z
            )
            roi.contour_slices.append(contour_slice)
    
    print(f"ROI created: {roi.name}")
    print(f"  Number of contour slices: {len(roi.contour_slices)}")
    
    # Get bounding box
    bbox_min, bbox_max = roi.get_bounding_box_mm()
    print(f"  Bounding box (mm):")
    print(f"    Min: {bbox_min}")
    print(f"    Max: {bbox_max}")
    
    # Rasterize to mask
    mask = rtstruct.rasterize_roi_to_mask(
        roi,
        origin=grid.origin,
        spacing=grid.spacing,
        size=grid.size,
        direction=grid.direction
    )
    
    # Calculate volume
    voxel_volume = np.prod(grid.spacing) / 1000.0  # mm³ to cc
    volume_cc = np.sum(mask) * voxel_volume
    
    print(f"Mask generated:")
    print(f"  Number of voxels: {np.sum(mask)}")
    print(f"  Volume: {volume_cc:.2f} cc")
    print(f"  Expected volume (sphere): {4/3 * np.pi * (radius/10)**3:.2f} cc")
    
    return roi, mask


def example_dvh_calculation(dose, mask, spacing):
    """Example 3: Calculate DVH and metrics."""
    print("\n" + "="*60)
    print("Example 3: Computing DVH and metrics")
    print("="*60)
    
    # Calculate voxel volume
    voxel_volume = np.prod(spacing) / 1000.0  # mm³ to cc
    
    # Compute DVH
    dose_bins, diff_dvh, cum_dvh = dvh.compute_dvh(
        dose, mask, voxel_volume, bin_width=0.5
    )
    
    print(f"DVH computed:")
    print(f"  Number of bins: {len(dose_bins)}")
    print(f"  Dose range: {dose_bins[0]:.2f} - {dose_bins[-1]:.2f} Gy")
    print(f"  Total volume: {cum_dvh[0]:.2f} cc")
    
    # Compute metrics
    metrics_spec = {
        'D_percent': [2, 50, 95, 98],
        'V_dose': [10, 20, 30, 40, 50]
    }
    
    metrics = dvh.compute_metrics(dose, mask, spacing, metrics_spec)
    
    print(f"\nDose metrics:")
    print(f"  Volume: {metrics['Volume_cc']:.2f} cc")
    print(f"  Dmean:  {metrics['Dmean']:.2f} Gy")
    print(f"  Dmax:   {metrics['Dmax']:.2f} Gy")
    print(f"  Dmin:   {metrics['Dmin']:.2f} Gy")
    
    print(f"\nDose coverage:")
    for key in ['D2%', 'D50%', 'D95%', 'D98%']:
        if key in metrics:
            print(f"  {key:6s}: {metrics[key]:.2f} Gy")
    
    print(f"\nVolume at dose:")
    for key in sorted([k for k in metrics.keys() if k.startswith('V') and 'Gy' in k]):
        print(f"  {key:8s}: {metrics[key]:.1f}%")
    
    return metrics


def example_comparison_with_reference(metrics):
    """Example 4: Compare with reference TPS."""
    print("\n" + "="*60)
    print("Example 4: Comparison with reference TPS")
    print("="*60)
    
    # Simulate reference metrics (slightly different)
    reference = {
        'Volume_cc': metrics['Volume_cc'] * 1.01,  # 1% higher
        'Dmean': metrics['Dmean'] + 0.2,            # +0.2 Gy
        'Dmax': metrics['Dmax'] + 0.5,              # +0.5 Gy
        'Dmin': metrics['Dmin'] - 0.1,              # -0.1 Gy
        'D2%': metrics.get('D2%', 0) + 0.3,
        'D95%': metrics.get('D95%', 0) + 0.1,
        'D98%': metrics.get('D98%', 0) + 0.1,
    }
    
    # Add V_dose metrics
    for key in metrics.keys():
        if key.startswith('V') and 'Gy' in key:
            reference[key] = metrics[key] + 0.5  # +0.5 percentage points
    
    print("Simulated reference metrics (from primary TPS):")
    print(f"  Dmean: {reference['Dmean']:.2f} Gy")
    print(f"  D95%:  {reference.get('D95%', 0):.2f} Gy")
    
    # Compare
    comparison = dvh.compare_dvh_metrics(
        metrics,
        reference,
        tolerance_abs=0.5,  # 0.5 Gy
        tolerance_rel=0.03  # 3%
    )
    
    print("\nComparison results:")
    all_pass = all(comp['pass'] for comp in comparison.values())
    print(f"  Overall status: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    
    print("\n  Metric          Calc      Ref       Diff      Status")
    print("  " + "-"*56)
    
    for metric_name, comp in sorted(comparison.items()):
        status = "✓ PASS" if comp['pass'] else "✗ FAIL"
        print(f"  {metric_name:12s}  {comp['calculated']:7.2f}  {comp['reference']:7.2f}  "
              f"{comp['diff']:+6.2f}  {status}")
    
    return comparison


def example_full_report(roi_name, metrics, comparison):
    """Example 5: Generate full report."""
    print("\n" + "="*60)
    print("Example 5: Generating full DVH report")
    print("="*60)
    
    report = dvh.generate_dvh_report(roi_name, metrics, comparison)
    print(report)


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# DoseCUDA Secondary Check Pipeline - Complete Example")
    print("#"*60)
    
    # Example 1: Load CT
    grid = example_ct_loading()
    
    # Example 2: Create and rasterize ROI
    roi, mask = example_rtstruct_rasterization(grid)
    
    # Create synthetic dose (for demonstration)
    # In real use, this would be: dose = grid.dose (after calculation)
    print("\n" + "="*60)
    print("Creating synthetic dose distribution...")
    print("="*60)
    
    # Create dose gradient (higher in center, lower at edges)
    dose = np.zeros(grid.size, dtype=np.single)
    center_idx = np.array(grid.size) // 2
    
    for k in range(grid.size[0]):
        for j in range(grid.size[1]):
            for i in range(grid.size[2]):
                # Distance from center
                dist = np.sqrt((k - center_idx[0])**2 + 
                             (j - center_idx[1])**2 + 
                             (i - center_idx[2])**2)
                
                # Dose falls off with distance
                dose[k, j, i] = max(0, 60.0 - dist * 0.5)
    
    print(f"Dose created: range {np.min(dose):.2f} - {np.max(dose):.2f} Gy")
    
    # Example 3: Calculate DVH and metrics
    metrics = example_dvh_calculation(dose, mask, grid.spacing)
    
    # Example 4: Compare with reference
    comparison = example_comparison_with_reference(metrics)
    
    # Example 5: Generate report
    example_full_report(roi.name, metrics, comparison)
    
    print("\n" + "#"*60)
    print("# Pipeline complete!")
    print("#"*60)
    print("\nThis example demonstrated:")
    print("  ✓ CT loading with validation (HU, direction, FrameOfReferenceUID)")
    print("  ✓ RTSTRUCT parsing and ROI rasterization")
    print("  ✓ DVH calculation (differential and cumulative)")
    print("  ✓ Dose metrics (Dmean, Dmax, D95, D98, V20, etc.)")
    print("  ✓ Comparison with reference TPS")
    print("  ✓ Report generation")
    print("\nFor real clinical use:")
    print("  1. Replace createCubePhantom() with grid.loadCTDCM('path/to/ct')")
    print("  2. Replace synthetic ROI with rtstruct.read_rtstruct('RTSTRUCT.dcm')")
    print("  3. Calculate dose using plan.computeIMRTPlan() or plan.computeIMPTPlan()")
    print("  4. Load reference RTDOSE for comparison")
    print("")


if __name__ == "__main__":
    main()
