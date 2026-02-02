# DoseCUDA - Secondary Dose Calculation System

## Clinical Patient Pipeline Documentation

This document describes the complete patient pipeline for clinical secondary dose calculation, including CT import, structure rasterization, and DVH analysis.

---

## Module Overview

### 1. CT Import (`plan.py`)

**Enhanced `loadCTDCM()` method:**

```python
from DoseCUDA.plan import DoseGrid

grid = DoseGrid()
grid.loadCTDCM('/path/to/ct/directory')

# Validated attributes:
# - grid.origin: (x, y, z) in mm
# - grid.spacing: (x, y, z) in mm  
# - grid.size: (z, y, x) in voxels
# - grid.HU: numpy array with corrected HU values
# - grid.direction: 3x3 direction cosine matrix
# - grid.FrameOfReferenceUID: for validation with RTSTRUCT
```

**Validations performed:**
- ✅ RescaleSlope/Intercept applied explicitly
- ✅ Oblique CT detection (raises error if direction ≠ identity)
- ✅ Slice spacing consistency check
- ✅ FrameOfReferenceUID extraction

---

### 2. RTSTRUCT Import and Rasterization (`rtstruct.py`)

#### Reading RTSTRUCT

```python
from DoseCUDA import rtstruct

# Read RTSTRUCT file
struct = rtstruct.read_rtstruct('/path/to/RTSTRUCT.dcm')

# Access ROIs
print(struct.get_roi_names())  # ['PTV', 'Bladder', 'Rectum', ...]

# Get specific ROI
ptv = struct.rois['PTV']
print(f"ROI: {ptv.name}")
print(f"Number of slices: {len(ptv.contour_slices)}")
print(f"Display color: {ptv.display_color}")

# Get bounding box
bbox_min, bbox_max = ptv.get_bounding_box_mm()
```

#### Validating Frame of Reference

```python
# Validate that RTSTRUCT matches CT
rtstruct.validate_rtstruct_with_ct(
    struct, 
    grid.FrameOfReferenceUID,
    strict=True  # Raises error on mismatch
)
```

#### Rasterizing ROI to Binary Mask

```python
# Rasterize ROI to mask on CT grid
mask = rtstruct.rasterize_roi_to_mask(
    roi=struct.rois['PTV'],
    origin=grid.origin,
    spacing=grid.spacing,
    size=grid.size,
    direction=grid.direction
)

# mask is numpy array of shape (z, y, x) with dtype=bool
# Calculate volume
voxel_volume = np.prod(grid.spacing) / 1000.0  # mm³ to cc
volume_cc = np.sum(mask) * voxel_volume
print(f"PTV volume: {volume_cc:.2f} cc")
```

**Rasterization algorithm:**
- Converts contour points from mm to voxel indices
- Uses matplotlib.path.Path for point-in-polygon testing
- Bounding box optimization for performance
- Multiple contours combined with OR operation

---

### 3. DVH and Dose Metrics (`dvh.py`)

#### Computing DVH

```python
from DoseCUDA import dvh
import numpy as np

# Assume dose calculated: grid.dose
dose = grid.dose  # numpy array (z, y, x)

# Calculate voxel volume
voxel_volume = np.prod(grid.spacing) / 1000.0  # cc

# Compute DVH
dose_bins, differential_dvh, cumulative_dvh = dvh.compute_dvh(
    dose=dose,
    mask=mask,
    voxel_volume=voxel_volume,
    bin_width=0.1  # Gy
)

print(f"Dose range: {dose_bins[0]:.2f} - {dose_bins[-1]:.2f} Gy")
print(f"Total volume: {cumulative_dvh[0]:.2f} cc")
```

#### Computing Dose Metrics

```python
# Specify metrics to calculate
metrics_spec = {
    'D_percent': [2, 50, 95, 98],    # D2%, D50%, D95%, D98%
    'V_dose': [10, 20, 30, 40, 50]   # V10Gy, V20Gy, ...
}

# Compute metrics
metrics = dvh.compute_metrics(
    dose=dose,
    mask=mask,
    spacing=grid.spacing,
    metrics_spec=metrics_spec
)

# Access metrics
print(f"Dmean: {metrics['Dmean']:.2f} Gy")
print(f"Dmax:  {metrics['Dmax']:.2f} Gy")
print(f"D95%:  {metrics['D95%']:.2f} Gy")
print(f"D98%:  {metrics['D98%']:.2f} Gy")
print(f"V20Gy: {metrics['V20Gy']:.1f}%")
print(f"Volume: {metrics['Volume_cc']:.2f} cc")
```

**Available metrics:**
- **Basic:** Dmean, Dmax, Dmin, Volume_cc
- **D_percent:** Dose received by X% of volume (sorted high to low)
  - D2%: dose to hottest 2%
  - D95%: dose covering 95% of volume (near-minimum)
- **V_dose:** Percentage of volume receiving ≥ specified dose
  - V20Gy: percent of volume receiving ≥ 20 Gy

#### Comparing with Reference

```python
# Reference metrics from primary TPS
reference_metrics = {
    'Dmean': 50.5,
    'D95%': 48.0,
    'V20Gy': 95.0,
    # ... other metrics
}

# Compare
comparison = dvh.compare_dvh_metrics(
    metrics_calculated=metrics,
    metrics_reference=reference_metrics,
    tolerance_abs=0.5,  # 0.5 Gy absolute
    tolerance_rel=0.03  # 3% relative
)

# Check overall pass/fail
all_pass = all(comp['pass'] for comp in comparison.values())
print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

# Detailed comparison
for metric_name, comp in comparison.items():
    print(f"{metric_name}: {comp['calculated']:.2f} vs {comp['reference']:.2f} "
          f"({comp['diff_percent']:+.1f}%) - {'PASS' if comp['pass'] else 'FAIL'}")
```

**Tolerance criteria:**
- **Dose metrics:** Pass if `|diff| < tol_abs` OR `|diff_rel| < tol_rel`
- **Volume metrics:** Pass if `|diff_rel| < tol_rel`

#### Generating Report

```python
# Generate formatted report
report = dvh.generate_dvh_report(
    roi_name='PTV',
    metrics=metrics,
    comparison=comparison  # optional
)

print(report)
```

**Example output:**
```
============================================================
DVH Metrics Report: PTV
============================================================

Volume: 150.25 cc

Dose Statistics:
  Dmean: 50.24 Gy
  Dmax:  62.30 Gy
  Dmin:  35.20 Gy

Dose Coverage:
  D2%: 61.50 Gy
  D95%: 45.20 Gy
  D98%: 42.10 Gy

Volume at Dose:
  V20Gy: 98.5%
  V40Gy: 85.2%

============================================================
Comparison vs Reference:
============================================================

Overall: PASS ✓

  ✓ Dmean       : Calc=  50.24, Ref=  50.50, Diff= -0.26 ( -0.5%)
  ✓ D95%        : Calc=  45.20, Ref=  45.50, Diff= -0.30 ( -0.7%)
  ✓ V20Gy       : Calc=  98.50, Ref=  98.00, Diff= +0.50 ( +0.5%)

============================================================
```

---

## Complete Workflow Example

```python
import numpy as np
from DoseCUDA.plan import DoseGrid
from DoseCUDA.plan_imrt import IMRTPlan, IMRTDoseGrid
from DoseCUDA import rtstruct, dvh

# 1. Load CT
print("Loading CT...")
ct_grid = DoseGrid()
ct_grid.loadCTDCM('/data/patient001/CT')

# 2. Load RTSTRUCT
print("Loading structures...")
struct = rtstruct.read_rtstruct('/data/patient001/RTSTRUCT.dcm')

# 3. Validate frame of reference
rtstruct.validate_rtstruct_with_ct(struct, ct_grid.FrameOfReferenceUID)

# 4. Rasterize structures of interest
print("Rasterizing structures...")
masks = {}
for roi_name in ['PTV', 'Bladder', 'Rectum']:
    if roi_name in struct.rois:
        masks[roi_name] = rtstruct.rasterize_roi_to_mask(
            struct.rois[roi_name],
            ct_grid.origin, ct_grid.spacing, ct_grid.size, ct_grid.direction
        )

# 5. Load plan and calculate dose
print("Calculating dose...")
plan = IMRTPlan(machine_name="VarianTrueBeamHF")
plan.readPlanDicom('/data/patient001/RTPLAN.dcm')

dose_grid = IMRTDoseGrid()
dose_grid.loadCTDCM('/data/patient001/CT')
dose_grid.computeIMRTPlan(plan, gpu_id=0)

# 6. Compute DVH and metrics for each structure
print("\nComputing DVH and metrics...")
voxel_volume = np.prod(dose_grid.spacing) / 1000.0

results = {}
for roi_name, mask in masks.items():
    # DVH
    dose_bins, diff_dvh, cum_dvh = dvh.compute_dvh(
        dose_grid.dose, mask, voxel_volume, bin_width=0.1
    )
    
    # Metrics
    metrics_spec = {
        'D_percent': [2, 50, 95, 98],
        'V_dose': [20, 30, 40, 50]
    }
    metrics = dvh.compute_metrics(
        dose_grid.dose, mask, dose_grid.spacing, metrics_spec
    )
    
    results[roi_name] = {
        'dose_bins': dose_bins,
        'cumulative_dvh': cum_dvh,
        'metrics': metrics
    }
    
    # Print summary
    print(f"\n{roi_name}:")
    print(f"  Volume: {metrics['Volume_cc']:.2f} cc")
    print(f"  Dmean:  {metrics['Dmean']:.2f} Gy")
    print(f"  D95%:   {metrics['D95%']:.2f} Gy")

# 7. Compare with reference (if available)
reference_metrics = {
    # Load from reference TPS or RTDOSE
}

comparison = dvh.compare_dvh_metrics(
    results['PTV']['metrics'],
    reference_metrics,
    tolerance_abs=0.5,
    tolerance_rel=0.03
)

# 8. Generate report
report = dvh.generate_dvh_report('PTV', results['PTV']['metrics'], comparison)
print(report)

# Save report to file
with open('/data/patient001/dvh_report.txt', 'w') as f:
    f.write(report)
```

---

## Error Handling and Warnings

### Common Warnings

**CT Loading:**
- `"CT sem FrameOfReferenceUID"` - CT missing FrameOfReferenceUID
- `"Slice spacing inconsistente"` - Irregular slice spacing detected
- `"CT oblíquo detectado"` - Oblique CT (raises error)

**RTSTRUCT Parsing:**
- `"ROI 'Name': ContourSequence ausente"` - ROI has no contours
- `"Contour com menos de 3 pontos"` - Invalid polygon
- `"RTSTRUCT e CT têm FrameOfReferenceUID diferentes"` - Mismatch

**Rasterization:**
- `"Contour em Z=... está fora do grid"` - Contour outside grid bounds
- `"MLC: Nenhuma posição MLCX encontrada"` - Missing MLC data

**DVH:**
- `"Máscara vazia: não há voxels na estrutura"` - Empty ROI mask

### Error Recovery

Most functions provide graceful degradation:
- Missing data → warnings + continue with valid data
- Invalid contours → skip + warning
- Empty masks → return NaN metrics + warning

---

## Performance Considerations

### Rasterization
- **Bounding box optimization:** Only test pixels within contour bounds
- **Typical performance:** ~0.1-1 second per ROI (depends on complexity)
- **Memory:** Mask requires `size_z × size_y × size_x × 1 byte`

### DVH Computation
- **Bin width:** Smaller bins = more detail but slower
- **Typical performance:** ~0.01-0.1 seconds per structure
- **Recommendation:** Use 0.1 Gy bin width for clinical use

---

## Testing

Run unit tests:
```bash
cd /path/to/DoseCUDA
source DoseCuda/bin/activate

# Test rasterization
pytest tests/test_rtstruct_rasterization.py -v

# Test DVH
pytest tests/test_dvh_metrics.py -v

# Run example
python tests/example_patient_pipeline.py
```

---

## Dependencies

Required packages:
- `numpy` - array operations
- `pydicom` - DICOM file reading
- `SimpleITK` - CT image processing
- `matplotlib` - point-in-polygon (rasterization)

---

## Limitations

### Current Version (v1.0)
- ❌ Oblique CTs not supported (must be axial)
- ❌ Holes in contours not supported
- ❌ RTDOSE import not yet implemented
- ❌ Gamma analysis not yet implemented

### Planned (v2.0)
- ✅ RTDOSE import for comparison
- ✅ Gamma index calculation
- ✅ Automated report generation
- ✅ Export calculated dose as RTDOSE

---

## References

Implementation based on:
- **OpenTPS** `dicomIO.py` - DICOM parsing best practices
- **DICOM Standard** Part 3 (Information Object Definitions)
- **TG-53** - Quality assurance for clinical radiotherapy treatment planning

---

## Support

For issues or questions:
1. Check unit tests for examples
2. Review `example_patient_pipeline.py`
3. Consult `TASK2_PATIENT_PIPELINE_COMPLETE.md`

---

## License

See LICENSE file in repository root.
