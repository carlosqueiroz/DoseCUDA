# Clinical Secondary Dose Check - Usage Guide

## Overview

This directory contains examples for **real clinical use** of DoseCUDA as a secondary dose calculation system.

For clinical validation of treatment plans from primary TPS.

---

## Quick Start

### 1. Organize Your Patient Data

```
/data/patients/patient_001/
├── CT/                      # Directory with CT DICOM slices
│   ├── CT.1.dcm
│   ├── CT.2.dcm
│   └── ...
├── RTPLAN.dcm              # Treatment plan from primary TPS
├── RTSTRUCT.dcm            # Structure set
└── RTDOSE_reference.dcm    # Reference dose from primary TPS
```

### 2. Run Secondary Check

```python
from examples.clinical_secondary_check import clinical_secondary_check_imrt

results = clinical_secondary_check_imrt(
    ct_path='/data/patients/patient_001/CT',
    rtplan_path='/data/patients/patient_001/RTPLAN.dcm',
    rtstruct_path='/data/patients/patient_001/RTSTRUCT.dcm',
    rtdose_ref_path='/data/patients/patient_001/RTDOSE_reference.dcm',
    roi_names=['PTV', 'Bladder', 'Rectum'],
    machine_name='VarianTrueBeamHF',
    gpu_id=0,
    tolerance_abs=0.5,  # ±0.5 Gy
    tolerance_rel=0.03  # ±3%
)

print(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
```

### 3. Review Report

```
============================================================
CLINICAL SECONDARY DOSE CHECK REPORT
============================================================

Plan Type: IMRT
Machine: VarianTrueBeamHF
Tolerance: ±0.5 Gy (abs) or ±3% (rel)

Overall Status: ✓✓✓ PASS ✓✓✓

------------------------------------------------------------

PTV: ✓ PASS
  Volume: 150.25 cc
  ✓ Dmean : Calc=  50.24 Gy, Ref=  50.50 Gy, Diff= -0.26 Gy ( -0.5%)
  ✓ Dmax  : Calc=  62.30 Gy, Ref=  62.80 Gy, Diff= -0.50 Gy ( -0.8%)
  ✓ D95%  : Calc=  48.20 Gy, Ref=  48.50 Gy, Diff= -0.30 Gy ( -0.6%)
  ✓ D98%  : Calc=  46.10 Gy, Ref=  46.30 Gy, Diff= -0.20 Gy ( -0.4%)

Bladder: ✓ PASS
  Volume: 85.30 cc
  ✓ Dmean : Calc=  25.40 Gy, Ref=  25.60 Gy, Diff= -0.20 Gy ( -0.8%)
  ...

============================================================
```

---

## Detailed Workflow

### Step 1: Load CT with Validation

```python
from DoseCUDA.plan import DoseGrid

grid = DoseGrid()
grid.loadCTDCM('/data/patients/patient_001/CT')

# Automatic validations:
# ✓ RescaleSlope/Intercept applied
# ✓ Oblique CT detection (raises error)
# ✓ FrameOfReferenceUID extracted
# ✓ Slice spacing validated
```

### Step 2: Load RTSTRUCT and Rasterize

```python
from DoseCUDA import rtstruct

# Read RTSTRUCT
struct = rtstruct.read_rtstruct('/data/patients/patient_001/RTSTRUCT.dcm')

# Validate frame of reference
rtstruct.validate_rtstruct_with_ct(struct, grid.FrameOfReferenceUID)

# Rasterize specific ROI
ptv_mask = rtstruct.rasterize_roi_to_mask(
    struct.rois['PTV'],
    grid.origin, grid.spacing, grid.size, grid.direction
)
```

### Step 3: Calculate Dose

**For IMRT:**
```python
from DoseCUDA.plan_imrt import IMRTPlan, IMRTDoseGrid

plan = IMRTPlan(machine_name='VarianTrueBeamHF')
plan.readPlanDicom('/data/patients/patient_001/RTPLAN.dcm')

dose_grid = IMRTDoseGrid()
dose_grid.loadCTDCM('/data/patients/patient_001/CT')
dose_grid.computeIMRTPlan(plan, gpu_id=0)
```

### Step 4: Load Reference RTDOSE

```python
from DoseCUDA import dvh

dose_ref, origin_ref, spacing_ref, frame_uid = dvh.read_reference_rtdose(
    '/data/patients/patient_001/RTDOSE_reference.dcm'
)

# Interpolate to calculated grid if geometries differ
dose_ref_interp = dvh.interpolate_dose_to_grid(
    dose_ref, origin_ref, spacing_ref,
    dose_grid.origin, dose_grid.spacing, dose_grid.size
)
```

### Step 5: Compute DVH and Compare

```python
from DoseCUDA import dvh

# Compute metrics for calculated dose
metrics_calc = dvh.compute_metrics(
    dose_grid.dose, ptv_mask, dose_grid.spacing,
    {'D_percent': [2, 95, 98], 'V_dose': [20, 30]}
)

# Compute metrics for reference dose
metrics_ref = dvh.compute_metrics(
    dose_ref_interp, ptv_mask, dose_grid.spacing,
    {'D_percent': [2, 95, 98], 'V_dose': [20, 30]}
)

# Compare with tolerances
comparison = dvh.compare_dvh_metrics(
    metrics_calc, metrics_ref,
    tolerance_abs=0.5,  # ±0.5 Gy
    tolerance_rel=0.03  # ±3%
)

# Check pass/fail
overall_pass = all(comp['pass'] for comp in comparison.values())
```

### Step 6: Generate Report

```python
report = dvh.generate_dvh_report('PTV', metrics_calc, comparison)
print(report)

# Save to file
with open('/data/patients/patient_001/secondary_check.txt', 'w') as f:
    f.write(report)
```

---

## Configuration

### Machine Names

Available machine configurations (in `lookuptables/photons/`):
- `VarianTrueBeamHF`
- `VarianTrueBeamSTD`
- (Add more as configured)

### Tolerance Criteria

Adjust based on institutional policy:

```python
# Conservative (stricter)
tolerance_abs = 0.3  # ±0.3 Gy
tolerance_rel = 0.02  # ±2%

# Standard (recommended)
tolerance_abs = 0.5  # ±0.5 Gy
tolerance_rel = 0.03  # ±3%

# Permissive (research only)
tolerance_abs = 1.0  # ±1.0 Gy
tolerance_rel = 0.05  # ±5%
```

### ROIs to Analyze

```python
# Prostate case
roi_names = ['PTV', 'Bladder', 'Rectum', 'Femur_L', 'Femur_R']

# Head & neck case
roi_names = ['PTV_Primary', 'PTV_Nodes', 'SpinalCord', 'Brainstem', 'Parotid_L', 'Parotid_R']

# Lung case
roi_names = ['PTV', 'Lung_L', 'Lung_R', 'Heart', 'Esophagus', 'SpinalCord']
```

---

## Troubleshooting

### Error: "CT oblíquo detectado"

**Problem:** CT is not axial (has oblique orientation)

**Solution:** Reorient CT to axial in primary TPS before export

### Error: "RTSTRUCT e CT têm FrameOfReferenceUID diferentes"

**Problem:** Structure set doesn't match CT geometry

**Solution:** 
1. Check that RTSTRUCT was created for this specific CT
2. Re-export RTSTRUCT from primary TPS
3. Use `strict=False` in validation (not recommended for clinical use)

### Warning: "Slice spacing inconsistente"

**Problem:** CT has irregular slice spacing

**Solution:**
1. Check for missing slices
2. Resample CT to uniform spacing in primary TPS
3. Proceed with caution (may affect accuracy)

### Error: "Beam model not found for beam energy 'X'"

**Problem:** Energy in RTPLAN doesn't match configured models

**Solution:**
1. Check `energy_labels.csv` for your machine
2. Add missing energy configuration
3. Verify RTPLAN has correct energy values

### Error: "MLC: Tamanho inválido do vetor"

**Problem:** MLC configuration mismatch

**Solution:**
1. Verify machine name matches RTPLAN
2. Check `mlc_geometry.csv` for correct MLC configuration
3. Ensure number of MLC pairs matches machine

---

## Validation Checklist

Before clinical use, validate with known test cases:

- [ ] Calculate dose for standard square fields (10×10, 5×5, 20×20)
- [ ] Compare with ion chamber measurements
- [ ] Validate DVH for simple geometries (cubes, cylinders)
- [ ] Test with clinical plans (at least 10 diverse cases)
- [ ] Compare with primary TPS within tolerance
- [ ] Document any systematic biases
- [ ] Establish institutional tolerance criteria
- [ ] Create standard operating procedure (SOP)
- [ ] Train physics staff on system use
- [ ] Implement regular QA checks

---

## Export Results

### Text Report
```python
with open('report.txt', 'w') as f:
    f.write(results['report'])
```

### CSV Export
```python
import csv

with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ROI', 'Metric', 'Calculated', 'Reference', 'Diff', 'Pass'])
    
    for roi_name, roi_result in results['roi_results'].items():
        for metric, comp in roi_result['comparison'].items():
            writer.writerow([
                roi_name, metric,
                f"{comp['calculated']:.2f}",
                f"{comp['reference']:.2f}",
                f"{comp['diff']:+.2f}",
                'PASS' if comp['pass'] else 'FAIL'
            ])
```

### JSON Export
```python
import json

# Remove non-serializable numpy arrays
results_json = {
    'overall_pass': results['overall_pass'],
    'roi_results': {
        roi: {
            'pass': data['pass'],
            'comparison': {
                k: {kk: float(vv) if isinstance(vv, (np.float32, np.float64)) else vv
                    for kk, vv in v.items()}
                for k, v in data['comparison'].items()
            }
        }
        for roi, data in results['roi_results'].items()
    }
}

with open('results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
```

---

## Safety and Regulatory Considerations

**IMPORTANT:** This software is for research and validation purposes.

### Before Clinical Use:

1. **Local Validation Required**
   - Perform comprehensive validation with your specific:
     - Machine configuration
     - Beam models
     - CT scanner protocols
   
2. **Quality Assurance**
   - Establish institutional QA program
   - Regular checks with reference dosimetry
   - Document validation results

3. **Regulatory Compliance**
   - Check local regulations for secondary check systems
   - Maintain audit trail of all calculations
   - Document software version and configuration

4. **Limitations**
   - Not approved for primary treatment planning
   - Results must be reviewed by qualified medical physicist
   - System limitations must be clearly documented

5. **Error Handling**
   - Never treat patient if secondary check fails
   - Investigate all discrepancies before treatment
   - Have clear escalation procedures

---

## Support

For issues or questions:
1. Review this documentation
2. Check test examples in `tests/`
3. Consult main documentation in parent directory

---

## License

See LICENSE file in repository root.

**DISCLAIMER:** This software is provided for research and validation purposes only. 
Users are responsible for local validation and compliance with all applicable regulations.
