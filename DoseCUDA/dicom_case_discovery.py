"""
Automatic DICOM case discovery and selection for patient data.

Recursively scans a directory for DICOM files (CT, RTPLAN, RTSTRUCT, RTDOSE),
classifies them by modality, and automatically selects the most appropriate
series/files for dose calculation based on clinical heuristics.

Key functions:
- scan_dicom_directory(): Recursive scan and classification
- select_rtplan(): Choose most likely treatment RTPLAN
- select_rtdose_template(): Choose PLAN RTDOSE for template
- select_ct_series(): Choose correct CT series (by RTSTRUCT refs, FrameOfReference, or slices)
- infer_machine_model(): Detect treatment machine from RTPLAN
- materialize_case(): Create output folder with selected files
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import pydicom
import numpy as np


@dataclass
class DicomFile:
    """Single DICOM file with metadata."""
    path: Path
    modality: str
    study_uid: str
    series_uid: str
    sop_uid: str
    frame_of_reference_uid: Optional[str] = None
    # CT-specific
    instance_number: Optional[int] = None
    slice_location: Optional[float] = None
    image_position: Optional[Tuple[float, float, float]] = None
    # RTPLAN-specific
    treatment_delivery_type: Optional[str] = None
    num_beams: Optional[int] = None
    machine_name: Optional[str] = None
    instance_creation_date: Optional[str] = None
    instance_creation_time: Optional[str] = None
    # RTDOSE-specific
    dose_summation_type: Optional[str] = None
    dose_grid_size: Optional[int] = None  # rows * cols * frames
    referenced_rtplan_uid: Optional[str] = None
    # RTSTRUCT-specific
    referenced_ct_sops: List[str] = field(default_factory=list)


@dataclass
class DicomCase:
    """Organized DICOM case with all modalities."""
    ct_files: List[DicomFile] = field(default_factory=list)
    rtplan_files: List[DicomFile] = field(default_factory=list)
    rtstruct_files: List[DicomFile] = field(default_factory=list)
    rtdose_files: List[DicomFile] = field(default_factory=list)
    other_files: List[DicomFile] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"DicomCase:\n"
            f"  CT series: {len(self.get_ct_series())} ({len(self.ct_files)} files)\n"
            f"  RTPLAN: {len(self.rtplan_files)}\n"
            f"  RTSTRUCT: {len(self.rtstruct_files)}\n"
            f"  RTDOSE: {len(self.rtdose_files)}\n"
            f"  Other: {len(self.other_files)}"
        )
    
    def get_ct_series(self) -> Dict[str, List[DicomFile]]:
        """Group CT files by SeriesInstanceUID."""
        series = defaultdict(list)
        for ct in self.ct_files:
            series[ct.series_uid].append(ct)
        return dict(series)


def scan_dicom_directory(root_dir: str) -> DicomCase:
    """
    Recursively scan directory for DICOM files and classify by modality.
    
    Parameters
    ----------
    root_dir : str
        Root directory to scan (will search recursively)
        
    Returns
    -------
    case : DicomCase
        Organized DICOM files by modality
        
    Notes
    -----
    - Uses stop_before_pixels=True for fast header-only reading
    - Silently skips non-DICOM files
    - Extracts metadata needed for automatic selection
    """
    case = DicomCase()
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    print(f"\n[DICOM Discovery] Scanning: {root_dir}")
    scanned_count = 0
    error_count = 0
    
    # Recursively find all files
    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        scanned_count += 1
        
        try:
            # Try to read as DICOM (header only)
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
            
            # Check if it has Modality
            if not hasattr(ds, 'Modality'):
                continue
            
            modality = ds.Modality
            
            # Extract common metadata
            dicom_file = DicomFile(
                path=file_path,
                modality=modality,
                study_uid=getattr(ds, 'StudyInstanceUID', 'UNKNOWN'),
                series_uid=getattr(ds, 'SeriesInstanceUID', 'UNKNOWN'),
                sop_uid=getattr(ds, 'SOPInstanceUID', 'UNKNOWN'),
                frame_of_reference_uid=getattr(ds, 'FrameOfReferenceUID', None)
            )
            
            # Modality-specific metadata
            if modality == 'CT':
                dicom_file.instance_number = getattr(ds, 'InstanceNumber', None)
                dicom_file.slice_location = getattr(ds, 'SliceLocation', None)
                if hasattr(ds, 'ImagePositionPatient'):
                    dicom_file.image_position = tuple(ds.ImagePositionPatient)
                case.ct_files.append(dicom_file)
                
            elif modality == 'RTPLAN':
                # Check for treatment delivery type
                if hasattr(ds, 'BeamSequence') and len(ds.BeamSequence) > 0:
                    dicom_file.num_beams = len(ds.BeamSequence)
                    # Check first beam for TreatmentDeliveryType
                    if hasattr(ds.BeamSequence[0], 'TreatmentDeliveryType'):
                        dicom_file.treatment_delivery_type = ds.BeamSequence[0].TreatmentDeliveryType
                    # Get machine name
                    if hasattr(ds.BeamSequence[0], 'TreatmentMachineName'):
                        dicom_file.machine_name = ds.BeamSequence[0].TreatmentMachineName
                
                # Also check plan-level machine name
                if not dicom_file.machine_name and hasattr(ds, 'TreatmentMachineName'):
                    dicom_file.machine_name = ds.TreatmentMachineName
                
                # Instance creation metadata
                dicom_file.instance_creation_date = getattr(ds, 'InstanceCreationDate', None)
                dicom_file.instance_creation_time = getattr(ds, 'InstanceCreationTime', None)
                
                case.rtplan_files.append(dicom_file)
                
            elif modality == 'RTDOSE':
                dicom_file.dose_summation_type = getattr(ds, 'DoseSummationType', None)
                
                # Calculate grid size
                rows = getattr(ds, 'Rows', 0)
                cols = getattr(ds, 'Columns', 0)
                frames = getattr(ds, 'NumberOfFrames', 0)
                dicom_file.dose_grid_size = rows * cols * frames
                
                # Referenced RTPLAN
                if hasattr(ds, 'ReferencedRTPlanSequence') and len(ds.ReferencedRTPlanSequence) > 0:
                    dicom_file.referenced_rtplan_uid = ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
                
                case.rtdose_files.append(dicom_file)
                
            elif modality == 'RTSTRUCT':
                # Extract referenced CT SOPInstanceUIDs from ContourImageSequence
                referenced_sops = set()
                if hasattr(ds, 'ROIContourSequence'):
                    for roi_contour in ds.ROIContourSequence:
                        if hasattr(roi_contour, 'ContourSequence'):
                            for contour in roi_contour.ContourSequence:
                                if hasattr(contour, 'ContourImageSequence'):
                                    for img_ref in contour.ContourImageSequence:
                                        if hasattr(img_ref, 'ReferencedSOPInstanceUID'):
                                            referenced_sops.add(img_ref.ReferencedSOPInstanceUID)
                
                dicom_file.referenced_ct_sops = list(referenced_sops)
                case.rtstruct_files.append(dicom_file)
                
            else:
                dicom_file.modality = modality
                case.other_files.append(dicom_file)
                
        except Exception as e:
            error_count += 1
            # Silently skip non-DICOM files
            continue
    
    print(f"[DICOM Discovery] Scanned {scanned_count} files, found:")
    print(f"  CT: {len(case.ct_files)} files in {len(case.get_ct_series())} series")
    print(f"  RTPLAN: {len(case.rtplan_files)}")
    print(f"  RTSTRUCT: {len(case.rtstruct_files)}")
    print(f"  RTDOSE: {len(case.rtdose_files)}")
    print(f"  Other: {len(case.other_files)}")
    if error_count > 0:
        print(f"  Skipped {error_count} non-DICOM files")
    
    return case


def select_rtplan(case: DicomCase) -> Optional[DicomFile]:
    """
    Select most appropriate RTPLAN for treatment.
    
    Selection criteria (in order):
    1. TreatmentDeliveryType == "TREATMENT"
    2. Highest number of beams
    3. Most recent (InstanceCreationDate/Time)
    
    Parameters
    ----------
    case : DicomCase
        DICOM case with all files
        
    Returns
    -------
    rtplan : DicomFile or None
        Selected RTPLAN, or None if no RTPLAN found
        
    Warns
    -----
    If multiple RTPLANs exist, warns about selection criteria used
    """
    if not case.rtplan_files:
        warnings.warn("No RTPLAN found in case")
        return None
    
    if len(case.rtplan_files) == 1:
        rtplan = case.rtplan_files[0]
        print(f"\n[RTPLAN Selection] Only 1 RTPLAN found: {rtplan.path.name}")
        return rtplan
    
    # Multiple RTPLANs - apply selection criteria
    print(f"\n[RTPLAN Selection] Multiple RTPLANs found ({len(case.rtplan_files)}), selecting best...")
    
    # 1. Prefer TreatmentDeliveryType == "TREATMENT"
    treatment_plans = [p for p in case.rtplan_files if p.treatment_delivery_type == "TREATMENT"]
    if treatment_plans:
        if len(treatment_plans) == 1:
            rtplan = treatment_plans[0]
            print(f"  ✓ Selected RTPLAN with TreatmentDeliveryType=TREATMENT: {rtplan.path.name}")
            return rtplan
        candidates = treatment_plans
    else:
        candidates = case.rtplan_files
    
    # 2. Prefer highest number of beams
    max_beams = max((p.num_beams or 0) for p in candidates)
    beam_candidates = [p for p in candidates if (p.num_beams or 0) == max_beams]
    
    if len(beam_candidates) == 1:
        rtplan = beam_candidates[0]
        print(f"  ✓ Selected RTPLAN with most beams ({max_beams}): {rtplan.path.name}")
        return rtplan
    
    candidates = beam_candidates
    
    # 3. Prefer most recent
    dated_candidates = [p for p in candidates if p.instance_creation_date]
    if dated_candidates:
        # Sort by date+time descending
        dated_candidates.sort(
            key=lambda p: (p.instance_creation_date or "", p.instance_creation_time or ""),
            reverse=True
        )
        rtplan = dated_candidates[0]
        print(f"  ✓ Selected most recent RTPLAN: {rtplan.path.name} ({rtplan.instance_creation_date})")
        return rtplan
    
    # Fallback: first in list
    rtplan = candidates[0]
    warnings.warn(f"Multiple RTPLANs matched all criteria, using first: {rtplan.path.name}")
    return rtplan


def select_rtdose_template(case: DicomCase) -> Optional[DicomFile]:
    """
    Select RTDOSE to use as template for grid/tags.
    
    Selection criteria:
    1. DoseSummationType == "PLAN"
    2. Largest grid (rows * cols * frames)
    
    Parameters
    ----------
    case : DicomCase
        DICOM case with all files
        
    Returns
    -------
    rtdose : DicomFile or None
        Selected RTDOSE template, or None if no RTDOSE found
        
    Warns
    -----
    If multiple RTPLANs found, warns about selection
    """
    if not case.rtdose_files:
        warnings.warn("No RTDOSE found - will not save DICOM output")
        return None
    
    if len(case.rtdose_files) == 1:
        rtdose = case.rtdose_files[0]
        print(f"\n[RTDOSE Template] Only 1 RTDOSE found: {rtdose.path.name}")
        return rtdose
    
    print(f"\n[RTDOSE Template] Multiple RTDOSE found ({len(case.rtdose_files)}), selecting PLAN...")
    
    # 1. Prefer DoseSummationType == "PLAN"
    plan_doses = [d for d in case.rtdose_files if d.dose_summation_type == "PLAN"]
    
    if not plan_doses:
        warnings.warn("No RTDOSE with DoseSummationType=PLAN, using largest grid")
        candidates = case.rtdose_files
    else:
        candidates = plan_doses
    
    # 2. Choose largest grid
    candidates.sort(key=lambda d: d.dose_grid_size or 0, reverse=True)
    rtdose = candidates[0]
    
    if len(candidates) > 1:
        print(f"  ✓ Selected RTDOSE: {rtdose.path.name} (grid size: {rtdose.dose_grid_size})")
        print(f"    (warning: {len(candidates)} candidates matched criteria)")
    else:
        print(f"  ✓ Selected RTDOSE: {rtdose.path.name}")
    
    return rtdose


def select_ct_series(
    case: DicomCase,
    rtstruct: Optional[DicomFile] = None,
    rtdose: Optional[DicomFile] = None
) -> Optional[List[DicomFile]]:
    """
    Select appropriate CT series for dose calculation.
    
    Selection criteria (in order):
    1. RTSTRUCT references: Choose CT series with most referenced SOPInstanceUIDs
    2. FrameOfReferenceUID: Match CT to RTDOSE/RTSTRUCT
    3. Fallback: Choose CT series with most slices
    
    Parameters
    ----------
    case : DicomCase
        DICOM case with all files
    rtstruct : DicomFile, optional
        RTSTRUCT file (if available) for SOPInstanceUID matching
    rtdose : DicomFile, optional
        RTDOSE file (if available) for FrameOfReferenceUID matching
        
    Returns
    -------
    ct_series : List[DicomFile] or None
        Selected CT series files (sorted by ImagePositionPatient Z), or None if no CT found
        
    Raises
    ------
    ValueError
        If selected CT series has files without ImagePositionPatient
        
    Warns
    -----
    About which selection criterion was used
    """
    if not case.ct_files:
        raise ValueError("No CT files found in case")
    
    ct_series_dict = case.get_ct_series()
    
    if len(ct_series_dict) == 1:
        series_uid = list(ct_series_dict.keys())[0]
        ct_series = ct_series_dict[series_uid]
        print(f"\n[CT Series] Only 1 CT series found: {len(ct_series)} slices")
        return _validate_and_sort_ct_series(ct_series)
    
    print(f"\n[CT Series] Multiple CT series found ({len(ct_series_dict)}), selecting best...")
    
    # 1. Try RTSTRUCT reference matching
    if rtstruct and rtstruct.referenced_ct_sops:
        print(f"  Criterion 1: RTSTRUCT references {len(rtstruct.referenced_ct_sops)} CT SOPs")
        
        # Count matches per series
        series_matches = {}
        for series_uid, ct_files in ct_series_dict.items():
            ct_sops = {ct.sop_uid for ct in ct_files}
            matches = len(ct_sops.intersection(rtstruct.referenced_ct_sops))
            series_matches[series_uid] = matches
        
        max_matches = max(series_matches.values())
        if max_matches > 0:
            # Choose series with most matches
            best_series_uid = max(series_matches, key=series_matches.get)
            ct_series = ct_series_dict[best_series_uid]
            print(f"  ✓ Selected CT series with {max_matches} RTSTRUCT matches ({len(ct_series)} slices)")
            return _validate_and_sort_ct_series(ct_series)
        else:
            print("  ⚠ RTSTRUCT references found but no CT SOPs matched")
    
    # 2. Try FrameOfReferenceUID matching
    reference_for = None
    if rtdose and rtdose.frame_of_reference_uid:
        reference_for = rtdose.frame_of_reference_uid
        print(f"  Criterion 2: FrameOfReferenceUID from RTDOSE")
    elif rtstruct and rtstruct.frame_of_reference_uid:
        reference_for = rtstruct.frame_of_reference_uid
        print(f"  Criterion 2: FrameOfReferenceUID from RTSTRUCT")
    
    if reference_for:
        matching_series = {
            uid: files for uid, files in ct_series_dict.items()
            if files[0].frame_of_reference_uid == reference_for
        }
        if matching_series:
            if len(matching_series) == 1:
                series_uid = list(matching_series.keys())[0]
                ct_series = matching_series[series_uid]
                print(f"  ✓ Selected CT series by FrameOfReferenceUID ({len(ct_series)} slices)")
                return _validate_and_sort_ct_series(ct_series)
            else:
                # Multiple matches - use slice count
                print(f"  Multiple CT series match FrameOfReferenceUID ({len(matching_series)})")
                ct_series_dict = matching_series
    
    # 3. Fallback: Choose series with most slices
    print(f"  Criterion 3 (fallback): Most slices")
    series_by_count = sorted(ct_series_dict.items(), key=lambda x: len(x[1]), reverse=True)
    series_uid, ct_series = series_by_count[0]
    print(f"  ✓ Selected CT series with most slices ({len(ct_series)} slices)")
    
    return _validate_and_sort_ct_series(ct_series)


def _validate_and_sort_ct_series(ct_series: List[DicomFile]) -> List[DicomFile]:
    """
    Validate CT series has ImagePositionPatient and sort by Z position.
    
    Parameters
    ----------
    ct_series : List[DicomFile]
        CT files in series
        
    Returns
    -------
    sorted_series : List[DicomFile]
        CT files sorted by Z position (ImagePositionPatient[2])
        
    Raises
    ------
    ValueError
        If any CT file is missing ImagePositionPatient
    """
    # Validate all have ImagePositionPatient
    missing_position = [ct for ct in ct_series if ct.image_position is None]
    if missing_position:
        raise ValueError(
            f"CT series has {len(missing_position)} files without ImagePositionPatient. "
            "Cannot use for dose calculation."
        )
    
    # Sort by Z position
    sorted_series = sorted(ct_series, key=lambda ct: ct.image_position[2])
    
    return sorted_series


def infer_machine_model(rtplan: DicomFile, default_model: str = "VarianTrueBeamHF") -> str:
    """
    Infer DoseCUDA machine model from RTPLAN metadata.
    
    Uses heuristics to map treatment machine name to DoseCUDA machine model.
    
    Parameters
    ----------
    rtplan : DicomFile
        Selected RTPLAN
    default_model : str
        Default model if inference fails (default: "VarianTrueBeamHF")
        
    Returns
    -------
    machine_model : str
        DoseCUDA machine model name
        
    Notes
    -----
    Heuristics:
    - "truebeam" -> "VarianTrueBeamHF"
    - "halcyon" -> "VarianHalcyon" (if implemented)
    - Otherwise: use default_model
    
    The function also logs which DICOM field/value was used for inference.
    """
    print(f"\n[Machine Model] Inferring from RTPLAN: {rtplan.path.name}")
    
    machine_name = rtplan.machine_name
    
    if not machine_name:
        print(f"  ⚠ No TreatmentMachineName found in RTPLAN")
        print(f"  Using default: {default_model}")
        return default_model
    
    print(f"  TreatmentMachineName: '{machine_name}'")
    
    # Apply heuristics
    name_lower = machine_name.lower()
    
    if "truebeam" in name_lower:
        model = "VarianTrueBeamHF"
        print(f"  ✓ Detected 'truebeam' -> {model}")
        return model
    elif "halcyon" in name_lower:
        model = "VarianHalcyon"
        print(f"  ✓ Detected 'halcyon' -> {model}")
        warnings.warn(f"Machine model '{model}' may not be implemented in DoseCUDA")
        return model
    else:
        print(f"  ⚠ No matching heuristic for '{machine_name}'")
        print(f"  Using default: {default_model}")
        return default_model


def materialize_case(
    output_dir: str,
    ct_series: List[DicomFile],
    rtplan: DicomFile,
    rtdose: Optional[DicomFile] = None,
    rtstruct: Optional[DicomFile] = None
) -> Dict[str, Path]:
    """
    Create output folder with selected files organized for dose calculation.
    
    Creates structure:
    output_dir/
        CT_ONLY/         # Symlinks (or copies) of CT series only
        RTPLAN.dcm       # Copy of selected RTPLAN
        RTDOSE.dcm       # Copy of template RTDOSE (if provided)
        RTSTRUCT.dcm     # Copy of RTSTRUCT (if provided)
    
    Parameters
    ----------
    output_dir : str
        Output directory path
    ct_series : List[DicomFile]
        Selected CT series files
    rtplan : DicomFile
        Selected RTPLAN
    rtdose : DicomFile, optional
        Template RTDOSE
    rtstruct : DicomFile, optional
        RTSTRUCT file
        
    Returns
    -------
    paths : Dict[str, Path]
        Dictionary with keys: 'ct_dir', 'rtplan', 'rtdose', 'rtstruct'
        Values are None if file not provided
    """
    import shutil
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Materialize] Creating case folder: {output_dir}")
    
    # Create CT_ONLY directory (clean start)
    ct_only_dir = output_path / "CT_ONLY"
    if ct_only_dir.exists():
        shutil.rmtree(ct_only_dir)  # Remove old directory to avoid stale symlinks
    ct_only_dir.mkdir(exist_ok=True)
    
    print(f"  Creating CT_ONLY with {len(ct_series)} slices...")
    
    # Try symlinks first, fallback to copy
    use_symlinks = True
    for i, ct_file in enumerate(ct_series):
        target = ct_only_dir / f"CT_{i+1:04d}.dcm"
        try:
            # Remove existing file/symlink (including broken symlinks)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(ct_file.path.resolve())
        except (OSError, NotImplementedError):
            # Symlinks not supported, use copy
            if use_symlinks:  # Only print once
                print("    (symlinks not supported, using copy)")
                use_symlinks = False
            # Remove broken symlink if it exists
            if target.is_symlink():
                target.unlink()
            shutil.copy2(ct_file.path, target)
    
    if use_symlinks:
        print("    ✓ CT files linked")
    else:
        print("    ✓ CT files copied")
    
    # Copy RTPLAN
    rtplan_path = output_path / "RTPLAN.dcm"
    shutil.copy2(rtplan.path, rtplan_path)
    print(f"  ✓ RTPLAN copied: {rtplan.path.name}")
    
    # Copy RTDOSE template (if provided)
    rtdose_path = None
    if rtdose:
        rtdose_path = output_path / "RTDOSE_template.dcm"
        shutil.copy2(rtdose.path, rtdose_path)
        print(f"  ✓ RTDOSE template copied: {rtdose.path.name}")
    
    # Copy RTSTRUCT (if provided)
    rtstruct_path = None
    if rtstruct:
        rtstruct_path = output_path / "RTSTRUCT.dcm"
        shutil.copy2(rtstruct.path, rtstruct_path)
        print(f"  ✓ RTSTRUCT copied: {rtstruct.path.name}")
    
    return {
        'ct_dir': ct_only_dir,
        'rtplan': rtplan_path,
        'rtdose': rtdose_path,
        'rtstruct': rtstruct_path
    }


# ============================================================================
# Multi-phase utilities
# ============================================================================

@dataclass
class DicomPhase:
    """
    Represents a coherent treatment phase:
    - one RTPLAN
    - zero/one RTDOSE that references that plan
    - optional RTSTRUCT sharing the same FrameOfReferenceUID
    - CT series chosen for that FrameOfReferenceUID
    """
    rtplan: DicomFile
    ct_series: List[DicomFile]
    rtdose: Optional[DicomFile] = None
    rtstruct: Optional[DicomFile] = None
    warnings: List[str] = field(default_factory=list)


def enumerate_phases(case: DicomCase) -> List[DicomPhase]:
    """
    Build deterministic plan-centric phases without user interaction.

    Matching logic (per RTPLAN):
    1) RTDOSE: any with ReferencedRTPlanUID == plan.sop_uid; prefer DoseSummationType=PLAN,
       then largest grid.
    2) RTSTRUCT: first with same FrameOfReferenceUID.
    3) CT: series whose FrameOfReferenceUID matches the plan; choose the series
       with most slices; validate & sort by z.

    Falls back to select_ct_series() if no CT shares the FOR (rare).
    """
    phases: List[DicomPhase] = []

    for plan in case.rtplan_files:
        notes: List[str] = []

        # --- RTDOSE match ---
        dose_candidates = [
            d for d in case.rtdose_files
            if d.referenced_rtplan_uid == plan.sop_uid
        ]
        if dose_candidates:
            plan_doses = [d for d in dose_candidates if d.dose_summation_type == "PLAN"]
            if plan_doses:
                rtdose = plan_doses[0]
            else:
                # largest grid as tie-breaker
                dose_candidates.sort(key=lambda d: d.dose_grid_size or 0, reverse=True)
                rtdose = dose_candidates[0]
        else:
            rtdose = None
            notes.append("No RTDOSE referencing this RTPLAN")

        # --- RTSTRUCT match on FrameOfReferenceUID ---
        rtstruct_candidates = [
            s for s in case.rtstruct_files
            if s.frame_of_reference_uid and s.frame_of_reference_uid == plan.frame_of_reference_uid
        ]
        rtstruct = rtstruct_candidates[0] if rtstruct_candidates else None
        if not rtstruct and case.rtstruct_files:
            notes.append("RTSTRUCT exists but none share FrameOfReferenceUID with RTPLAN")

        # --- CT series match on FrameOfReferenceUID ---
        ct_series = _select_ct_series_by_for(case, plan.frame_of_reference_uid)
        if not ct_series:
            try:
                # fallback to existing heuristic using rtstruct/dose hints
                ct_series = select_ct_series(case, rtstruct=rtstruct, rtdose=rtdose)
                notes.append("CT selected via fallback heuristics (no FOR match)")
            except Exception as e:
                notes.append(f"CT selection failed: {e}")
                ct_series = []

        phases.append(
            DicomPhase(
                rtplan=plan,
                ct_series=ct_series,
                rtdose=rtdose,
                rtstruct=rtstruct,
                warnings=notes
            )
        )

    return phases


def _select_ct_series_by_for(case: DicomCase, for_uid: Optional[str]) -> Optional[List[DicomFile]]:
    """Pick CT series whose FrameOfReferenceUID matches; choose the series with most slices."""
    if not for_uid:
        return None

    series = {}
    for ct in case.ct_files:
        if ct.frame_of_reference_uid == for_uid:
            series.setdefault(ct.series_uid, []).append(ct)

    if not series:
        return None

    # choose the series with most slices
    _, ct_series = max(series.items(), key=lambda item: len(item[1]))
    return _validate_and_sort_ct_series(ct_series)
