import numpy as np
import SimpleITK as sitk
import pydicom as pyd


class VolumeObject:

    def __init__(self):
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.spacing = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.voxel_data = []


class Prescription:

    def __init__(self):
        self.TargetPrescriptionDose = 0.0
        self.ROIName = None
        self.TargetUnderdoseVolumeFraction = 0.0


class DoseGrid:

    def __init__(self):
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.spacing = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.size = np.array([0, 0, 0])
        self.HU = []
        self.dose = []
        self.beam_doses = []
        self.FrameOfReferenceUID = ""
        self.direction = np.eye(3, dtype=np.single)  # Direction cosine matrix (3x3 identity for axial)

    def loadCTNRRD(self, ct_path):
        fr = sitk.ImageFileReader()
        fr.SetFileName(ct_path)
        ct_img = fr.Execute()

        self.origin = np.array(ct_img.GetOrigin())
        self.spacing = np.array(ct_img.GetSpacing())
        self.HU = np.array(sitk.GetArrayFromImage(ct_img), dtype=np.single)
        self.size = np.array(self.HU.shape)

    def loadCTDCM(self, ct_path):
        """
        Load CT DICOM series with robust handling for clinical secondary dose calculation.
        
        - Applies RescaleSlope/Intercept explicitly to ensure correct HU values
        - Validates and stores direction cosines (oblique CTs detected and rejected)
        - Stores FrameOfReferenceUID for validation with RTSTRUCT
        - Validates slice spacing consistency
        
        Parameters
        ----------
        ct_path : str
            Path to directory containing CT DICOM files
        """
        import warnings
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ct_path)

        if len(dicom_names) == 0:
            raise ValueError(f"Nenhum arquivo DICOM encontrado em: {ct_path}")

        # Sort by ImagePositionPatient[2] (Z coordinate)
        # Read Z positions once to avoid multiple pydicom reads (performance)
        dicom_names = list(dicom_names)
        z_positions = []
        for fname in dicom_names:
            dcm = pyd.dcmread(fname, force=True, stop_before_pixels=True)
            z_positions.append(float(dcm.ImagePositionPatient[2]))
        
        # Sort by Z position
        sorted_indices = np.argsort(z_positions)
        dicom_names = [dicom_names[i] for i in sorted_indices]
        z_positions_sorted = [z_positions[i] for i in sorted_indices]
        
        # Store Z positions for later use (e.g., RTSTRUCT rasterization)
        self._ct_z_positions = np.array(z_positions_sorted)

        # Read first slice to get metadata
        first_dcm = pyd.dcmread(dicom_names[0], force=True)
        
        # Get RescaleSlope and RescaleIntercept
        rescale_slope = float(first_dcm.RescaleSlope) if hasattr(first_dcm, 'RescaleSlope') else 1.0
        rescale_intercept = float(first_dcm.RescaleIntercept) if hasattr(first_dcm, 'RescaleIntercept') else 0.0
        
        if rescale_slope != 1.0 or rescale_intercept != 0.0:
            warnings.warn(
                f"CT com RescaleSlope={rescale_slope}, RescaleIntercept={rescale_intercept}. "
                "Aplicando correção explicitamente para garantir HU corretos."
            )
        
        # Get FrameOfReferenceUID
        if hasattr(first_dcm, 'FrameOfReferenceUID'):
            self.FrameOfReferenceUID = first_dcm.FrameOfReferenceUID
        else:
            warnings.warn("CT sem FrameOfReferenceUID. Isso pode dificultar validação com RTSTRUCT.")
            self.FrameOfReferenceUID = ""
        
        # Load CT with SimpleITK
        reader.SetFileNames(dicom_names)
        ct_img = reader.Execute()

        # Store geometric information
        self.origin = np.array(ct_img.GetOrigin(), dtype=np.single)
        self.spacing = np.array(ct_img.GetSpacing(), dtype=np.single)
        
        # Get and validate direction matrix
        direction_flat = np.array(ct_img.GetDirection())
        self.direction = direction_flat.reshape(3, 3).astype(np.single)
        
        # Validate if CT is approximately axial (direction ~ identity)
        identity = np.eye(3)
        direction_diff = np.abs(self.direction - identity)
        max_deviation = np.max(direction_diff)
        
        if max_deviation > 0.01:  # Tolerance of 1% (~0.6 degrees)
            raise ValueError(
                f"CT oblíquo detectado (desvio máximo da identidade: {max_deviation:.4f}). "
                f"Direction matrix:\n{self.direction}\n"
                "Cálculo secundário requer CT reorientado para axial. "
                "Por favor, reoriente o CT antes de importar."
            )
        
        # Get HU array from SimpleITK
        hu_array = np.array(sitk.GetArrayFromImage(ct_img), dtype=np.single)
        
        # Apply rescale explicitly (SimpleITK may or may not have applied it)
        # To be safe, we read pixel_array from first slice and compare
        first_pixel_raw = first_dcm.pixel_array[0, 0]
        first_pixel_sitk = hu_array[0, 0, 0]
        first_pixel_expected = first_pixel_raw * rescale_slope + rescale_intercept
        
        # CRITICAL FIX: Actually apply the correction if SimpleITK didn't
        if abs(first_pixel_sitk - first_pixel_expected) > 0.1:
            warnings.warn(
                f"SimpleITK não aplicou RescaleSlope/Intercept corretamente. "
                f"Pixel[0,0]: SITK={first_pixel_sitk:.1f}, esperado={first_pixel_expected:.1f}. "
                f"Aplicando manualmente: HU = pixel * {rescale_slope} + {rescale_intercept}"
            )
            hu_array = hu_array * rescale_slope + rescale_intercept
        
        # Additional validation: check if HU values are plausible
        hu_min, hu_max = np.min(hu_array), np.max(hu_array)
        if hu_min < -2000 or hu_max > 5000:
            warnings.warn(
                f"HU fora de range plausível: min={hu_min:.0f}, max={hu_max:.0f}. "
                "Verifique se RescaleSlope/Intercept estão corretos."
            )
        
        # Store HU and clip to valid range
        self.HU = np.clip(hu_array, -1000.0, None)
        self.size = np.array(self.HU.shape)
        
        # Validate slice spacing consistency
        self._validate_slice_spacing(dicom_names)

    def _validate_slice_spacing(self, dicom_names):
        """
        Validate consistency of slice spacing by checking ImagePositionPatient.
        
        Warns if spacing varies by more than 1% from mean spacing.
        """
        import warnings
        
        if len(dicom_names) < 2:
            return
        
        z_positions = []
        for dcm_path in dicom_names:
            dcm = pyd.dcmread(dcm_path, force=True)
            z_positions.append(float(dcm.ImagePositionPatient[2]))
        
        z_positions = np.array(z_positions)
        z_diffs = np.diff(z_positions)
        
        mean_spacing = np.mean(z_diffs)
        max_deviation = np.max(np.abs(z_diffs - mean_spacing))
        relative_deviation = max_deviation / mean_spacing if mean_spacing > 0 else 0
        
        if relative_deviation > 0.01:  # 1% tolerance
            warnings.warn(
                f"Slice spacing inconsistente: spacing médio = {mean_spacing:.3f} mm, "
                f"desvio máximo = {max_deviation:.3f} mm ({relative_deviation*100:.2f}%). "
                "Isso pode indicar slices faltando ou espaçamento irregular."
            )
        
        # Also check against declared SliceThickness
        first_dcm = pyd.dcmread(dicom_names[0], force=True)
        if hasattr(first_dcm, 'SliceThickness'):
            declared_thickness = float(first_dcm.SliceThickness)
            if abs(mean_spacing - declared_thickness) > 0.001:
                warnings.warn(
                    f"Spacing real entre slices ({mean_spacing:.3f} mm) difere de "
                    f"SliceThickness declarado ({declared_thickness:.3f} mm). "
                    "Usando spacing real calculado."
                )

    def resampleCT(self, new_spacing, new_size, new_origin):
        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        rf = sitk.ResampleImageFilter()
        rf.SetOutputOrigin(new_origin)
        rf.SetOutputSpacing(new_spacing)
        rf.SetSize(new_size)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)
        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)

        self.origin = new_origin
        self.spacing = new_spacing
        self.size = np.array(self.HU.shape)

    def resampleCTfromSpacing(self, spacing):

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(tuple(float(x) for x in self.origin))
        HU_img.SetSpacing(tuple(float(x) for x in self.spacing))

        rf = sitk.ResampleImageFilter()
        rf.SetOutputOrigin(tuple(float(x) for x in self.origin))
        sp_new = (spacing, spacing, spacing)
        sz_new = (int(self.size[2] * self.spacing[0] / sp_new[0]),
                  int(self.size[1] * self.spacing[1] / sp_new[1]),
                  int(self.size[0] * self.spacing[2] / sp_new[2]))
        rf.SetOutputSpacing(sp_new)
        rf.SetSize(sz_new)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)
        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)
        
        self.spacing = np.array(sp_new, dtype=np.single)
        self.size = np.array(self.HU.shape)

    def resampleCTfromReferenceDose(self, ref_dose_path):

        ref_dose = pyd.dcmread(ref_dose_path, force=True)
        slice_thickness = float(ref_dose.GridFrameOffsetVector[1]) - float(ref_dose.GridFrameOffsetVector[0])
        ref_spacing = np.array([float(ref_dose.PixelSpacing[0]), float(ref_dose.PixelSpacing[1]), slice_thickness])
        ref_origin = np.array(ref_dose.ImagePositionPatient)

        ref_dose_img = sitk.GetImageFromArray(ref_dose.pixel_array)
        ref_dose_img.SetOrigin(ref_origin)
        ref_dose_img.SetSpacing(ref_spacing)

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        rf = sitk.ResampleImageFilter()
        rf.SetReferenceImage(ref_dose_img)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)

        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)

        self.size = np.array(self.HU.shape)
        self.origin = ref_origin
        self.spacing = ref_spacing

    def applyCouchModel(self, couch_wet=8.0):
        spacing = self.spacing[0]
        n_voxels = int(50.0 / spacing)
        hu_override_value = ((couch_wet / (n_voxels * spacing)) - 1.0) * 1000.0

        self.HU[:, -n_voxels:, :] = hu_override_value

    def writeDoseDCM(self, dose_path, ref_dose_path, dose_type="PHYSICAL", individual_beams=False, 
                     rtplan_sop_uid=None):
        """
        Write dose distribution as DICOM RTDOSE file, cloning template grid/tags.
        
        CRITICAL: self.dose must already be resampled to match template grid shape.
        
        Parameters
        ----------
        dose_path : str
            Output DICOM file path (must end in .dcm)
        ref_dose_path : str
            Path to template RTDOSE (for grid geometry and DICOM tags)
        dose_type : str
            "PHYSICAL" (default) or "EFFECTIVE" (applies RBE=1.1)
        individual_beams : bool
            If True, save individual beam doses (uses self.beam_doses)
        rtplan_sop_uid : str, optional
            SOPInstanceUID of RTPLAN to reference. If None, keeps template reference.
            
        Raises
        ------
        ValueError
            If dose shape doesn't match template (must resample first)
        """
        if not dose_path.endswith(".dcm"):
            raise ValueError("dose_path must have .dcm extension")

        if dose_type == "EFFECTIVE":
            RBE = 1.1
        elif dose_type == "PHYSICAL":
            RBE = 1.0
        else:
            raise ValueError(f"Unknown dose_type: {dose_type}. Use 'PHYSICAL' or 'EFFECTIVE'.")

        # Read template to get expected shape
        ref_dose = pyd.dcmread(ref_dose_path, force=True)
        expected_shape = (
            int(ref_dose.NumberOfFrames) if hasattr(ref_dose, 'NumberOfFrames') else 1,
            int(ref_dose.Rows),
            int(ref_dose.Columns)
        )

        if individual_beams:
            for i, beam_dose in enumerate(self.beam_doses):
                # Validate shape
                if beam_dose.shape != expected_shape:
                    raise ValueError(
                        f"Beam {i+1} dose shape {beam_dose.shape} != template {expected_shape}. "
                        "Must resample dose to template grid before calling writeDoseDCM."
                    )
                
                # Clone template
                ds = pyd.dcmread(ref_dose_path, force=True)
                
                # Calculate proper scaling
                dose_gy = beam_dose * RBE
                max_dose = np.max(dose_gy)
                scaling = max(max_dose / 65535.0, 1e-8)  # Avoid divide by zero
                
                # Convert to uint16
                pixels = np.clip(np.rint(dose_gy / scaling), 0, 65535).astype(np.uint16)
                
                # Update DICOM tags
                ds.PixelData = pixels.tobytes()
                ds.DoseGridScaling = float(scaling)
                ds.DoseSummationType = "BEAM"
                ds.DoseType = dose_type
                
                # Update identifiers
                ds.SOPInstanceUID = pyd.uid.generate_uid()
                ds.SeriesInstanceUID = pyd.uid.generate_uid()
                if hasattr(ds, 'SeriesDescription'):
                    ds.SeriesDescription = ds.SeriesDescription + f"_DoseCUDA_Beam{i+1}"
                else:
                    ds.SeriesDescription = f"DoseCUDA_Beam{i+1}"
                
                # Update RTPLAN reference if provided
                if rtplan_sop_uid and hasattr(ds, 'ReferencedRTPlanSequence'):
                    if len(ds.ReferencedRTPlanSequence) > 0:
                        ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = rtplan_sop_uid
                
                # Save
                output_path = dose_path.replace(".dcm", f"_beam{i+1:02d}.dcm")
                ds.save_as(output_path)
                print(f"  Saved beam {i+1} RTDOSE: {output_path} (scaling={scaling:.6f}, max={max_dose:.2f} Gy)")
                
        else:
            # Validate shape
            if self.dose.shape != expected_shape:
                raise ValueError(
                    f"Dose shape {self.dose.shape} != template {expected_shape}. "
                    "Must resample dose to template grid before calling writeDoseDCM.\n"
                    f"Use: from DoseCUDA.grid_utils import resample_dose_linear, GridInfo"
                )
            
            # Clone template
            ds = pyd.dcmread(ref_dose_path, force=True)
            
            # Calculate proper scaling
            dose_gy = self.dose * RBE
            max_dose = np.max(dose_gy)
            scaling = max(max_dose / 65535.0, 1e-8)  # Avoid divide by zero
            
            # Convert to uint16
            pixels = np.clip(np.rint(dose_gy / scaling), 0, 65535).astype(np.uint16)
            
            # Update DICOM tags
            ds.PixelData = pixels.tobytes()
            ds.DoseGridScaling = float(scaling)
            ds.DoseSummationType = "PLAN"
            ds.DoseType = dose_type
            
            # Update identifiers
            ds.SOPInstanceUID = pyd.uid.generate_uid()
            ds.SeriesInstanceUID = pyd.uid.generate_uid()
            if hasattr(ds, 'SeriesDescription'):
                ds.SeriesDescription = ds.SeriesDescription + "_DoseCUDA"
            else:
                ds.SeriesDescription = "DoseCUDA"
            
            # Update RTPLAN reference if provided
            if rtplan_sop_uid:
                if hasattr(ds, 'ReferencedRTPlanSequence') and len(ds.ReferencedRTPlanSequence) > 0:
                    ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = rtplan_sop_uid
                else:
                    # Create reference if doesn't exist
                    import warnings
                    warnings.warn(
                        "Template RTDOSE has no ReferencedRTPlanSequence. "
                        "Cannot set RTPLAN reference."
                    )
            
            # Save
            ds.save_as(dose_path)
            print(f"  ✓ Saved RTDOSE: {dose_path}")
            print(f"    Shape: {self.dose.shape}, Scaling: {scaling:.6f}, Max: {max_dose:.2f} Gy")

    def writeDoseNRRD(self, dose_path, individual_beams=False, dose_type="EFFECTIVE"):

        if not dose_path.endswith(".nrrd"):
            raise Exception("Dose path must have .nrrd extension")
        
        if dose_type == "EFFECTIVE":
            RBE = 1.1
        elif dose_type == "PHYSICAL":
            RBE = 1.0
        else:
            raise Exception("Unknown dose type: %s" % dose_type)

        fw = sitk.ImageFileWriter()
        dose_img = sitk.GetImageFromArray(np.array(self.dose * RBE, dtype=np.single))
        # SimpleITK requires native Python tuple/list of floats, not numpy arrays
        dose_img.SetOrigin(tuple(float(x) for x in self.origin))
        dose_img.SetSpacing(tuple(float(x) for x in self.spacing))

        if individual_beams:
            for i, beam_dose in enumerate(self.beam_doses):
                dose_img = sitk.GetImageFromArray(np.array(beam_dose * RBE, dtype=np.single))
                dose_img.SetOrigin(tuple(float(x) for x in self.origin))
                dose_img.SetSpacing(tuple(float(x) for x in self.spacing))
                fw.SetFileName(dose_path.replace(".nrrd", "_beam%02i.nrrd" % (i+1)))
                fw.Execute(dose_img)
        else:
            fw.SetFileName(dose_path)
            fw.Execute(dose_img)

    def writeCTNRRD(self, ct_path):

        if not ct_path.endswith(".nrrd"):
            raise Exception("CT path must have .nrrd extension")

        fw = sitk.ImageFileWriter()
        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        fw.SetFileName(ct_path)
        fw.Execute(HU_img)

    def writeCTNIFTI(self, ct_path):
        if not ct_path.endswith(".nii.gz"):
            raise Exception("CT path must have .nii.gz extension")

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        fw = sitk.ImageFileWriter()
        fw.SetFileName(ct_path)
        fw.Execute(HU_img)

    def createCubePhantom(self, size=[138, 138, 138], spacing=3.0):
        self.origin = np.array([-size[0] * spacing / 2.0, -size[1] * spacing / 2.0, -size[2] * spacing / 2.0])
        self.spacing = np.array([spacing, spacing, spacing])
        self.size = np.array(size)
        edge = round(10.0 / spacing)
        self.HU = np.ones(size, dtype=np.single) * -1000.0
        self.HU[edge:-edge, edge:-edge, edge:-edge] = 0.0


class Beam:

    def __init__(self):
        self.iso = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.gantry_angle = 0.0
        self.collimator_angle = 0.0
        self.couch_angle = 0.0
        self.BeamName = None
        self.BeamDescription = None


class Plan:

    def __init__(self):
        self.n_beams = 0
        self.n_fractions = 1
        self.beam_list = []
        self.RTPlanLabel = None
        self.Prescriptions = []

    def addPrescription(self, TargetPrescriptionDose, ROIName, TargetUnderdoseVolumeFraction):
        rx = Prescription()
        rx.TargetPrescriptionDose = TargetPrescriptionDose
        rx.ROIName = ROIName
        rx.TargetUnderdoseVolumeFraction = TargetUnderdoseVolumeFraction
        self.Prescriptions.append(rx)

    def addBeam(self, beam):
        if not beam.BeamName:
            beam.BeamName = f'PBS_Beam{self.n_beams + 1}'
        if not beam.BeamDescription:
            beam.BeamDescription = f'PBS_Beam {self.n_beams + 1}'
        self.beam_list.append(beam)
        self.n_beams += 1
