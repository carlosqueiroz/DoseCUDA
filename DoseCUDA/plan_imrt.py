from .plan import Plan, Beam, DoseGrid, VolumeObject
import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pydicom as pyd
import pkg_resources
try:
    import dose_kernels
except ModuleNotFoundError:
    from DoseCUDA import dose_kernels
from dataclasses import dataclass

@dataclass
class IMRTPhotonEnergy:

    def __init__(self, dicom_energy_label):
        self.dicom_energy_label = dicom_energy_label
        self.output_factor_equivalent_squares = None
        self.output_factor_values = None
        self.mu_calibration = None
        self.primary_source_distance = None
        self.scatter_source_distance = None
        self.mlc_distance = None
        self.scatter_source_weight = None
        self.electron_attenuation = None
        self.primary_source_size = None
        self.scatter_source_size = None
        self.profile_radius = None
        self.profile_intensities = None
        self.profile_softening = None
        self.spectrum_attenuation_coefficients = None
        self.spectrum_primary_weights = None
        self.spectrum_scatter_weights = None
        self.electron_source_weight = None
        self.has_xjaws = None
        self.has_yjaws = None
        self.electron_fitted_dmax = None
        self.jaw_transmission = None
        self.mlc_transmission = None
        self.mlc_index = None
        self.mlc_widths = None
        self.mlc_offsets = None
        self.n_mlc_pairs = None
        self.kernel = None
        
        # FASE 2: Kernel dependente de profundidade
        self.kernel_depths = None  # [0, 5, 10, 15, 20, 25, 30] cm
        self.kernel_params = None  # [n_depths x 24] para 6 ângulos x 4 params
        self.use_depth_dependent_kernel = False

    def validate_parameters(self):
        """Validate that all required parameters are set"""
        required_params = [
            'output_factor_equivalent_squares', 'output_factor_values', 'mu_calibration',
            'primary_source_distance', 'scatter_source_distance', 'mlc_distance',
            'scatter_source_weight', 'electron_attenuation', 'primary_source_size',
            'scatter_source_size', 'profile_radius', 'profile_intensities',
            'profile_softening', 'spectrum_attenuation_coefficients', 'spectrum_primary_weights',
            'spectrum_scatter_weights', 'electron_source_weight', 'has_xjaws', 'has_yjaws',
            'electron_fitted_dmax', 'jaw_transmission', 'mlc_transmission'
        ]
        
        for param in required_params:
            if getattr(self, param) is None:
                raise Exception(f"{param} not set in beam model")
        
    def outputFactor(self, cp):
        """
        Calcula o output factor baseado no campo equivalente.
        
        Considera interseção de jaws e MLC para geometria real do campo.
        Valida área e perímetro para evitar divisões por zero ou valores inválidos.
        """
        import warnings
        
        # ========== ETAPA 1: Boundaries do MLC ==========
        min_y_mlc = 10000.0
        max_y_mlc = -10000.0
        min_x_mlc = 10000.0
        max_x_mlc = -10000.0
        area_mlc = 0.0
        
        for i in range(cp.mlc.shape[0]):
            x1 = cp.mlc[i, 0]
            x2 = cp.mlc[i, 1]
            y_offset = cp.mlc[i, 2]
            y_width = cp.mlc[i, 3]

            # Acumular área do MLC (ignora gaps <3mm como antes)
            if (x2 - x1) > 3.0:
                area_mlc += (x2 - x1) * y_width
                
                # Atualizar boundaries X do MLC
                min_x_mlc = min(min_x_mlc, x1)
                max_x_mlc = max(max_x_mlc, x2)
                
                # Atualizar boundaries Y do MLC
                min_y_mlc = min(min_y_mlc, y_offset - y_width / 2.0)
                max_y_mlc = max(max_y_mlc, y_offset + y_width / 2.0)
        
        # ========== ETAPA 2: Interseção com JAWS ==========
        # Se jaws estão definidas, aplicar interseção
        if hasattr(cp, 'xjaws') and hasattr(cp, 'yjaws'):
            if cp.xjaws is not None and cp.yjaws is not None:
                # Interseção: limitar MLC pelas jaws
                min_x = max(min_x_mlc, cp.xjaws[0])
                max_x = min(max_x_mlc, cp.xjaws[1])
                min_y = max(min_y_mlc, cp.yjaws[0])
                max_y = min(max_y_mlc, cp.yjaws[1])
                
                # Se jaws cortam completamente o MLC, campo efetivo pode ser menor
                if max_x > min_x and max_y > min_y:
                    # Recomputar área efetiva (aproximação retangular)
                    area = (max_x - min_x) * (max_y - min_y)
                    # Limitar pela área do MLC (não pode ser maior)
                    area = min(area, area_mlc)
                else:
                    # Jaws bloqueiam completamente
                    area = 0.0
            else:
                # Jaws None: usar MLC puro
                min_x = min_x_mlc
                max_x = max_x_mlc
                min_y = min_y_mlc
                max_y = max_y_mlc
                area = area_mlc
        else:
            # Sem jaws: usar MLC puro (backward compatibility)
            min_x = min_x_mlc
            max_x = max_x_mlc
            min_y = min_y_mlc
            max_y = max_y_mlc
            area = area_mlc

        # ========== ETAPA 3: Validação e Equivalent Square ==========
        eps = 1e-6
        
        if area < eps:
            warnings.warn(
                "Output Factor: Área do campo muito pequena ou zero. "
                "Retornando output factor mínimo (0.0). "
                "Verifique se o segmento tem abertura válida (MLC ou jaws podem estar fechados)."
            )
            return 0.0
        
        perimeter = 2 * (max_y - min_y) + 2 * (max_x - min_x)
        
        if perimeter < eps:
            warnings.warn(
                "Output Factor: Perímetro do campo muito pequeno ou zero. "
                "Retornando output factor mínimo (0.0). "
                "Verifique se o segmento tem abertura válida no MLC."
            )
            return 0.0

        equivalent_square = 4 * area / perimeter

        output_factor = np.interp(equivalent_square, self.output_factor_equivalent_squares, self.output_factor_values)

        return output_factor
       

class IMRTControlPoint:

    def __init__(self, iso, mu, mlc, ga, ca, ta, xjaws, yjaws):
        self.iso = iso
        self.mu = mu
        self.mlc = mlc
        self.ga = ga
        self.ca = ca
        self.ta = ta
        self.xjaws = xjaws
        self.yjaws = yjaws


class IMRTDoseGrid(DoseGrid):
        
    def __init__(self):
        super().__init__()
        self.Density = []

    def DensityFromHU(self, machine_name):
                
        density_table_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "photons", machine_name, "HU_Density.csv"))
        df_density = pd.read_csv(density_table_path)

        hu_curve = df_density["HU"].to_numpy()
        density_curve = df_density["Density"].to_numpy()
        
        density = np.array(np.interp(self.HU, hu_curve, density_curve), dtype=np.single)
        
        return density

    def computeIMRTPlan(self, plan, gpu_id=0):
            
        self.beam_doses = []
        self.dose = np.zeros(self.size, dtype=np.single)
        self.Density = self.DensityFromHU(plan.machine_name)

        if self.spacing[0] != self.spacing[1] or self.spacing[0] != self.spacing[2]:
            raise Exception("Spacing must be isotropic for IMPT dose calculation - consider resampling CT")
        
        density_object = VolumeObject()
        density_object.voxel_data = np.array(self.Density, dtype=np.single)
        density_object.origin = np.array(self.origin, dtype=np.single)
        density_object.spacing = np.array(self.spacing, dtype=np.single)

        for beam in plan.beam_list:
            beam_dose = np.zeros(self.Density.shape, dtype=np.single)
            
            try:
                # Garantir que ambos são strings para comparação
                available_energies = [str(e) for e in plan.dicom_energy_label]
                beam_energy_str = str(beam.dicom_energy_label) if beam.dicom_energy_label is not None else "None"
                model_index = available_energies.index(beam_energy_str)
            except ValueError:
                print(f"Beam model not found for beam energy '{beam.dicom_energy_label}'.")
                print(f"Available energies in models: {', '.join(available_energies)}")
                sys.exit(1)
            beam_model = plan.beam_models[model_index]

            for cp in beam.cp_list:
                output_factor = beam_model.outputFactor(cp)
                cp_dose = dose_kernels.photon_dose_cuda(beam_model, density_object, cp, gpu_id)
                beam_dose += cp_dose * output_factor
                
            self.beam_doses.append(beam_dose)
            self.dose += beam_dose

        self.dose *= plan.n_fractions


class IMRTBeam(Beam):

    def __init__(self):
        super().__init__()
        self.cp_list = []
        self.n_cps = 0
        self.dicom_energy_label = None

    def addControlPoint(self, cp):
        self.cp_list.append(cp)
        self.n_cps += 1


class IMRTPlan(Plan):

    def __init__(self, machine_name = "VarianTrueBeamHF"):
        
        super().__init__()

        self.machine_name = machine_name

        energy_list = pd.read_csv(pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "photons", machine_name, "energy_labels.csv")))
        self.dicom_energy_label = energy_list["dicom_energy_label"]
        self.folder_energy_label = energy_list["folder_energy_label"]

        self.beam_models = []
        for d, f in zip(self.dicom_energy_label, self.folder_energy_label):
            beam_model = IMRTPhotonEnergy(d)
            self._load_beam_model_parameters(beam_model, machine_name, f)
            self.beam_models.append(beam_model)

    def _load_beam_model_parameters(self, beam_model, machine_name, folder_energy_label, mlc_geometry_from_dicom=None):
        """Load beam model parameters from lookup tables
        
        Args:
            mlc_geometry_from_dicom: tuple (n_pairs, offsets, widths) from DICOM or None to use CSV
        """
        path_to_model = os.path.join("lookuptables", "photons", machine_name)
        
        # Load MLC geometry: priorizar DICOM, usar CSV como fallback
        if mlc_geometry_from_dicom is not None:
            n_pairs, offsets, widths = mlc_geometry_from_dicom
            beam_model.mlc_index = np.arange(n_pairs, dtype=np.int32)
            beam_model.mlc_offsets = offsets
            beam_model.mlc_widths = widths
            beam_model.n_mlc_pairs = n_pairs
            print(f"✓ MLC geometry carregada do DICOM: {n_pairs} pares (LeafPositionBoundaries)")
        else:
            # Fallback: carregar do CSV
            mlc_geometry_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, "mlc_geometry.csv"))
            mlc_geometry = pd.read_csv(mlc_geometry_path)
            
            beam_model.mlc_index = mlc_geometry["mlc_pair_index"].to_numpy()
            beam_model.mlc_widths = mlc_geometry["width"].to_numpy()
            beam_model.mlc_offsets = mlc_geometry["center_offset"].to_numpy()
            beam_model.n_mlc_pairs = len(beam_model.mlc_index)
            print(f"⚠ MLC geometry carregada do CSV (fallback): {beam_model.n_mlc_pairs} pares")

        # Load kernel
        kernel_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, folder_energy_label, "kernel.csv"))
        kernel = pd.read_csv(kernel_path)
        beam_model.kernel = np.array(kernel.to_numpy(), dtype=np.single)
        
        # FASE 2: Tentar carregar kernel z-dependente (se existir)
        kernel_zdep_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, folder_energy_label, "kernel_depth_dependent.csv"))
        if os.path.exists(kernel_zdep_path):
            kernel_zdep = pd.read_csv(kernel_zdep_path)
            # Formato esperado: depth, angle_idx, Am, am, Bm, bm
            # Reorganizar para [n_depths x 24]
            depths = kernel_zdep["depth"].unique()
            beam_model.kernel_depths = np.array(depths, dtype=np.single)
            beam_model.n_kernel_depths = len(depths)
            
            # Criar array [n_depths x 6_angles x 4_params]
            n_angles = 6
            n_params = 4
            kernel_params = np.zeros((len(depths), n_angles * n_params), dtype=np.single)
            
            for i, depth in enumerate(depths):
                depth_data = kernel_zdep[kernel_zdep["depth"] == depth]
                for j in range(n_angles):
                    angle_data = depth_data[depth_data["angle_idx"] == j]
                    if len(angle_data) > 0:
                        kernel_params[i, j*4 + 0] = angle_data["Am"].values[0]
                        kernel_params[i, j*4 + 1] = angle_data["am"].values[0]
                        kernel_params[i, j*4 + 2] = angle_data["Bm"].values[0]
                        kernel_params[i, j*4 + 3] = angle_data["bm"].values[0]
            
            beam_model.kernel_params = kernel_params.flatten()  # Linearizar para passar ao CUDA
            beam_model.use_depth_dependent_kernel = True
            print(f"✓ Loaded depth-dependent kernel for {folder_energy_label} ({len(depths)} depths)")
        else:
            beam_model.use_depth_dependent_kernel = False

        # Load machine geometry
        machine_geometry_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, "machine_geometry.csv"))
        self._load_machine_geometry(beam_model, machine_geometry_path)

        # Load beam parameters
        beam_parameter_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, folder_energy_label, "beam_parameters.csv"))
        self._load_beam_parameters(beam_model, beam_parameter_path)

        # Validate all parameters are loaded
        beam_model.validate_parameters()

    def _load_machine_geometry(self, beam_model, machine_geometry_path):
        """Load machine geometry parameters"""
        for line in open(machine_geometry_path):
            if line.startswith('primary_source_distance'):
                beam_model.primary_source_distance = float(line.split(',')[1])
            elif line.startswith('scatter_source_distance'):
                beam_model.scatter_source_distance = float(line.split(',')[1])
            elif line.startswith('mlc_distance'):
                beam_model.mlc_distance = float(line.split(',')[1])
            elif line.startswith('has_xjaws'):
                beam_model.has_xjaws = int(line.split(',')[1].strip()) != 0
            elif line.startswith('has_yjaws'):
                beam_model.has_yjaws = int(line.split(',')[1].strip()) != 0

    def _load_beam_parameters(self, beam_model, beam_parameter_path):
        """Load beam-specific parameters"""
        for line in open(beam_parameter_path):
            if line.startswith('output_factor_equivalent_squares'):
                beam_model.output_factor_equivalent_squares = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('output_factor_values'):
                beam_model.output_factor_values = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('mu_calibration'):
                beam_model.mu_calibration = float(line.split(',')[1])
            elif line.startswith('scatter_source_weight'):
                beam_model.scatter_source_weight = float(line.split(',')[1])
            elif line.startswith('electron_attenuation'):
                beam_model.electron_attenuation = float(line.split(',')[1])
            elif line.startswith('primary_source_size'):
                beam_model.primary_source_size = float(line.split(',')[1])
            elif line.startswith('scatter_source_size'):
                beam_model.scatter_source_size = float(line.split(',')[1])
            elif line.startswith('profile_radius'):
                beam_model.profile_radius = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('profile_intensities'):
                beam_model.profile_intensities = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('profile_softening'):
                beam_model.profile_softening = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('spectrum_attenuation_coefficients'):
                beam_model.spectrum_attenuation_coefficients = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('spectrum_primary_weights'):
                beam_model.spectrum_primary_weights = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('spectrum_scatter_weights'):
                beam_model.spectrum_scatter_weights = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('electron_source_weight'):
                beam_model.electron_source_weight = float(line.split(',')[1])
            elif line.startswith('electron_fitted_dmax'):
                beam_model.electron_fitted_dmax = float(line.split(',')[1])
            elif line.startswith('jaw_transmission'):
                beam_model.jaw_transmission = float(line.split(',')[1])
            elif line.startswith('mlc_transmission'):
                beam_model.mlc_transmission = float(line.split(',')[1])

    def _normalize_beam_energy(self, nominal_energy):
        """
        Normaliza a energia do feixe para o formato esperado pelos modelos.
        
        Converte valores numéricos (6.0, 10.0) para strings ("6", "10").
        Remove decimais desnecessários.
        
        Args:
            nominal_energy: valor de NominalBeamEnergy do DICOM (pode ser float ou string)
            
        Returns:
            String normalizada da energia
        """
        if nominal_energy is None:
            return None
            
        # Converter para string se for numérico
        energy_str = str(nominal_energy)
        
        # Remover '.0' no final se presente (ex: "6.0" -> "6")
        if '.' in energy_str and energy_str.endswith('.0'):
            energy_str = energy_str[:-2]
        
        return energy_str
    
    def _validate_beam_modifiers(self, beam, beam_number):
        """
        Detecta modificadores de feixe não suportados (wedges, compensators, etc.).
        Lança exceção se encontrar modificadores que comprometem o cálculo secundário.
        
        Args:
            beam: objeto do beam DICOM
            beam_number: número do beam para mensagem de erro
        """
        import warnings
        
        unsupported_modifiers = []
        
        # Verificar wedges
        if hasattr(beam, 'WedgeSequence') and len(beam.WedgeSequence) > 0:
            unsupported_modifiers.append('Physical/Dynamic Wedge')
        
        # Verificar compensadores
        if hasattr(beam, 'CompensatorSequence') and len(beam.CompensatorSequence) > 0:
            unsupported_modifiers.append('Compensator')
        
        # Verificar blocos
        if hasattr(beam, 'BlockSequence') and len(beam.BlockSequence) > 0:
            unsupported_modifiers.append('Block')
        
        # Verificar aplicadores
        if hasattr(beam, 'ApplicatorSequence') and len(beam.ApplicatorSequence) > 0:
            unsupported_modifiers.append('Applicator')
        
        # Verificar bolus (pode estar em diferentes lugares)
        for cp in getattr(beam, 'ControlPointSequence', []):
            if hasattr(cp, 'ReferencedBolusSequence') and len(cp.ReferencedBolusSequence) > 0:
                unsupported_modifiers.append('Bolus')
                break
        
        if unsupported_modifiers:
            modifier_list = ', '.join(unsupported_modifiers)
            raise ValueError(
                f"Beam {beam_number}: Modificadores não suportados detectados ({modifier_list}). "
                f"Este cálculo secundário não pode validar planos com estes dispositivos. "
                f"Por favor, use apenas feixes com MLC/Jaws."
            )

    def readPlanDicom(self, plan_path):
        """
        Read RTPLAN DICOM file and construct beam/control point structures.

        Robust parsing for clinical secondary dose calculation:
        - MU per beam from BeamMeterset via FractionGroupSequence
        - MU per segment = delta_CMW × scalingFactor
        - Jaws/MLC parsed by RTBeamLimitingDeviceType (not fixed indices)
        - Fallback "last-known" for omitted CP data
        - Validation of unsupported modifiers (wedges, compensators, etc.)
        - Energy normalization for model matching
        """
        import warnings

        ds = pyd.dcmread(plan_path, force=True)

        # Validate FractionGroupSequence
        if not hasattr(ds, 'FractionGroupSequence') or len(ds.FractionGroupSequence) == 0:
            raise ValueError("RTPLAN inválido: FractionGroupSequence ausente ou vazia")

        fg = ds.FractionGroupSequence[0]
        self.n_fractions = float(fg.NumberOfFractionsPlanned)

        # Build BeamNumber -> BeamMeterset mapping
        beam_meterset_map = {}
        if hasattr(fg, 'ReferencedBeamSequence'):
            for ref_beam in fg.ReferencedBeamSequence:
                beam_num = int(ref_beam.ReferencedBeamNumber)
                beam_meterset_map[beam_num] = float(ref_beam.BeamMeterset)
        else:
            raise ValueError("RTPLAN inválido: ReferencedBeamSequence ausente em FractionGroupSequence")

        self.beam_list = []
        self.n_beams = 0
        
        # P1.5: Variável para armazenar MLC geometry do DICOM (extraída do primeiro beam)
        mlc_geometry_from_dicom = None

        for beam in ds.BeamSequence:
            # Skip non-treatment beams (e.g., setup fields)
            if not hasattr(beam, 'TreatmentDeliveryType') or beam.TreatmentDeliveryType != "TREATMENT":
                continue

            # Get beam number and corresponding BeamMeterset
            beam_number = int(beam.BeamNumber) if hasattr(beam, 'BeamNumber') else None
            if beam_number is None or beam_number not in beam_meterset_map:
                warnings.warn(f"BeamMeterset não encontrado para BeamNumber={beam_number}. Pulando feixe.")
                continue

            # Validar modificadores não suportados (wedges, compensators, etc.)
            self._validate_beam_modifiers(beam, beam_number)

            # P1.5: Extrair MLC geometry do DICOM (se disponível)
            # Isso é feito uma vez no primeiro beam e reutilizado
            if self.n_beams == 0 and len(self.beam_models) == 0:
                mlc_geometry_from_dicom = self._extract_mlc_geometry_from_dicom(beam)
                if mlc_geometry_from_dicom is not None:
                    # Recarregar beam models com MLC geometry do DICOM
                    self.beam_models = []
                    for energy_label in self.dicom_energy_label:
                        beam_model = IMRTPhotonEnergy(self.machine_name, energy_label)
                        folder_energy_label = self._normalize_energy_for_folder(energy_label)
                        self._load_beam_model_parameters(beam_model, self.machine_name, folder_energy_label, mlc_geometry_from_dicom)
                        self.beam_models.append(beam_model)

            beam_meterset = beam_meterset_map[beam_number]

            # Get FinalCumulativeMetersetWeight with consistency validation
            final_cmw_from_beam = None
            final_cmw_from_cp = None
            
            if hasattr(beam, 'FinalCumulativeMetersetWeight'):
                final_cmw_from_beam = float(beam.FinalCumulativeMetersetWeight)
            
            if hasattr(beam, 'ControlPointSequence') and len(beam.ControlPointSequence) > 0:
                last_cp = beam.ControlPointSequence[-1]
                if hasattr(last_cp, 'CumulativeMetersetWeight'):
                    final_cmw_from_cp = float(last_cp.CumulativeMetersetWeight)
            
            # Determinar final_cmw com validação de consistência
            if final_cmw_from_beam is not None and final_cmw_from_cp is not None:
                # Ambos disponíveis: validar consistência
                cmw_diff = abs(final_cmw_from_beam - final_cmw_from_cp)
                cmw_relative_diff = cmw_diff / max(final_cmw_from_beam, 1e-9)
                
                if cmw_relative_diff > 0.01:  # Tolerância de 1%
                    warnings.warn(
                        f"Beam {beam_number}: Inconsistência entre FinalCumulativeMetersetWeight "
                        f"({final_cmw_from_beam:.6f}) e último CP CMW ({final_cmw_from_cp:.6f}). "
                        f"Diferença relativa: {cmw_relative_diff*100:.2f}%. Usando valor do beam."
                    )
                final_cmw = final_cmw_from_beam
            elif final_cmw_from_beam is not None:
                final_cmw = final_cmw_from_beam
            elif final_cmw_from_cp is not None:
                final_cmw = final_cmw_from_cp
                warnings.warn(f"Beam {beam_number}: FinalCumulativeMetersetWeight ausente, usando último CMW={final_cmw}")
            else:
                final_cmw = 1.0
                warnings.warn(f"Beam {beam_number}: Nenhum CMW disponível, assumindo finalCmw=1.0")
            
            # Normalizar final_cmw se muito próximo de 1.0 (arredondamento)
            if abs(final_cmw - 1.0) < 1e-6:
                final_cmw = 1.0

            if final_cmw <= 0:
                raise ValueError(f"Beam {beam_number}: FinalCumulativeMetersetWeight inválido ({final_cmw})")

            scaling_factor = beam_meterset / final_cmw

            # Validate control point sequence
            cps = beam.ControlPointSequence
            if len(cps) < 2:
                warnings.warn(f"Beam {beam_number}: menos de 2 control points ({len(cps)}). Pulando feixe.")
                continue

            imrt_beam = IMRTBeam()

            # Initialize last-known values from first CP
            first_cp = cps[0]

            # Energy - normalizar para comparação com modelos
            if hasattr(first_cp, 'NominalBeamEnergy'):
                raw_energy = first_cp.NominalBeamEnergy
                imrt_beam.dicom_energy_label = self._normalize_beam_energy(raw_energy)

                # Avisar se energia não está nos modelos disponíveis
                if hasattr(self, 'dicom_energy_label') and self.dicom_energy_label is not None:
                    available_energies = [str(e) for e in self.dicom_energy_label]
                    if imrt_beam.dicom_energy_label not in available_energies:
                        warnings.warn(
                            f"Beam {beam_number}: Energia {imrt_beam.dicom_energy_label} "
                            f"(do DICOM: {raw_energy}) não encontrada nos modelos disponíveis. "
                            f"Energias disponíveis: {', '.join(available_energies)}. "
                            f"O cálculo falhará se esta energia for usada."
                        )
            else:
                warnings.warn(f"Beam {beam_number}: NominalBeamEnergy ausente no primeiro CP")
                imrt_beam.dicom_energy_label = None

            # Isocenter
            if hasattr(first_cp, 'IsocenterPosition'):
                last_iso = np.array(first_cp.IsocenterPosition, dtype=np.single)
            else:
                last_iso = np.array([0.0, 0.0, 0.0], dtype=np.single)
                warnings.warn(f"Beam {beam_number}: IsocenterPosition ausente, usando [0,0,0]")
            imrt_beam.iso = last_iso

            # Angles
            last_ga = float(first_cp.GantryAngle) if hasattr(first_cp, 'GantryAngle') else 0.0
            last_ca = float(first_cp.BeamLimitingDeviceAngle) if hasattr(first_cp, 'BeamLimitingDeviceAngle') else 0.0
            last_ta = float(first_cp.PatientSupportAngle) if hasattr(first_cp, 'PatientSupportAngle') else 0.0

            # Jaws and MLC from first CP
            last_xjaws, last_yjaws, last_mlc_raw = self._parse_beam_limiting_devices(first_cp)

            # Process segments: segment i uses geometry from CP[i] and MU from delta to CP[i+1]
            for i in range(len(cps) - 1):
                cp = cps[i]
                cp_next = cps[i + 1]

                # Calculate segment MU
                cmw_current = float(cp.CumulativeMetersetWeight) if hasattr(cp, 'CumulativeMetersetWeight') else 0.0
                cmw_next = float(cp_next.CumulativeMetersetWeight) if hasattr(cp_next, 'CumulativeMetersetWeight') else 0.0

                delta_cmw = cmw_next - cmw_current
                if delta_cmw < -1e-9:
                    raise ValueError(f"Beam {beam_number}, CP {i}: CMW decresce ({cmw_current} -> {cmw_next})")

                seg_mu = delta_cmw * scaling_factor

                # Skip zero-MU segments (no dose delivered)
                if seg_mu <= 1e-9:
                    continue

                # Update geometry from current CP (with last-known fallback)
                if hasattr(cp, 'IsocenterPosition'):
                    last_iso = np.array(cp.IsocenterPosition, dtype=np.single)
                if hasattr(cp, 'GantryAngle'):
                    last_ga = float(cp.GantryAngle)
                if hasattr(cp, 'BeamLimitingDeviceAngle'):
                    last_ca = float(cp.BeamLimitingDeviceAngle)
                if hasattr(cp, 'PatientSupportAngle'):
                    last_ta = float(cp.PatientSupportAngle)

                # Parse jaws/MLC from current CP (with last-known fallback)
                if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
                    xjaws, yjaws, mlc_raw = self._parse_beam_limiting_devices(cp)
                    if xjaws is not None:
                        last_xjaws = xjaws
                    if yjaws is not None:
                        last_yjaws = yjaws
                    if mlc_raw is not None:
                        last_mlc_raw = mlc_raw

                # Build MLC array in DoseCUDA format
                mlc_formatted = self._format_mlc_for_dosecuda(last_mlc_raw, last_xjaws, last_yjaws)

                # Create control point
                control_point = IMRTControlPoint(
                    last_iso, seg_mu, mlc_formatted,
                    last_ga, last_ca, last_ta,
                    last_xjaws, last_yjaws
                )
                imrt_beam.addControlPoint(control_point)

            if imrt_beam.n_cps > 0:
                self.addBeam(imrt_beam)
            else:
                warnings.warn(f"Beam {beam_number}: nenhum segmento com MU > 0. Pulando feixe.")

    def _parse_beam_limiting_devices(self, cp):
        """
        Parse BeamLimitingDevicePositionSequence by RTBeamLimitingDeviceType.
        Returns (xjaws, yjaws, mlc_raw) - each can be None if not present.
        Does NOT use fixed indices.
        """
        import warnings

        xjaws = None
        yjaws = None
        mlc_raw = None

        if not hasattr(cp, 'BeamLimitingDevicePositionSequence'):
            return xjaws, yjaws, mlc_raw

        for device in cp.BeamLimitingDevicePositionSequence:
            device_type = device.RTBeamLimitingDeviceType
            positions = np.array(device.LeafJawPositions, dtype=np.single)

            if device_type in ('X', 'ASYMX'):
                xjaws = positions
            elif device_type in ('Y', 'ASYMY'):
                yjaws = positions
            elif device_type in ('MLCX', 'MLC'):
                mlc_raw = positions
            elif device_type == 'MLCY':
                warnings.warn("MLCY encontrado mas não suportado pelo modelo atual. Ignorando.")
            # Other device types are silently ignored

        return xjaws, yjaws, mlc_raw

    def _format_mlc_for_dosecuda(self, mlc_raw, xjaws, yjaws):
        """
        Format MLC data for DoseCUDA kernel.

        If mlc_raw is provided: reshape to (n_pairs, 4) with [x1, x2, offset, width]
        If mlc_raw is None: create synthetic MLC from jaw positions

        Returns: np.array shape (n_mlc_pairs, 4), dtype=np.single
        """
        import warnings
        
        n_mlc_pairs = self.beam_models[0].n_mlc_pairs
        mlc_offsets = self.beam_models[0].mlc_offsets
        mlc_widths = self.beam_models[0].mlc_widths

        if mlc_raw is not None:
            # MLC data from DICOM: validar tamanho antes do reshape
            expected_size = 2 * n_mlc_pairs
            actual_size = len(mlc_raw)
            
            if actual_size != expected_size:
                raise ValueError(
                    f"Tamanho inválido do vetor MLC: esperado {expected_size} "
                    f"(2 × {n_mlc_pairs} pares), mas recebeu {actual_size} valores. "
                    f"Verifique se a máquina configurada ({self.machine_name}) "
                    f"corresponde ao plano DICOM."
                )
            
            # Reshape (2*n_pairs,) -> (2, n_pairs)
            mlc = np.reshape(mlc_raw, (2, n_mlc_pairs))
            
            # Garantir ordenação correta: x1 <= x2 (esquerda <= direita)
            # Detectar inversão e corrigir
            x1 = mlc[0, :]
            x2 = mlc[1, :]
            
            # Contar pares invertidos (x2 < x1)
            inverted_pairs = np.sum(x2 < x1)
            if inverted_pairs > 0:
                inversion_fraction = inverted_pairs / n_mlc_pairs
                
                if inversion_fraction > 0.5:
                    # Maioria invertida: provavelmente os bancos estão trocados
                    warnings.warn(
                        f"MLC: {inverted_pairs}/{n_mlc_pairs} pares com x2 < x1. "
                        f"Os bancos parecem estar invertidos. Corrigindo automaticamente."
                    )
                    mlc[0, :], mlc[1, :] = np.minimum(x1, x2), np.maximum(x1, x2)
                elif inversion_fraction > 0.1:
                    # Alguns invertidos: problema mais sério
                    raise ValueError(
                        f"MLC: {inverted_pairs}/{n_mlc_pairs} pares com x2 < x1. "
                        f"Isso pode indicar erro no export DICOM ou dados corrompidos. "
                        f"Verifique o plano original."
                    )
                else:
                    # Poucos invertidos: corrigir silenciosamente
                    mlc[0, :], mlc[1, :] = np.minimum(x1, x2), np.maximum(x1, x2)
            
            mlc = np.array(np.vstack((
                mlc,
                mlc_offsets.reshape(1, -1),
                mlc_widths.reshape(1, -1)
            )), dtype=np.single)
            mlc = np.transpose(mlc)  # (n_pairs, 4)
        else:
            # No MLC: create synthetic field from jaws
            warnings.warn(
                "MLC: Nenhuma posição MLCX encontrada no DICOM. "
                "Criando campo sintético a partir dos jaws. "
                "ATENÇÃO: Isto pode mascarar um erro no parsing. "
                "Verifique se o plano realmente não possui MLC ou se houve falha na leitura."
            )
            mlc = np.zeros((2, n_mlc_pairs), dtype=np.single)

            if xjaws is not None:
                mlc[0, :] = xjaws[0]  # Left jaw position
                mlc[1, :] = xjaws[1]  # Right jaw position

            # Close leaves outside Y-jaw field
            if yjaws is not None:
                for i in range(n_mlc_pairs):
                    if mlc_offsets[i] < yjaws[0] or mlc_offsets[i] > yjaws[1]:
                        mlc[0, i] = 0.0
                        mlc[1, i] = 0.0

            mlc = np.array(np.vstack((
                mlc,
                mlc_offsets.reshape(1, -1),
                mlc_widths.reshape(1, -1)
            )), dtype=np.single)
            mlc = np.transpose(mlc)  # (n_pairs, 4)

        return mlc

    def addSquareField(self, dicom_energy_label='6', dimx=10, dimy=10, mu=100, gantry_angle=0.0, collimator_angle=0.0, table_angle=0.0):
    

        imrt_beam = IMRTBeam()
        imrt_beam.dicom_energy_label = dicom_energy_label
        iso = np.array([0.0, 0.0, 0.0], dtype=np.single)

        dimx *= 10.0 # convert to mm
        dimy *= 10.0 # convert to mm

        # create a square field
        mlc = np.zeros((2, self.beam_models[0].n_mlc_pairs), dtype=np.single)
        mlc[0, :] = -dimx / 2.0
        mlc[1, :] = dimx / 2.0

        for i in range(self.beam_models[0].n_mlc_pairs):
            if(self.beam_models[0].mlc_offsets[i] < -(dimy / 2.0)) or (self.beam_models[0].mlc_offsets[i] > (dimy / 2.0)):
                mlc[0, i] = 0.0
                mlc[1, i] = 0.0

        mlc = np.array(np.vstack((mlc, self.beam_models[0].mlc_offsets.reshape(1, -1), self.beam_models[0].mlc_widths.reshape(1, -1))), dtype=np.single)
        mlc = np.transpose(mlc)

        jawx = np.array([-dimx / 2.0, dimx / 2.0], dtype=np.single)
        jawy = np.array([-dimy / 2.0, dimy / 2.0], dtype=np.single)


        cp = IMRTControlPoint(iso, mu, mlc, gantry_angle, collimator_angle, table_angle, jawx, jawy)
        imrt_beam.addControlPoint(cp)
        self.addBeam(imrt_beam)