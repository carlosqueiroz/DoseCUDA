from .plan import Plan, Beam, DoseGrid, VolumeObject
from .ct_calibration import CTCalibration, CTCalibrationManager
import sys
import os
import logging
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pydicom as pyd
import pkg_resources
import warnings
try:
    import dose_kernels
except ModuleNotFoundError:
    from DoseCUDA import dose_kernels
from dataclasses import dataclass


logger = logging.getLogger(__name__)

# MU handling tolerances
_MIN_SEG_MU_ABS = 1e-12          # Gy-equivalent MU floor (absolute)
_MIN_SEG_MU_REL = 1e-6           # Relative to BeamMeterset (per segment)

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
        self.kernel_len = 0
        self.kernel_weights = None
        self.n_kernel_depths = 0
        self.kernel_depths = None
        self.kernel_params = None

        # P1.2: Dosimetric Leaf Gap (DLG) correction [mm]
        # Typical values: 1.0-2.0 mm for Varian MLCs
        self.dlg = 0.0  # Default: no DLG correction
        # Tongue-and-groove over-blocking [mm] applied when adjacent leaves
        # differ in opening (approximate fluence shadowing). Leave 0 to disable.
        self.tg_ext = 0.0

        # FASE 2: Kernel dependente de profundidade
        self.kernel_depths = None  # [0, 5, 10, 15, 20, 25, 30] cm
        self.kernel_params = None  # [n_depths x 24] para 6 ângulos x 4 params
        self.use_depth_dependent_kernel = False
        # P2.4: suavização opcional de heterogeneidade (filtro exponencial ao longo do raio)
        self.heterogeneity_alpha = 0.0  # 0 desativa; 0.2-0.4 suaviza interfaces

    def validate_parameters(self):
        """
        Validate beam model parameters (P3.1).

        Performs comprehensive sanity checks:
        - Required parameters are set
        - Array lengths match (spectrum, profile)
        - Parameter ranges are physically valid
        - Monotonicity constraints are satisfied

        Raises:
            ValueError: If validation fails with descriptive error
        """
        import warnings

        # Required scalar parameters
        required_params = [
            'output_factor_equivalent_squares', 'output_factor_values', 'mu_calibration',
            'primary_source_distance', 'scatter_source_distance', 'mlc_distance',
            'scatter_source_weight', 'electron_attenuation', 'primary_source_size',
            'scatter_source_size', 'profile_radius', 'profile_intensities',
            'profile_softening', 'spectrum_attenuation_coefficients', 'spectrum_primary_weights',
            'spectrum_scatter_weights', 'electron_source_weight', 'has_xjaws', 'has_yjaws',
            'electron_fitted_dmax', 'jaw_transmission', 'mlc_transmission', 'heterogeneity_alpha'
        ]

        for param in required_params:
            if getattr(self, param) is None:
                raise ValueError(f"Beam model '{self.dicom_energy_label}': {param} not set")

        # Array length consistency checks
        n_spectrum = len(self.spectrum_attenuation_coefficients)
        if len(self.spectrum_primary_weights) != n_spectrum:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': spectrum_primary_weights length "
                f"({len(self.spectrum_primary_weights)}) != attenuation coefficients ({n_spectrum})"
            )
        if len(self.spectrum_scatter_weights) != n_spectrum:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': spectrum_scatter_weights length "
                f"({len(self.spectrum_scatter_weights)}) != attenuation coefficients ({n_spectrum})"
            )

        n_profile = len(self.profile_radius)
        if len(self.profile_intensities) != n_profile:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': profile_intensities length "
                f"({len(self.profile_intensities)}) != profile_radius ({n_profile})"
            )
        if len(self.profile_softening) != n_profile:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': profile_softening length "
                f"({len(self.profile_softening)}) != profile_radius ({n_profile})"
            )

        # Profile radius must be monotonically increasing
        if not np.all(np.diff(self.profile_radius) > 0):
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': profile_radius must be strictly increasing"
            )

        # Kernel shape checks
        if self.kernel is None or self.kernel_len not in (36,):
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': kernel must contain 36 floats (6 angles x 6 columns). "
                f"Got len={self.kernel_len}"
            )
        if self.kernel_weights is not None and len(self.kernel_weights) != 6:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': kernel_weights must have length 6"
            )
        if self.kernel_weights is not None:
            wsum = float(np.sum(self.kernel_weights))
            if abs(wsum - 1.0) > 0.1:
                warnings.warn(
                    f"Beam model '{self.dicom_energy_label}': kernel_weights sum to {wsum:.3f}, "
                    f"expected ~1.0 for unbiased quadrature."
                )
        if self.use_depth_dependent_kernel:
            if self.kernel_depths is None or self.kernel_params is None:
                raise ValueError(
                    f"Beam model '{self.dicom_energy_label}': depth-dependent kernel enabled but data missing"
                )
            if self.kernel_params.size != (len(self.kernel_depths) * 24):
                raise ValueError(
                    f"Beam model '{self.dicom_energy_label}': kernel_params size "
                    f"{self.kernel_params.size} != n_depths*24 ({len(self.kernel_depths)*24})"
                )
        # heterogeneity_alpha in [0,1)
        if not (0.0 <= self.heterogeneity_alpha < 1.0 + 1e-6):
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': heterogeneity_alpha "
                f"({self.heterogeneity_alpha}) must be in [0,1)"
            )

        # Transmission values in [0, 1]
        if not (0 <= self.jaw_transmission <= 1):
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': jaw_transmission ({self.jaw_transmission}) "
                f"must be in [0, 1]"
            )
        if not (0 <= self.mlc_transmission <= 1):
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': mlc_transmission ({self.mlc_transmission}) "
                f"must be in [0, 1]"
            )

        # Distances must be positive
        if self.primary_source_distance <= 0:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': primary_source_distance must be positive"
            )
        if self.scatter_source_distance <= 0:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': scatter_source_distance must be positive"
            )
        if self.mlc_distance < 0:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': mlc_distance must be non-negative"
            )

        # Tongue-and-groove extension must be non-negative
        if self.tg_ext < 0:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': tg_ext ({self.tg_ext}) "
                f"must be >= 0"
            )

        # Source sizes must be positive
        if self.primary_source_size <= 0:
            warnings.warn(
                f"Beam model '{self.dicom_energy_label}': primary_source_size ({self.primary_source_size}) "
                f"should be positive for finite source modeling"
            )
        if self.scatter_source_size <= 0:
            warnings.warn(
                f"Beam model '{self.dicom_energy_label}': scatter_source_size ({self.scatter_source_size}) "
                f"should be positive for head scatter modeling"
            )

        # Scatter source weight in [0, 1]
        if not (0 <= self.scatter_source_weight <= 1):
            warnings.warn(
                f"Beam model '{self.dicom_energy_label}': scatter_source_weight ({self.scatter_source_weight}) "
                f"outside typical range [0, 1]"
            )

        # MU calibration should be positive
        if self.mu_calibration <= 0:
            raise ValueError(
                f"Beam model '{self.dicom_energy_label}': mu_calibration must be positive"
            )

    def get_model_hash(self):
        """
        Compute a hash of key model parameters for energy uniqueness check.

        Returns:
            String hash representing the model's key parameters
        """
        import hashlib

        # Combine key arrays and scalars into a hashable representation
        key_data = []
        key_data.append(self.spectrum_attenuation_coefficients.tobytes())
        key_data.append(self.spectrum_primary_weights.tobytes())
        key_data.append(self.kernel.tobytes() if self.kernel is not None else b'')
        key_data.append(np.float32(self.tg_ext).tobytes())
        if self.kernel_weights is not None:
            key_data.append(self.kernel_weights.tobytes())
        if self.kernel_params is not None:
            key_data.append(self.kernel_params.tobytes())
        key_data.append(np.float32(self.heterogeneity_alpha).tobytes())
        key_data.append(str(self.mu_calibration).encode())

        combined = b''.join(key_data)
        return hashlib.md5(combined).hexdigest()[:16]
        
    def outputFactor(self, cp):
        """
        Calcula o output factor baseado no campo equivalente.
        
        Considera interseção de jaws e MLC por folha (geometria real).
        Usa área efetiva somando aberturas de cada par de folhas dentro das jaws e
        perímetro aproximado por 2*(largura_média + altura_total), evitando ruído
        em campos pequenos/VMAT.
        """
        import warnings
        
        # Limites de jaws (ou infinito se não presentes)
        jaw_xmin = -1e6
        jaw_xmax = 1e6
        jaw_ymin = -1e6
        jaw_ymax = 1e6
        if getattr(cp, "xjaws", None) is not None:
            jaw_xmin, jaw_xmax = float(cp.xjaws[0]), float(cp.xjaws[1])
        if getattr(cp, "yjaws", None) is not None:
            jaw_ymin, jaw_ymax = float(cp.yjaws[0]), float(cp.yjaws[1])

        area = 0.0
        min_x = 1e6
        max_x = -1e6
        min_y = 1e6
        max_y = -1e6

        # Soma da área folha a folha, já clipada pelas jaws
        for i in range(cp.mlc.shape[0]):
            x1 = cp.mlc[i, 0]
            x2 = cp.mlc[i, 1]
            y_offset = cp.mlc[i, 2]
            y_width = cp.mlc[i, 3]

            y0 = y_offset - 0.5 * y_width
            y1 = y_offset + 0.5 * y_width

            # Interseção com jaws em Y
            y_low = max(y0, jaw_ymin)
            y_high = min(y1, jaw_ymax)
            if y_high <= y_low:
                continue

            # Interseção em X
            x_left = max(x1, jaw_xmin)
            x_right = min(x2, jaw_xmax)
            if x_right <= x_left:
                continue

            dx = x_right - x_left
            dy = y_high - y_low

            area += dx * dy

            min_x = min(min_x, x_left)
            max_x = max(max_x, x_right)
            min_y = min(min_y, y_low)
            max_y = max(max_y, y_high)

        # ========== Validação e Equivalent Square ==========
        eps = 1e-6
        
        if area < eps:
            warnings.warn(
                "Output Factor: Área do campo muito pequena ou zero. "
                "Retornando output factor mínimo (0.0). "
                "Verifique se o segmento tem abertura válida (MLC ou jaws podem estar fechados)."
            )
            return 0.0
        
        height = max(max_y - min_y, eps)
        width_mean = area / height
        perimeter = 2 * (width_mean + height)
        
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
        self.ct_calibration = None  # Will hold CTCalibration instance

    def DensityFromHU(self, machine_name, ct_calibration=None, calibration_name=None):
        """
        Convert HU to density using CTCalibration object.
        
        Args:
            machine_name: Machine name (for legacy CSV lookup)
            ct_calibration: Optional CTCalibration instance. If None, loads from CSV.
            calibration_name: Optional calibration curve name (if multiple CSVs available)
            
        Returns:
            density: numpy array of density values [g/cc]
        """
        # Use provided calibration or load from CSV
        if ct_calibration is not None:
            if not isinstance(ct_calibration, CTCalibration):
                raise TypeError("ct_calibration deve ser uma instância de CTCalibration")
            self.ct_calibration = ct_calibration
        else:
            # Load calibration(s) from lookuptables (supports multiple curves)
            machine_dir = pkg_resources.resource_filename(
                __name__,
                os.path.join("lookuptables", "photons", machine_name)
            )
            manager = CTCalibrationManager()
            manager.load_from_directory(machine_dir, pattern="HU_Density*.csv")

            if calibration_name is None:
                # Prefer explicitly named curve if present, else default HU_Density.csv
                default_name = f"{machine_name}_CT"
                try:
                    self.ct_calibration = manager.get_calibration(default_name)
                except KeyError:
                    # Fallback to first available
                    if len(manager.calibrations) == 0:
                        raise FileNotFoundError(f"Nenhuma curva HU_Density*.csv encontrada em {machine_dir}")
                    first_name = manager.list_calibrations()[0]
                    warnings.warn(f"Curva padrão '{default_name}' não encontrada. Usando '{first_name}'.")
                    self.ct_calibration = manager.get_calibration(first_name)
            else:
                self.ct_calibration = manager.get_calibration(calibration_name)
        
        # Convert HU to density using calibration object
        density = self.ct_calibration.convert(self.HU)
        
        return density

    def computeIMRTPlan(self, plan, gpu_id=0, ct_calibration_name=None):
            
        self.beam_doses = []
        self.dose = np.zeros(self.size, dtype=np.single)
        self.Density = self.DensityFromHU(plan.machine_name, calibration_name=ct_calibration_name)

        if self.spacing[0] != self.spacing[1] or self.spacing[0] != self.spacing[2]:
            raise Exception("Spacing must be isotropic for dose calculation - consider resampling CT")
        
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

        # P3.1: Check that different energy models are actually different
        self._validate_energy_model_uniqueness()

    def _validate_energy_model_uniqueness(self):
        """
        Verify that different energy models have distinct parameters (P3.1).

        Identical energy models for different nominal energies indicate
        a commissioning error and could lead to incorrect dose calculation.
        """
        import warnings

        if len(self.beam_models) < 2:
            return

        # Compute hash for each energy model
        hashes = {}
        for model in self.beam_models:
            model_hash = model.get_model_hash()
            energy = model.dicom_energy_label

            if model_hash in hashes:
                other_energy = hashes[model_hash]
                warnings.warn(
                    f"⚠ BEAM MODEL WARNING: Energy '{energy}' has identical parameters "
                    f"to energy '{other_energy}'. This is physically implausible. "
                    f"Each energy should have unique spectrum, kernel, and calibration. "
                    f"Check your beam model commissioning data.",
                    category=UserWarning
                )
            else:
                hashes[model_hash] = energy

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

        # Load kernel (CUDA expects column-grouped layout: theta[0..5], Am[0..5], am[0..5], Bm[0..5], bm[0..5], ray_length[0..5])
        kernel_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, folder_energy_label, "kernel.csv"))
        kernel = pd.read_csv(kernel_path)
        # Normalize column names (strip spaces/BOM)
        kernel.columns = [str(c).strip() for c in kernel.columns]
        required_cols = ["theta", "Am", "am", "Bm", "bm", "ray_length"]
        missing = [c for c in required_cols if c not in kernel.columns]
        if missing:
            raise ValueError(f"kernel.csv missing column(s): {', '.join(missing)}")

        # Build column-grouped vector to match CUDA indexing
        theta = kernel["theta"].to_numpy(dtype=np.single)
        Am = kernel["Am"].to_numpy(dtype=np.single)
        am = kernel["am"].to_numpy(dtype=np.single)
        Bm = kernel["Bm"].to_numpy(dtype=np.single)
        bm = kernel["bm"].to_numpy(dtype=np.single)
        ray_length = kernel["ray_length"].to_numpy(dtype=np.single)

        beam_model.kernel = np.concatenate([theta, Am, am, Bm, bm, ray_length]).astype(np.single)
        beam_model.kernel_len = beam_model.kernel.size

        if "weight" in kernel.columns:
            weights = np.array(kernel["weight"].to_numpy(), dtype=np.single)
            wsum = float(np.sum(weights))
            if wsum <= 0:
                warnings.warn(
                    f"Kernel weights for {folder_energy_label} sum to {wsum:.3f}; "
                    f"disabling directional weighting."
                )
                beam_model.kernel_weights = None
            else:
                if abs(wsum - 1.0) > 1e-3:
                    warnings.warn(
                        f"Kernel weights for {folder_energy_label} sum to {wsum:.3f}; "
                        f"normalizing to 1.0."
                    )
                beam_model.kernel_weights = (weights / wsum).astype(np.single)
        else:
            beam_model.kernel_weights = None
        # heterogeneity smoothing (optional)
        beam_model.heterogeneity_alpha = 0.0  # default off; overridden by beam_parameters.csv
        
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
            
            beam_model.kernel_params = kernel_params.flatten().astype(np.single)  # Linearizar para passar ao CUDA
            beam_model.use_depth_dependent_kernel = True
            print(f"✓ Loaded depth-dependent kernel for {folder_energy_label} ({len(depths)} depths)")
        else:
            beam_model.use_depth_dependent_kernel = False
            beam_model.n_kernel_depths = 0
            beam_model.kernel_params = None
            beam_model.kernel_depths = None

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
            elif line.startswith('dlg'):
                # P1.2: Dosimetric Leaf Gap [mm]
                beam_model.dlg = float(line.split(',')[1])
            elif line.startswith('tg_ext'):
                # P1.2: Tongue-and-groove over-blocking [mm]
                beam_model.tg_ext = float(line.split(',')[1])
            elif line.startswith('heterogeneity_alpha'):
                # P2.4: history smoothing factor [0..1)
                beam_model.heterogeneity_alpha = float(line.split(',')[1])

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

    def _extract_mlc_geometry_from_dicom(self, beam):
        """
        Extract MLC geometry from DICOM BeamLimitingDeviceSequence (P1.3).

        Reads LeafPositionBoundaries to compute y_offset and y_width for each leaf pair.
        This provides machine-accurate MLC geometry without relying on CSV.

        Args:
            beam: DICOM beam object from BeamSequence

        Returns:
            tuple (n_pairs, offsets, widths) if LeafPositionBoundaries present, else None
        """
        import warnings

        if not hasattr(beam, 'BeamLimitingDeviceSequence'):
            return None

        for device in beam.BeamLimitingDeviceSequence:
            device_type = getattr(device, 'RTBeamLimitingDeviceType', '')

            if device_type in ('MLCX', 'MLC'):
                # LeafPositionBoundaries defines edges between leaves (n_pairs + 1 values)
                if not hasattr(device, 'LeafPositionBoundaries'):
                    warnings.warn(
                        f"MLCX device encontrado mas LeafPositionBoundaries ausente. "
                        f"Usando fallback CSV para geometria MLC."
                    )
                    return None

                boundaries = np.array(device.LeafPositionBoundaries, dtype=np.float32)
                n_boundaries = len(boundaries)
                n_pairs = n_boundaries - 1

                if n_pairs < 1:
                    warnings.warn(f"LeafPositionBoundaries inválido: {n_boundaries} valores")
                    return None

                # Compute y_offset (center of each leaf) and y_width
                offsets = np.zeros(n_pairs, dtype=np.float32)
                widths = np.zeros(n_pairs, dtype=np.float32)

                for i in range(n_pairs):
                    y_bottom = boundaries[i]
                    y_top = boundaries[i + 1]
                    offsets[i] = (y_bottom + y_top) / 2.0  # Center position
                    widths[i] = y_top - y_bottom           # Width

                # Validate: widths should all be positive
                if np.any(widths <= 0):
                    warnings.warn(
                        f"LeafPositionBoundaries produzem larguras negativas/zero. "
                        f"Verifique se boundaries estão ordenados."
                    )
                    return None

                return (n_pairs, offsets, widths)

        # No MLCX device found
        return None

    def _normalize_energy_for_folder(self, energy_label):
        """
        Normalize energy label for folder lookup.

        Maps DICOM energy labels to folder names in lookuptables.
        Handles variations like "6" vs "6MV" vs "6 MV", FFF variants, etc.

        Args:
            energy_label: Energy label from DICOM or energy_labels.csv

        Returns:
            Normalized folder name string
        """
        # Convert to string and strip whitespace
        label = str(energy_label).strip()

        # Remove decimal point for integer values ("6.0" -> "6")
        if '.' in label:
            try:
                val = float(label)
                if val == int(val):
                    label = str(int(val))
            except ValueError:
                pass

        # Common normalizations
        label = label.upper().replace(' ', '')

        # Map common DICOM formats to folder names
        # The folder_energy_label from energy_labels.csv should be used directly
        # This is just a fallback normalization
        return label

    def _resample_vmat_control_points(self, cp_list, max_gantry_spacing=2.0):
        """
        Resample VMAT control points to finer angular spacing (P1.5).

        For VMAT arcs with large gantry angle intervals, interpolates geometry
        parameters to improve dose calculation accuracy.

        Args:
            cp_list: List of IMRTControlPoint objects
            max_gantry_spacing: Maximum gantry angle spacing in degrees (default 2.0)

        Returns:
            Resampled list of IMRTControlPoint objects
        """
        import warnings

        if len(cp_list) < 2:
            return cp_list

        # Check if this is a VMAT beam (gantry angles change)
        gantry_angles = np.array([cp.ga for cp in cp_list])
        angle_changes = np.abs(np.diff(gantry_angles))

        # Handle wraparound (e.g., 359 -> 1)
        angle_changes = np.minimum(angle_changes, 360.0 - angle_changes)

        # If no significant gantry motion, return original
        if np.max(angle_changes) < 0.1:
            return cp_list

        resampled_cps = []
        total_original_mu = sum(cp.mu for cp in cp_list)

        for i in range(len(cp_list) - 1):
            cp1 = cp_list[i]
            cp2 = cp_list[i + 1]

            # Calculate angle difference
            ga1, ga2 = cp1.ga, cp2.ga
            delta_ga = ga2 - ga1

            # Handle wraparound
            if delta_ga > 180:
                delta_ga -= 360
            elif delta_ga < -180:
                delta_ga += 360

            abs_delta_ga = abs(delta_ga)

            if abs_delta_ga <= max_gantry_spacing or abs_delta_ga < 0.1:
                # No resampling needed for this interval
                resampled_cps.append(cp1)
                continue

            # Number of sub-intervals
            n_intervals = int(np.ceil(abs_delta_ga / max_gantry_spacing))
            segment_mu = cp1.mu / n_intervals  # Distribute MU equally

            for j in range(n_intervals):
                frac = j / n_intervals

                # Interpolate gantry angle
                new_ga = ga1 + frac * delta_ga
                if new_ga >= 360:
                    new_ga -= 360
                elif new_ga < 0:
                    new_ga += 360

                # Interpolate other angles
                new_ca = cp1.ca + frac * (cp2.ca - cp1.ca)
                new_ta = cp1.ta + frac * (cp2.ta - cp1.ta)

                # Interpolate jaws
                new_xjaws = cp1.xjaws + frac * (cp2.xjaws - cp1.xjaws) if cp1.xjaws is not None and cp2.xjaws is not None else cp1.xjaws
                new_yjaws = cp1.yjaws + frac * (cp2.yjaws - cp1.yjaws) if cp1.yjaws is not None and cp2.yjaws is not None else cp1.yjaws

                # Interpolate MLC positions
                new_mlc = cp1.mlc + frac * (cp2.mlc - cp1.mlc)

                # Create new control point
                new_cp = IMRTControlPoint(
                    iso=cp1.iso.copy(),
                    mu=segment_mu,
                    mlc=new_mlc,
                    ga=new_ga,
                    ca=new_ca,
                    ta=new_ta,
                    xjaws=new_xjaws.copy() if new_xjaws is not None else None,
                    yjaws=new_yjaws.copy() if new_yjaws is not None else None
                )
                resampled_cps.append(new_cp)

        # Add the last control point's contribution (if any MU remaining)
        # Actually, the last CP in the segment list has its MU already distributed
        # So we don't need to add it again

        total_resampled_mu = sum(cp.mu for cp in resampled_cps)

        # Warn if MU conservation is violated
        if abs(total_resampled_mu - total_original_mu) > 0.1:
            warnings.warn(
                f"VMAT resampling: MU mismatch. Original: {total_original_mu:.2f}, "
                f"Resampled: {total_resampled_mu:.2f}. Diff: {total_resampled_mu - total_original_mu:.4f}"
            )

        return resampled_cps

    def _resample_leaf_travel(self, cp_list, max_leaf_motion_mm=1.0):
        """
        Resample segments so that maximum leaf/jaw motion between successive CPs
        is limited (P1.5, sliding-window fidelity).

        Args:
            cp_list: list of IMRTControlPoint objects (segment MU already set)
            max_leaf_motion_mm: maximum allowed motion per subdivision

        Returns:
            new_cp_list with MU conserved
        """
        if len(cp_list) < 2:
            return cp_list

        resampled = []
        total_mu_before = sum(cp.mu for cp in cp_list)

        for idx in range(len(cp_list) - 1):
            cp1 = cp_list[idx]
            cp2 = cp_list[idx + 1]

            # Compute max motion across leaves and jaws
            max_motion = 0.0
            if cp1.mlc is not None and cp2.mlc is not None:
                max_motion = max(max_motion, float(np.max(np.abs(cp2.mlc[:, :2] - cp1.mlc[:, :2]))))
            if cp1.xjaws is not None and cp2.xjaws is not None:
                max_motion = max(max_motion, float(np.max(np.abs(cp2.xjaws - cp1.xjaws))))
            if cp1.yjaws is not None and cp2.yjaws is not None:
                max_motion = max(max_motion, float(np.max(np.abs(cp2.yjaws - cp1.yjaws))))

            n_parts = int(np.ceil(max_motion / max_leaf_motion_mm)) if max_motion > max_leaf_motion_mm else 1
            part_mu = cp1.mu / n_parts

            for j in range(n_parts):
                frac = (j + 0.5) / n_parts  # mid-interval
                new_mlc = cp1.mlc + frac * (cp2.mlc - cp1.mlc)
                new_xjaws = cp1.xjaws + frac * (cp2.xjaws - cp1.xjaws) if cp1.xjaws is not None and cp2.xjaws is not None else cp1.xjaws
                new_yjaws = cp1.yjaws + frac * (cp2.yjaws - cp1.yjaws) if cp1.yjaws is not None and cp2.yjaws is not None else cp1.yjaws
                new_ga = cp1.ga + frac * (cp2.ga - cp1.ga)
                new_ca = cp1.ca + frac * (cp2.ca - cp1.ca)
                new_ta = cp1.ta + frac * (cp2.ta - cp1.ta)

                resampled.append(IMRTControlPoint(
                    iso=cp1.iso.copy(),
                    mu=part_mu,
                    mlc=new_mlc,
                    ga=new_ga,
                    ca=new_ca,
                    ta=new_ta,
                    xjaws=new_xjaws.copy() if new_xjaws is not None else None,
                    yjaws=new_yjaws.copy() if new_yjaws is not None else None
                ))

        total_mu_after = sum(cp.mu for cp in resampled)
        if abs(total_mu_after - total_mu_before) > 1e-3:
            import warnings
            warnings.warn(f"Leaf travel resampling MU mismatch: before {total_mu_before:.3f}, after {total_mu_after:.3f}")

        return resampled

    def readPlanDicom(self, plan_path, vmat_resampling=True, max_gantry_spacing=2.0,
                      leaf_travel_resampling=True, max_leaf_motion_mm=1.0):
        """
        Read RTPLAN DICOM file and construct beam/control point structures.

        Robust parsing for clinical secondary dose calculation:
        - MU per beam from BeamMeterset via FractionGroupSequence
        - MU per segment = delta_CMW × scalingFactor
        - Jaws/MLC parsed by RTBeamLimitingDeviceType (not fixed indices)
        - Fallback "last-known" for omitted CP data
        - Validation of unsupported modifiers (wedges, compensators, etc.)
        - Energy normalization for model matching
        - P1.5: Optional VMAT arc resampling for delivery fidelity

        Args:
            plan_path: Path to RTPLAN DICOM file
            vmat_resampling: Enable VMAT control point resampling (default True)
            max_gantry_spacing: Maximum gantry angle spacing in degrees (default 2.0)
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
        
        # P1.5/P1.3: armazenar geometria MLC do DICOM (primeiro beam que fornecer boundaries)
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

            # P1.3: Extract MLC geometry from DICOM (primary source). Apply once when first found.
            if mlc_geometry_from_dicom is None:
                mlc_geometry_from_dicom = self._extract_mlc_geometry_from_dicom(beam)
                if mlc_geometry_from_dicom is not None:
                    # Reload beam models with DICOM geometry for all energies
                    self.beam_models = []
                    for idx, d_label in enumerate(self.dicom_energy_label):
                        beam_model = IMRTPhotonEnergy(d_label)
                        f_label = self.folder_energy_label[idx]
                        self._load_beam_model_parameters(
                            beam_model, self.machine_name, f_label, mlc_geometry_from_dicom
                        )
                        self.beam_models.append(beam_model)
                else:
                    # No boundaries found yet; continue with existing models (CSV)
                    pass

            beam_meterset = beam_meterset_map[beam_number]

            # Control points
            cps = beam.ControlPointSequence if hasattr(beam, 'ControlPointSequence') else []

            # Synthesize CMW if missing/constant/non-monotonic to avoid MU=0 segments
            if len(cps) == 0:
                warnings.warn(f"Beam {beam_number}: ControlPointSequence vazia. Pulando feixe.")
                continue

            cmw_vals = []
            missing_cmw = False
            for cp in cps:
                if hasattr(cp, 'CumulativeMetersetWeight'):
                    cmw_vals.append(float(cp.CumulativeMetersetWeight))
                else:
                    cmw_vals.append(None)
                    missing_cmw = True

            cmw_numeric = np.array([v for v in cmw_vals if v is not None], dtype=float)
            all_equal = cmw_numeric.size > 0 and (np.max(cmw_numeric) - np.min(cmw_numeric) < 1e-9)
            non_monotonic = cmw_numeric.size > 1 and np.any(np.diff(cmw_numeric) < -1e-9)
            last_too_small = cmw_numeric.size > 0 and cmw_numeric[-1] <= 1e-6

            need_synth = missing_cmw or all_equal or non_monotonic or last_too_small

            if need_synth:
                n_cps = len(cps)
                if n_cps == 1:
                    synth = [1.0]
                else:
                    synth = [i / (n_cps - 1) for i in range(n_cps)]
                for cp, val in zip(cps, synth):
                    cp.CumulativeMetersetWeight = float(val)
                warnings.warn(
                    f"Beam {beam_number}: CumulativeMetersetWeight ausente/constante; "
                    f"gerando rampa 0→1 com {n_cps} CPs (último={synth[-1]:.3f})."
                )

            # Get FinalCumulativeMetersetWeight with consistency validation
            final_cmw_from_beam = None
            final_cmw_from_cp = None
            
            if hasattr(beam, 'FinalCumulativeMetersetWeight'):
                final_cmw_from_beam = float(beam.FinalCumulativeMetersetWeight)
            
            if len(cps) > 0:
                last_cp = cps[-1]
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
            skipped_small_mu = 0
            total_segments = max(len(cps) - 1, 0)
            skip_floor = max(_MIN_SEG_MU_ABS, beam_meterset * _MIN_SEG_MU_REL)

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

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Beam %s CP%d delta_cmw=%.6e scaling_factor=%.6e seg_mu=%.6e "
                        "(skip_floor=%.3e)",
                        beam_number, i, delta_cmw, scaling_factor, seg_mu, skip_floor
                    )

                # Skip near-zero MU segments using absolute + relative floor
                if seg_mu <= skip_floor:
                    skipped_small_mu += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Skipping beam %s CP%d seg_mu=%.6e <= floor %.3e",
                            beam_number, i, seg_mu, skip_floor
                        )
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

            if skipped_small_mu:
                logger.info(
                    "Beam %s: skipped %d/%d segments with MU <= %.3e",
                    beam_number, skipped_small_mu, total_segments, skip_floor
                )

            if imrt_beam.n_cps > 0:
                # P1.5: Optional leaf-travel resampling for dynamic MLC/sliding window
                if leaf_travel_resampling and imrt_beam.n_cps > 1:
                    imrt_beam.cp_list = self._resample_leaf_travel(imrt_beam.cp_list, max_leaf_motion_mm)
                    imrt_beam.n_cps = len(imrt_beam.cp_list)
                # P1.5: Apply VMAT resampling if enabled
                if vmat_resampling and imrt_beam.n_cps > 1:
                    original_n_cps = imrt_beam.n_cps
                    imrt_beam.cp_list = self._resample_vmat_control_points(
                        imrt_beam.cp_list, max_gantry_spacing
                    )
                    imrt_beam.n_cps = len(imrt_beam.cp_list)
                    if imrt_beam.n_cps > original_n_cps:
                        print(f"  VMAT resampling: {original_n_cps} -> {imrt_beam.n_cps} CPs "
                              f"(max spacing {max_gantry_spacing}°)")
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

            # P1.2: Apply Dosimetric Leaf Gap (DLG) correction
            # DLG models the rounded leaf ends by shifting leaf tips
            # x1_eff = x1 - DLG/2 (left leaf retracts to simulate transmission)
            # x2_eff = x2 + DLG/2 (right leaf retracts)
            dlg = self.beam_models[0].dlg
            if dlg > 0:
                half_dlg = dlg / 2.0
                mlc[0, :] = mlc[0, :] - half_dlg  # Left bank retracts
                mlc[1, :] = mlc[1, :] + half_dlg  # Right bank retracts
                # Ensure x1 <= x2 after DLG correction (for closed leaves)
                mlc[0, :], mlc[1, :] = np.minimum(mlc[0, :], mlc[1, :]), np.maximum(mlc[0, :], mlc[1, :])

            # Tongue-and-groove approximation: widen leaves near boundaries where
            # adjacent openings differ, producing extra blocking (shadowing).
            tg_ext = getattr(self.beam_models[0], "tg_ext", 0.0)
            tg_threshold = 0.5  # mm difference in opening to trigger TG shadow
            mlc_widths_eff = mlc_widths.copy()
            if tg_ext > 0.0:
                openings = mlc[1, :] - mlc[0, :]
                for idx in range(n_mlc_pairs - 1):
                    if abs(openings[idx] - openings[idx + 1]) > tg_threshold:
                        # Spread the over-blocking equally to the two leaves
                        mlc_widths_eff[idx] += 0.5 * tg_ext
                        mlc_widths_eff[idx + 1] += 0.5 * tg_ext
                mlc_widths_eff = np.clip(mlc_widths_eff, 0.1, None)
            else:
                mlc_widths_eff = mlc_widths

            mlc = np.array(np.vstack((
                mlc,
                mlc_offsets.reshape(1, -1),
                mlc_widths_eff.reshape(1, -1)
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
