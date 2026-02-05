#!/usr/bin/env python3
"""
=============================================================================
EDK Kernel Processor for DoseCUDA
=============================================================================

Reads monoenergetic kernel data K(r, theta) from EGSnrc simulations and:
1. Loads and validates raw kernel data
2. Computes polyenergetic kernels via spectral weighting
3. Compresses angular bins to DoseCUDA format (6 or 12 polar angles)
4. Fits the two-term exponential model (Am, am, Bm, bm)
5. Exports kernel.csv and optional kernel_depth_dependent.csv

Reference Model (DoseCUDA CCC):
    K_line(r) = Am * exp(-am * r) + Bm * exp(-bm * r)

    Where:
    - Am, am: Primary (short-range) component
    - Bm, bm: Scatter (long-range) component
    - r: Distance along ray in cm

Author: DoseCUDA Team
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, RegularGridInterpolator
from pathlib import Path
import argparse
import json
import logging
from typing import Tuple, List, Dict, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Kernel Model Functions
# =============================================================================

def kernel_two_term(r: np.ndarray, Am: float, am: float,
                    Bm: float, bm: float) -> np.ndarray:
    """
    Two-term exponential kernel model used by DoseCUDA.

    K(r) = Am * exp(-am * r) + Bm * exp(-bm * r)

    Parameters:
        r: Distance array (cm)
        Am: Amplitude of primary (short-range) component
        am: Attenuation of primary component (cm^-1)
        Bm: Amplitude of scatter (long-range) component
        bm: Attenuation of scatter component (cm^-1)

    Returns:
        Kernel values at distances r
    """
    # Clip r to avoid numerical issues at r=0
    r_safe = np.maximum(r, 1e-6)
    return Am * np.exp(-am * r_safe) + Bm * np.exp(-bm * r_safe)


def kernel_single_term(r: np.ndarray, A: float, a: float) -> np.ndarray:
    """Single exponential for initial guess."""
    r_safe = np.maximum(r, 1e-6)
    return A * np.exp(-a * r_safe)


# =============================================================================
# Kernel Data Loading
# =============================================================================

class MonoenergeticKernel:
    """
    Container for monoenergetic kernel data K(r, theta).

    Attributes:
        energy: Photon energy in MeV
        r_bins: Radial bin centers (cm)
        theta_bins: Angular bin centers (degrees, 0=forward, 180=backward)
        kernel: 2D array [n_r, n_theta] of normalized kernel values
    """

    def __init__(self, energy: float, r_bins: np.ndarray,
                 theta_bins: np.ndarray, kernel: np.ndarray):
        self.energy = energy
        self.r_bins = r_bins
        self.theta_bins = theta_bins
        self.kernel = kernel

        # Validate
        assert kernel.shape == (len(r_bins), len(theta_bins)), \
            f"Kernel shape {kernel.shape} doesn't match bins ({len(r_bins)}, {len(theta_bins)})"

    @classmethod
    def from_csv(cls, filepath: str, energy: float) -> 'MonoenergeticKernel':
        """
        Load kernel from CSV file (EGSnrc output format).

        Expected CSV columns: r_cm, theta_deg, K_normalized
        """
        logger.info(f"Loading kernel from {filepath}")
        df = pd.read_csv(filepath, comment='#')

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Get unique r and theta values
        r_unique = np.sort(df['r_cm'].unique())
        theta_unique = np.sort(df['theta_deg'].unique())

        # Pivot to 2D array
        kernel_2d = np.zeros((len(r_unique), len(theta_unique)))
        for i, r in enumerate(r_unique):
            for j, theta in enumerate(theta_unique):
                mask = (df['r_cm'] == r) & (df['theta_deg'] == theta)
                if mask.sum() > 0:
                    kernel_2d[i, j] = df.loc[mask, 'k_normalized'].values[0]

        return cls(energy, r_unique, theta_unique, kernel_2d)

    @classmethod
    def from_egsnrc_3ddose(cls, filepath: str, energy: float) -> 'MonoenergeticKernel':
        """
        Load kernel from EGSnrc .3ddose format (spherical phantom output).
        """
        logger.info(f"Loading kernel from 3ddose: {filepath}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse .3ddose header
        idx = 0
        nx, ny, nz = map(int, lines[idx].split())
        idx += 1

        # Read boundaries
        xbounds = np.array([float(x) for x in lines[idx].split()])
        idx += 1
        ybounds = np.array([float(x) for x in lines[idx].split()])
        idx += 1
        zbounds = np.array([float(x) for x in lines[idx].split()])
        idx += 1

        # Read dose values
        dose_values = []
        while idx < len(lines) and len(dose_values) < nx * ny * nz:
            dose_values.extend([float(x) for x in lines[idx].split()])
            idx += 1

        dose_3d = np.array(dose_values).reshape((nx, ny, nz), order='F')

        # Convert Cartesian to spherical coordinates
        # (This is simplified - actual implementation needs proper binning)
        x_centers = 0.5 * (xbounds[:-1] + xbounds[1:])
        y_centers = 0.5 * (ybounds[:-1] + ybounds[1:])
        z_centers = 0.5 * (zbounds[:-1] + zbounds[1:])

        # Create spherical grid
        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        THETA = np.degrees(np.arccos(Z / np.maximum(R, 1e-10)))

        # Bin into spherical shells and cones
        r_bins = np.linspace(0.1, 30, 60)
        theta_bins = np.linspace(0, 180, 48)

        kernel_2d = np.zeros((len(r_bins), len(theta_bins)))
        counts = np.zeros_like(kernel_2d)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r = R[i, j, k]
                    theta = THETA[i, j, k]

                    r_idx = np.searchsorted(r_bins, r) - 1
                    theta_idx = np.searchsorted(theta_bins, theta) - 1

                    if 0 <= r_idx < len(r_bins) and 0 <= theta_idx < len(theta_bins):
                        kernel_2d[r_idx, theta_idx] += dose_3d[i, j, k]
                        counts[r_idx, theta_idx] += 1

        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            kernel_2d = np.where(counts > 0, kernel_2d / counts, 0)

        return cls(energy, r_bins, theta_bins, kernel_2d)

    def get_radial_kernel(self, theta_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract radial kernel K(r) at a specific polar angle.

        Parameters:
            theta_deg: Polar angle in degrees

        Returns:
            (r_array, K_array) tuple
        """
        # Find closest theta bin
        theta_idx = np.argmin(np.abs(self.theta_bins - theta_deg))
        return self.r_bins, self.kernel[:, theta_idx]

    def integrate_angular_bin(self, theta_min: float, theta_max: float) -> np.ndarray:
        """
        Integrate kernel over angular range [theta_min, theta_max].

        This compresses multiple fine angular bins into one DoseCUDA bin.

        Returns:
            K(r) integrated over the angular range
        """
        mask = (self.theta_bins >= theta_min) & (self.theta_bins < theta_max)
        if not mask.any():
            # Return interpolated value
            theta_center = 0.5 * (theta_min + theta_max)
            theta_idx = np.argmin(np.abs(self.theta_bins - theta_center))
            return self.kernel[:, theta_idx]

        # Weight by solid angle (sin(theta) * dtheta)
        theta_rad = np.radians(self.theta_bins[mask])
        weights = np.sin(theta_rad)
        weights /= weights.sum()

        return np.dot(self.kernel[:, mask], weights)


# =============================================================================
# Spectral Weighting
# =============================================================================

class SpectralWeighter:
    """
    Computes polyenergetic kernels from monoenergetic data using spectral weights.

    The spectral weights W(E) can be extracted from IAEA phase space files
    using energy histogram analysis.
    """

    def __init__(self, energies: np.ndarray, weights: np.ndarray):
        """
        Parameters:
            energies: Energy bins in MeV
            weights: Normalized spectral weights (sum to 1)
        """
        self.energies = np.array(energies)
        self.weights = np.array(weights)

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

        logger.info(f"Spectral weighter initialized with {len(energies)} energy bins")
        logger.info(f"Energy range: {energies.min():.3f} - {energies.max():.3f} MeV")
        logger.info(f"Mean energy: {np.dot(energies, weights):.3f} MeV")

    @classmethod
    def from_csv(cls, filepath: str) -> 'SpectralWeighter':
        """
        Load spectral weights from CSV file.

        Expected columns: energy_MeV, weight
        """
        df = pd.read_csv(filepath)
        return cls(df['energy_MeV'].values, df['weight'].values)

    @classmethod
    def from_phsp_histogram(cls, phsp_file: str, n_bins: int = 50) -> 'SpectralWeighter':
        """
        Extract spectral weights from IAEA phase space file.

        This is a placeholder - actual implementation depends on PHSP format.
        """
        logger.warning("PHSP extraction not implemented - using dummy spectrum")

        # Generate a typical 6MV spectrum shape (for demo)
        energies = np.linspace(0.1, 6.0, n_bins)
        weights = energies * np.exp(-energies / 1.5)  # Typical bremsstrahlung shape
        weights /= weights.sum()

        return cls(energies, weights)

    @classmethod
    def for_fff_beam(cls, max_energy: float, n_bins: int = 50) -> 'SpectralWeighter':
        """
        Generate approximate FFF (Flattening Filter Free) spectrum.

        FFF beams have harder spectra with more high-energy photons.
        """
        energies = np.linspace(0.1, max_energy, n_bins)

        # FFF spectrum: harder than filtered, more forward-peaked
        weights = (energies / max_energy)**0.5 * np.exp(-energies / (max_energy * 0.4))
        weights /= weights.sum()

        return cls(energies, weights)

    def compute_polyenergetic_kernel(
        self,
        mono_kernels: Dict[float, MonoenergeticKernel]
    ) -> MonoenergeticKernel:
        """
        Compute polyenergetic kernel by spectral weighting.

        K_poly(r, theta) = sum_E [ W(E) * K_mono(r, theta, E) ]

        Parameters:
            mono_kernels: Dict mapping energy (MeV) to MonoenergeticKernel

        Returns:
            Weighted polyenergetic kernel
        """
        # Get common grid from first kernel
        ref_kernel = list(mono_kernels.values())[0]
        r_bins = ref_kernel.r_bins
        theta_bins = ref_kernel.theta_bins

        poly_kernel = np.zeros_like(ref_kernel.kernel)
        total_weight = 0.0

        for energy, weight in zip(self.energies, self.weights):
            # Find closest monoenergetic kernel
            available_energies = np.array(list(mono_kernels.keys()))
            closest_E = available_energies[np.argmin(np.abs(available_energies - energy))]

            # Interpolate if exact energy not available
            if abs(closest_E - energy) > 0.1:
                logger.debug(f"Interpolating E={energy:.3f} MeV from E={closest_E:.3f} MeV")

            mono = mono_kernels[closest_E]

            # Interpolate to common grid if needed
            if not np.allclose(mono.r_bins, r_bins):
                interpolator = RegularGridInterpolator(
                    (mono.r_bins, mono.theta_bins),
                    mono.kernel,
                    bounds_error=False,
                    fill_value=0.0
                )
                R, THETA = np.meshgrid(r_bins, theta_bins, indexing='ij')
                points = np.stack([R.ravel(), THETA.ravel()], axis=1)
                kernel_interp = interpolator(points).reshape(len(r_bins), len(theta_bins))
                poly_kernel += weight * kernel_interp
            else:
                poly_kernel += weight * mono.kernel

            total_weight += weight

        # Normalize
        poly_kernel /= total_weight

        # Mean energy for labeling
        mean_energy = np.dot(self.energies, self.weights)

        return MonoenergeticKernel(mean_energy, r_bins, theta_bins, poly_kernel)


# =============================================================================
# Angular Compression
# =============================================================================

class AngularCompressor:
    """
    Compresses fine angular bins to DoseCUDA format.

    DoseCUDA uses discrete polar angles (default 6, optionally 12).
    The standard angles are chosen to provide good coverage:

    6 angles (default):
        theta = [1.875, 20.625, 43.125, 61.875, 88.125, 106.875] degrees

    12 angles:
        theta = [3.75, 11.25, 18.75, 26.25, 33.75, 41.25,
                 48.75, 56.25, 63.75, 71.25, 78.75, 86.25] degrees
    """

    # Standard DoseCUDA angle configurations
    ANGLES_6 = np.array([1.875, 20.625, 43.125, 61.875, 88.125, 106.875])
    ANGLES_12 = np.array([3.75, 11.25, 18.75, 26.25, 33.75, 41.25,
                          48.75, 56.25, 63.75, 71.25, 78.75, 86.25])

    def __init__(self, n_angles: int = 6):
        """
        Parameters:
            n_angles: Number of output angles (6 or 12)
        """
        if n_angles == 6:
            self.output_angles = self.ANGLES_6
            self.angle_bins = self._compute_angle_bins_6()
        elif n_angles == 12:
            self.output_angles = self.ANGLES_12
            self.angle_bins = self._compute_angle_bins_12()
        else:
            raise ValueError(f"n_angles must be 6 or 12, got {n_angles}")

        logger.info(f"Angular compressor: {n_angles} output angles")
        logger.info(f"Angles: {self.output_angles}")

    def _compute_angle_bins_6(self) -> List[Tuple[float, float]]:
        """Compute angular bin boundaries for 6 angles."""
        # Based on DoseCUDA kernel representation
        return [
            (0, 11.25),      # Forward
            (11.25, 33.75),
            (33.75, 52.5),
            (52.5, 75.0),
            (75.0, 97.5),
            (97.5, 180.0),   # Backward
        ]

    def _compute_angle_bins_12(self) -> List[Tuple[float, float]]:
        """Compute angular bin boundaries for 12 angles."""
        bins = []
        for i in range(12):
            theta_min = i * 15.0
            theta_max = (i + 1) * 15.0
            bins.append((theta_min, theta_max))
        return bins

    def compress(self, kernel: MonoenergeticKernel) -> Dict[float, np.ndarray]:
        """
        Compress kernel to output angular bins.

        Parameters:
            kernel: Full-resolution kernel

        Returns:
            Dict mapping output angle to radial kernel K(r)
        """
        compressed = {}

        for theta_out, (theta_min, theta_max) in zip(self.output_angles, self.angle_bins):
            K_r = kernel.integrate_angular_bin(theta_min, theta_max)
            compressed[theta_out] = (kernel.r_bins, K_r)

        return compressed


# =============================================================================
# Two-Term Exponential Fitting
# =============================================================================

class KernelFitter:
    """
    Fits the DoseCUDA two-term exponential model to kernel data.

    Model: K(r) = Am * exp(-am * r) + Bm * exp(-bm * r)

    Fitting Strategy:
    1. Initial fit with single exponential to get rough scale
    2. Fit full two-term model with bounded parameters
    3. Validate fit quality and physics constraints

    Physical Constraints:
    - All parameters > 0
    - am > bm (primary attenuates faster than scatter)
    - Am > Bm typically (primary dominates at short range)
    """

    def __init__(self, r_max: float = 30.0, r_min: float = 0.1):
        """
        Parameters:
            r_max: Maximum radius for fitting (cm)
            r_min: Minimum radius for fitting (cm)
        """
        self.r_max = r_max
        self.r_min = r_min

    def fit(self, r: np.ndarray, K: np.ndarray) -> Dict[str, float]:
        """
        Fit two-term model to kernel data.

        Parameters:
            r: Radial distances (cm)
            K: Kernel values

        Returns:
            Dict with keys: Am, am, Bm, bm, ray_length, fit_quality
        """
        # Filter to fitting range
        mask = (r >= self.r_min) & (r <= self.r_max) & (K > 0)
        r_fit = r[mask]
        K_fit = K[mask]

        if len(r_fit) < 10:
            logger.warning(f"Insufficient data points ({len(r_fit)}) for fitting")
            return self._default_params()

        # Normalize K for numerical stability
        K_max = K_fit.max()
        K_norm = K_fit / K_max

        # Initial guess from single exponential fit
        try:
            popt_single, _ = curve_fit(
                kernel_single_term, r_fit, K_norm,
                p0=[1.0, 0.5],
                bounds=([0, 0.01], [10, 10]),
                maxfev=5000
            )
            A0, a0 = popt_single
        except RuntimeError:
            A0, a0 = 1.0, 0.5

        # Two-term fit with initial guess
        # Primary: higher attenuation, scatter: lower attenuation
        p0 = [A0 * 0.7, a0 * 2, A0 * 0.3, a0 * 0.3]

        # Bounds: [Am, am, Bm, bm]
        bounds_low = [1e-6, 0.1, 1e-8, 0.001]
        bounds_high = [100, 20, 10, 5]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    kernel_two_term, r_fit, K_norm,
                    p0=p0,
                    bounds=(bounds_low, bounds_high),
                    maxfev=10000
                )

            Am, am, Bm, bm = popt

            # Unnormalize
            Am *= K_max
            Bm *= K_max

            # Ensure am > bm (primary attenuates faster)
            if bm > am:
                Am, Bm = Bm, Am
                am, bm = bm, am

            # Calculate fit quality
            K_pred = kernel_two_term(r_fit, Am, am, Bm, bm)
            ss_res = np.sum((K_fit - K_pred)**2)
            ss_tot = np.sum((K_fit - K_fit.mean())**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Ray length: distance where kernel drops to 1% of max
            ray_length = self._compute_ray_length(Am, am, Bm, bm)

        except RuntimeError as e:
            logger.warning(f"Fitting failed: {e}")
            return self._default_params()

        return {
            'Am': Am,
            'am': am,
            'Bm': Bm,
            'bm': bm,
            'ray_length': ray_length,
            'fit_quality': r_squared
        }

    def _compute_ray_length(self, Am: float, am: float,
                            Bm: float, bm: float) -> float:
        """
        Compute ray length where kernel drops to threshold.

        Uses 1% of peak value as cutoff.
        """
        K0 = Am + Bm
        threshold = 0.01 * K0

        # Solve numerically
        r_test = np.linspace(0.1, 50, 500)
        K_test = kernel_two_term(r_test, Am, am, Bm, bm)

        idx = np.where(K_test < threshold)[0]
        if len(idx) > 0:
            return min(r_test[idx[0]], 30.0)  # Cap at 30 cm
        else:
            return 30.0

    def _default_params(self) -> Dict[str, float]:
        """Return default parameters when fitting fails."""
        return {
            'Am': 1.0,
            'am': 2.0,
            'Bm': 0.01,
            'bm': 0.1,
            'ray_length': 3.0,
            'fit_quality': 0.0
        }


# =============================================================================
# DoseCUDA Kernel Exporter
# =============================================================================

class DoseCUDAExporter:
    """
    Exports kernels in DoseCUDA format.

    Output Files:
    1. kernel.csv: Standard kernel with columns
       theta, Am, am, Bm, bm, ray_length

    2. kernel_depth_dependent.csv (optional): Depth-varying kernel
       depth, angle_idx, Am, am, Bm, bm
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_kernel(self, angles: np.ndarray,
                      params: List[Dict[str, float]]) -> str:
        """
        Export kernel.csv file.

        Parameters:
            angles: Polar angles in degrees
            params: List of parameter dicts (one per angle)

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / 'kernel.csv'

        data = {
            'theta': angles,
            'Am': [p['Am'] for p in params],
            'am': [p['am'] for p in params],
            'Bm': [p['Bm'] for p in params],
            'bm': [p['bm'] for p in params],
            'ray_length': [p['ray_length'] for p in params]
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        logger.info(f"Exported kernel to {filepath}")
        return str(filepath)

    def export_depth_dependent(
        self,
        depths: np.ndarray,
        angles: np.ndarray,
        params_by_depth: Dict[float, List[Dict[str, float]]]
    ) -> str:
        """
        Export kernel_depth_dependent.csv file.

        Parameters:
            depths: Depth values in cm (WET)
            angles: Polar angles in degrees
            params_by_depth: Dict mapping depth to list of params per angle

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / 'kernel_depth_dependent.csv'

        rows = []
        for depth in depths:
            for angle_idx, params in enumerate(params_by_depth[depth]):
                rows.append({
                    'depth': depth,
                    'angle_idx': angle_idx,
                    'Am': params['Am'],
                    'am': params['am'],
                    'Bm': params['Bm'],
                    'bm': params['bm']
                })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        logger.info(f"Exported depth-dependent kernel to {filepath}")
        return str(filepath)

    def export_validation_report(
        self,
        params: List[Dict[str, float]],
        angles: np.ndarray
    ) -> str:
        """
        Export validation report with sanity checks.
        """
        filepath = self.output_dir / 'kernel_validation.txt'

        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DoseCUDA Kernel Validation Report\n")
            f.write("=" * 60 + "\n\n")

            # Check parameter ranges
            f.write("Parameter Ranges:\n")
            f.write("-" * 40 + "\n")

            Am_values = [p['Am'] for p in params]
            am_values = [p['am'] for p in params]
            Bm_values = [p['Bm'] for p in params]
            bm_values = [p['bm'] for p in params]

            f.write(f"Am: {min(Am_values):.4e} - {max(Am_values):.4e}\n")
            f.write(f"am: {min(am_values):.4f} - {max(am_values):.4f}\n")
            f.write(f"Bm: {min(Bm_values):.4e} - {max(Bm_values):.4e}\n")
            f.write(f"bm: {min(bm_values):.4f} - {max(bm_values):.4f}\n\n")

            # Sanity checks
            f.write("Sanity Checks:\n")
            f.write("-" * 40 + "\n")

            # Check 1: All parameters positive
            all_positive = all(
                p['Am'] > 0 and p['am'] > 0 and p['Bm'] > 0 and p['bm'] > 0
                for p in params
            )
            f.write(f"[{'PASS' if all_positive else 'FAIL'}] All parameters positive\n")

            # Check 2: am > bm (primary attenuates faster)
            am_gt_bm = all(p['am'] > p['bm'] for p in params)
            f.write(f"[{'PASS' if am_gt_bm else 'WARN'}] am > bm (primary > scatter attenuation)\n")

            # Check 3: Fit quality
            fit_qualities = [p.get('fit_quality', 0) for p in params]
            avg_quality = np.mean(fit_qualities)
            f.write(f"[{'PASS' if avg_quality > 0.9 else 'WARN'}] Average fit R² = {avg_quality:.4f}\n")

            # Check 4: Monotonic decrease with angle
            Am_mono = all(Am_values[i] >= Am_values[i+1] for i in range(len(Am_values)-1))
            f.write(f"[{'PASS' if Am_mono else 'INFO'}] Am decreases with angle (expected)\n")

            # Per-angle details
            f.write("\nPer-Angle Parameters:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'theta':>8} {'Am':>12} {'am':>8} {'Bm':>12} {'bm':>8} {'R²':>6}\n")

            for theta, p in zip(angles, params):
                f.write(f"{theta:>8.2f} {p['Am']:>12.4e} {p['am']:>8.4f} "
                       f"{p['Bm']:>12.4e} {p['bm']:>8.4f} {p.get('fit_quality', 0):>6.3f}\n")

        logger.info(f"Exported validation report to {filepath}")
        return str(filepath)


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_monoenergetic_kernels(
    kernel_dir: str,
    energies: List[float],
    output_dir: str,
    n_angles: int = 6
) -> str:
    """
    Process monoenergetic kernel files into DoseCUDA format.

    Parameters:
        kernel_dir: Directory containing kernel CSV files (one per energy)
        energies: List of energies in MeV
        output_dir: Output directory for processed kernels
        n_angles: Number of output angles (6 or 12)

    Returns:
        Path to output kernel.csv
    """
    kernel_dir = Path(kernel_dir)

    # Initialize processors
    compressor = AngularCompressor(n_angles)
    fitter = KernelFitter()
    exporter = DoseCUDAExporter(output_dir)

    # Process each energy
    for energy in energies:
        logger.info(f"Processing E = {energy:.3f} MeV")

        # Find kernel file
        kernel_file = kernel_dir / f"kernel_{energy:.3f}MeV.csv"
        if not kernel_file.exists():
            kernel_file = kernel_dir / f"kernel_{energy}MeV.csv"

        if not kernel_file.exists():
            logger.warning(f"Kernel file not found: {kernel_file}")
            continue

        # Load kernel
        kernel = MonoenergeticKernel.from_csv(str(kernel_file), energy)

        # Compress angles
        compressed = compressor.compress(kernel)

        # Fit parameters
        params = []
        for theta in compressor.output_angles:
            r, K = compressed[theta]
            fit_params = fitter.fit(r, K)
            params.append(fit_params)
            logger.debug(f"  theta={theta:.2f}°: Am={fit_params['Am']:.4e}, "
                        f"am={fit_params['am']:.3f}, Bm={fit_params['Bm']:.4e}, "
                        f"bm={fit_params['bm']:.3f}")

        # Export
        energy_dir = Path(output_dir) / f"{energy}MeV"
        energy_dir.mkdir(parents=True, exist_ok=True)
        energy_exporter = DoseCUDAExporter(str(energy_dir))
        energy_exporter.export_kernel(compressor.output_angles, params)
        energy_exporter.export_validation_report(params, compressor.output_angles)

    return output_dir


def process_polyenergetic_kernel(
    mono_kernel_dir: str,
    spectrum_file: str,
    output_dir: str,
    beam_name: str = "clinical",
    n_angles: int = 6
) -> str:
    """
    Create polyenergetic kernel from monoenergetic data and spectrum.

    Parameters:
        mono_kernel_dir: Directory with monoenergetic kernels
        spectrum_file: CSV file with spectral weights
        output_dir: Output directory
        beam_name: Name for output (e.g., "6MV", "10MV_FFF")
        n_angles: Number of output angles

    Returns:
        Path to output kernel.csv
    """
    mono_dir = Path(mono_kernel_dir)

    # Load spectral weights
    weighter = SpectralWeighter.from_csv(spectrum_file)

    # Load all monoenergetic kernels
    mono_kernels = {}
    for kernel_file in mono_dir.glob("kernel_*.csv"):
        # Extract energy from filename
        name = kernel_file.stem
        try:
            energy = float(name.replace("kernel_", "").replace("MeV", ""))
            mono_kernels[energy] = MonoenergeticKernel.from_csv(str(kernel_file), energy)
            logger.info(f"Loaded monoenergetic kernel: {energy:.3f} MeV")
        except ValueError:
            continue

    if not mono_kernels:
        raise ValueError(f"No monoenergetic kernels found in {mono_dir}")

    # Compute polyenergetic kernel
    poly_kernel = weighter.compute_polyenergetic_kernel(mono_kernels)

    # Compress and fit
    compressor = AngularCompressor(n_angles)
    fitter = KernelFitter()

    compressed = compressor.compress(poly_kernel)

    params = []
    for theta in compressor.output_angles:
        r, K = compressed[theta]
        fit_params = fitter.fit(r, K)
        params.append(fit_params)

    # Export
    beam_dir = Path(output_dir) / beam_name
    beam_dir.mkdir(parents=True, exist_ok=True)
    exporter = DoseCUDAExporter(str(beam_dir))

    exporter.export_kernel(compressor.output_angles, params)
    exporter.export_validation_report(params, compressor.output_angles)

    logger.info(f"Polyenergetic kernel exported to {beam_dir}")
    return str(beam_dir / "kernel.csv")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process EDK kernels for DoseCUDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process monoenergetic kernels
  python kernel_processor.py mono --kernel-dir ./raw_kernels --energies 0.5,1.0,2.0,6.0 --output ./processed

  # Create polyenergetic kernel with spectrum
  python kernel_processor.py poly --mono-dir ./processed --spectrum spectrum_6MV.csv --output ./clinical/6MV

  # Create FFF kernel
  python kernel_processor.py poly --mono-dir ./processed --spectrum spectrum_6FFF.csv --output ./clinical/6MV_FFF --beam-name 6MV_FFF
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Processing mode')

    # Monoenergetic processing
    mono_parser = subparsers.add_parser('mono', help='Process monoenergetic kernels')
    mono_parser.add_argument('--kernel-dir', required=True, help='Directory with raw kernel files')
    mono_parser.add_argument('--energies', required=True, help='Comma-separated energies in MeV')
    mono_parser.add_argument('--output', required=True, help='Output directory')
    mono_parser.add_argument('--n-angles', type=int, default=6, choices=[6, 12],
                            help='Number of output angles')

    # Polyenergetic processing
    poly_parser = subparsers.add_parser('poly', help='Create polyenergetic kernel')
    poly_parser.add_argument('--mono-dir', required=True, help='Directory with monoenergetic kernels')
    poly_parser.add_argument('--spectrum', required=True, help='Spectral weights CSV file')
    poly_parser.add_argument('--output', required=True, help='Output directory')
    poly_parser.add_argument('--beam-name', default='clinical', help='Beam name for output folder')
    poly_parser.add_argument('--n-angles', type=int, default=6, choices=[6, 12],
                            help='Number of output angles')

    args = parser.parse_args()

    if args.command == 'mono':
        energies = [float(e) for e in args.energies.split(',')]
        process_monoenergetic_kernels(
            args.kernel_dir, energies, args.output, args.n_angles
        )

    elif args.command == 'poly':
        process_polyenergetic_kernel(
            args.mono_dir, args.spectrum, args.output,
            args.beam_name, args.n_angles
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
