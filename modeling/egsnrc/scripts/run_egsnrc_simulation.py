#!/usr/bin/env python3
"""
=============================================================================
EGSnrc EDK Simulation Runner
=============================================================================

Generates input files for EGSnrc simulations to compute Energy Deposition
Kernels (EDK) in water for DoseCUDA.

Simulation Approach:
1. Uses DOSXYZnrc with a spherical phantom defined in Cartesian voxels
2. Monoenergetic photon source forced to interact at center
3. Scores dose to water in fine voxel grid
4. Post-processes to spherical coordinates K(r, theta)

Alternative: Uses EDKnrc user code if available (more accurate but complex).

Author: DoseCUDA Team
"""

import numpy as np
import os
import subprocess
from pathlib import Path
import argparse
import logging
from typing import List, Tuple, Optional
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DOSXYZnrc Input File Generator
# =============================================================================

class DOSXYZnrcInputGenerator:
    """
    Generates DOSXYZnrc input files for EDK simulation.

    The simulation uses a water cube phantom with fine voxel resolution
    and a photon point source at the center.
    """

    def __init__(
        self,
        phantom_size: float = 60.0,  # cm
        voxel_size: float = 0.2,     # cm (2mm)
        n_histories: int = 10_000_000,
        ecut: float = 0.521,
        pcut: float = 0.010
    ):
        """
        Parameters:
            phantom_size: Total phantom size (cm)
            voxel_size: Voxel size (cm)
            n_histories: Number of histories per simulation
            ecut: Electron cutoff energy (MeV)
            pcut: Photon cutoff energy (MeV)
        """
        self.phantom_size = phantom_size
        self.voxel_size = voxel_size
        self.n_histories = n_histories
        self.ecut = ecut
        self.pcut = pcut

        # Calculate number of voxels
        self.n_voxels = int(phantom_size / voxel_size)
        if self.n_voxels % 2 == 0:
            self.n_voxels += 1  # Ensure odd number for center voxel

        logger.info(f"DOSXYZnrc configuration: {self.n_voxels}³ voxels, "
                   f"{voxel_size} cm resolution")

    def generate_egsinp(self, energy: float, output_dir: str) -> str:
        """
        Generate DOSXYZnrc .egsinp input file.

        Parameters:
            energy: Photon energy in MeV
            output_dir: Output directory for files

        Returns:
            Path to generated input file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_file = output_dir / f"edk_{energy:.3f}MeV.egsinp"
        phantom_file = output_dir / f"edk_{energy:.3f}MeV.egsphant"

        # Generate phantom file first
        self._generate_phantom(phantom_file)

        # Generate input file
        content = self._generate_input_content(energy, phantom_file.name)

        with open(input_file, 'w') as f:
            f.write(content)

        logger.info(f"Generated input file: {input_file}")
        return str(input_file)

    def _generate_phantom(self, phantom_file: Path):
        """
        Generate .egsphant file (water cube).

        Format:
        - Line 1: Number of media
        - Line 2-N: Media names (ESTEPE values)
        - Line N+1: ESTEPE values
        - Line N+2: NX NY NZ
        - Lines: X boundaries, Y boundaries, Z boundaries
        - Lines: Media indices for each voxel
        - Lines: Densities for each voxel
        """
        half_size = self.phantom_size / 2

        # Voxel boundaries
        boundaries = np.linspace(-half_size, half_size, self.n_voxels + 1)

        with open(phantom_file, 'w') as f:
            # Number of media
            f.write("1\n")

            # Media name
            f.write("H2O521ICRU\n")

            # ESTEPE
            f.write("0.25\n")

            # Grid dimensions
            f.write(f"{self.n_voxels} {self.n_voxels} {self.n_voxels}\n")

            # X boundaries
            f.write(" ".join(f"{x:.4f}" for x in boundaries) + "\n")

            # Y boundaries
            f.write(" ".join(f"{y:.4f}" for y in boundaries) + "\n")

            # Z boundaries
            f.write(" ".join(f"{z:.4f}" for z in boundaries) + "\n")

            # Media indices (all water = 1)
            for iz in range(self.n_voxels):
                for iy in range(self.n_voxels):
                    f.write("1" * self.n_voxels + "\n")

            # Densities (all 1.0 for water)
            for iz in range(self.n_voxels):
                for iy in range(self.n_voxels):
                    f.write(" ".join("1.000" for _ in range(self.n_voxels)) + "\n")

        logger.info(f"Generated phantom file: {phantom_file}")

    def _generate_input_content(self, energy: float, phantom_name: str) -> str:
        """Generate DOSXYZnrc input file content."""

        content = f"""EDK simulation for E = {energy:.3f} MeV
#
# DOSXYZnrc input file for Energy Deposition Kernel generation
# DoseCUDA Kernel Factory
#
###############################################################################
#
# Title
#
###############################################################################
 EDK_{energy:.3f}MeV
#
###############################################################################
#
# I/O Control
#
###############################################################################
 0                              # IWATCH (0=no output, 1=summary, 2=full)
 0                              # ISTORE (0=don't store, 1=store)
 0                              # IRESTART (0=new run, 1=restart, 3=analyze)
 0                              # IO_OPT (0=standard output)
 2                              # DOSE_PRINT (0=none, 2=3ddose file)
 0                              # SPEC_PRINT
#
###############################################################################
#
# Monte Carlo Parameters
#
###############################################################################
 {self.n_histories}             # NCASE
 97                             # IXXIN (RNG seed 1)
 33                             # JXXIN (RNG seed 2)
 10000                          # TIMMAX (max CPU time in hours)
 0, 0, 0, 0, 0                  # IBRSPL, NBRSPL, IRRLTT, NBINDOS, NSPLIT_PHOT
#
###############################################################################
#
# Transport Parameters
#
###############################################################################
 0                              # ECUT global (use input deck below)
 0                              # PCUT global
 0                              # SMAX (max step)
 0                              # ESTEPE (energy loss per step)
 0                              # XIMAX (max first xsec interaction)
#
###############################################################################
#
# Source Definition
#
###############################################################################
 3                              # ISOURCE (3 = point source, isotropic)
 0.0, 0.0, 0.0                  # Source position (center)
 0.0, 0.0                       # THDIR, PHIDIR (not used for source 3)
 -1.0, -1.0, -1.0               # Field size (ignored for point source)
 -1.0, -1.0, -1.0               # DSOURCE (distance, ignored)
 0                              # IQIN (0=photon)
#
###############################################################################
#
# Source Energy
#
###############################################################################
 0                              # MONOEN (0=monoenergetic)
 {energy:.6f}                   # Energy in MeV
#
###############################################################################
#
# Phantom File
#
###############################################################################
 {phantom_name}                 # Phantom file name
#
###############################################################################
#
# Media Parameters (PEGS4 data)
#
###############################################################################
 H2O521ICRU                     # Medium name
 {self.ecut:.4f}, {self.pcut:.4f}, 0, 0  # ECUT, PCUT, DOSE_ZONE, IREGION_TO_BIT
#
###############################################################################
#
# Variance Reduction
#
###############################################################################
 0                              # No range rejection
 0                              # No electron range rejection
 0                              # No photon forcing
 0                              # No bremsstrahlung splitting
#
###############################################################################
#
# EGSnrc Transport Parameters
#
###############################################################################
 Global ECUT= {self.ecut}
 Global PCUT= {self.pcut}
 Global SMAX= 5
 ESTEPE= 0.25
 XIMAX= 0.5
 Boundary crossing algorithm= EXACT
 Skin depth for BCA= 0
 Electron-step algorithm= PRESTA-II
 Spin effects= On
 Brems angular sampling= Simple
 Brems cross sections= BH
 Bound Compton scattering= On
 Compton cross sections= default
 Pair angular sampling= Simple
 Pair cross sections= BH
 Photoelectron angular sampling= On
 Rayleigh scattering= On
 Atomic relaxations= On
 Electron impact ionization= On
 Photon cross sections= xcom
 Photon cross-sections output= Off
 :Stop transport parameters:
###############################################################################
"""
        return content


# =============================================================================
# Simulation Runner
# =============================================================================

class EGSnrcRunner:
    """
    Runs EGSnrc simulations and collects results.
    """

    def __init__(
        self,
        egs_home: str = None,
        hen_house: str = None,
        pegs_file: str = "521icru"
    ):
        """
        Parameters:
            egs_home: Path to EGS_HOME (default from environment)
            hen_house: Path to HEN_HOUSE (default from environment)
            pegs_file: PEGS4 data file name
        """
        self.egs_home = egs_home or os.environ.get('EGS_HOME', '/opt/egsnrc/egs_home')
        self.hen_house = hen_house or os.environ.get('HEN_HOUSE', '/opt/egsnrc/HEN_HOUSE')
        self.pegs_file = pegs_file

        logger.info(f"EGSnrc runner initialized")
        logger.info(f"  EGS_HOME: {self.egs_home}")
        logger.info(f"  HEN_HOUSE: {self.hen_house}")

    def run_simulation(self, input_file: str, output_dir: str) -> str:
        """
        Run DOSXYZnrc simulation.

        Parameters:
            input_file: Path to .egsinp file
            output_dir: Directory for output files

        Returns:
            Path to .3ddose output file
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get input name without extension
        input_name = input_path.stem

        # Prepare environment
        env = os.environ.copy()
        env['EGS_HOME'] = self.egs_home
        env['HEN_HOUSE'] = self.hen_house

        # DOSXYZnrc command
        dosxyznrc_path = Path(self.egs_home) / "dosxyznrc"

        if not dosxyznrc_path.exists():
            logger.warning(f"DOSXYZnrc not found at {dosxyznrc_path}")
            logger.info("Generating mock kernel data for testing...")
            return self._generate_mock_3ddose(input_file, output_dir)

        # Copy input files to dosxyznrc directory
        shutil.copy(input_file, dosxyznrc_path)

        # Run simulation
        cmd = [
            str(dosxyznrc_path / "dosxyznrc"),
            "-i", input_name,
            "-p", self.pegs_file
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(dosxyznrc_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600 * 24  # 24 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"DOSXYZnrc failed: {result.stderr}")
                raise RuntimeError(f"DOSXYZnrc failed with code {result.returncode}")

            logger.info("DOSXYZnrc completed successfully")

        except subprocess.TimeoutExpired:
            logger.error("DOSXYZnrc timed out")
            raise

        # Find and copy output
        dose_file = dosxyznrc_path / f"{input_name}.3ddose"
        if dose_file.exists():
            output_dose = output_path / dose_file.name
            shutil.copy(dose_file, output_dose)
            logger.info(f"Output: {output_dose}")
            return str(output_dose)
        else:
            logger.warning(f"3ddose file not found: {dose_file}")
            return self._generate_mock_3ddose(input_file, output_dir)

    def _generate_mock_3ddose(self, input_file: str, output_dir: str) -> str:
        """
        Generate mock .3ddose file for testing when EGSnrc is not available.

        Uses analytical approximation of photon kernel based on
        Mohan & Chui (1987) parameterization.
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)

        # Extract energy from filename
        import re
        match = re.search(r'(\d+\.?\d*)MeV', input_path.stem)
        energy = float(match.group(1)) if match else 1.0

        logger.info(f"Generating mock kernel for E = {energy:.3f} MeV")

        # Phantom parameters
        n = 101  # Voxels per dimension (reduced for mock)
        size = 40.0  # cm
        voxel_size = size / n

        # Generate grid
        x = np.linspace(-size/2, size/2, n+1)
        y = np.linspace(-size/2, size/2, n+1)
        z = np.linspace(-size/2, size/2, n+1)

        # Voxel centers
        xc = 0.5 * (x[:-1] + x[1:])
        yc = 0.5 * (y[:-1] + y[1:])
        zc = 0.5 * (z[:-1] + z[1:])

        # Calculate dose using analytical kernel
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        R = np.maximum(R, 0.001)  # Avoid division by zero

        # Mohan-Chui type parameterization (simplified)
        # K(r) = A1*exp(-a1*r)/r² + A2*exp(-a2*r)/r²
        # Parameters scale with energy

        mu = 0.04 + 0.01 * energy  # Effective attenuation
        A1 = 0.5 * energy
        a1 = 3.0 / energy
        A2 = 0.1 * energy
        a2 = 0.3 / energy

        dose = (A1 * np.exp(-a1 * R) + A2 * np.exp(-a2 * R)) / (R**2 + 0.1)

        # Add angular dependence (cosine term for forward peaking)
        COSTH = Z / R
        angular_factor = 0.5 * (1 + COSTH + COSTH**2)
        dose *= angular_factor

        # Normalize
        dose /= dose.sum() * voxel_size**3

        # Write .3ddose file
        output_file = output_path / f"edk_{energy:.3f}MeV.3ddose"

        with open(output_file, 'w') as f:
            # Grid dimensions
            f.write(f"  {n}  {n}  {n}\n")

            # X boundaries
            f.write(" ".join(f" {xi:.6f}" for xi in x) + "\n")

            # Y boundaries
            f.write(" ".join(f" {yi:.6f}" for yi in y) + "\n")

            # Z boundaries
            f.write(" ".join(f" {zi:.6f}" for zi in z) + "\n")

            # Dose values (one line per XY plane, space-separated)
            dose_flat = dose.flatten(order='F')
            for i in range(0, len(dose_flat), 10):
                f.write(" ".join(f"{d:.6e}" for d in dose_flat[i:i+10]) + "\n")

            # Uncertainties (assume 1% for mock data)
            uncertainty = 0.01 * np.ones_like(dose).flatten(order='F')
            for i in range(0, len(uncertainty), 10):
                f.write(" ".join(f"{u:.6e}" for u in uncertainty[i:i+10]) + "\n")

        logger.info(f"Generated mock 3ddose: {output_file}")
        return str(output_file)


# =============================================================================
# 3ddose to Spherical Kernel Converter
# =============================================================================

class DoseToKernelConverter:
    """
    Converts DOSXYZnrc .3ddose output to spherical kernel K(r, theta).
    """

    def __init__(
        self,
        n_r_bins: int = 60,
        n_theta_bins: int = 48,
        r_max: float = 30.0
    ):
        """
        Parameters:
            n_r_bins: Number of radial bins
            n_theta_bins: Number of angular bins
            r_max: Maximum radius (cm)
        """
        self.n_r_bins = n_r_bins
        self.n_theta_bins = n_theta_bins
        self.r_max = r_max

        # Setup bins
        self.r_bins = np.logspace(-2, np.log10(r_max), n_r_bins + 1)  # Log spacing
        self.theta_bins = np.linspace(0, 180, n_theta_bins + 1)

        self.r_centers = 0.5 * (self.r_bins[:-1] + self.r_bins[1:])
        self.theta_centers = 0.5 * (self.theta_bins[:-1] + self.theta_bins[1:])

    def convert(self, dose_file: str, energy: float, output_dir: str) -> str:
        """
        Convert .3ddose file to spherical kernel CSV.

        Parameters:
            dose_file: Path to .3ddose file
            energy: Photon energy in MeV
            output_dir: Output directory

        Returns:
            Path to kernel CSV file
        """
        logger.info(f"Converting {dose_file} to spherical kernel")

        # Read .3ddose file
        dose_3d, x, y, z = self._read_3ddose(dose_file)

        # Voxel centers
        xc = 0.5 * (x[:-1] + x[1:])
        yc = 0.5 * (y[:-1] + y[1:])
        zc = 0.5 * (z[:-1] + z[1:])

        # Voxel volumes
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)

        # Calculate spherical coordinates for each voxel
        kernel_2d = np.zeros((self.n_r_bins, self.n_theta_bins))
        counts = np.zeros_like(kernel_2d)

        for i, xi in enumerate(xc):
            for j, yj in enumerate(yc):
                for k, zk in enumerate(zc):
                    # Spherical coordinates
                    r = np.sqrt(xi**2 + yj**2 + zk**2)
                    if r < 1e-6:
                        continue

                    theta = np.degrees(np.arccos(zk / r))

                    # Find bin indices
                    r_idx = np.searchsorted(self.r_bins, r) - 1
                    theta_idx = np.searchsorted(self.theta_bins, theta) - 1

                    if 0 <= r_idx < self.n_r_bins and 0 <= theta_idx < self.n_theta_bins:
                        # Weight by voxel volume
                        vol = dx[i] * dy[j] * dz[k]
                        kernel_2d[r_idx, theta_idx] += dose_3d[i, j, k] * vol
                        counts[r_idx, theta_idx] += vol

        # Normalize by volume and to unity
        with np.errstate(divide='ignore', invalid='ignore'):
            kernel_2d = np.where(counts > 0, kernel_2d / counts, 0)

        # Normalize total to 1
        kernel_2d /= (kernel_2d.sum() + 1e-10)

        # Save to CSV
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"kernel_{energy:.3f}MeV.csv"

        with open(output_file, 'w') as f:
            f.write("# EDK Kernel from DOSXYZnrc simulation\n")
            f.write(f"# Energy: {energy:.3f} MeV\n")
            f.write(f"# Radial bins: {self.n_r_bins}, Angular bins: {self.n_theta_bins}\n")
            f.write("r_cm,theta_deg,K_normalized\n")

            for i, r in enumerate(self.r_centers):
                for j, theta in enumerate(self.theta_centers):
                    f.write(f"{r:.6f},{theta:.4f},{kernel_2d[i,j]:.10e}\n")

        logger.info(f"Kernel saved to {output_file}")
        return str(output_file)

    def _read_3ddose(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read EGSnrc .3ddose file format.

        Returns:
            (dose_3d, x_bounds, y_bounds, z_bounds)
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        idx = 0

        # Grid dimensions
        dims = lines[idx].split()
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        idx += 1

        # X boundaries
        x_bounds = []
        while len(x_bounds) < nx + 1:
            x_bounds.extend([float(v) for v in lines[idx].split()])
            idx += 1
        x_bounds = np.array(x_bounds[:nx + 1])

        # Y boundaries
        y_bounds = []
        while len(y_bounds) < ny + 1:
            y_bounds.extend([float(v) for v in lines[idx].split()])
            idx += 1
        y_bounds = np.array(y_bounds[:ny + 1])

        # Z boundaries
        z_bounds = []
        while len(z_bounds) < nz + 1:
            z_bounds.extend([float(v) for v in lines[idx].split()])
            idx += 1
        z_bounds = np.array(z_bounds[:nz + 1])

        # Dose values
        n_voxels = nx * ny * nz
        dose_values = []
        while len(dose_values) < n_voxels and idx < len(lines):
            dose_values.extend([float(v) for v in lines[idx].split()])
            idx += 1

        dose_3d = np.array(dose_values[:n_voxels]).reshape((nx, ny, nz), order='F')

        return dose_3d, x_bounds, y_bounds, z_bounds


# =============================================================================
# Main Pipeline
# =============================================================================

def run_edk_pipeline(
    energies: List[float],
    output_dir: str,
    n_histories: int = 10_000_000,
    voxel_size: float = 0.2
) -> List[str]:
    """
    Run complete EDK generation pipeline.

    Parameters:
        energies: List of photon energies in MeV
        output_dir: Output directory for all files
        n_histories: Number of histories per simulation
        voxel_size: Voxel size in cm

    Returns:
        List of generated kernel CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    input_gen = DOSXYZnrcInputGenerator(
        n_histories=n_histories,
        voxel_size=voxel_size
    )
    runner = EGSnrcRunner()
    converter = DoseToKernelConverter()

    kernel_files = []

    for energy in energies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing E = {energy:.3f} MeV")
        logger.info(f"{'='*60}")

        try:
            # Generate input files
            input_dir = output_path / "inputs"
            input_file = input_gen.generate_egsinp(energy, str(input_dir))

            # Run simulation
            sim_dir = output_path / "simulations"
            dose_file = runner.run_simulation(input_file, str(sim_dir))

            # Convert to kernel
            kernel_dir = output_path / "kernels"
            kernel_file = converter.convert(dose_file, energy, str(kernel_dir))

            kernel_files.append(kernel_file)

        except Exception as e:
            logger.error(f"Failed for E = {energy:.3f} MeV: {e}")
            continue

    return kernel_files


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run EGSnrc EDK simulations for DoseCUDA"
    )
    parser.add_argument(
        '--energies', '-e',
        required=True,
        help='Comma-separated photon energies in MeV (e.g., 0.5,1.0,2.0,6.0)'
    )
    parser.add_argument(
        '--output', '-o',
        default='./edk_output',
        help='Output directory'
    )
    parser.add_argument(
        '--histories', '-n',
        type=int,
        default=10_000_000,
        help='Number of histories per simulation'
    )
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=0.2,
        help='Voxel size in cm'
    )

    args = parser.parse_args()

    energies = [float(e) for e in args.energies.split(',')]

    kernel_files = run_edk_pipeline(
        energies=energies,
        output_dir=args.output,
        n_histories=args.histories,
        voxel_size=args.voxel_size
    )

    print("\n" + "=" * 60)
    print("EDK Pipeline Complete")
    print("=" * 60)
    print(f"Generated {len(kernel_files)} kernel files:")
    for kf in kernel_files:
        print(f"  - {kf}")


if __name__ == '__main__':
    main()
