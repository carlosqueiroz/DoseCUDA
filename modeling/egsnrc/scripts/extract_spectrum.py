#!/usr/bin/env python3
"""
=============================================================================
Spectral Weight Extractor for DoseCUDA Kernel Generation
=============================================================================

Extracts photon energy spectra from IAEA phase space files for use in
polyenergetic kernel generation.

IAEA Phase Space Format:
- Binary file with particle records
- Each record contains: type, energy, position, direction, weight
- Reference: IAEA Technical Reports Series No. 461

This script reads IAEA PHSP files and generates spectral weight files
in the format required by the kernel processor.

Author: DoseCUDA Team
"""

import numpy as np
import struct
from pathlib import Path
import argparse
import logging
from typing import Tuple, Dict, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# IAEA Phase Space Reader
# =============================================================================

class IAEAPhaseSpaceReader:
    """
    Reader for IAEA phase space files.

    IAEA PHSP format consists of:
    1. Header file (.IAEAheader) - ASCII with metadata
    2. Phase space file (.IAEAphsp) - Binary particle records
    """

    def __init__(self, phsp_path: str):
        """
        Parameters:
            phsp_path: Path to PHSP file (without extension)
        """
        self.base_path = Path(phsp_path).with_suffix('')
        self.header_path = self.base_path.with_suffix('.IAEAheader')
        self.phsp_path = self.base_path.with_suffix('.IAEAphsp')

        # Check for alternative naming
        if not self.header_path.exists():
            self.header_path = Path(str(self.base_path) + '.IAEAheader')
        if not self.phsp_path.exists():
            self.phsp_path = Path(str(self.base_path) + '.IAEAphsp')

        self.header = {}
        self.record_size = 0
        self.n_particles = 0
        self.extra_floats = 0
        self.extra_ints = 0

    def read_header(self) -> Dict:
        """Read IAEA header file and extract metadata."""

        if not self.header_path.exists():
            logger.warning(f"Header file not found: {self.header_path}")
            # Try alternative extensions
            for ext in ['.IAEAheader', '.header', '_header.txt']:
                alt_path = self.base_path.with_suffix(ext)
                if alt_path.exists():
                    self.header_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"No header file found for {self.base_path}")

        logger.info(f"Reading header: {self.header_path}")

        with open(self.header_path, 'r') as f:
            content = f.read()

        # Parse key-value pairs
        for line in content.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ''
                self.header[key] = value

        # Extract key parameters
        try:
            self.n_particles = int(self.header.get('PARTICLES', self.header.get('ORIG_HISTORIES', 0)))
            self.extra_floats = int(self.header.get('EXTRA_FLOATS', 0))
            self.extra_ints = int(self.header.get('EXTRA_INTS', 0))
        except (ValueError, TypeError):
            logger.warning("Could not parse particle count from header")

        # Calculate record size
        # Standard IAEA record: type(1) + E(4) + x,y,z(12) + u,v,w(12) + weight(4) = 33 bytes minimum
        # Plus extra floats and ints
        self.record_size = 33 + 4 * self.extra_floats + 4 * self.extra_ints

        return self.header

    def read_energies(self, max_particles: int = None) -> np.ndarray:
        """
        Read photon energies from phase space file.

        Parameters:
            max_particles: Maximum number of particles to read (None = all)

        Returns:
            Array of photon energies in MeV
        """
        if not self.phsp_path.exists():
            raise FileNotFoundError(f"PHSP file not found: {self.phsp_path}")

        if not self.header:
            self.read_header()

        logger.info(f"Reading PHSP: {self.phsp_path}")

        energies = []
        weights = []

        with open(self.phsp_path, 'rb') as f:
            count = 0

            while True:
                if max_particles and count >= max_particles:
                    break

                # Read particle type (1 byte: 1=photon, 2=electron, 3=positron)
                type_byte = f.read(1)
                if not type_byte:
                    break

                particle_type = struct.unpack('b', type_byte)[0]

                # Read energy (4 bytes, float)
                energy_bytes = f.read(4)
                if len(energy_bytes) < 4:
                    break
                energy = struct.unpack('f', energy_bytes)[0]

                # Read position (3 floats = 12 bytes)
                pos_bytes = f.read(12)
                if len(pos_bytes) < 12:
                    break

                # Read direction (3 floats = 12 bytes)
                dir_bytes = f.read(12)
                if len(dir_bytes) < 12:
                    break

                # Read weight (4 bytes, float)
                weight_bytes = f.read(4)
                if len(weight_bytes) < 4:
                    break
                weight = struct.unpack('f', weight_bytes)[0]

                # Skip extra floats and ints
                f.read(4 * self.extra_floats + 4 * self.extra_ints)

                # Only count photons (type = 1 or positive)
                if particle_type == 1 or particle_type > 0:
                    energies.append(abs(energy))
                    weights.append(abs(weight))

                count += 1

                if count % 1000000 == 0:
                    logger.info(f"  Read {count:,} particles, {len(energies):,} photons")

        energies = np.array(energies)
        weights = np.array(weights)

        logger.info(f"Total particles read: {count:,}")
        logger.info(f"Photons extracted: {len(energies):,}")
        logger.info(f"Energy range: {energies.min():.4f} - {energies.max():.4f} MeV")

        return energies, weights


# =============================================================================
# Spectral Analysis
# =============================================================================

class SpectralAnalyzer:
    """
    Analyzes photon energy spectrum and generates binned weights.
    """

    def __init__(self, energies: np.ndarray, weights: np.ndarray = None):
        """
        Parameters:
            energies: Array of photon energies (MeV)
            weights: Optional array of particle weights
        """
        self.energies = energies
        self.weights = weights if weights is not None else np.ones_like(energies)

    def compute_histogram(
        self,
        n_bins: int = 50,
        e_min: float = None,
        e_max: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute energy histogram.

        Parameters:
            n_bins: Number of energy bins
            e_min: Minimum energy (default: min of data)
            e_max: Maximum energy (default: max of data)

        Returns:
            (bin_centers, normalized_weights)
        """
        if e_min is None:
            e_min = max(0.01, self.energies.min())
        if e_max is None:
            e_max = self.energies.max()

        # Create bins
        bin_edges = np.linspace(e_min, e_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Compute weighted histogram
        hist, _ = np.histogram(self.energies, bins=bin_edges, weights=self.weights)

        # Normalize to sum = 1
        hist = hist / hist.sum()

        return bin_centers, hist

    def get_statistics(self) -> Dict:
        """Compute spectrum statistics."""
        weighted_mean = np.average(self.energies, weights=self.weights)
        weighted_std = np.sqrt(np.average((self.energies - weighted_mean)**2, weights=self.weights))

        return {
            'mean_energy': weighted_mean,
            'std_energy': weighted_std,
            'min_energy': self.energies.min(),
            'max_energy': self.energies.max(),
            'n_photons': len(self.energies),
            'total_weight': self.weights.sum(),
        }

    def save_spectrum(self, output_path: str, n_bins: int = 50):
        """
        Save spectrum to CSV file.

        Parameters:
            output_path: Path to output CSV
            n_bins: Number of energy bins
        """
        bin_centers, weights = self.compute_histogram(n_bins)
        stats = self.get_statistics()

        df = pd.DataFrame({
            'energy_MeV': bin_centers,
            'weight': weights
        })

        # Add header comments
        with open(output_path, 'w') as f:
            f.write(f"# Spectral weights extracted from IAEA PHSP\n")
            f.write(f"# Mean energy: {stats['mean_energy']:.4f} MeV\n")
            f.write(f"# Energy range: {stats['min_energy']:.4f} - {stats['max_energy']:.4f} MeV\n")
            f.write(f"# Number of photons: {stats['n_photons']}\n")

        df.to_csv(output_path, mode='a', index=False)

        logger.info(f"Spectrum saved to: {output_path}")
        logger.info(f"  Mean energy: {stats['mean_energy']:.4f} MeV")
        logger.info(f"  {n_bins} energy bins")


# =============================================================================
# Alternative: Read from TOPAS/Geant4 Phase Space
# =============================================================================

def read_topas_phsp(phsp_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read energies from TOPAS phase space ASCII output.

    TOPAS can output phase space in various formats including ASCII.
    This reads the common ASCII format with columns for particle properties.
    """
    logger.info(f"Reading TOPAS PHSP: {phsp_path}")

    # Try to determine format from file
    with open(phsp_path, 'r') as f:
        first_lines = [f.readline() for _ in range(10)]

    # Check if it's a CSV or space-separated
    if ',' in first_lines[0]:
        df = pd.read_csv(phsp_path, comment='#')
    else:
        df = pd.read_csv(phsp_path, delim_whitespace=True, comment='#')

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Find energy column
    energy_col = None
    for col in ['energy', 'e', 'kinetic_energy', 'ek']:
        if col in df.columns:
            energy_col = col
            break

    if energy_col is None:
        raise ValueError(f"Could not find energy column. Columns: {df.columns.tolist()}")

    # Find weight column
    weight_col = None
    for col in ['weight', 'w', 'statistical_weight']:
        if col in df.columns:
            weight_col = col
            break

    energies = df[energy_col].values
    weights = df[weight_col].values if weight_col else np.ones_like(energies)

    # Filter photons if particle type column exists
    for col in ['particle_type', 'type', 'pdg']:
        if col in df.columns:
            # Photon PDG code is 22
            photon_mask = (df[col] == 22) | (df[col] == 1) | (df[col].astype(str).str.lower() == 'gamma')
            energies = energies[photon_mask]
            weights = weights[photon_mask]
            break

    logger.info(f"Read {len(energies)} photon energies")

    return energies, weights


# =============================================================================
# Generate Standard Beam Spectra
# =============================================================================

def generate_standard_spectrum(
    beam_type: str,
    output_path: str,
    n_bins: int = 50
):
    """
    Generate approximate spectrum for standard beam types.

    These are analytical approximations useful when PHSP data is not available.

    Parameters:
        beam_type: One of '6MV', '10MV', '15MV', '6FFF', '10FFF'
        output_path: Path to output CSV
        n_bins: Number of energy bins
    """
    logger.info(f"Generating standard spectrum for {beam_type}")

    # Beam parameters (approximate)
    beam_params = {
        '6MV': {'max_e': 6.0, 'mean_e': 2.0, 'filter': 'flattened'},
        '10MV': {'max_e': 10.0, 'mean_e': 3.5, 'filter': 'flattened'},
        '15MV': {'max_e': 15.0, 'mean_e': 5.0, 'filter': 'flattened'},
        '18MV': {'max_e': 18.0, 'mean_e': 6.0, 'filter': 'flattened'},
        '6FFF': {'max_e': 6.0, 'mean_e': 2.5, 'filter': 'fff'},
        '10FFF': {'max_e': 10.0, 'mean_e': 4.5, 'filter': 'fff'},
    }

    if beam_type not in beam_params:
        raise ValueError(f"Unknown beam type: {beam_type}. Available: {list(beam_params.keys())}")

    params = beam_params[beam_type]
    max_e = params['max_e']
    mean_e = params['mean_e']
    is_fff = params['filter'] == 'fff'

    # Generate energy bins
    energies = np.linspace(0.1, max_e, n_bins)

    if is_fff:
        # FFF spectrum: harder, more high-energy photons
        # Approximate Bremsstrahlung shape without filter hardening
        weights = (energies / max_e)**0.3 * np.exp(-energies / (max_e * 0.5))
    else:
        # Flattened beam: filtered spectrum
        # Softer due to flattening filter absorption of low-energy photons
        weights = (energies / max_e)**0.8 * np.exp(-energies / (mean_e * 0.8))

    # Normalize
    weights /= weights.sum()

    # Calculate actual mean
    actual_mean = np.dot(energies, weights)

    # Save
    df = pd.DataFrame({
        'energy_MeV': energies,
        'weight': weights
    })

    with open(output_path, 'w') as f:
        f.write(f"# Standard {beam_type} spectrum (analytical approximation)\n")
        f.write(f"# Max energy: {max_e:.1f} MeV\n")
        f.write(f"# Mean energy: {actual_mean:.3f} MeV\n")
        f.write(f"# Filter type: {params['filter']}\n")
        f.write(f"# WARNING: This is an approximation. Use PHSP data for clinical accuracy.\n")

    df.to_csv(output_path, mode='a', index=False)

    logger.info(f"Spectrum saved: {output_path}")
    logger.info(f"  Mean energy: {actual_mean:.3f} MeV")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract photon spectrum from phase space files"
    )

    subparsers = parser.add_subparsers(dest='command', help='Mode')

    # IAEA PHSP extraction
    iaea_parser = subparsers.add_parser('iaea', help='Extract from IAEA PHSP')
    iaea_parser.add_argument('--phsp', required=True, help='Path to IAEA PHSP file (without extension)')
    iaea_parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    iaea_parser.add_argument('--bins', '-n', type=int, default=50, help='Number of energy bins')
    iaea_parser.add_argument('--max-particles', type=int, help='Max particles to read')

    # TOPAS PHSP extraction
    topas_parser = subparsers.add_parser('topas', help='Extract from TOPAS PHSP')
    topas_parser.add_argument('--phsp', required=True, help='Path to TOPAS PHSP file')
    topas_parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    topas_parser.add_argument('--bins', '-n', type=int, default=50, help='Number of energy bins')

    # Standard spectrum generation
    standard_parser = subparsers.add_parser('standard', help='Generate standard beam spectrum')
    standard_parser.add_argument('--beam', required=True,
                                 choices=['6MV', '10MV', '15MV', '18MV', '6FFF', '10FFF'],
                                 help='Beam type')
    standard_parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    standard_parser.add_argument('--bins', '-n', type=int, default=50, help='Number of energy bins')

    args = parser.parse_args()

    if args.command == 'iaea':
        reader = IAEAPhaseSpaceReader(args.phsp)
        reader.read_header()
        energies, weights = reader.read_energies(args.max_particles)

        analyzer = SpectralAnalyzer(energies, weights)
        analyzer.save_spectrum(args.output, args.bins)

    elif args.command == 'topas':
        energies, weights = read_topas_phsp(args.phsp)

        analyzer = SpectralAnalyzer(energies, weights)
        analyzer.save_spectrum(args.output, args.bins)

    elif args.command == 'standard':
        generate_standard_spectrum(args.beam, args.output, args.bins)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
