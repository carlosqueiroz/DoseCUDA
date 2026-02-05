#!/bin/bash
# =============================================================================
# EGSnrc Kernel Factory - Main Entrypoint
# =============================================================================
#
# Generates Energy Deposition Kernels (EDK) for DoseCUDA using EGSnrc.
#
# Usage:
#   make_kernels.sh --energies 0.5,1.0,2.0,4.0,6.0,10.0 [options]
#   make_kernels.sh --energies 0.5,1.0,2.0,4.0,6.0,10.0 --spectrum spectrum_6MV.csv --beam 6MV
#   make_kernels.sh --help
#
# Options:
#   --energies, -e    Comma-separated monoenergetic values (MeV) (required)
#   --spectrum, -s    Spectral weights CSV for polyenergetic kernel (optional)
#   --beam, -b        Beam name for output (default: clinical)
#   --output, -o      Output directory (default: /output)
#   --histories, -n   Number of histories per simulation (default: 10M)
#   --angles          Number of output angles: 6 or 12 (default: 6)
#   --fff             Use FFF beam spectral approximation
#   --skip-sim        Skip EGSnrc simulation (use existing kernel files)
#   --help, -h        Show this help
#
# =============================================================================

set -e

# Default values
OUTPUT_DIR="/output"
N_HISTORIES=10000000
N_ANGLES=6
BEAM_NAME="clinical"
SKIP_SIM=false
USE_FFF=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

show_help() {
    cat << 'EOF'
EGSnrc Kernel Factory for DoseCUDA
===================================

Generates Energy Deposition Kernels (EDK) from EGSnrc Monte Carlo simulations
and processes them into the DoseCUDA kernel.csv format.

USAGE:
    make_kernels.sh --energies <energies> [options]

REQUIRED:
    --energies, -e <list>
        Comma-separated list of monoenergetic values in MeV.
        Example: --energies 0.2,0.5,1.0,2.0,4.0,6.0,10.0,15.0,20.0

OPTIONAL:
    --spectrum, -s <file>
        Path to spectral weights CSV file for polyenergetic kernel.
        Format: energy_MeV,weight
        If provided, creates a polyenergetic kernel by weighting.

    --beam, -b <name>
        Beam name for output folder (default: clinical).
        Examples: 6MV, 10MV_FFF, 6FFF

    --output, -o <dir>
        Output directory (default: /output).
        Creates subdirectories: kernels/, processed/, logs/

    --histories, -n <number>
        Number of Monte Carlo histories per energy (default: 10M).
        More histories = better statistics, longer runtime.

    --angles <6|12>
        Number of polar angles for DoseCUDA kernel (default: 6).
        Use 12 for SBRT with 1.5mm grid.

    --fff
        Use FFF beam spectral approximation (harder spectrum).
        Automatically generates FFF-like spectral weights.

    --skip-sim
        Skip EGSnrc simulation phase.
        Use existing kernel files in kernels/ directory.

    --help, -h
        Show this help message.

EXAMPLES:

    1. Generate monoenergetic kernels:
       make_kernels.sh -e 0.5,1.0,2.0,6.0,10.0 -o /output

    2. Generate 6MV clinical beam kernel:
       make_kernels.sh -e 0.2,0.5,1.0,2.0,4.0,6.0 \
                       -s spectrum_6MV.csv \
                       -b 6MV \
                       -o /output

    3. Generate 6FFF kernel with 12 angles for SBRT:
       make_kernels.sh -e 0.2,0.5,1.0,2.0,4.0,6.0 \
                       --fff \
                       -b 6MV_FFF \
                       --angles 12 \
                       -o /output

OUTPUT STRUCTURE:
    /output/
    ├── kernels/                    # Raw spherical kernels K(r,θ)
    │   ├── kernel_0.500MeV.csv
    │   ├── kernel_1.000MeV.csv
    │   └── ...
    ├── processed/                  # DoseCUDA-format kernels
    │   ├── 6MV/
    │   │   ├── kernel.csv          # Main kernel file
    │   │   └── kernel_validation.txt
    │   └── 6MV_FFF/
    │       └── kernel.csv
    └── logs/                       # Simulation logs
        └── edk_simulation.log

SPECTRAL WEIGHTS FILE FORMAT:
    energy_MeV,weight
    0.1,0.05
    0.5,0.15
    1.0,0.25
    2.0,0.20
    4.0,0.15
    6.0,0.20

    Notes:
    - Weights should sum to 1.0 (will be normalized if not)
    - Extract from IAEA phase space using energy histogram
    - FFF beams have harder spectra (more high-energy photons)

DOSECUDA INTEGRATION:
    Copy the generated kernel.csv to:
    DoseCUDA/lookuptables/photons/<Machine>/<Energy>/kernel.csv

    Example:
    cp /output/processed/6MV/kernel.csv \
       /path/to/DoseCUDA/lookuptables/photons/VarianTrueBeamHF/6MV/

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--energies)
            ENERGIES="$2"
            shift 2
            ;;
        -s|--spectrum)
            SPECTRUM_FILE="$2"
            shift 2
            ;;
        -b|--beam)
            BEAM_NAME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--histories)
            N_HISTORIES="$2"
            shift 2
            ;;
        --angles)
            N_ANGLES="$2"
            shift 2
            ;;
        --fff)
            USE_FFF=true
            shift
            ;;
        --skip-sim)
            SKIP_SIM=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$ENERGIES" ]; then
    log_error "Missing required argument: --energies"
    show_help
    exit 1
fi

# Validate angles
if [ "$N_ANGLES" != "6" ] && [ "$N_ANGLES" != "12" ]; then
    log_error "Invalid --angles value: $N_ANGLES (must be 6 or 12)"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"/{kernels,processed,logs,inputs,simulations}

# Log file
LOG_FILE="$OUTPUT_DIR/logs/edk_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

log_section "EGSnrc Kernel Factory for DoseCUDA"

echo "Configuration:"
echo "  Energies:    $ENERGIES MeV"
echo "  Output:      $OUTPUT_DIR"
echo "  Histories:   $N_HISTORIES"
echo "  Angles:      $N_ANGLES"
echo "  Beam:        $BEAM_NAME"
echo "  Spectrum:    ${SPECTRUM_FILE:-None (monoenergetic)}"
echo "  FFF mode:    $USE_FFF"
echo "  Skip sim:    $SKIP_SIM"
echo ""

# =============================================================================
# Step 1: Run EGSnrc Simulations (or skip)
# =============================================================================

if [ "$SKIP_SIM" = false ]; then
    log_section "Step 1: Running EGSnrc Simulations"

    # Check if EGSnrc is available
    if [ -z "$HEN_HOUSE" ]; then
        log_warn "HEN_HOUSE not set. Using mock kernel generation."
        export HEN_HOUSE=/opt/egsnrc/HEN_HOUSE
        export EGS_HOME=/opt/egsnrc/egs_home
    fi

    # Run simulation for each energy
    python3 /scripts/run_egsnrc_simulation.py \
        --energies "$ENERGIES" \
        --output "$OUTPUT_DIR" \
        --histories "$N_HISTORIES"

    log_info "Simulations complete. Raw kernels in $OUTPUT_DIR/kernels/"
else
    log_info "Skipping simulation (--skip-sim). Using existing kernel files."

    # Verify kernel files exist
    KERNEL_COUNT=$(ls "$OUTPUT_DIR/kernels/"kernel_*.csv 2>/dev/null | wc -l)
    if [ "$KERNEL_COUNT" -eq 0 ]; then
        log_error "No kernel files found in $OUTPUT_DIR/kernels/"
        log_error "Run without --skip-sim to generate kernels first."
        exit 1
    fi
    log_info "Found $KERNEL_COUNT kernel files."
fi

# =============================================================================
# Step 2: Process Monoenergetic Kernels
# =============================================================================

log_section "Step 2: Processing Monoenergetic Kernels"

python3 /scripts/kernel_processor.py mono \
    --kernel-dir "$OUTPUT_DIR/kernels" \
    --energies "$ENERGIES" \
    --output "$OUTPUT_DIR/processed/monoenergetic" \
    --n-angles "$N_ANGLES"

log_info "Monoenergetic kernels processed."

# =============================================================================
# Step 3: Generate Polyenergetic Kernel (if spectrum provided)
# =============================================================================

if [ -n "$SPECTRUM_FILE" ] || [ "$USE_FFF" = true ]; then
    log_section "Step 3: Creating Polyenergetic Kernel"

    if [ "$USE_FFF" = true ]; then
        # Generate FFF spectrum approximation
        log_info "Generating FFF spectrum approximation..."

        # Extract max energy from energy list
        MAX_E=$(echo "$ENERGIES" | tr ',' '\n' | sort -n | tail -1)

        # Create FFF spectrum file
        python3 << PYEOF
import numpy as np

max_e = $MAX_E
energies = np.linspace(0.1, max_e, 30)

# FFF spectrum: harder than filtered, forward-peaked bremsstrahlung
weights = (energies / max_e)**0.5 * np.exp(-energies / (max_e * 0.4))
weights /= weights.sum()

with open('$OUTPUT_DIR/spectrum_fff.csv', 'w') as f:
    f.write('energy_MeV,weight\n')
    for e, w in zip(energies, weights):
        f.write(f'{e:.4f},{w:.6f}\n')

print(f"Generated FFF spectrum: mean energy = {np.dot(energies, weights):.3f} MeV")
PYEOF

        SPECTRUM_FILE="$OUTPUT_DIR/spectrum_fff.csv"
    fi

    # Verify spectrum file exists
    if [ ! -f "$SPECTRUM_FILE" ]; then
        log_error "Spectrum file not found: $SPECTRUM_FILE"
        exit 1
    fi

    python3 /scripts/kernel_processor.py poly \
        --mono-dir "$OUTPUT_DIR/kernels" \
        --spectrum "$SPECTRUM_FILE" \
        --output "$OUTPUT_DIR/processed" \
        --beam-name "$BEAM_NAME" \
        --n-angles "$N_ANGLES"

    log_info "Polyenergetic kernel created: $BEAM_NAME"
fi

# =============================================================================
# Step 4: Validation and Sanity Checks
# =============================================================================

log_section "Step 4: Validation and Sanity Checks"

python3 << PYEOF
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

output_dir = Path("$OUTPUT_DIR/processed")
beam_name = "$BEAM_NAME"
n_angles = $N_ANGLES

# Find all kernel.csv files
kernel_files = list(output_dir.rglob("kernel.csv"))

if not kernel_files:
    print("ERROR: No kernel.csv files found!")
    sys.exit(1)

print(f"Found {len(kernel_files)} kernel file(s):")

all_passed = True

for kf in kernel_files:
    print(f"\n  Validating: {kf}")

    try:
        df = pd.read_csv(kf)

        # Check columns
        required = ['theta', 'Am', 'am', 'Bm', 'bm', 'ray_length']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"    FAIL: Missing columns: {missing}")
            all_passed = False
            continue

        # Check row count
        if len(df) != n_angles:
            print(f"    WARN: Expected {n_angles} rows, got {len(df)}")

        # Check parameter ranges
        checks = [
            ('Am > 0', (df['Am'] > 0).all()),
            ('am > 0', (df['am'] > 0).all()),
            ('Bm > 0', (df['Bm'] > 0).all()),
            ('bm > 0', (df['bm'] > 0).all()),
            ('ray_length > 0', (df['ray_length'] > 0).all()),
            ('am > bm', (df['am'] > df['bm']).all()),
            ('theta increasing', (df['theta'].diff()[1:] > 0).all()),
        ]

        for name, passed in checks:
            status = "PASS" if passed else "WARN"
            print(f"    [{status}] {name}")
            if not passed:
                all_passed = False

        # Summary statistics
        print(f"    Am range: {df['Am'].min():.4e} - {df['Am'].max():.4e}")
        print(f"    am range: {df['am'].min():.4f} - {df['am'].max():.4f}")
        print(f"    ray_length: {df['ray_length'].mean():.2f} cm (mean)")

    except Exception as e:
        print(f"    FAIL: {e}")
        all_passed = False

if all_passed:
    print("\n✓ All validations passed!")
else:
    print("\n⚠ Some validations failed. Check warnings above.")
PYEOF

# =============================================================================
# Step 5: Summary and Next Steps
# =============================================================================

log_section "Summary"

echo ""
echo "Generated files:"
find "$OUTPUT_DIR/processed" -name "kernel.csv" -exec echo "  {}" \;

echo ""
echo "To use with DoseCUDA, copy kernel files to:"
echo "  DoseCUDA/lookuptables/photons/<Machine>/<Energy>/kernel.csv"
echo ""
echo "Example:"
echo "  cp $OUTPUT_DIR/processed/$BEAM_NAME/kernel.csv \\"
echo "     /path/to/DoseCUDA/lookuptables/photons/VarianTrueBeamHF/$BEAM_NAME/"
echo ""
echo "Log file: $LOG_FILE"
echo ""

log_info "Kernel Factory complete!"
