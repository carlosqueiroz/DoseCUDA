#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-/app/configs/config_10X.yaml}
SPECTRUM=${2:?Provide spectrum CSV path}
OUTDIR=${OUTDIR:-/out}

set +e
if [[ -f "$HEN_HOUSE/scripts/egsnrc_bashrc_additions" ]]; then
  # shellcheck source=/dev/null
  source "$HEN_HOUSE/scripts/egsnrc_bashrc_additions"
fi
set -e

python3 /app/scripts/build_poly_kernel.py \
  --config "$CONFIG" \
  --numpy-dir "/scratch/edk_work/numpy" \
  --spectrum "$SPECTRUM" \
  --out "$OUTDIR/kernel_poly.bin" \
  --cumulative

echo "[make_poly] done -> $OUTDIR/kernel_poly.bin"
