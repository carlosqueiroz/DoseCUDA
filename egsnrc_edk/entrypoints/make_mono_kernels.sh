#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-/app/configs/config_10X.yaml}
WORKDIR=${WORKDIR:-/scratch/edk_work}
OUTDIR=${OUTDIR:-/out}
JOBS=${JOBS:-$(nproc)}
CSV_DEBUG=${CSV_DEBUG:-0}
EDK_EXEC=${EDK_EXEC:-/opt/EGSnrc/egs_homebin/docker/edknrc}

mkdir -p "$WORKDIR" "$OUTDIR" /scratch/logs

# Ensure EGSnrc env (disable errexit while sourcing to tolerate internal tests)
set +e
if [[ -f "$HEN_HOUSE/scripts/egsnrc_bashrc_additions" ]]; then
  # shellcheck source=/dev/null
  source "$HEN_HOUSE/scripts/egsnrc_bashrc_additions"
fi
set -e

# Use isolated EGS_HOME under workdir so outputs land in the mounted volume
export EGS_HOME="$WORKDIR/egs_home"
mkdir -p "$EGS_HOME/edknrc"
# Keep EGSnrc temp files inside workdir (so .egsdat não some em /tmp)
export EGS_TMP="${EGS_TMP:-$WORKDIR/tmp}"
mkdir -p "$EGS_TMP"

echo "[make_mono] generating inputs from $CONFIG"
python3 /app/scripts/generate_edknrc_inputs.py --config "$CONFIG" --work-dir "$WORKDIR/inputs"

echo "[make_mono] running EDKnrc batch with $JOBS jobs"
python3 /app/scripts/run_edknrc_batch.py \
  --config "$CONFIG" \
  --work-dir "$WORKDIR/inputs" \
  --log-dir /scratch/logs \
  --jobs "$JOBS" \
  --edk-exec "$EDK_EXEC"

echo "[make_mono] parsing outputs"
parse_args=(--config "$CONFIG" --work-dir "$WORKDIR/inputs" --out-dir "$WORKDIR/numpy")
if [[ "$CSV_DEBUG" == "1" ]]; then
  parse_args+=(--csv-dir "$OUTDIR/csv_debug")
fi
python3 /app/scripts/parse_edknrc_output.py "${parse_args[@]}"

echo "[make_mono] packing binary"
python3 /app/scripts/pack_kernels.py \
  --config "$CONFIG" \
  --numpy-dir "$WORKDIR/numpy" \
  --out "$OUTDIR/kernels_mono.bin" \
  --use-cumulative

echo "[make_mono] copying logs to $OUTDIR/logs"
mkdir -p "$OUTDIR/logs"
cp -r /scratch/logs/* "$OUTDIR/logs/" || true

echo "[make_mono] done -> $OUTDIR/kernels_mono.bin"
