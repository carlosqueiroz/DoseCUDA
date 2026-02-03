#!/bin/bash
set -euo pipefail

# Seta ambiente Geant4 caso o shell não tenha carregado /etc/profile.d
if [ -f /etc/profile.d/geant4.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/geant4.sh
fi

macro_default="/workflow/topas_inputs/TrueBeam_10X_WaterPhantom.txt"
macro_path="${1:-$macro_default}"

if [ ! -f "$macro_path" ]; then
  echo "ERRO: macro TOPAS não encontrada: $macro_path" >&2
  exit 1
fi

echo "[run.sh] Usando macro: $macro_path"
echo "[run.sh] Saída em: /output"
cd /output

exec topas "$macro_path"
