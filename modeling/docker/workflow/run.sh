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

# Escolhe número de threads: TOPAS_THREADS (se setado) ou nproc do container
threads="${TOPAS_THREADS:-$(nproc)}"
export G4FORCENUMBEROFTHREADS="$threads"
export OMP_NUM_THREADS="$threads"

# Cria macro temporário injetando NumberOfThreads, sem tocar o original
tmp_macro=$(mktemp /tmp/topas_macro.XXXX.txt)
if grep -Eq '^i:Ts/(Run/)?NumberOfThreads' "$macro_path"; then
  # Normaliza para a sintaxe suportada pelo TOPAS (i:Ts/NumberOfThreads)
  sed -E "s#^i:Ts/(Run/)?NumberOfThreads *= *[-0-9]+#i:Ts/NumberOfThreads = ${threads}#g" "$macro_path" > "$tmp_macro"
else
  {
    printf 'i:Ts/NumberOfThreads = %s\n' "$threads"
    cat "$macro_path"
  } > "$tmp_macro"
fi

echo "[run.sh] Threads solicitadas: $threads"
exec topas "$tmp_macro"
