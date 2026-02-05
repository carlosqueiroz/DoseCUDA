# EGSnrc/EDKnrc EDK pipeline (CCCS-ready)

Pipeline para gerar **kernels de deposição de energia (EDK) monoenergéticos em água**, no formato clássico de cones colapsados (θ × cascas radiais), normalizados pela **energia incidente que interage na origem**. CUDA/DoseCUDA não é tocado aqui; o alvo é produzir arquivos prontos para ingestão posterior pelo CCCS.

## O que vem pronto
- Docker Ubuntu que compila **EGSnrc v2025a + EDKnrc**, com opção de usar tarball local em `egsnrc_edk/sources/`.
- Entrypoints: `make_mono_kernels.sh` (banco mono) e `make_poly_kernel.sh` (opcional, espectro → kernel poli).
- Scripts Python: geram entradas EDKnrc, executam em lote, fazem parsing, normalizam, checam sanidade, empacotam binário/CSV.
- Configs YAML: grade padrão de energias 10X (0.2–20 MeV), 48 cones de 3,75°, **radial segmentado** para SBRT (0–2 cm @0,5 mm; 2–5 cm @1 mm; 5–20 cm @2,5 mm).

## Build
```bash
docker build -t egs-edk -f egsnrc_edk/Dockerfile .
```
- Offline: coloque o tarball `EGSnrc-v2025a.tar.gz` (ou equivalente) em `egsnrc_edk/sources/` antes do build. O Dockerfile usa o tarball local se existir; caso contrário, baixa de `${EGSNRC_REPO}`.
- Outros tags: `--build-arg EGSNRC_VERSION=v2024` etc.

## Rodar (banco mono 10X)
```bash
docker run --rm \
  -v $(pwd)/out:/out \
  egs-edk /app/make_mono_kernels.sh /app/configs/config_10X.yaml
```
Env overrides: `WORKDIR` (scratch, default `/scratch/edk_work`), `OUTDIR` (`/out`), `JOBS` (paralelo), `CSV_DEBUG=1` (grava CSVs).

Saída em `$OUTDIR`:
- `kernels_mono.bin` – binário float32 LE, header + payload E-major.
- `csv_debug/` – opcional, `E_MeV,theta_deg,r_cm,value,kind` (diff e cum).
- Logs por energia em `$OUTDIR/logs`.

Smoke test (histórias reduzidas para validar o fluxo):
```bash
docker run --rm -v $(pwd)/egsnrc_edk/out_smoke:/out \
  egs-edk /app/make_mono_kernels.sh /app/configs/config_smoke.yaml
```
Com 20k histórias a integral fica ~0.5 (alerta esperado); a cadeia geração→parse→pack roda completa.
Sanity extra:
```bash
docker run --rm -v $(pwd)/egsnrc_edk/out_smoke:/out egs-edk \
  python3 /app/scripts/sanity_check.py /out/kernels_mono.bin
```

## Discretização e normalização
- **Ângulo (θ):** 48 cones, `Δθ = 3.75°`, cobrindo 0–180°. Se seu CCCS usa 0–90°, dobre a simetria downstream ou mude `theta_max_deg`.
- **Raio (segmentado default 10X):**  
  - 0–2.0 cm, `dr=0.5 mm` (40 bins)  
  - 2.0–5.0 cm, `dr=1.0 mm` (30 bins)  
  - 5.0–20.0 cm, `dr=2.5 mm` (60 bins)  
  Total `nR=130`. `r_edges` (armazenados no header) incluem 0 e as bordas externas `r_edges[1:]`. Para uniforme, use `radial.mode: uniform` no YAML.
- **Cumulativo e diferencial:** salvamos `Kcum(θ,r)` (monotônico) e `Kdiff(θ,r)`. Recuperar `Kdiff` por diferença radial: `Kdiff_i = Kcum(r_out) - Kcum(r_in)` (Cho 2012).
- **Normalização:** padrão `per_incident_energy_interacting_at_origin` (Mackie 1988). Ou seja, `Σ Kdiff ≈ 1` (admite perdas por cutoffs/fuga). Gravado no header.
- **Grade de energias (10X default, sobrescrevível por CLI/YAML):**
  ```
  0.20, 0.30, 0.40, 0.50, 0.60, 0.80,
  1.00, 1.25, 1.50, 1.75, 2.00,
  2.50, 3.00, 3.50, 4.00, 4.50, 5.00,
  6.00, 7.00, 8.00, 9.00, 10.0,
  12.0, 14.0, 16.0, 18.0, 20.0
  ```
  Editável via YAML ou CLI (`--energies`).

## Formato do binário `kernels_mono.bin` (versão 2)
Header LE:
- `magic[8] = "EDKCCCS\0"`, `uint32 version=2`
- `uint32 nE, nTheta, nR`
- `float32 energies[nE]` (MeV)
- `float32 theta_edges[nTheta+1]` (graus)
- `float32 dr_cm, r0_cm, rmax_cm`
- `uint32 flags` (bit0 = payload cumulativo)
- `char layout[16]` (ASCII, ex. `E-major`)
- `char normalization[64]` (ASCII, ex. `per_incident_energy_interacting_at_origin`)
- `float32 r_edges[nR+1]` (cm)
- payload `float32 k[E][theta][r]` em ordem C (E-major).

## Checagens de sanidade
- NaN/Inf barred; kdiff negativo grave ⇒ erro; negativos muito pequenos são clipados em 0.
- Cumulativo monotônico em `r`.
- Integral `Σ Kdiff` reportada; alerta se fora de [0.95, 1.05] (com poucas histórias, espere valores <1).

## Fluxo interno (resumido)
1. `generate_edknrc_inputs.py` → um input por energia (`ANGLES` usa um único Δθ, `CAVITY ZONES=0`, raios exportados como lista explícita com quebras de linha seguras para EDKnrc).
2. `run_edknrc_batch.py` roda EDKnrc em paralelo, com `EGS_HOME=$WORKDIR/egs_home`; cada energia grava `.egsdat/.errors` e copia de volta para o diretório da energia.
3. `parse_edknrc_output.py` lê `.egsdat` (ou `.egslst`, se existir), divide por `N_hist` e por `E_inc` → `Kdiff` já normalizado; constrói `Kcum`, valida (NaN/Inf, monotonia, integral), salva `.npz/.npy` e CSV debug opcional.
4. `pack_kernels.py` empacota header+payload (`kcum` por default) em `kernels_mono.bin`.
5. `sanity_check.py` (invocado no entrypoint, pode ser rodado isolado) relê o binário e checa integral, NaN/Inf, monotonicidade do cumulativo e não‑negatividade.

## Poly (opcional)
```bash
docker run --rm \
  -v $(pwd)/out:/out -v $(pwd)/spectra:/spectra \
  egs-edk /app/make_poly_kernel.sh /app/configs/config_10X.yaml /spectra/beam.csv
```
`beam.csv` com colunas `E_MeV,weight`. Interpola pesos para a grade e compõe `kernel_poly.bin` (mesmo header, payload com `nE=1`).

## Conceito rápido (para README clínico)
- EDK ≠ PHSP: aqui o fóton é forçado a interagir na origem e propagamos a cascata em água homogênea → kernel de espalhamento de energia.
- CCCS: dose é a soma, por cone colapsado, de `TERMA × K(θ,r) × fatores de densidade/volume` (ver Cho 2012). `Kcum` melhora estabilidade no miolo; `Kdiff` é obtido por diferença radial.

## Config knobs
- `ncones_theta`, `theta_max_deg`, `dr_mm`, `rmax_cm`, `ncases`, `ecut/pcut`, `rng_seeds`, `normalization`.
- CLI overrides: `generate_edknrc_inputs.py --energies ... --ncones ... --dr-mm ... --rmax-cm ...`.

## Estrutura de pastas
```
egsnrc_edk/
  Dockerfile
  configs/ (config_10X.yaml, config_smoke.yaml)
  entrypoints/ (make_mono_kernels.sh, make_poly_kernel.sh)
  scripts/ (generate_edknrc_inputs.py, run_edknrc_batch.py,
            parse_edknrc_output.py, pack_kernels.py, build_poly_kernel.py, common.py)
  examples/ (placeholder spectrum CSV)
  sources/ (coloque aqui EGSnrc-<versao>.tar.gz para build offline)
```
