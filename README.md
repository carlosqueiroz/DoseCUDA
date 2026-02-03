# DoseCUDA
# DoseCUDA — GPU Collapsed-Cone Convolution (CCC) para Linux

**Resumo:** DoseCUDA é um motor de cálculo de dose fotônica baseado em CUDA (NVIDIA) que implementa o algoritmo Collapsed-Cone Convolution/Superposition (TERMA + correções de heterogeneidade). Este repositório contém o núcleo CUDA em `DoseCUDA/dose_kernels` e a camada Python que prepara entradas, carrega modelos e exporta resultados.

**Objetivo deste README:** fornecer instruções completas e detalhadas para compilar, instalar e executar DoseCUDA em Linux com GPU NVIDIA, explicar a API mínima necessária e listar checagens e soluções para problemas comuns.

**Importante:** o motor atual requer GPUs NVIDIA compatíveis com o CUDA Toolkit; não há suporte CUDA para AMD/Intel no código atualmente.

**Principais arquivos e locais:**
- **`DoseCUDA/dose_kernels/dosemodule.cu`**: binding Python → CUDA, exporta `photon_dose_cuda(model, volume, cp, gpu_id)`.
- **`DoseCUDA/dose_kernels/IMRTClasses.cu`**, **`CudaClasses.cu`**, **`TextureClasses.cu`**: implementação CUDA (ray-trace, TERMA, CCC kernel).
- **`setup.py`** / **`pyproject.toml`**: configuração de build via scikit-build (CMake) e `pip`.
- **`DoseCUDA/plan_imrt.py`**, **`DoseCUDA/plan.py`**, **`DoseCUDA/grid_utils.py`**, **`DoseCUDA/rtstruct.py`**: API Python para montar grids, resample CT, rasterizar RTSTRUCT e orquestrar cálculo.

**Versão mínima de ferramentas**
- **CMake** ≥ 3.15
- **Python** 3.6+ (suporta 3.12 no ambiente presente)
- **CUDA Toolkit**: use a versão compatível com seu driver e GPU (ex.: 11.x ou 12.x conforme seu stack). Testado com toolkits recentes — verifique `nvcc --version`.

**Recomendações de sistema**
- Driver NVIDIA instalado e funcionando (ver `nvidia-smi`).
- NVCC (CUDA compiler) disponível no PATH.
- Compilador host compatível com `nvcc` (GCC ou `clang++` apropriado). Se necessário, use `CUDAHOSTCXX` (ex.: `export CUDAHOSTCXX=/usr/bin/clang++`).

**Passo a passo: preparar ambiente Linux (exemplo Ubuntu/Debian)**

1) Verificações prévias (GPU / CUDA / compilador):

```bash
# verifique driver e GPUs
nvidia-smi

# versão do nvcc
nvcc --version

# versão do compilador C++ (gcc/clang)
gcc --version || clang --version
```

2) Recomenda-se um virtualenv isolado (opcional, mas útil):

```bash
sudo apt-get install -y cmake
cd /home/rt/scripts/DoseCUDA
pip install . --no-build-isolation

python3 -m venv dosecuda-env
source dosecuda-env/bin/activate
python -m pip install -U pip setuptools wheel


```

3) Dependências Python (instalar antes do build):

```bash
python -m pip install numpy pandas pydicom SimpleITK matplotlib scikit-build-core
```

4) Compilar e instalar o pacote (CMake + CUDA via scikit-build):

```bash
# Caso nvcc exija um host compiler específico
export CUDAHOSTCXX=/usr/bin/g++

# Instalação (build + instalação via pip)
python -m pip install .
```

Observações:
- O build usa `scikit-build` que invoca CMake nos sources em `DoseCUDA/dose_kernels` (veja `setup.py` e `pyproject.toml`).
- Se encontrar erros do `nvcc` referentes ao host compiler, a variável `CUDAHOSTCXX` costuma resolver.

5) Teste rápido (phantom/example)

```bash
python tests/test_phantom_imrt.py
```

Se o módulo nativo `dose_kernels` não estiver disponível, o import falhará; verifique que a build criou e instalou o módulo Python (`dose_kernels`), tipicamente dentro do seu ambiente (`site-packages/DoseCUDA/dose_kernels.*.so`).

Como usar a API Python (exemplo resumido)

- Função-chave exportada pelo módulo nativo: `dose_kernels.photon_dose_cuda(model, volume, control_point, gpu_id)`.
- Em alto nível, o fluxo é:
  - Carregar CT → `DoseGrid` / `IMRTDoseGrid` (ver `DoseCUDA/plan.py`).
  - Garantir voxels isotrópicos (usar `grid_utils.resample_ct_to_isotropic`).
  - Construir `VolumeObject` com `voxel_data`, `origin`, `spacing` (todos `np.single`).
  - Montar `IMRTPhotonEnergy` (beam model) e `IMRTControlPoint` (CP) com arrays `np.single` e chamadas apropriadas.
  - Chamar `IMRTDoseGrid.computeIMRTPlan(plan, gpu_id=0)` que orquestra `dose_kernels.photon_dose_cuda(...)` internamente.

Exemplo mínimo (pseudo-code):

```python
from DoseCUDA.plan_imrt import IMRTPlan, IMRTDoseGrid

plan = IMRTPlan(machine_name='VarianTrueBeamHF')
dose_grid = IMRTDoseGrid()
dose_grid.loadCTNRRD('/path/to/ct.nrrd')
dose_grid.resampleCTfromSpacing(2.5)  # garante isotropia
plan.load_from_rtplan_or_build(...)   # preparar beams e CPs
dose_grid.computeIMRTPlan(plan, gpu_id=0)
print('Dose calculada:', dose_grid.dose.shape)
```

Entrada do Beam Model — requisitos críticos
- `profile_radius`, `profile_intensities`, `profile_softening`: arrays 1-D (floats).
- `spectrum_attenuation_coefficients`, `spectrum_primary_weights`, `spectrum_scatter_weights`: arrays 1-D (mesmo comprimento).
- `kernel`: array 1-D com 36 floats (layout crítico descrito abaixo).
- `use_depth_dependent_kernel`: ativa parâmetros adicionais `kernel_depths` e `kernel_params` (n_depths × 24 floats).
- Transmissões: `jaw_transmission`, `mlc_transmission` (0..1).
- `mu_calibration`: fator de escala para dose absoluta.

Kernel layout (CRÍTICO para CUDA)

- O CUDA espera a ordem agrupada por coluna: `[theta(6), Am(6), am(6), Bm(6), bm(6), ray_length(6)]`.
- A carga do CSV nos loaders já monta este layout; se criar manualmente, siga exatamente essa ordem.

Restrições importantes do motor
- Voxel spacing deve ser isotrópico; se não for, use `resample_ct_to_isotropic` antes de calcular.
- O uso de `SimpleITK` é necessário para reamostragem e operações DICOM mais robustas.

Seleção de GPU e múltiplas GPUs
- A API aceita `gpu_id` (inteiro) e chama `cudaSetDevice(gpu_id)` internamente (`IMRTClasses.cu`).
- Use `nvidia-smi` para listar GPUs e IDs.

Diagnóstico e resolução de problemas comuns
- Erro de import do módulo nativo (`ModuleNotFoundError` para `dose_kernels`): provavelmente a extensão nativa não foi compilada/instalada. Refaça `python -m pip install .` e verifique logs do CMake/nvcc.
- Erros `nvcc` relacionados a símbolos C++/STL: verifique compatibilidade entre `nvcc` e `gcc`/`clang` (use `CUDAHOSTCXX` para apontar um compilador suportado).
- Erros de runtime CUDA (exceções std::runtime_error levantadas do CUDA_CHECK): verifique `nvidia-smi` e se o `gpu_id` está correto; confira memória GPU (picos de alocação). Reduce grid size ou use GPU com mais memória.
- Diferença de HU após leitura DICOM: `plan.loadCTDCM()` aplica checagens e corrige `RescaleSlope/Intercept` quando necessário.

Performance & recomendações
- Use GPUs modernas com memória suficiente (≥8–16 GB para grids clínicos grandes).
- Experimente espaçamentos maiores (ex.: 2.5 mm) para acelerar cálculo em verificações secundárias clínicas.
- O código usa texturas 3D e kernels com blocos cúbicos (`TILE_WIDTH`) — ajustar `TILE_WIDTH` no código CUDA requer recompilação.

Desenvolvimento e depuração
- Para desenvolver/depurar CUDA, habilite build debug via CMake args (modifique `setup.py`/`pyproject.toml` temporariamente):

```bash
python -m pip install . --global-option=--cmake-args -DCMAKE_BUILD_TYPE=Debug
```

- Os arquivos de interesse para inspeção rápida: [DoseCUDA/dose_kernels/dosemodule.cu](DoseCUDA/dose_kernels/dosemodule.cu), [DoseCUDA/dose_kernels/IMRTClasses.cu](DoseCUDA/dose_kernels/IMRTClasses.cu), e [DoseCUDA/plan_imrt.py](DoseCUDA/plan_imrt.py).

Checklist pré-execução (rápido)
- [ ] `nvidia-smi` mostra GPU(s) e driver OK
- [ ] `nvcc --version` retorna toolchain instalado
- [ ] `python -m pip install .` completou sem erros e `dose_kernels` importa
- [ ] CT reamostrado para voxels isotrópicos
- [ ] Modelo de feixe (`IMRTPhotonEnergy`) validado (`validate_parameters()`)

Contato e contribuições
- Autores originais: Tom Hrinivich, Calin Reamy, Mahasweta Bhattacharya (veja `pyproject.toml`).
- Abra issues / PRs no repositório principal para bugfixes e melhorias de performance.

Licença
- MIT (ver arquivo `LICENSE`).

-----

## Secondary Check Pipeline

DoseCUDA includes a complete secondary dose check workflow for clinical QA, comparing calculated dose against TPS reference.

### Features

- **Gamma 3D Analysis**: Global and local modes with configurable criteria (3%/3mm, 2%/2mm)
- **DVH Comparison**: Per-ROI metrics comparison with automatic target/OAR classification
- **MU Sanity Check**: Informational dose/MU ratio check at isocenter
- **Structured Reports**: JSON (with schema validation) and CSV formats

### Quick Start

```python
from DoseCUDA.secondary_report import evaluate_secondary_check, SecondaryCheckCriteria
from DoseCUDA.secondary_report import generate_json_report, generate_csv_report

# Run evaluation
result = evaluate_secondary_check(
    dose_calc=my_calculated_dose,     # Resampled to ref grid
    dose_ref=tps_dose,
    grid_origin=grid_origin,
    grid_spacing=grid_spacing,
    rois=roi_masks,                   # Dict of ROI name -> boolean mask
    roi_classification=classified_rois,
    plan=loaded_plan,
    patient_id="PAT001",
    plan_name="Prostate VMAT",
    plan_uid="1.2.3..."
)

# Generate reports
generate_json_report(result, "secondary_report.json")
generate_csv_report(result, "secondary_report.csv")

print(f"Overall Status: {result.overall_status}")
```

### Components

| Module | Purpose |
|--------|---------|
| `DoseCUDA.gamma` | 3D gamma index computation |
| `DoseCUDA.roi_selection` | Pattern-based ROI classification |
| `DoseCUDA.mu_sanity` | MU sanity check at isocenter |
| `DoseCUDA.secondary_report` | Report generation and orchestration |

### Default Criteria (Configurable)

| Check | Parameter | Default Threshold |
|-------|-----------|-------------------|
| Gamma 3%/3mm | Pass rate | >= 95% |
| Gamma 2%/2mm | Pass rate | >= 90% |
| Gamma threshold | Dose cutoff | 10% of max |
| Target D95 | Relative diff | <= 3% |
| Target Dmean | Relative diff | <= 2% |
| OAR Dmean | Absolute diff | <= 1 Gy |
| OAR Dmax | Absolute diff | <= 2 Gy |
| MU ratio | Deviation | <= 5% (INFO) |

### Custom Criteria

```python
from DoseCUDA.secondary_report import SecondaryCheckCriteria

# Stricter criteria
criteria = SecondaryCheckCriteria(
    gamma_3_3_pass_rate=0.98,    # Require 98% pass rate
    gamma_2_2_pass_rate=0.95,    # Require 95% pass rate
    target_d95_tolerance_rel=0.02,  # 2% tolerance
    oar_dmax_tolerance_abs=1.5   # 1.5 Gy tolerance
)

result = evaluate_secondary_check(..., criteria=criteria)
```

### Gamma Analysis Standalone

```python
from DoseCUDA.gamma import compute_gamma_3d, GammaCriteria

# Custom gamma criteria
criteria = GammaCriteria(
    dta_mm=3.0,
    dd_percent=3.0,
    local=False,              # Global mode
    dose_threshold_percent=10.0
)

result = compute_gamma_3d(
    dose_eval=dose_calculated,
    dose_ref=dose_reference,
    spacing_mm=(2.5, 2.5, 2.5),
    criteria=criteria,
    return_map=True           # Get full gamma map
)

print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Mean gamma: {result.mean_gamma:.3f}")
```

### ROI Classification

```python
from DoseCUDA.roi_selection import classify_rois, ROIClassificationConfig

# Use default patterns
classification = classify_rois(['PTV_70', 'Bladder', 'Rectum', 'BODY'])
print(classification.targets)   # ['PTV_70']
print(classification.oars)      # ['Bladder', 'Rectum']
print(classification.excluded)  # ['BODY']

# Custom patterns
config = ROIClassificationConfig(
    target_patterns=[r'^PTV', r'^CTV', r'TARGET'],
    exclude_patterns=[r'^BODY', r'^EXTERNAL', r'^SUPPORT']
)
classification = classify_rois(roi_names, config)
```

### Running Tests

**Testes unitários (sem GPU, dados sintéticos):**

```bash
# Testes do módulo gamma
pytest tests/test_gamma_metrics.py -v

# Testes do relatório secundário
pytest tests/test_secondary_report_smoke.py -v

# Todos os testes unitários
pytest tests/test_gamma_metrics.py tests/test_secondary_report_smoke.py -v
```

**Teste end-to-end completo (requer GPU + dados DICOM):**

```bash
# 1. Defina o diretório com dados DICOM do paciente
#    (deve conter: CT, RTPLAN, RTSTRUCT, RTDOSE do TPS)
export DOSECUDA_PATIENT_DICOM_DIR=/home/rt/scripts/DoseCUDA/tests/PATIENT/TRUEBEAM

# 2. (Opcional) Configure GPU e espaçamento
export DOSECUDA_GPU_ID=0           # GPU a usar (default: 0)
export DOSECUDA_ISO_MM=2.5         # Espaçamento isotrópico em mm (default: 2.5)

# 3. Execute o teste end-to-end
pytest tests/test_patient_end2end.py -v -s

# Ou execute testes específicos:
pytest tests/test_patient_end2end.py::test_8_gamma_analysis -v -s
pytest tests/test_patient_end2end.py::test_9_dvh_comparison -v -s
pytest tests/test_patient_end2end.py::test_11_generate_report -v -s
```

**Estrutura esperada do diretório DICOM:**

```
PATIENT/TRUEBEAM/
├── CT.1.2.3...dcm         # Slices do CT
├── CT.1.2.4...dcm
├── ...
├── RP.1.2.5...dcm         # RTPLAN
├── RS.1.2.6...dcm         # RTSTRUCT
└── RD.1.2.7...dcm         # RTDOSE (referência TPS)
```

**Saídas geradas (em `tests/test_patient_output/`):**

| Arquivo | Descrição |
|---------|-----------|
| `DoseCUDA_RD.dcm` | Dose calculada pelo DoseCUDA |
| `RTDOSE_template.dcm` | Cópia do RTDOSE TPS (referência) |
| `secondary_check_report.json` | Relatório completo em JSON |
| `secondary_check_report.csv` | Relatório em CSV |
| `gamma_summary.json` | Resumo da análise gamma |
| `dvh_comparison.txt` | Comparação DVH por ROI |

### Output Files

After running the end-to-end test with RTDOSE template:
- `tests/test_patient_output/secondary_check_report.json` - Full JSON report
- `tests/test_patient_output/secondary_check_report.csv` - CSV for spreadsheet import
- `tests/test_patient_output/gamma_summary.json` - Gamma analysis summary
- `tests/test_patient_output/dvh_comparison.txt` - DVH metrics report

-----

Se desejar, eu posso tambem:
- Gerar um script `build_linux.sh` que executa as checagens e faz `pip install .` automaticamente;
- Adicionar instrucoes para cross-compile/CI (GitHub Actions) com GPUs virtuais ou imagens Docker;
- Criar exemplos praticos executaveis que calculem a dose de um pequeno phantom e salvem saida NRRD/RTDOSE.

Diga qual desses itens quer que eu faca agora.

