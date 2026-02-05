# EGSnrc Kernel Factory for DoseCUDA

Pipeline Docker para geração de Energy Deposition Kernels (EDK) usando EGSnrc Monte Carlo, com pós-processamento para o formato DoseCUDA CCC/CCCS.

## Índice

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Build da Imagem Docker](#build-da-imagem-docker)
4. [Gerando Kernels Monoenergéticos](#gerando-kernels-monoenergéticos)
5. [Gerando Kernels Clínicos (Polienergéticos)](#gerando-kernels-clínicos-polienergéticos)
6. [Formato de Saída](#formato-de-saída)
7. [Integração com DoseCUDA](#integração-com-dosecuda)
8. [Recomendações para SBRT](#recomendações-para-sbrt)
9. [Referências](#referências)

---

## Overview

### O que este pipeline faz

1. **Simula** deposição de energia em água para fótons monoenergéticos usando EGSnrc (ou gera kernels aproximados se EGSnrc não estiver disponível)
2. **Converte** os dados de dose 3D para coordenadas esféricas K(r, θ)
3. **Comprime** os ângulos finos para os ângulos discretos do DoseCUDA (6 ou 12)
4. **Ajusta** o modelo de dois termos exponenciais: K(r) = Am·exp(-am·r) + Bm·exp(-bm·r)
5. **Exporta** `kernel.csv` no formato exato do DoseCUDA

### Modelo Físico

O kernel de deposição de energia (EDK) descreve como a energia de um fóton é depositada no meio após interação. O DoseCUDA usa um modelo colapsado com dois componentes:

```
K_linha(r) = Am * exp(-am * r) + Bm * exp(-bm * r)
```

Onde:
- **Am, am**: Componente primário (curto alcance, elétrons de alta energia)
- **Bm, bm**: Componente scatter (longo alcance, elétrons de baixa energia)
- **r**: Distância ao longo do raio em cm

### Metodologia SCASPH (Mackie 1988)

A simulação segue o método clássico:
1. Fóton monoenergético forçado a interagir no centro de uma esfera de água
2. Partículas secundárias transportadas pelo Monte Carlo
3. Energia depositada scored em shells radiais e cones angulares
4. Kernel normalizado por energia incidente

---

## Quick Start

```bash
# 1. Build da imagem (requer EGSnrc tarball)
cd modeling/egsnrc
docker build -f Dockerfile.egsnrc -t egsnrc-kernels .

# 2. Gerar kernels monoenergéticos
docker run -v $(pwd)/output:/output egsnrc-kernels \
    --energies 0.5,1.0,2.0,4.0,6.0 \
    --output /output

# 3. Gerar kernel 6MV com espectro clínico
docker run -v $(pwd)/output:/output \
           -v $(pwd)/spectrum_6MV.csv:/spectrum.csv \
    egsnrc-kernels \
    --energies 0.2,0.5,1.0,2.0,4.0,6.0 \
    --spectrum /spectrum.csv \
    --beam 6MV

# 4. Copiar para DoseCUDA
cp output/processed/6MV/kernel.csv \
   ../../DoseCUDA/lookuptables/photons/VarianTrueBeamHF/6MV/
```

---

## Build da Imagem Docker

### Requisitos

1. **EGSnrc tarball** (ex: `EGSnrc_v2024.tar.gz`)
   - Download de: https://github.com/nrc-cnrc/EGSnrc/releases
   - Coloque no diretório de build

2. **Docker** (sem necessidade de privilégios especiais)

### Build

```bash
cd modeling/egsnrc

# Colocar o tarball do EGSnrc aqui
cp /path/to/EGSnrc_v2024.tar.gz .

# Build
docker build -f Dockerfile.egsnrc -t egsnrc-kernels .
```

Se o EGSnrc tarball não estiver disponível, a imagem ainda será construída e usará **kernels analíticos aproximados** (útil para teste/desenvolvimento, mas não para uso clínico).

### Verificar instalação

```bash
docker run egsnrc-kernels --help
```

---

## Gerando Kernels Monoenergéticos

Para gerar kernels para um conjunto de energias discretas:

```bash
docker run -v $(pwd)/output:/output egsnrc-kernels \
    --energies 0.2,0.5,1.0,2.0,4.0,6.0,10.0,15.0,20.0 \
    --histories 10000000 \
    --output /output
```

### Parâmetros

| Parâmetro | Descrição | Default |
|-----------|-----------|---------|
| `--energies, -e` | Energias em MeV (obrigatório) | - |
| `--histories, -n` | Histórias por energia | 10M |
| `--output, -o` | Diretório de saída | /output |
| `--angles` | Ângulos polares (6 ou 12) | 6 |

### Energias Recomendadas

Para cobrir espectros clínicos típicos:

```bash
# Feixe filtrado (6MV, 10MV, 15MV)
--energies 0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0,15.0

# FFF (6FFF, 10FFF) - precisa de mais energias altas
--energies 0.1,0.2,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0
```

### Saída

```
output/
├── kernels/                    # Kernels esféricos K(r,θ)
│   ├── kernel_0.200MeV.csv
│   ├── kernel_0.500MeV.csv
│   └── kernel_1.000MeV.csv
├── processed/
│   └── monoenergetic/
│       ├── 0.2MeV/kernel.csv   # Formato DoseCUDA
│       └── 1.0MeV/kernel.csv
└── logs/
    └── edk_*.log
```

---

## Gerando Kernels Clínicos (Polienergéticos)

### 1. Extrair espectro do Phase Space IAEA

Se você tem um phase space IAEA (ex: do TOPAS):

```bash
# Usando script Python diretamente
python3 scripts/extract_spectrum.py iaea \
    --phsp /path/to/phsp/TrueBeam_6MV \
    --output spectrum_6MV.csv \
    --bins 50
```

### 2. Usar espectro pré-definido (aproximação)

```bash
python3 scripts/extract_spectrum.py standard \
    --beam 6MV \
    --output spectrum_6MV.csv

# Disponíveis: 6MV, 10MV, 15MV, 18MV, 6FFF, 10FFF
```

### 3. Gerar kernel polienergético

```bash
docker run -v $(pwd)/output:/output \
           -v $(pwd)/spectrum_6MV.csv:/spectrum.csv \
    egsnrc-kernels \
    --energies 0.2,0.5,1.0,2.0,4.0,6.0 \
    --spectrum /spectrum.csv \
    --beam 6MV \
    --output /output
```

### 4. Para feixes FFF

```bash
# Usa aproximação automática de espectro FFF
docker run -v $(pwd)/output:/output egsnrc-kernels \
    --energies 0.2,0.5,1.0,2.0,4.0,6.0 \
    --fff \
    --beam 6MV_FFF \
    --output /output
```

---

## Formato de Saída

### kernel.csv (formato DoseCUDA)

```csv
theta,Am,am,Bm,bm,ray_length
1.875,1.2800e+00,1.9100,1.2400e-02,2.8100e-01,3.000
20.625,6.0300e-01,1.8300,6.6600e-03,2.5300e-02,2.000
43.125,1.8300e-01,2.0500,2.1200e-03,2.9100e-02,2.000
61.875,8.4800e-02,2.8500,9.5400e-04,2.4400e-02,2.000
88.125,1.5800e-02,3.7100,4.5000e-04,1.8400e-02,2.000
106.875,1.4500e-02,7.7100,3.3300e-04,1.8500e-02,1.000
```

### Colunas

| Coluna | Unidade | Descrição |
|--------|---------|-----------|
| `theta` | graus | Ângulo polar (0° = forward, 180° = backward) |
| `Am` | MeV/cm³/MeV | Amplitude do componente primário |
| `am` | cm⁻¹ | Atenuação do componente primário |
| `Bm` | MeV/cm³/MeV | Amplitude do componente scatter |
| `bm` | cm⁻¹ | Atenuação do componente scatter |
| `ray_length` | cm | Comprimento máximo do raio (cutoff) |

### kernel_depth_dependent.csv (opcional)

Para SBRT/VMAT com variação por profundidade:

```csv
depth,angle_idx,Am,am,Bm,bm
0.0,0,0.0123,0.456,0.0789,1.234
0.0,1,0.0234,0.567,0.0890,1.345
...
30.0,5,0.0589,0.912,0.1145,1.700
```

---

## Integração com DoseCUDA

### Estrutura de diretórios

```
DoseCUDA/lookuptables/photons/
└── <Machine>/
    ├── energy_labels.csv
    ├── machine_geometry.csv
    ├── mlc_geometry.csv
    ├── HU_Density.csv
    ├── 6MV/
    │   ├── kernel.csv          ← Coloque aqui
    │   └── beam_parameters.csv
    └── 6MV_FFF/
        ├── kernel.csv          ← Coloque aqui
        └── beam_parameters.csv
```

### Copiar kernels

```bash
# 6MV filtrado
cp output/processed/6MV/kernel.csv \
   DoseCUDA/lookuptables/photons/VarianTrueBeamHF/6MV/

# 6MV FFF
cp output/processed/6MV_FFF/kernel.csv \
   DoseCUDA/lookuptables/photons/VarianTrueBeamHF/6MV_FFF/
```

### Verificar integração

```python
from DoseCUDA import plan_imrt

# O kernel será carregado automaticamente
beam_model = plan_imrt.load_beam_model("VarianTrueBeamHF", "6MV")
print(f"Kernel loaded: {beam_model.kernel.shape}")  # Esperado: (36,)
```

---

## Recomendações para SBRT

### Referência: Cho et al. 2012

> *"For CCC/S, using M=24×12=288 directions provides accurate dose distributions
> for small fields. Energy deposited within 1-2mm of the primary interaction site.
> Large voxels (>2mm) cause systematic errors in the dose near the center of
> small fields."*

### Configurações de Ângulos

| Tier | Polar × Azimute | Total | Uso Recomendado |
|------|-----------------|-------|-----------------|
| **Baseline** | 6 × 12 | 72 | IMRT padrão, grid ≥3mm |
| **Medium** | 12 × 12 | 144 | SBRT, grid 2mm |
| **High** | 12 × 24 | 288 | SBRT, grid 1.5mm |

### Gerar kernels para SBRT

```bash
# 12 ângulos polares para SBRT
docker run -v $(pwd)/output:/output egsnrc-kernels \
    --energies 0.2,0.5,1.0,2.0,4.0,6.0,10.0 \
    --angles 12 \
    --fff \
    --beam 6MV_FFF_SBRT \
    --output /output
```

### Mudanças necessárias no CUDA (para tiers superiores)

Para suportar mais direções no DoseCUDA:

```cpp
// Em IMRTClasses.cu / dosemodule.cu

// Tier Baseline (atual): 6 polar × 12 azimute = 72 direções
#define N_POLAR_ANGLES 6
#define N_AZIMUTHAL_ANGLES 12

// Tier Medium: 12 polar × 12 azimute = 144 direções
#define N_POLAR_ANGLES 12
#define N_AZIMUTHAL_ANGLES 12

// Tier High: 12 polar × 24 azimute = 288 direções
#define N_POLAR_ANGLES 12
#define N_AZIMUTHAL_ANGLES 24

// Arrays de trigunometria
__constant__ float cos_theta[N_POLAR_ANGLES];
__constant__ float sin_theta[N_POLAR_ANGLES];
__constant__ float cos_phi[N_AZIMUTHAL_ANGLES];
__constant__ float sin_phi[N_AZIMUTHAL_ANGLES];

// Pesos sólidos ângulos (integração numérica)
__constant__ float solid_angle_weights[N_POLAR_ANGLES];
```

### Validação para SBRT

```python
# Verificar que kernel tem 12 ângulos
import pandas as pd

kernel = pd.read_csv("kernel.csv")
assert len(kernel) == 12, f"Expected 12 angles, got {len(kernel)}"

# Verificar cobertura angular
thetas = kernel['theta'].values
print(f"Angular coverage: {thetas.min():.1f}° - {thetas.max():.1f}°")

# Para SBRT: garantir boa cobertura em forward direction
assert thetas.min() < 10, "Need fine angular sampling near forward direction"
```

---

## Troubleshooting

### "EGSnrc not found" - kernels analíticos sendo usados

```
[WARN] HEN_HOUSE not set. Using mock kernel generation.
```

**Causa**: EGSnrc tarball não foi incluído no build.

**Solução**: Baixe o EGSnrc e rebuild:
```bash
wget https://github.com/nrc-cnrc/EGSnrc/releases/download/v2024/EGSnrc_v2024.tar.gz
docker build -f Dockerfile.egsnrc -t egsnrc-kernels .
```

### Kernel validation warnings

```
[WARN] am > bm (primary > scatter attenuation)
```

**Causa**: Parâmetros de ajuste podem estar invertidos.

**Solução**: Verifique se os dados de entrada têm estatística suficiente. Aumente `--histories`.

### Memória insuficiente

Para simulações com muitas energias:

```bash
# Rodar em batches
for E in 0.5 1.0 2.0; do
    docker run -v $(pwd)/output:/output egsnrc-kernels \
        --energies $E --output /output
done
```

---

## Referências

1. **Mackie TR, Bielajew AF, Rogers DWO, Battista JJ.** Generation of photon energy deposition kernels using the EGS Monte Carlo code. *Phys Med Biol.* 1988;33(1):1-20.

2. **Cho BC, Schwarz M, Mijnheer BJ, Oelfke U.** Simplified intensity modulated radiotherapy using pre-defined segments. *Phys Med Biol.* 2012;57(4):985-1002.

3. **Ahnesjö A.** Collapsed cone convolution of radiant energy for photon dose calculation in heterogeneous media. *Med Phys.* 1989;16(4):577-592.

4. **EGSnrc Documentation**: https://nrc-cnrc.github.io/EGSnrc/

5. **IAEA Phase Space Database**: https://www-nds.iaea.org/phsp/

---

## Licença

Este código é parte do projeto DoseCUDA e segue a mesma licença do projeto principal.
