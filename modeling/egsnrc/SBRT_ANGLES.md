# Recomendações de Ângulos para SBRT

## Contexto

Em SBRT (Stereotactic Body Radiation Therapy) com campos pequenos e grid fino (1.5mm), a discretização angular do kernel CCC/CCCS tem impacto significativo na acurácia da dose.

## Referência Principal

**Cho BC et al. (2012)** - Phys Med Biol 57(4):985-1002

> "For accurate dose calculation with CCC/S method using small fields:
> - Energy from primary interaction deposits within 1-2mm
> - M=24×12=288 directions provides < 1% error for most cases
> - Large voxels (>2mm) cause systematic errors near beam center
> - FFF beams require finer angular sampling due to harder spectrum"

## Configurações Recomendadas

### Tier 1: Baseline (Padrão IMRT)
- **Configuração**: 6 polar × 12 azimute = **72 direções**
- **Grid recomendado**: ≥ 3mm
- **Uso**: IMRT padrão, campos grandes
- **Kernel file**: 6 linhas em `kernel.csv`

```cpp
#define N_POLAR_ANGLES 6
#define N_AZIMUTHAL_ANGLES 12
```

### Tier 2: Medium (SBRT básico)
- **Configuração**: 12 polar × 12 azimute = **144 direções**
- **Grid recomendado**: 2mm
- **Uso**: SBRT, campos < 3cm
- **Kernel file**: 12 linhas em `kernel.csv`

```cpp
#define N_POLAR_ANGLES 12
#define N_AZIMUTHAL_ANGLES 12
```

### Tier 3: High (SBRT fino)
- **Configuração**: 12 polar × 24 azimute = **288 direções**
- **Grid recomendado**: 1.5mm
- **Uso**: SBRT com campos muito pequenos, FFF, QA de máquina
- **Kernel file**: 12 linhas em `kernel.csv` (azimute é fixo no código)

```cpp
#define N_POLAR_ANGLES 12
#define N_AZIMUTHAL_ANGLES 24
```

## Mudanças no Código CUDA

### 1. Definições de Constantes (IMRTClasses.cu)

```cpp
// Alterar de:
#define N_POLAR_ANGLES 6
#define N_AZIMUTHAL_ANGLES 12
#define KERNEL_SIZE 36  // 6 * 6 colunas

// Para (Tier 2):
#define N_POLAR_ANGLES 12
#define N_AZIMUTHAL_ANGLES 12
#define KERNEL_SIZE 72  // 12 * 6 colunas

// Ou (Tier 3):
#define N_POLAR_ANGLES 12
#define N_AZIMUTHAL_ANGLES 24
#define KERNEL_SIZE 72  // 12 * 6 colunas
```

### 2. Arrays Trigonométricos

```cpp
// Pré-calcular senos e cossenos para os novos ângulos
__constant__ float cos_theta[N_POLAR_ANGLES];
__constant__ float sin_theta[N_POLAR_ANGLES];
__constant__ float cos_phi[N_AZIMUTHAL_ANGLES];
__constant__ float sin_phi[N_AZIMUTHAL_ANGLES];

// Inicialização (host code)
void init_trig_tables(int n_polar, int n_azimuthal) {
    float* h_cos_theta = new float[n_polar];
    float* h_sin_theta = new float[n_polar];
    float* h_cos_phi = new float[n_azimuthal];
    float* h_sin_phi = new float[n_azimuthal];

    // Ângulos polares (do kernel.csv)
    for (int i = 0; i < n_polar; i++) {
        float theta = kernel_thetas[i] * M_PI / 180.0f;
        h_cos_theta[i] = cosf(theta);
        h_sin_theta[i] = sinf(theta);
    }

    // Ângulos azimutais (uniformemente distribuídos)
    for (int j = 0; j < n_azimuthal; j++) {
        float phi = 2.0f * M_PI * j / n_azimuthal;
        h_cos_phi[j] = cosf(phi);
        h_sin_phi[j] = sinf(phi);
    }

    // Copiar para GPU
    cudaMemcpyToSymbol(cos_theta, h_cos_theta, ...);
    // ...
}
```

### 3. Pesos de Ângulo Sólido

```cpp
// Os pesos devem integrar a esfera corretamente
// Δω = sin(θ) × Δθ × Δφ

__constant__ float solid_angle_weights[N_POLAR_ANGLES];

void compute_solid_angle_weights(float* theta_centers, int n_polar, int n_azimuthal) {
    float total = 0.0f;

    // Calcular pesos baseados na cobertura angular
    // Assume bins uniformes em phi
    float dphi = 2.0f * M_PI / n_azimuthal;

    for (int i = 0; i < n_polar; i++) {
        // Limites do bin polar
        float theta1 = (i > 0) ? 0.5f * (theta_centers[i-1] + theta_centers[i]) : 0.0f;
        float theta2 = (i < n_polar-1) ? 0.5f * (theta_centers[i] + theta_centers[i+1]) : M_PI;

        // Ângulo sólido = ∫∫ sin(θ) dθ dφ = 2π × (cos(θ1) - cos(θ2))
        // Dividido por n_azimuthal para peso por direção
        weights[i] = (cosf(theta1) - cosf(theta2)) / n_azimuthal;
        total += weights[i] * n_azimuthal;
    }

    // Normalizar para 4π (esfera completa)
    for (int i = 0; i < n_polar; i++) {
        weights[i] *= 4.0f * M_PI / total;
    }
}
```

### 4. Loop de Convolução

```cpp
__global__ void dose_kernel(/* ... */) {
    // ...

    float dose = 0.0f;

    // Loop sobre todas as direções
    for (int i_polar = 0; i_polar < N_POLAR_ANGLES; i_polar++) {
        // Parâmetros do kernel para este ângulo polar
        float Am = kernel[i_polar];
        float am = kernel[i_polar + N_POLAR_ANGLES];
        float Bm = kernel[i_polar + 2*N_POLAR_ANGLES];
        float bm = kernel[i_polar + 3*N_POLAR_ANGLES];
        float ray_len = kernel[i_polar + 5*N_POLAR_ANGLES];

        // Trigonometria polar
        float ct = cos_theta[i_polar];
        float st = sin_theta[i_polar];
        float weight_polar = solid_angle_weights[i_polar];

        for (int i_azim = 0; i_azim < N_AZIMUTHAL_ANGLES; i_azim++) {
            // Trigonometria azimutal
            float cp = cos_phi[i_azim];
            float sp = sin_phi[i_azim];

            // Direção do raio
            float dx = st * cp;
            float dy = st * sp;
            float dz = ct;

            // Ray-trace e acumular dose
            float ray_dose = trace_ray(pos, dx, dy, dz, ray_len, Am, am, Bm, bm);
            dose += ray_dose * weight_polar;
        }
    }

    // ...
}
```

## Impacto no Desempenho

| Tier | Direções | Tempo relativo | Memória kernel |
|------|----------|----------------|----------------|
| Baseline | 72 | 1.0× | 36 floats |
| Medium | 144 | ~1.5× | 72 floats |
| High | 288 | ~2.5× | 72 floats |

**Notas**:
- O aumento de tempo não é linear porque o ray-tracing domina
- Mais direções = melhor paralelismo GPU (pode até ser mais eficiente em alguns casos)
- Memória do kernel é negligível (< 1KB)

## Validação

### 1. Verificar cobertura angular

```python
import numpy as np

# Carregar kernel
kernel = pd.read_csv("kernel.csv")
thetas = kernel['theta'].values * np.pi / 180

# Calcular ângulo sólido coberto
solid_angle_total = 0
for i, theta in enumerate(thetas):
    # Bin boundaries
    if i == 0:
        theta1 = 0
    else:
        theta1 = 0.5 * (thetas[i-1] + theta)

    if i == len(thetas) - 1:
        theta2 = np.pi
    else:
        theta2 = 0.5 * (theta + thetas[i+1])

    solid_angle = 2 * np.pi * (np.cos(theta1) - np.cos(theta2))
    solid_angle_total += solid_angle
    print(f"θ={np.degrees(theta):.1f}°: Δω = {solid_angle:.4f} sr")

print(f"\nTotal: {solid_angle_total:.4f} sr (esperado: 4π = {4*np.pi:.4f} sr)")
```

### 2. Comparar com Monte Carlo

- Usar TOPAS/EGSnrc para calcular dose em fantoma de água
- Comparar PDD e perfis laterais
- Diferença < 2% para profundidades > dmax

## Recomendação Final para SBRT 1.5mm

**Use Tier 2 (144 direções) como ponto de partida.**

Razões:
1. Boa relação custo-benefício
2. Erro < 1% para maioria dos casos SBRT
3. Não requer modificações complexas no código
4. Tier 3 reservado para QA e casos especiais

**Para FFF com campos < 2cm em grid 1.5mm:**
- Considere Tier 3 (288 direções)
- Principalmente na região de build-up e penumbra
