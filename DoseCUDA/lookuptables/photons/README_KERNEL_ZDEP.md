# Formato do Kernel Dependente de Profundidade

## Motivação (FASE 2)

O kernel CCC tradicional usa parâmetros **fixos** (Am, am, Bm, bm) independentes da profundidade. Isso é uma aproximação que funciona razoavelmente para IMRT mas **falha em VMAT** e em casos com:
- Campos superficiais vs profundos
- Heterogeneidades (pulmão, osso)
- Build-up region

A correção correta é fazer os parâmetros do kernel **dependentes de z' (profundidade em WET)**:
- **z' = WET acumulado até o voxel** (em cm de água equivalente)
- Cada ângulo polar do kernel tem parâmetros que mudam com z'

---

## Estrutura do Arquivo CSV

**Nome do arquivo:** `kernel_depth_dependent.csv`

**Localização:** `lookuptables/photons/<MACHINE>/<ENERGY>/`

Exemplo: `lookuptables/photons/VarianTrueBeamHF/6MV_FFF/kernel_depth_dependent.csv`

### Colunas:

```csv
depth,angle_idx,Am,am,Bm,bm
0.0,0,0.0123,0.456,0.0789,1.234
0.0,1,0.0234,0.567,0.0890,1.345
...
0.0,5,0.0567,0.890,0.1123,1.678
5.0,0,0.0125,0.458,0.0791,1.236
5.0,1,0.0236,0.569,0.0892,1.347
...
30.0,5,0.0589,0.912,0.1145,1.700
```

### Descrição:

- **depth**: Profundidade em cm de WET (0, 5, 10, 15, 20, 25, 30)
- **angle_idx**: Índice do ângulo polar (0-5, total 6 ângulos)
- **Am, am**: Parâmetros do componente primário (exponencial)
- **Bm, bm**: Parâmetros do componente scatter (linear)

---

## Como Usar

1. **Coletar dados:**
   - Monte Carlo ou comissionamento
   - Para cada profundidade z', ajustar kernel CCC

2. **Criar CSV:**
   ```python
   import pandas as pd
   
   data = []
   depths = [0, 5, 10, 15, 20, 25, 30]  # cm
   angles = range(6)
   
   for depth in depths:
       for angle in angles:
           # Ajustar parâmetros para este depth + angle
           Am, am, Bm, bm = fit_kernel_at_depth(depth, angle)
           data.append([depth, angle, Am, am, Bm, bm])
   
   df = pd.DataFrame(data, columns=['depth', 'angle_idx', 'Am', 'am', 'Bm', 'bm'])
   df.to_csv('kernel_depth_dependent.csv', index=False)
   ```

3. **Código carrega automaticamente:**
   - Se existir `kernel_depth_dependent.csv`, usa interpolação
   - Senão, usa `kernel.csv` (fallback para compatibilidade)

---

## Comportamento do Código

### Python (`plan_imrt.py`):
```python
if os.path.exists('kernel_depth_dependent.csv'):
    # Carrega e reorganiza para [n_depths x 24]
    beam_model.use_depth_dependent_kernel = True
else:
    # Usa kernel fixo
    beam_model.use_depth_dependent_kernel = False
```

### CUDA (`IMRTClasses.cu`):
```cuda
// Em cccKernel:
float z_prime = dose->WETArray[vox_index] / 10.0f;  // mm → cm
beam->interpolateKernelParams(angle_idx, z_prime, &Am, &am, &Bm, &bm);

// Interpola linearmente entre profundidades tabeladas
```

---

## Exemplo de Dados (6MV FFF)

```csv
depth,angle_idx,Am,am,Bm,bm
0.0,0,0.0123,0.456,0.0789,1.234
0.0,1,0.0134,0.467,0.0800,1.245
0.0,2,0.0145,0.478,0.0811,1.256
0.0,3,0.0156,0.489,0.0822,1.267
0.0,4,0.0167,0.500,0.0833,1.278
0.0,5,0.0178,0.511,0.0844,1.289
5.0,0,0.0125,0.458,0.0791,1.236
5.0,1,0.0136,0.469,0.0802,1.247
...
```

---

## Validação

Execute após criar o arquivo:
```bash
cd tests
python test_kernel_zdep.py
```

Deve verificar:
- ✅ Interpolação funciona corretamente
- ✅ Dose superficial vs profunda muda
- ✅ Backward compatibility (sem z-dep ainda funciona)

---

## Referências

- Ahnesjö A. (1989) - CCC original
- Papanikolaou N. (2004) - "Tissue Inhomogeneity Corrections for Megavoltage Photon Beams"
- Eclipse Algorithm Reference Guide (Varian) - Seção sobre depth-dependent kernels
