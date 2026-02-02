# Estrutura de Lookup Tables por Energia (FASE 4)

## Problema Original

O código anterior tinha **um único conjunto de parâmetros** para todas as energias, o que é fisicamente incorreto:
- 6MV tem espectro diferente de 10MV
- FFF tem espectro mais "mole" que flattened
- Cada energia precisa de kernel, espectro e parâmetros únicos

---

## Nova Estrutura de Diretórios

```
DoseCUDA/lookuptables/photons/
├── VarianTrueBeamHF/
│   ├── energy_labels.csv          # Mapeia DICOM → pastas
│   ├── machine_geometry.csv        # Geometria comum (SAD, etc)
│   ├── mlc_geometry.csv            # MLC (comum todas energias)
│   ├── HU_Density.csv              # HU → ρ (comum)
│   │
│   ├── 6MV_FFF/                    # ← Específico da energia
│   │   ├── kernel.csv              # Kernel fixo (6x6)
│   │   ├── kernel_depth_dependent.csv  # (Opcional) Kernel z-dep
│   │   ├── spectrum.csv            # Espectro policromático
│   │   ├── profile.csv             # Perfil off-axis
│   │   └── beam_parameters.csv     # Parâmetros únicos desta energia
│   │
│   ├── 6MV_Flattened/
│   │   ├── kernel.csv
│   │   ├── ...
│   │
│   ├── 10MV_FFF/
│   │   └── ...
│   │
│   └── 10MV_Flattened/
│       └── ...
│
└── ElektaSynergyAgility/
    └── ...
```

---

## Arquivos por Energia

### 1. `beam_parameters.csv`

Parâmetros únicos desta energia:

```csv
parameter,value
mu_calibration,1.000
primary_source_distance,1000.0
scatter_source_distance,500.0
primary_source_size,1.5
scatter_source_size,15.0
mlc_distance,450.0
scatter_source_weight,0.03
electron_attenuation,0.045
electron_source_weight,0.02
electron_fitted_dmax,15.0
jaw_transmission,0.02
mlc_transmission,0.015
has_xjaws,True
has_yjaws,True
```

### 2. `spectrum.csv`

Espectro policromático (10-15 energias):

```csv
energy_MeV,attenuation_coeff_cm2g,primary_weight,scatter_weight
0.5,0.0456,0.012,0.015
1.0,0.0389,0.089,0.095
1.5,0.0345,0.145,0.152
2.0,0.0312,0.178,0.180
2.5,0.0289,0.156,0.150
3.0,0.0271,0.123,0.115
4.0,0.0245,0.089,0.082
5.0,0.0228,0.067,0.058
6.0,0.0215,0.052,0.043
8.0,0.0198,0.039,0.032
10.0,0.0187,0.028,0.022
12.0,0.0179,0.015,0.012
15.0,0.0169,0.007,0.005
```

**Importante:** Cada linha representa uma energia no espectro. Os pesos devem somar ~1.0.

### 3. `profile.csv`

Perfil off-axis (intensidade + hardening):

```csv
radius_cm,intensity,softening
0.0,1.000,1.000
2.0,0.998,1.002
4.0,0.995,1.005
6.0,0.990,1.010
8.0,0.983,1.018
10.0,0.975,1.028
12.0,0.965,1.040
15.0,0.950,1.060
20.0,0.920,1.100
25.0,0.880,1.150
30.0,0.830,1.200
```

### 4. `kernel.csv`

Kernel CCC tradicional (6 ângulos x 6 parâmetros):

```csv
angle_0,angle_1,angle_2,angle_3,angle_4,angle_5
0.0,15.0,30.0,45.0,60.0,75.0
0.0123,0.0134,0.0145,0.0156,0.0167,0.0178
0.456,0.467,0.478,0.489,0.500,0.511
0.0789,0.0800,0.0811,0.0822,0.0833,0.0844
1.234,1.245,1.256,1.267,1.278,1.289
50.0,50.0,50.0,50.0,50.0,50.0
```

Linhas:
1. Ângulos polares (graus)
2. Am (parâmetro primário exponencial)
3. am (coef atenuação primário)
4. Bm (parâmetro scatter linear)
5. bm (coef atenuação scatter)
6. ray_length_init (cm)

### 5. `kernel_depth_dependent.csv` (Opcional)

Ver `README_KERNEL_ZDEP.md` para formato completo.

---

## Como Criar para Nova Máquina

1. **Crie diretório:**
   ```bash
   mkdir -p lookuptables/photons/NewMachine/{6MV_FFF,10MV_FFF}
   ```

2. **Copie templates:**
   ```bash
   cp VarianTrueBeamHF/6MV_FFF/*.csv NewMachine/6MV_FFF/
   ```

3. **Edite `energy_labels.csv`:**
   ```csv
   dicom_energy_label,folder_energy_label
   6,6MV_FFF
   10,10MV_FFF
   ```

4. **Commissioning:**
   - Medir IDD, perfis, output factors
   - Monte Carlo para espectro
   - Ajustar kernel por energia

---

## Código Python (Carregamento)

```python
# Em IMRTPlan.__init__:
energy_list = pd.read_csv("energy_labels.csv")
for dicom_label, folder_label in zip(energy_list.dicom_energy_label, energy_list.folder_energy_label):
    beam_model = IMRTPhotonEnergy(dicom_label)
    self._load_beam_model_parameters(beam_model, machine_name, folder_label)
    self.beam_models.append(beam_model)

# Em _load_beam_model_parameters:
path = f"lookuptables/photons/{machine_name}/{folder_energy_label}/"
beam_model.kernel = pd.read_csv(path + "kernel.csv")
beam_model.spectrum = pd.read_csv(path + "spectrum.csv")
beam_model.profile = pd.read_csv(path + "profile.csv")
beam_model.params = pd.read_csv(path + "beam_parameters.csv")
```

---

## Vantagens

✅ **Física correta:** Cada energia tem modelo próprio  
✅ **Fácil adicionar:** Só criar pasta nova  
✅ **Backward compatible:** Código antigo funciona (kernel.csv sempre presente)  
✅ **VMAT-ready:** Espectro correto = dose correta  

---

## Checklist de Validação

- [ ] Todas as energias têm pasta
- [ ] Cada pasta tem 5 CSVs mínimos
- [ ] Espectro soma ~1.0
- [ ] Perfil monotônico decrescente
- [ ] Kernel física razoável (Am, am > 0)
- [ ] Teste com phantom água
