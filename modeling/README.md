# Pipeline de modelagem (PHSP → DoseCUDA)

Este diretório guarda tudo o que é **específico de modelagem**: onde colocar os phase spaces (PHSP), macros do TOPAS e o Dockerfile da imagem usada para gerar dados de comissionamento do DoseCUDA.

## Estrutura proposta

```
modeling/
├── phsp/                          # arquivos grandes (.IAEAphsp/.IAEAheader) — mantidos fora do git
│   └── VarianTrueBeam_v2/10X/
│       └── TrueBeam_v2_10X_00.IAEA{phsp,header}
├── topas_inputs/                  # macros TOPAS prontos para rodar
│   └── TrueBeam_10X_WaterPhantom.txt
└── docker/
    ├── Dockerfile.modeling        # imagem TOPAS 4.0 + Geant4 11.1.3 + Celeritas 0.6 (CUDA)
    └── workflow/run.sh            # entrypoint que chama o TOPAS dentro do contêiner
```

> Os PHSP não devem ser versionados. Há regra no `.gitignore` para `modeling/phsp/`.

## Onde colocar o PHSP da IAEA

1) Copie os dois arquivos baixados para a pasta abaixo (pode criar se não existir):

```
modeling/phsp/VarianTrueBeam_v2/10X/
  ├── TrueBeam_v2_10X_00.IAEAheader
  └── TrueBeam_v2_10X_00.IAEAphsp
```

2) Para o TOPAS, mantenha também os nomes sem a extensão `.IAEA*` (link simbólico ou cópia):

```bash
cd modeling/phsp/VarianTrueBeam_v2/10X
ln -sf TrueBeam_v2_10X_00.IAEAheader TrueBeam_v2_10X_00.header
ln -sf TrueBeam_v2_10X_00.IAEAphsp  TrueBeam_v2_10X_00.phsp
```

O macro de exemplo já aponta para `/phsp/VarianTrueBeam_v2/10X/TrueBeam_v2_10X_00` (sem extensão).

## Construir a imagem de modelagem (TOPAS + Celeritas/CUDA)

O Dockerfile em `modeling/docker/Dockerfile.modeling` recria o ambiente descrito na mensagem do projeto (TOPAS 4.0, Geant4 11.1.3, Celeritas 0.6.1, CUDA 12.6). Para construir:

```bash
docker build -f modeling/docker/Dockerfile.modeling -t topas-celeritas .
```

Pré-requisitos: host com `nvidia-container-toolkit`, driver NVIDIA e suporte a `--gpus all`.

## Rodar o TOPAS consumindo o PHSP

1) Garanta que os arquivos `.IAEA*` (e os links `.header/.phsp`) estão em `modeling/phsp/...`.

2) Rode o contêiner montando PHSP e pasta de saída:

```bash
mkdir -p modeling/output
docker run --rm --gpus all \
  -v $(pwd)/modeling/phsp:/phsp \
  -v $(pwd)/modeling/output:/output \
  topas-celeritas \
  /workflow/topas_inputs/TrueBeam_10X_WaterPhantom.txt
```

O entrypoint (`/workflow/run.sh`) muda o diretório de trabalho para `/output`; os CSVs gerados pelo scorer do TOPAS aparecerão em `modeling/output/` no host.

## Próximos passos (ligação com DoseCUDA)

- A partir do `DoseToWater` em CSV, extraia PDD e perfis para alimentar `lookuptables/photons/<Machine>/<Energy>/profile.csv` e ajuste espectro/kernels conforme `README_STRUCTURE.md`.
- Se desejar, adicione novos macros em `modeling/topas_inputs/` para outras energias (FFF, 6 MV) usando a mesma convenção de pastas.
