TOPAS
Entendido. O TOPAS gera “matéria‑prima” (ex.: dose_water.csv), e o DoseCUDA só
  consome os CSVs em DoseCUDA/lookuptables/photons/<MACHINE>/<ENERGY>/. O que já
  deixei pronto:

  - Estrutura de modelagem em modeling/ (phsp/, topas_inputs/, docker/) e Dockerfile
    para TOPAS+Celeritas.
  - Macro modeling/topas_inputs/TrueBeam_10X_WaterPhantom.txt lendo TrueBeam_v2_10X_00
    e salvando dose_water.csv em modeling/output/.
  - .gitignore já ignora modeling/phsp/.

  O que falta (pós-processamento TOPAS → lookup tables DoseCUDA):

  - Do PHSP: histogramar fótons → spectrum.csv (colunas: energy_MeV,
    attenuation_coeff_cm2g, primary_weight, scatter_weight).
  - Do dose_water.csv: extrair PDD e perfis (eixo X/Y em profundidades típicas) →
    profile.csv; usar também para calibrar beam_parameters.csv (mu_calibration,
    distâncias, transmissões etc.).
  - Ajustar kernel CCC: via MC + ajuste (opcional depth-dependent) → kernel.csv ou
    kernel_depth_dependent.csv.
  - Preencher arquivos da máquina uma vez (energy_labels.csv, machine_geometry.csv,
    mlc_geometry.csv, HU_Density.csv), copiando o formato dos exemplos já no repo.

  Sugestão imediata:

  1. Rode o contêiner com seu PHSP (com os links .header/.phsp já indicados).
  2. Me envie as primeiras ~20 linhas de modeling/output/dose_water.csv; te devolvo o
     layout exato e um script Python curto para gerar PDD + perfis no formato do
     DoseCUDA.
  3. Se quiser, também gero o script para ler o PHSP e montar spectrum.csv nos bins
     que você escolher.

  Próximos passos que posso executar já:

  - Adicionar um scorer de plano (fluência/energia) ao macro para facilitar
    profile.csv.
  - Esqueleto de script tools/phsp_to_spectrum.py e tools/dosecsv_to_profiles.py
    salvando direto nos caminhos lookuptables/photons/<MACHINE>/<ENERGY>/.

  Diga se quer que eu já adicione esses dois scripts e um scorer de fluência ao macro.

