# Modelagem Clarkson — análise do código e uso dos CSVs de comissionamento

Resumo curto
- Os scripts em `rt/connect/app/Libraries/Nanodicom` implementam um algoritmo de Clarkson (e uma versão modificada) que utiliza tabelas medidas (CSV) de PDP, Sp, Sc e afins presentes em `rt/connect/storage/app/maquinas/<equip>/<energy>/`.
- O `DoseCUDA` implementa um algoritmo do tipo CCC (convolution / collapsed-cone) que usa modelos de feixe (kernel.csv, beam_parameters.csv, spectrum.csv) em `DoseCUDA/lookuptables/photons/<machine>/<energy>/`.

Como o código Clarkson usa os dados (detalhes técnicos)
- `CSVManager` (rt): carrega CSVs da pasta `storage/app/maquinas/<Maquina>_<Energia>/...` e fornece `interpolate()` para consulta 2D (profundidade x campo) e 1D.
- Arquivos observados (ex.: `primus5770_pdp_open.csv`, `primus5770_sp.csv`, `primus5770_sc_open.csv`): formato com `;` como delimitador; primeira linha = colunas (tamanhos equivalentes / campos), primeira coluna = profundidades (mm) ou ângulos conforme o arquivo.
- `TMR.php`: fornece `TMR()` e `TMR_0()` que calculam valores de TMR a partir das tabelas PDP e do fator Sp (carregado com `CSV_Table->loadFromSuffix('_Sp')`). A rotina usa IQD (inverso do quadrado da distância) e interpola PDP por profundidade/campo.
- `Clarkson.php` e `ModifiedClarkson.php`: extraem forma do campo (imagem B/W), usam ray-tracing polar (`getImageRads`) para obter raios em todos os ângulos e integram SMR/TMR por setor (`calculateMediumSMR` ou lógica em `ModifiedClarkson`) para obter o contributo de espalhamento de campo irregular (modelagem de penumbra, MLCs, blocos). O fluxo principal:
  - monta `CSV_Table` com `setSearchFolder(<Maquina>_<Energia>)`
  - carrega `_Sp` e `_PDP_<open|filtro>` (sufixos usados no código)
  - interpola Sp e PDP, chama `TMR()` para cada raio/segmento e soma para obter SPR/TPR/SMR médios
  - calcula MU/Dose usando fatores de calibração, Sc, Sp, IQD e correções off-axis

Como o DoseCUDA (CCC) usa dados de comissionamento
- Em `DoseCUDA/plan_imrt.py` o `IMRTPlan` carrega lookuptables sob `DoseCUDA/lookuptables/photons/<machine>/<energy>/`:
  - `beam_parameters.csv` — parâmetros do modelo: `output_factor_equivalent_squares`, `output_factor_values`, `spectrum_*`, `mu_calibration`, `mlc_transmission`, `dlg`, etc.
  - `kernel.csv` — parâmetros do kernel usado pela convolução/CCC (parâmetros do kernel angular/ametrias) ou `kernel_depth_dependent.csv` se houver dependência com profundidade.
  - `spectrum.csv` / `beam_parameters.csv` — definem espectro, pesos e parâmetros que o núcleo CUDA consome.
- `IMRTBeamModel.outputFactor()` usa geometria (folhas/mandíbulas) para calcular uma área equivalente e interpola `output_factor_equivalent_squares` ↔ `output_factor_values` carregadas de `beam_parameters.csv`.
- O kernel e espectro são usados diretamente na função CUDA `photon_dose_cuda` (kernels em `dose_kernels/`), que separa primário e espalhado e aplica correções de transmissão/MLC.

Comparação direta: Clarkson vs CCC
- Natureza:
  - Clarkson: método baseado em medidas (PDP/Sp/Sc/TMR) e integração setorial (SMR) — bom para cálculos rápidos de campos planos e campo irregular via integração angular. Modelo empiricista (medidas diretas de PDD/TMR/Sp).
  - CCC (DoseCUDA): modelo físico-híbrido (convolução com kernel + espectro) — precisa de kernel, espectro e fatores de saída; é volumétrico e trata heterogeneidades via transporte baseado em kernel.
- Entradas necessárias:
  - Clarkson: precisa de PDP (PDD/TMR), Sp (output factors por campo) e, idealmente, Sc para correções da cabeça.
  - CCC: precisa de kernel.csv (função resposta no espaço), parâmetros de feixe (spectrum, weights, transmissões) e tabelas de output factor equivalentes.
- Escopo de uso dos CSVs `maquinas` no CCC:
  - O conteúdo das tabelas `maquinas` inclui PDP (profundidade x campo) e Sp (fatores de saída por campo) — estes dados são exatamente o que o Clarkson usa.
  - CCC em DoseCUDA já consome um conjunto de arquivos diferentes (kernel, beam_parameters.csv, spectrum.csv). Não existe uma leitura direta dos CSVs sem conversão.

Viabilidade prática de reutilizar os CSVs `maquinas` no CCC (conclusão)
- Sim, parcialmente: os CSVs medidos (PDP / Sp / Sc) podem ser reconvertidos para preencher partes de `beam_parameters.csv` e tabelas de output-factor do DoseCUDA:
  - `primus5770_sp.csv` (Sp) → pode ser mapeado para `output_factor_equivalent_squares` (cabecalhos) e `output_factor_values` (valores interpolados) depois de adaptar separador/formatos.
  - `primus5770_pdp_open.csv` (PDP/PDD por campo) → pode ser usado para derivar TMR/PDD usados para calibrar a profundidade/normalização; DoseCUDA usa `mu_calibration` + espectro para calibrar, mas PDP pode servir para ajuste/refino da calibração.
  - `sc` e `foa` presentes podem ser usados para compor correções de cabeça/BIAS quando for necessário.
- Limitações e trabalho necessário:
  - Formato: arquivos `maquinas` usam `;` e layout (primeira linha = campos) compatível com o `CSVManager` do RT, mas o `DoseCUDA` espera arquivos text/CSV com vírgula e formatos específicos (`beam_parameters.csv`, `kernel.csv`, `kernel_depth_dependent.csv`, `spectrum.csv`). Será preciso converter/gerar esses arquivos no formato esperado.
  - Kernels e espectro: os CSVs de `maquinas` não contêm kernels de convolução nem parâmetros espectrais necessários ao CCC. Esses precisam ser gerados (fitting) ou fornecidos pelo fabricante/modelo (Monte Carlo / ajustes). Sem kernel+espectro o CCC não funciona apenas com PDP/Sp.
  - Unidades / indexação: `CSVManager` do RT trabalha com mm e pode usar escala x10 em alguns CSVs (ver `valueAt(1,1)>=10` checks). A conversão deve respeitar essas convenções (profundidade em mm ou *10 em alguns arquivos xml-like).

Recomendações práticas — como proceder para usar os CSVs no CCC
1) Conversão inicial (rápida): converter `primus5770_sp.csv` para um `beam_parameters.csv` com duas linhas relevantes:
   - `output_factor_equivalent_squares, <lista de campos>`  (ex.: 20,30,50...) 
   - `output_factor_values, <valores correspondentes obtidos do CSV SP para campo= valor de referência>`
   - ajustar `mu_calibration` a partir de calibração clínica ou `primus5770_pdp_open.csv` (usando profundidade 10 cm).
2) Para maior fidelidade: usar `primus5770_pdp_open.csv` para estimar TMR/PDD e comparar com a resposta do modelo CCC em um conjunto de campos de validação; ajustar `mu_calibration` e `spectrum_*` até obter concordância.
3) Kernel: idealmente obter `kernel.csv` (ou `kernel_depth_dependent.csv`) a partir de Monte Carlo ou do fornecedor; como alternativa, tentar ajuste inverso: usar um conjunto de medidas (perfil + PDD + output factors) e otimizar parâmetros do kernel para reproduzir medições (trabalho não trivial).
4) Automatizar conversão: criar um script Python que leia os CSVs em `rt/connect/storage/app/maquinas/<equip>/<energy>/` e gere a árvore `DoseCUDA/lookuptables/photons/<machine>/<energy>/` com:
   - `beam_parameters.csv` (output_factor_equivalent_squares, output_factor_values, mu_calibration, mlc_transmission, etc.)
   - (opcional) `pdp_<...>.csv` usado apenas para documentação/ajuste
5) Validar: rodar calculos de campo aberto e alguns campos pequenos/irregulares comparando Clarkson (já presente) e CCC para ajustar parâmetros.

Conclusão curta
- Os CSVs de comissionamento do RT têm dados úteis (PDP, Sp, Sc) que podem e devem ser aproveitados para comissionar ou ajustar o modelo CCC do `DoseCUDA`, mas não são plug-and-play: é necessário converter formatos, mapear colunas/unidades e, sobretudo, fornecer ou ajustar o kernel e espectro exigidos pelo CCC.

Se quiser, eu posso:
- gerar um script de conversão automático (leitura `;` → escrita `beam_parameters.csv`) mapeando `primus*_sp.csv` → `output_factor_*` e `primus*_pdp_open.csv` → arquivo de suporte; ou
- começar a implementar uma rotina de fitting para extrair `kernel.csv` aproximado a partir de medidas (requer dados adicionais: perfis, PDDs e pontos de validação).

Arquivo gerado automaticamente pelo analisador de código em /home/rt/scripts/DoseCUDA
