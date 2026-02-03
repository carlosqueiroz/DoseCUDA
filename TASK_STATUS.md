# DoseCUDA Photon Dose Engine – Status Tracker (Feb 3, 2026)

## Feito (implementado nesta rodada)
- **P0**: Removidos `__syncthreads()` inseguros (IMPT kernels) para evitar deadlock.
- **P1.1**: Jaws + leakage aplicados no `headTransmission` (jaws ∩ MLC, transmissões de jaw/MLC).
- **P1.2**: DLG aplicado (shift de tips) e tongue‑and‑groove aproximado (`tg_ext`) em largura efetiva de folhas.
- **P1.3**: Geometria MLC passa a usar `LeafPositionBoundaries` do DICOM como fonte primária; CSV apenas fallback, modelos recarregados ao primeiro boundary encontrado.
- **P1.4**: TERMA multi‑fonte separado (primário vs extra‑focal) com 1/r² adequado para scatter.
- **P1.5**: Reamostragem adicional por viagem de folha/jaw (≤1 mm) além do refinamento VMAT por ângulo.
- **P2.1**: Pesos angulares na CCC (`weight` em kernel.csv), default uniforme 1/72 se ausente.
- **P2.2**: Kernel dependente de profundidade (bins WET) suportado via `kernel_depth_dependent.csv`.
- **P2.4**: Suavização de heterogeneidade opcional (`heterogeneity_alpha`) aplicada ao TERMA na CCC.
- **P3 (parcial)**:
  - Output Factor reescrito: área por folha com interseção jaws∩MLC; perímetro robusto 2*(largura_média+altura).
  - HU: `computeIMRTPlan` aceita `ct_calibration_name`; múltiplas curvas HU_Density*.csv carregadas via `CTCalibrationManager`.
  - Validações ampliadas (kernel_len, kernel_weights, heterogeneity_alpha).
  - Hash de modelo inclui novos campos.
- **Testes**: `tests/test_output_factor_smallfield.py` (campo fechado = OF 0; campo 1x1 cm >0).

## O que falta para fechar todas as especificações
- **QA/Comissionamento (P3.x)**:
  - Suite automatizada de fantomas: água (PDD/prof), campos pequenos (0.5–2 cm), slabs água‑pulmão‑água e água‑osso‑água, VMAT convergência, teste de fuga MLC fechado.
  - Gamma targets (2%/2 mm, 2%/1 mm SRS) e checagem de regressão em CI.
  - Documentar/validar múltiplas curvas HU (seleção por scanner/protocolo) e fornecer exemplos.
  - Refinar Output Factor pequeno campo com dados comissionados (opcional espectro/TFd).
- **Modelos / LUTs**:
  - Atualizar `beam_parameters.csv` com `heterogeneity_alpha`, `tg_ext`, `dlg` calibrados.
  - Garantir `kernel.csv` possui coluna `weight` somando ~1; fornecer `kernel_depth_dependent.csv` se disponível.
  - Verificar energies distintas não compartilham arquivos idênticos (avisos já existem, mas criar teste).
- **Documentação**:
  - README/LOOKUPTABLES: descrever novos campos, formatos de kernel, depth bins, heterogeneity_alpha, tg_ext.
  - Passo‑a‑passo de recompilação (`pip install -e .`) e de seleção de curva HU.
- **Validação final**:
  - Rodar smoke tests GPU/CPU (campo aberto, fechado, VMAT simples) após recompilar a extensão.
  - Adicionar testes unitários para geometria MLC via LeafPositionBoundaries e para seleção de curva HU.

## Notas de uso imediato
- Para ativar kernel z‑dependente: fornecer `kernel_depth_dependent.csv` (cols: depth, angle_idx, Am, am, Bm, bm).
- Para angular weights: adicione coluna `weight` em `kernel.csv` (6 linhas, soma ≈ 1).
- Parâmetros novos em `beam_parameters.csv`: `dlg`, `tg_ext`, `heterogeneity_alpha`.
- Selecionar curva HU: `computeIMRTPlan(..., ct_calibration_name="NomeDaCurva")`; nomes vêm dos arquivos HU_Density*.csv.
