# Decision Log (ADR)

## ADR-001: Wave 0A Environment Isolation, Data Freeze, and Anti-Leakage Gates (2026-11-23)

- **Context**: Local machine has limited C-drive space (~17 GiB free) and must remain CPU-only (`< 500 MB` `.venv`, no PyTorch/CUDA). ProteinGym v1.3 benchmark data and 6 anti-leakage pytest gates must be verified before any Colab GPU runs.
- **Decisions**:
  1. **D-Drive Cache Isolation**: Pointed `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `PIP_CACHE_DIR`, `HF_HOME`, and `MPLCONFIGDIR` (User env vars) to `D:\Antigravity\cache\`.
  2. **Source Data & Manifest Freeze**: Downloaded ProteinGym v1.3 `DMS_substitutions.csv`, `DMS_substitutions_Spearman_DMS_level.csv`, `DMS_substitutions_Spearman_Uniprot_level.csv`, and `cv_folds_singles_substitutions.zip` (SHA256 `920F0BE936233B96B5052CD23679E42355CFD2B4E6F45B4F571EB79C0B2F9C35`). Extracted the 10 locked panel proteins (23,816 single mutants) plus `OXDA_RHOTO_Vanella_2023_expression` (6,387 shared mutants with activity) and `CASP3_HUMAN_Roychowdhury_2020`.
  3. **Numbering Check**: Confirmed `OXDA_RHOTO_Vanella_2023_activity` `target_seq` length is 364 aa and catalytic pocket residues align at 1-based DMS positions `Y222 / Y237 / R284`.
  4. **Anti-Leakage Gates**: Implemented `src/esm2_ood/` (`splits.py`, `features.py`, `metrics.py`, `baselines.py`, `manifest.py`) and verified all 6 pytest gates plus full 285-task CPU baselines (`site_mean`, `aa44_ridge`).
- **Status**: Accepted and verified.

## ADR-002: Wave 0B Colab L4 End-to-End Verification & Timing Calibration (2026-11-23)

- **Execution**: Controlled Colab Pro `--gpu L4` via WSL `google-colab-cli` using the detached watchdog runner (`scripts/colab_detached_wave0b.sh`, `docs/COLAB_CLI_SAFETY.md`). Uninstalled preinstalled `torchao==0.10.0` before `peft`.
- **Cost & Runtime**: Total GPU script elapsed `446.08s`; total session cost **`0.23 CU`** (`198.65 -> 198.42 CU`); post-flight verified `Active assignments: 0`, `Usage rate: 0.00/hr`.
- **Zero-Shot Calibration (`masked-marginals`)**:
  - `CASP3_HUMAN_Roychowdhury_2020` (out-of-panel pure single, 258 aa, 1,567): `8M = 0.2641` (official `0.264`), `650M = 0.6376` (official `0.638`).
  - `OXDA_RHOTO_Vanella_2023_activity` (pure single, 364 aa, 6,396): `8M = 0.1216` (official `0.122`), `650M = 0.3702` (official `0.370`).
  - `GFP_AEQVI_Sarkisyan_2016` (1,084 single-mutant subset): `8M = 0.0175`, `650M = 0.0514` (official `0.078 / 0.108` includes multi-mutants).
- **Per-Fold `650M LoRA-LLR` Timing & Baselines (`Contiguous`, `fold=0`, `N=256`, L4)**:
  - `CASP3_HUMAN` (258 aa): `zs=0.4679`, `hsu_aug=0.4679`, `frozen_ridge=0.4308` (`MSE=0.409`), `lora_llr=0.4483` (`MSE=9.9762`, `epoch=1`, `33.97s`).
  - `GFP` (238 aa): `zs=0.1914`, `hsu_aug=0.1914`, `frozen_ridge=0.1037` (`MSE=0.4687`), `lora_llr=0.2359` (`MSE=0.3983`, `epoch=9`, `84.98s`).
  - `OXDA_RHOTO` (364 aa): `zs=0.3846`, `hsu_aug=0.3846`, `frozen_ridge=0.2980` (`MSE=0.0063`), `lora_llr=0.3825` (`MSE=0.2211`, `epoch=1`, `47.69s`).
- **Action Item for Wave 1**: Initialize `scale_param` / `bias_param` via train-only closed-form OLS on epoch-0 `LLR` before LoRA optimization so `LoRA-LLR` MSE starts calibrated, and add `LoRA-LinearHead` (matched to Ridge readout) alongside `LoRA-LLR`.
- **Status**: Accepted; Wave 0B complete, ready for Wave 1.
