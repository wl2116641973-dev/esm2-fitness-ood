# Decision Log (ADR)

## ADR-001: Wave 0A Environment Isolation, Data Freeze, and Anti-Leakage Gates (2026-11-23)

- **Context**: Local machine has limited C-drive space (~17 GiB free) and must remain CPU-only (`< 500 MB` `.venv`, no PyTorch/CUDA). ProteinGym v1.3 benchmark data and 6 anti-leakage pytest gates must be verified before any Colab GPU runs.
- **Decisions**:
  1. **D-Drive Cache Isolation**: Pointed `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `PIP_CACHE_DIR`, `HF_HOME`, and `MPLCONFIGDIR` (User env vars) to `D:\Antigravity\cache\`.
  2. **Source Data & Manifest Freeze**: Downloaded ProteinGym v1.3 `DMS_substitutions.csv`, `DMS_substitutions_Spearman_DMS_level.csv`, `DMS_substitutions_Spearman_Uniprot_level.csv`, and `cv_folds_singles_substitutions.zip` (SHA256 `920F0BE936233B96B5052CD23679E42355CFD2B4E6F45B4F571EB79C0B2F9C35`). Extracted the 10 locked panel proteins (23,816 single mutants) plus `OXDA_RHOTO_Vanella_2023_expression` (6,387 shared mutants with activity) and `CASP3_HUMAN_Roychowdhury_2020`.
  3. **Numbering Check**: Confirmed `OXDA_RHOTO_Vanella_2023_activity` `target_seq` length is 364 aa and catalytic pocket residues align at 1-based DMS positions `Y222 / Y237 / R284`.
  4. **Anti-Leakage Gates**: Implemented `src/esm2_ood/` (`splits.py`, `features.py`, `metrics.py`, `baselines.py`, `manifest.py`) and verified all 6 pytest gates plus full 285-task CPU baselines (`site_mean`, `aa44_ridge`).
- **Status**: Accepted and verified.
