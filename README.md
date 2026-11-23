# esm2-fitness-ood

**有限实验标签下 ESM-2 对未见位点单氨基酸替换效应的预测——ProteinGym 分层评测与 D-氨基酸氧化酶案例**  
*Predicting Unseen-Site Single Amino Acid Substitution Effects with ESM-2 under Limited Experimental Labels*

## 环境红线 / Environment Rules

- **本地仅限纯 CPU 轻量分析栈**（`.venv` `< 500 MB`，严禁在本地安装 `torch` / `CUDA`）。
- **所有缓存强制隔离至 `D:\Antigravity\cache\`**（`UV_CACHE_DIR`、`UV_PYTHON_INSTALL_DIR`、`PIP_CACHE_DIR`、`HF_HOME`、`MPLCONFIGDIR`）。
- **GPU 训练与推理统一在 Colab Pro 无状态运行**，仓库体积严格控制在 `< 30 MB`。

## Windows 本地入口

用 `scripts/project.ps1` 启动本地 Python、uv 或其他项目命令。它为子进程设置 D 盘缓存和临时目录。
使用 `scripts/run_cpu_baselines.py --export-only <新目录>` 从已验收记录生成 Wave 1 报表。

## 目录结构 / Repository Layout

- `src/esm2_ood/`: 核心数据处理、防泄露切分校验、基线与统计评估模块
- `tests/`: 自动化防泄露门禁测试（`pytest`）
- `configs/`: 10 蛋白锁定面板与实验超参配置
- `docs/`: 项目最终定稿方案（`PROJECT_PLAN_v3.1.md`）、Colab CLI 防白烧硬规则（`COLAB_CLI_SAFETY.md`）、ADR 决策日志与答辩资料
- `notebooks/`: Colab 5-Cell 无状态执行入口
- `scripts/`: 复算、分析与 300 dpi 绘图脚本
