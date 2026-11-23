# Colab CLI 防白烧硬规则（Anti-Idle-Burn Protocol）

为彻底杜绝 WebSocket 断连、CLI 进程卡死或脚本异常导致的 GPU 空转扣费（历史教训：2026-11-23 `wave0b` 长连接断连卡死空转 33 分钟消耗 `1.14 CU`），所有通过 `google-colab-cli` 调度的云端 GPU 任务**必须严格遵守以下 6 条硬规则**：

1. **单条 CLI 命令硬超时（Hard CLI Timeout）**
   - 任何 `colab` 子命令外层必须包裹 `timeout 30`（文件下载/上传允许 `timeout 90`），严禁任何 `colab` 调用无限期阻塞。
2. **严禁长任务阻塞在 `colab exec` WebSocket 上**
   - `colab exec` 只允许用来发送 **2–5 秒即可返回** 的短指令（如启动 `nohup` 后台进程、检查 `DONE/FAIL` 标志、读取最后 20 行日志）。
   - 所有训练/推理任务必须在 Colab VM 内通过 `nohup python3 -u ... > job.log 2>&1 &` 后台运行，并在结束时写入原子标志文件 `DONE.flag` 或 `FAIL.flag`（含退出码）。
3. **强制设定单次运行时间上限（`MAX_WALL_MINUTES`）与本地看门狗**
   - 每次开机必须预设硬时间上限（如 Wave 0B 为 `12` 分钟；Wave 1–3 单批不超过 `45` 分钟）。
   - 本地调度循环每 `30` 秒短连查询一次状态；一旦满足以下任一条件，**无条件执行 `colab stop`**：
     - 检测到 `DONE.flag`（先拉回产物再停卡）；
     - 检测到 `FAIL.flag`（拉回 `job.log` 后立即停卡）；
     - 总挂钟时间超过 `MAX_WALL_MINUTES`；
     - 连续 `4` 次短连探活均失败。
4. **开机前先清孤儿、开机后 3 秒热机门禁**
   - 开机前先跑 `colab sessions` 确认没有残留会话；
   - `colab new` 成功后等待 `3` 秒，先跑一条 `nvidia-smi` 短探测（最多重试 3 次），不通则立刻 `colab stop` 退出，绝不带病挂机。
5. **关机后必须双核验（`Active assignments: 0`）与记账**
   - 每次 `EXIT` trap 必须执行 `timeout 30 colab stop -s <name>` 并紧接 `colab sessions` + `colab usage`，核实 `Active assignments: 0` 与 `Usage rate: 0.00/hr`，记录 `CU_before -> CU_after`。
6. **分批断点续跑（禁止单次大包一把梭）**
   - 正式 Wave 1–3 任务必须按蛋白/切分拆小批（每批预估 15–35 分钟），每完成一折立即落盘，即使中途触顶时间上限或网络中断也可无缝续跑。
