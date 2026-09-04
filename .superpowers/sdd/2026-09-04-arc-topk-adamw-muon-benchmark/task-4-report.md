# Task 4 report

## 完成内容

- 新增 `benchmark/compressed_muon/benchmark_arc_2x2.py`：精确模型预设、CLI/config 校验、AdamW/Muon dense 与 ARC optimizer factory、BF16 synthetic workload、CUDA event timing、rank-max 汇总、显存/逻辑通信字节及 JSON artifact skeleton。
- 新增 `benchmark/compressed_muon/profiler_trace.py`：Chrome trace 的 NCCL parent-range attribution，支持 correlation/external ID 与时间包含回退；按区间 union 计算 raw、overlap、trace-derived exposed 时间，并保留未归类 kernel。
- 新增 `benchmark/compressed_muon/summarize_arc_2x2.py`：三次以上重复结果的 mean/sample std/CV、dense-vs-ARC 的 `R_bytes`、`R_grad_comm`、`R_step`，CV 超阈值时保留输出并返回非零退出码。
- 新增 `tests/test_benchmark_arc_2x2.py`：预设、CLI/config、schema、合成 trace attribution 与汇总数学测试。
- 在 `dion/arc_topk.py` 与 `dion/megabatch_base.py` 增加仅用于 profiler 的 `record_function` ranges；未改变数学、collective 或调度顺序。

## 验证

- `/home/wyr/dion/.venv/bin/python -m pytest tests/test_benchmark_arc_2x2.py -q`：10 passed。
- `/home/wyr/dion/.venv/bin/python -m compileall -q benchmark/compressed_muon dion tests`：通过。
- `git diff --check`：通过。
- 相关 ARC/Muon 回归测试通过；首次 sandbox 运行受 localhost Gloo bind EPERM 影响，使用批准的本地 socket 权限复核通过（其中原有跳过项保持不变）。

## 限制与 concerns

- 本任务按要求未启动正式 GPU/2x2 实验；runner/profile 的 CUDA 路径仅完成代码级实现。
- trace attribution 对不同 profiler 版本采用 correlation 与时间包含回退；无法识别 parent 的 NCCL kernel 会显式归为 `unattributed`。
- profiler 的 exposed communication 是 trace-derived estimate，不应解读为独立的端到端通信时间。

## Round-1 修复

- 修正结果 communication schema keys，并确保逻辑字节估算使用压缩阶段 step（默认 step 2）。
- 增加独立 `--profile` 模式：fresh model/optimizer，固定 3 wait + 3 warmup + 5 active steps，解析并写入 profiler summary。
- 修复 NCCL/compute interval union 尾段 flush、并发区间合并及 exposed 非负约束；补齐 c10d correlation 传递和 seed/sketch/selected/dense/Muon ranges。
- summarizer 现在要求每个 optimizer/sync cell 至少 3 个独立结果，严格校验 invariant 与 profiler 通信数据。
- 接受 `CM002b-m001-adamw-arc-...` / `CM002d-m001-muon-arc-...` 语义 ID，拒绝非 1 gradient accumulation，校验 P2P transport 环境。
- 结果新增有限性、参数 checksum agreement 和 per-rank collective signature 字段；模型初始化使用 `config.seed`，Muon 显式记录 accelerated kernel 选择。

Round-1 focused verification：`71 passed, 14 warnings`（benchmark + ARC/Muon 回归）；compileall 与 `git diff --check` 通过。正式 GPU 实验仍未启动。

## Round-2 修复

- 新增 process-local collective observer，记录实际 `(category, operation, numel, dtype, bytes)`；dense DDP 使用 SUM/world-size averaging comm hook，ARC/Muon collective 调用均记录真实事件。
- `_run_steps()` 保存实际 detached loss；correctness 输出真实 finite-loss、参数有限性、跨 rank checksum agreement 与 observed per-rank signature agreement。
- trace attribution 支持 named user range → nested c10d launch/op → GPU NCCL correlation/external ID，补齐 seed 与 dense-uncompressed 分类；不从 kernel 时长推断通信字节。
- summarizer 支持独立 timing JSON 与 profiler-summary JSON 输入：step/CV 仅来自 timing，NCCL 与 `R_grad_comm` 仅来自 profiler，严格拒绝缺失/不足/不匹配的 profile cell。
- 非 profile timing 路径不再隐式 profile；profile-only 路径使用 fresh model/optimizer、固定 3 wait + 3 warmup + 5 active，并写入实际通信 metadata/signatures。

Round-2 focused verification：`75 passed, 14 warnings`；compileall 与 `git diff --check` 通过。未启动正式 GPU 实验。
