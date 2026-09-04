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
