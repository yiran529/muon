# M005 PowerSGD-Muon Worklog

## 2026-09-16：实现记录与验证

### 实现提交

M005 的实现和修订由以下提交组成：

- `a42c78b`：PowerSGD tensor foundations；
- `125d1e6`：tensor validation；
- `6d88fea`：stable parameter layout；
- `9d1ca8e`：DDP state lifecycle；
- `78519c0`：PowerSGD DDP communication hook；
- `7eac1a0`：cancellation/failure lifecycle；
- `c9f240f`：bucket pipeline；
- `073a6c7`：CUDA completion export；
- `e42c43b`：PowerSGD-Muon training integration；
- `83415f5`：make the PowerSGD-Muon recipe runnable。

方法语义固定为：DDP gradient compression before unchanged ordinary Muon；默认
只压缩 Muon 的可获益二维矩阵，dense auxiliary 与不满足收益条件的矩阵精确
同步；每个矩阵的逻辑因子 payload 为 `r(m+n)`，但协议包含两次 All-Reduce。
这是 approximate Muon，不宣称 dense-equivalent、speedup 或 convergence gain。

### 实际验证

- `.venv/bin/python -m compileall -q dion train_powersgd.py`：exit 0。
- `.venv/bin/python -m pytest -q tests/test_power_sgd.py tests/test_power_sgd_layout.py tests/test_power_sgd_ddp_checkpoint.py tests/test_power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook_distributed.py tests/test_train_powersgd.py`：105 passed，14 warnings，28.36 s。
- `.venv/bin/python -m pytest -q -m 'not multi_gpu'`，并排除显式 NCCL、CUDA graph 和 sharded-GPU tests：957 passed，16 skipped，2 failed，15 warnings，11:00.30。

两项 failure 均为 unrelated GreedyLore/profiler contract drift，随后单独重跑仍为
`2 failed, 14 warnings in 3.69s`：

1. `tests/test_greedy_lore_profiler_launcher.py::test_print_plan_rotates_greedylore_modes_and_parameterizes_resources`
   期望 mapping 没有当前实现新增的 `calibrated_bucket_cap_mb_list: None`；
2. `tests/test_training_profiler_trace.py::test_greedylore_bucket_timeline_reports_queue_launch_and_completion_offsets`
   期望 timeline 没有当前实现新增的 `prepare_*` / `chain_wait_*` 字段。

没有启动 formal training、timing、profiler、benchmark 或 CM run；没有改动上述
GreedyLore 文件，也没有为 M005 编造 experiment ID、指标或质量结果。NCCL suite
保留为后续 exclusive-GPU correctness gate，本次没有在占用 GPU 的条件下运行。

### 结论与下一步

实现、checkpoint schema、CPU/Gloo hook lifecycle 和 DDP integration 已有测试
覆盖，方法记录进入 `testing`。下一步必须先完成可独占 GPU 的 NCCL correctness
gate，再进行 profiler-off paired timing 和公平 quality/convergence runs；在此
之前不应把 payload 公式写成端到端 speedup，也不应把 PowerSGD 的 SGD 证据写成
Muon convergence 证据。
