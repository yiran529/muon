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

没有启动 formal training、timing、profiler trace capture、benchmark 或 CM run；
没有改动上述 GreedyLore 文件，也没有为 M005 编造 experiment ID、指标或质量
结果。Task 5 最终 CUDA review fix 在只读检查确认可安全使用的本地 GPU 上运行了
`.venv/bin/pytest -q -m multi_gpu tests/test_power_sgd_ddp_hook_nccl.py`：
`3 passed, 14 warnings in 22.59s`。FP32 与 BF16 case 均为真实本地 two-rank NCCL
correctness，覆盖 rank-skewed delay、allocator churn、returned-Future 可见性、三轮
压缩、mixed dense auxiliary、跨 rank collective signature 一致性与真实 DDP
backward。第三个 warmed CUDA regression 在所选两张 GPU 上分别验证：未等待任一
bucket Future 时，`finish_step` aggregate 仍能向 caller stream 暴露多 bucket、独立
reconstruction stream 上的 gradient、EF14 error、Q memory 和 `q_initialized`
写入。该 suite 是 correctness smoke，不构成 timing 或性能实验。

### 结论与下一步

实现、checkpoint schema、CPU/Gloo hook lifecycle 和 DDP integration 已有测试
覆盖，本地 two-rank NCCL correctness 与 aggregate CUDA visibility 也已有上述
短测试证据，方法记录进入 `testing`。下一步仍需完成 multi-node correctness、
端到端 training、profiler-off paired timing 和公平 quality/convergence runs；在此
之前不应把 payload 公式写成端到端 speedup，也不应把 PowerSGD 的 SGD 证据写成
Muon convergence 证据。

## 2026-09-16：CM093/CM094 rank32 主实验启动

按 GreedyLore 论文 Table IV 与 CM070 的低秩口径，只安排 M005 rank32 主实验：
CM093 为 GPT-60M/10,000 updates，CM094 为 GPT-130M/20,000 updates；均使用
4 GPU DDP、BF16参数、FineWeb10B、seq256、global/device batch512/128、
bucket160 MiB、EF14、warm start、step1000后压缩和 seed1234。dense 与 M002
对照复用 CM070，不重跑 high-rank PowerSGD。

## 2026-09-16：warm-start 与 batched orthogonalization 性能修复

针对 CM093/CM094 在压缩阶段约 3 倍于 dense 的 step time，修复两条明确的本地
开销路径：warm start 仅在首个压缩 phase 生成随机 Q，后续直接复用 `q_memory`；
相同 shape 的 Q/P 按组堆叠，以 3-D FP32 Gram–Schmidt 合并矩阵间 CUDA kernel，
保留原逐列算法、epsilon、BF16输入输出和 P/Q All-Reduce 语义。

新增 CM095 timing-only 实验，复用 CM089 的 GPT-130M BF16 训练几何，20 步
PowerSGD warmup 后测量 800 个稳态 rank32 PowerSGD step。CM089 的 calibrated role-isolation
是 GreedyLore 专属布局，CM095 不声称与其 bucket role layout 完全一致。

首次 controller 启动后按用户要求关闭 checkpoint 保存；训练 worker 被定向停止，
原 controller 与 CM093 早期产物以 `-aborted-20260916T181005-checkpoint-enabled`
后缀保留。两份正式配置均改为 `checkpoint_freq: 0`，随后使用原实验编号重新启动；
本次按要求未执行额外脚本测试。
