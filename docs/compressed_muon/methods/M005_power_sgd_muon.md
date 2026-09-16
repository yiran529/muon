# M005：PowerSGD-Muon

## 定位与状态

M005 是 `testing` 阶段的近似 Muon 变体。它只压缩 DDP 梯度同步阶段的部分
二维矩阵梯度，然后把每个 rank 重建出的相同近似梯度交给 unchanged ordinary
Muon。它不压缩 Muon 正交化结果、Muon 更新或 Muon 结果通信；这些步骤仍由
原有 Muon 路径负责。

由于正交化是非线性操作，
`Ortho(Average(G)) != Average(Ortho(G))`，所以 M005 不是 dense Muon 的逐元素
等价通信实现，而是一个有损的近似优化器路径。当前没有收敛、端到端吞吐或
speedup claim；这些结论必须等正式 timing 和 quality experiments 完成后再写入
结果文档。

## 方法语义

对 rank `i` 的可压缩矩阵梯度 `G_i`，EF14 开启时使用 rank-local error
`E_i` 构造：

```text
H_i = G_i + E_i
Q   = orthogonalize(Q_init)
P_i = H_i @ Q
P   = orthogonalize(AllReduceSUM(P_i))
Q_i = H_i.T @ P
Q̄   = AllReduceSUM(Q_i) / world_size
Ĝ   = P @ Q̄.T
E_i = H_i - Ĝ
```

首个压缩阶段的 `Q_init` 来自按 base seed、压缩 phase 和 stable parameter ID
确定性派生的 Gaussian factor；warm start 开启时，之后复用上一轮的 `Q`。关闭
EF14 时不维护 error；关闭 warm start 时每个压缩阶段重新生成 factor。`Ĝ` 在
各 rank 上相同，但一般不等于 dense DDP 平均梯度。

默认只把 Muon parameter group 中的二维参数视为 `matrix`。形状为 `m x n`、
有效秩为 `r = min(config.rank, m, n)` 的矩阵只有在

```text
min_compression_rate * r * (m + n) < m * n
```

时才压缩。embedding、LM head、scalar-optimizer、向量参数以及不满足条件的
矩阵都走精确 dense average。当前版本只支持固定 process group、静态参数集合、
`find_unused_parameters=False` 的 DDP；FSDP/HSDP、动态参数集合、world-size
变化恢复、三维矩阵 batch 和 AMP GradScaler skip/retry 不在支持范围内。

## 通信与 payload 口径

每个可压缩矩阵发送两个低秩因子：`P` 的形状为 `[m, r]`，`Q` 的形状为
`[n, r]`。因此单矩阵的因子 payload 是 `r(m+n)` 个元素；实际 bucket 还会
包含同 bucket 的 dense auxiliary 项和所有 dense fallback 项，且要经过两次
All-Reduce：

1. 第一次 All-Reduce 同时处理 dense entries 和各矩阵的 `P`；
2. 第二次 All-Reduce 处理各矩阵的 `Q`，之后重建近似梯度并更新 EF14 状态。

通信 buffer 默认使用 DDP bucket dtype；正交化在 FP32 中累积，再转换回通信
dtype。因而 `r(m+n)` 是逻辑因子 payload 公式，不是端到端通信时间、带宽收益
或训练 speedup 的承诺。两次 collective、dense fallback、额外正交化与重建开销
都必须在后续测量中单独计入。

## 实现路径

- Tensor 原语与配置：`dion/power_sgd.py`
- 稳定参数布局与跨 rank fingerprint：`dion/power_sgd_layout.py`
- DDP state、EF14/warm-start 状态、Future、packed collectives、CUDA stream
  pipeline 和 checkpoint：`dion/power_sgd_ddp_hook.py`
- DDP-only Muon 集成与 `GradientSyncRuntime`：`train_powersgd.py`
- 示例配置：`configs/compressed_muon/m005_power_sgd_muon_ddp.yaml`
- 测试：`tests/test_power_sgd.py`、`tests/test_power_sgd_layout.py`、
  `tests/test_power_sgd_ddp_checkpoint.py`、`tests/test_power_sgd_ddp_hook.py`、
  `tests/test_power_sgd_ddp_hook_distributed.py`，以及可选 CUDA/NCCL 测试
  `tests/test_power_sgd_ddp_hook_nccl.py`

状态生命周期为 `begin_step -> note_bucket -> finish_step -> commit_step`。
checkpoint 在 committed step 边界保存 error、warm-start `Q`、初始化标志和
schema metadata；transient packed buffers、CUDA events 与 Futures 不进入保存
状态。参数顺序、角色、dtype、shape、world size、group membership、配置
fingerprint 和 seed scheme 必须一致，避免恢复后静默改变通信布局。

## 当前证据与未完成 gate

Task 7 在当前实现上执行了：

- `.venv/bin/python -m compileall -q dion train_powersgd.py`：exit 0；
- PowerSGD focused suite（CPU tensor、布局、checkpoint、fake-bucket hook、
  两 rank Gloo、训练入口）：105 passed，14 warnings，28.36 s；
- repository-wide 的 unit/integration 子集（排除显式 multi-GPU、NCCL、CUDA
  graph 和 sharded-GPU tests）：957 passed，16 skipped，2 unrelated failures。

本次按任务约束没有启动训练、timing、formal profiler run、benchmark 或 CM
experiments，也没有捕获 profiler trace。Task 5 的最终 CUDA review fix 已在安全
空闲的本地 GPU 上执行
`.venv/bin/pytest -q -m multi_gpu tests/test_power_sgd_ddp_hook_nccl.py`：
`3 passed, 14 warnings in 22.59s`。其中 FP32 和 BF16 用例使用真实本地 two-rank
NCCL，覆盖 rank-skewed delay、allocator churn、returned-Future 可见性、三轮压缩、
mixed dense auxiliary、跨 rank collective signature 一致性以及真实 DDP backward；
另一个 warmed CUDA regression 在所选两张 GPU 上分别验证 `finish_step` aggregate
对多 bucket、独立 reconstruction stream 的梯度、EF14 error、Q memory 和
`q_initialized` 写入可见性。这是 correctness evidence，不是 formal experiment 或
性能结论。两项 broader-suite failure 属于既有 GreedyLore/profiler contract 漂移，
不涉及 M005：

- `tests/test_greedy_lore_profiler_launcher.py::test_print_plan_rotates_greedylore_modes_and_parameterizes_resources`
  预期的 `greedy_lore` mapping 缺少当前 plan 的
  `calibrated_bucket_cap_mb_list: None`；
- `tests/test_training_profiler_trace.py::test_greedylore_bucket_timeline_reports_queue_launch_and_completion_offsets`
  预期 timeline 缺少当前输出中的 `prepare_*` 和 `chain_wait_*` 字段。

仍待完成的 gate 是 multi-node correctness、端到端 training、profiler-off paired
timing、端到端通信/step timing，以及 dense Muon/M002/M005 的公平
quality/convergence 比较。PowerSGD 的 SGD 收敛结果不能直接外推到 nonlinear
Muon。
