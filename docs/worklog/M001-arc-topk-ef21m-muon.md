# M001 ARC-TopK-EF21M-Muon Worklog

本文件按时间追加 M001 的实现、验证和实验记录。单元测试、临时调试与 smoke test 不分配正式实验编号；正式比较实验启动后再关联 `CMxxx` 编号和 `artifacts/compressed_muon/<experiment-id>/` 产物目录。

## 2026-09-04：DDP 原型实现与自动化验证

### 目的与假设

目标是在不改变原始 Muon baseline 的前提下，将 ARC-TopK Algorithm 1 和 EF21M 公式 11a–11c 接入 DDP Muon。假设是：DDP backward 保留各 rank 的本地梯度后，可以用共享随机 sketch 对齐 Top-K 行支持集，通过 All-Reduce 聚合选中值，并将所有 rank 一致的 EF21M 全局梯度估计送入现有 Muon momentum、正交化和参数更新路径。

该实现属于新的 ARC-TopK-EF21M-Muon 优化器，而不是原始 Muon 的严格等价通信实现。EF21M tracker 后仍保留 Muon momentum，因此存在双动量。第一版只支持 DDP，不支持 FSDP、HSDP、TP 或 CUDA Graph capture。

### 修改与配置

- 新增 `dion/arc_topk.py`：参数校验、Gaussian projection、row sketch、共享 Top-K support、row gather/scatter、EF21M 状态递推，以及 seed broadcast、sketch All-Reduce 和 selected-values All-Reduce。
- 新增 `dion/muon_arctopk.py`：`ArcTopKMuon`、rank-symmetric task creation、矩阵梯度 ARC-TopK-EF21M 同步，以及 AdamW/Lion 标量参数的 dense All-Reduce。
- `dion/__init__.py` 导出 `ArcTopKMuon`。
- `train.py` 只增加通用 parser、hyperparameter factory 和 optimizer factory 注入点；没有增加 ARC 专属参数或分支，原始 `dion/muon.py` 未修改。
- 新增独立入口 `train_arctopk.py`，明确拒绝非 DDP `DeviceMesh`。
- 新增配置 `configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`，主要方法参数为：

  ```yaml
  arc_topk_ratio: 0.2
  arc_projection_rank: 4
  arc_eta: 0.1
  arc_seed: 42
  replicate_mesh_grad_sync: true
  dp_size: null
  fs_size: null
  tp_size: null
  checkpoint_freq: 0
  ```

- 新增本地、两 rank 分布式、optimizer state、训练入口和回归测试。
- 新增只读验证说明 `docs/compressed_muon/VALIDATE_M001_PROMPT.md`，供独立 agent 复核。

关联提交：

- `330ae73`：方法设计。
- `19348c1`：训练入口通用注入点。
- `4e8633e`：ARC-TopK 与 EF21M 本地张量操作。
- `efc82cf`：完整分布式 ARC-TopK collective。
- `1af1667`：接入 Muon 和状态恢复。
- `ba258ca`：AdamW/Lion 标量参数同步与两 rank 一致性。
- `ed4d7ee`：独立训练入口和 DDP 配置。
- `8c736af`：验证 prompt、实施计划和方法状态更新。

### 验证

#### M001 聚焦测试

执行以下测试集合，包含两 rank Gloo collective：

```text
tests/test_train_factories.py
tests/test_arc_topk.py
tests/test_arc_topk_distributed.py
tests/test_muon_arctopk.py
tests/test_muon_arctopk_distributed.py
tests/test_train_arctopk.py
```

结果：`37 passed, 0 failed, 0 skipped`。覆盖固定 seed、共享 support、selected-values 平均、`ratio=1` dense-average 语义、EF21M 多步递推、局部梯度缺失、两 rank 参数一致性、AdamW/Lion dense 同步，以及 optimizer `state_dict()` 保存恢复。

#### 原有相关回归

M001 聚焦测试与以下原有回归在 GPU 可见的授权环境中合并执行：

```text
tests/test_configs.py
tests/test_state_prepopulation.py
tests/test_optimizers.py
tests/test_dion3_alias.py
```

结果：`172 passed, 0 failed, 0 skipped`。其中 37 项为 M001 聚焦测试，135 项为相关回归测试。

独立验证 agent 在默认沙箱中重复执行时，M001 聚焦测试仍为 `37 passed`；相关回归为 `34 passed, 101 skipped`。101 个 skip 全部源于默认沙箱隐藏 CUDA 设备，随后已由上述 GPU 可见运行覆盖，并非功能失败。

#### 两卡 NCCL smoke test

启动前检查宿主机 GPU：共 8 张 RTX 4090；GPU 0、1 已有其他进程占用，未干扰；选择基本空闲的 GPU 2、3。使用临时脚本执行两卡 CUDA DDP：

```text
DDP no_sync 下生成不同本地梯度
-> ARC-TopK sketch NCCL All-Reduce
-> selected-values NCCL All-Reduce
-> EF21M 状态更新
-> Muon momentum、正交化和参数更新
```

配置使用 `arc_topk_ratio=0.5`、`arc_projection_rank=2`、`arc_eta=1.0`。结果：两个 rank 的本地梯度不同，更新后的 `arc_g_global`、Muon momentum 和参数一致；参数发生有限值更新。临时脚本已删除，没有保留正式产物，也没有分配实验编号。

#### 静态检查

- `python -m compileall`：通过。
- `git diff --check`：通过。
- 独立 agent 按 `VALIDATE_M001_PROMPT.md` 完成人工执行路径核查，最低验证门结论为通过。

#### 完整仓库测试中的既有失败

完整测试尝试观察到 8 个 `tests/test_dion2_post_ortho_triton.py::test_post_ortho_triton_falls_back_for_wrapper_subclass[...]` 失败。失败在 CPU 沙箱和真实 GPU 环境均可复现，堆栈位于 PyTorch 2.11 Dynamo/AOTAutograd 对 traceable wrapper subclass 的编译，错误为 `GuardOnDataDependentSymNode`。

M001 未修改 `dion/dion2.py`、`dion/dion2_triton.py`、对应测试、PyTorch 依赖或锁文件；`ArcTopKMuon` 也不调用 Dion2 post-orthogonalize 路径。因此该失败不归因于 M001，但会导致当前仓库完整 pytest 不能报告全绿。

### 结果与观察

- 完整 ARC-TopK + EF21M 已进入独立 DDP Muon 正常执行路径，没有修改原 Muon baseline。
- Gloo 测试验证了算法递推和 collective 语义；NCCL smoke test 验证了两卡 CUDA 通信与参数一致性。
- `ratio=1` 在指定首步、`eta=1` 条件下与 dense DDP gradient average 一致；有损比例不要求与原 Muon 逐元素一致。
- 当前验证只能说明实现和主要通信路径可执行，不能说明训练收敛、最终精度、吞吐或端到端通信收益。
- ARC 降低的是 DDP 矩阵梯度同步的载荷，但 Muon 正交化任务分配与结果收集通信仍然存在；实际收益需要 profiler 或 benchmark 测量。

### 结论

M001 DDP 原型达到当前阶段的自动化测试和两卡 NCCL smoke-test 完成标准，方法状态保持 `testing`。尚不足以进入论文结果或宣称通信加速。

### 下一步

1. 使用小模型进行短训练，检查多步 loss、梯度/状态有限性和 rank 一致性。
2. 增加 optimizer-step 与端到端 profiler，分别测量 sketch、selected-values 和 Muon 原有通信成本。
3. 与原 Muon dense DDP baseline 比较吞吐、通信时间、显存和收敛趋势。
4. 对 `arc_topk_ratio`、`arc_projection_rank`、`arc_eta` 和双动量组合进行消融。
5. 若要启用长训练，先设计并验证 rank-local `arc_h_local`、`arc_g_local` 的分布式 checkpoint 语义。

### 关联位置

- 方法说明：`docs/compressed_muon/methods/M001_arc_topk_ef21m_muon.md`
- 方法索引：`docs/compressed_muon/METHOD_INDEX.md`
- 实施计划：`docs/superpowers/plans/2026-09-03-arc-topk-ef21m-muon.md`
- 验证 prompt：`docs/compressed_muon/VALIDATE_M001_PROMPT.md`
- 配置：`configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`
- 正式实验编号：无。
- 正式产物路径：无。

## 2026-09-04：对齐论文与官方仓库的初始化、warmup 和 dtype

### 目的与假设

对照 ARC-TopK 论文与官方发布仓库，修正第一版原型的三处语义或实现差异：EF21M 从零状态直接压缩导致首步通常不满足 `g_0 = h_0`；缺少论文实验使用的压缩 warmup；BF16 梯度的 sketch 被固定提升到 FP32，增加了实际通信字节。

### 修改与配置

- 首个 optimizer step 直接令本地 `h_0 = g_0 = grad_0`，并通过一次 dense All-Reduce 得到全局 `g_0`，使 `V_0 = ||g_0-h_0||^2 = 0`。
- 新增 `arc_start_compress_step`，默认值和 M001 配置均为 `1000`。第 1–1000 步保持 dense tracker 同步，第 1001 步开始 ARC-TopK；设为 `0` 时仅保留首步 dense 初始化。
- warmup 期间仍按 `eta` 更新 `h_t`，随后令本地 `g_t=h_t` 并 dense 聚合全局 tracker，不生成 ARC seed 或 sketch。
- Gaussian projection 和 sketch 改为跟随梯度/EF21M 状态 dtype，保留论文的 `1/sqrt(r)` 缩放。
- 旧 M001 checkpoint 加载时若缺少新字段，将 `arc_start_compress_step` 迁移为 `0`，保持旧版本立即压缩的恢复语义。
- 更新 M001 方法说明、验证 prompt、配置以及本地和两 rank 分布式测试。

### 验证

- TDD 红测：新增行为测试在修改前出现 `17 failed, 20 passed`，失败原因分别为缺少 warmup 参数、首步仍执行稀疏压缩、BF16 sketch 被提升为 FP32。
- M001 聚焦测试（含两 rank Gloo）：`49 passed, 0 failed, 0 skipped`。
- 原 Muon、配置、状态预创建和 Dion3 alias 回归：在 GPU 2、3 可见环境中得到 `131 passed, 0 failed, 4 skipped`；4 项需要超过 2 张 GPU 的参数化测试因设备数不足而跳过。
- 两卡 BF16/NCCL smoke test：GPU 2、3 上验证首步 dense 初始化、warmup 后进入 ARC、BF16 输出 dtype 和两 rank 全局估计一致，结果通过；临时脚本已删除。
- `python -m py_compile dion/arc_topk.py dion/muon_arctopk.py train_arctopk.py`：通过。
- `git diff --check`：通过。
- 独立代码审查发现旧 M001 checkpoint 缺少新增 warmup 字段会在恢复后触发 `KeyError`；增加迁移和回归测试后完成复审。

### 结果与观察

- 首步不再受 `ratio` 和 `eta` 缩放影响，`h_0`、本地 `g_0` 等于完整本地初始梯度，全局 `g_0` 等于各 rank 初始梯度平均。
- warmup 截止步仍走 dense tracker 同步，超过阈值后才执行 seed broadcast、sketch All-Reduce 和 selected-values All-Reduce。
- BF16 输入的 projection、sketch、selected values 和全局估计保持 BF16；不再因固定 FP32 sketch 产生双倍 sketch 字节。

### 结论和下一步

三处已知不一致已在 M001 正常执行路径中修正。当前结论仍只覆盖算法路径与短 smoke test，不代表端到端吞吐、通信收益或训练收敛已经得到验证。下一步应使用 dense DDP Muon 作为基线进行短训练和 profiler 对比。

### 关联位置

- 代码：`dion/arc_topk.py`、`dion/muon_arctopk.py`、`train_arctopk.py`
- 配置：`configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`
- 测试：`tests/test_arc_topk.py`、`tests/test_arc_topk_distributed.py`、`tests/test_muon_arctopk.py`、`tests/test_train_arctopk.py`
- 正式实验编号：无。
- 正式产物路径：无。

## 2026-09-04：将 GPT compression warmup 调整为 300 并启动 CM001

### 目的与假设

论文实验统一在 1000 iterations 后开始压缩，但其 C4 训练明显长于当前 3000-step GPT 配置。为避免三分之一训练都使用 dense 通信，当前配置改用约占总步数 10% 的 300-step compression warmup。除 ARC-TopK-EF21M 专属参数和 optimizer-side DDP gradient sync 外，训练超参数沿用此前完成的 4 卡 DDP Muon 160M baseline。

### 修改与实验配置

- `arc_start_compress_step` 的 M001 默认值和正式配置从 `1000` 改为 `300`，第 301 个 optimizer step 开始 ARC-TopK。
- 实验编号：`CM001-m001-gpt160m-ddp-ws4-s42`。
- 4 GPU DDP，模型 162M 参数，`batch_size=1024`、`device_batch_size=32`、`sequence_length=1024`、`num_iterations=3000`。
- `lr=0.02`、`mu=0.95`、`weight_decay=0.01`、`adjust_lr=spectral_norm`、scalar optimizer 为 AdamW。
- ARC 参数：`ratio=0.2`、`projection_rank=4`、`eta=0.1`、seed `42`、compression warmup `300`。
- 启用 W&B；不保存 checkpoint；训练在 `tmux` 中运行。

### 验证、结果与观察

- warmup 默认值和 M001 配置的聚焦测试：`2 passed`。
- 提交：`d0fbd11`（`config: use 300-step ARC warmup`）。
- 宿主环境确认 CUDA 可用且共有 8 张 GPU；GPU 0、1 已被其他任务占用，因此沿用此前 DDP baseline 使用的空闲 GPU 2、3、4、5，没有干扰既有进程。
- 2026-09-04 01:13 CST 在 tmux 会话 `cm001_m001_arctopk` 启动，W&B run id 为 `21qkn1do`。
- step 0 validation loss 为 `11.2487`；首个 optimizer step 完成，step 1 train loss 为 `11.2486`。首步包含 `torch.compile` 开销，用时约 107.67 秒。
- 停止日志跟踪后再次检查，进度已继续到 step 19，近期步耗约 2.3 秒，进程与 tmux 会话均仍在运行；未因启动问题修改 batch size 或其他 baseline 超参数。

### 结论和下一步

启动阶段已确认第一个 optimizer step 成功且后续训练继续运行，之后按约定不持续盯守。完整结果待训练自行结束后整理。

### 关联位置

- 配置：`configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`
- 实验登记：`docs/compressed_muon/EXPERIMENTS.md`
- 产物：`artifacts/compressed_muon/CM001-m001-gpt160m-ddp-ws4-s42/`

## 2026-09-04：CM001 完成及与论文 wall-clock 结果的解释

### 目的与假设

整理 CM001 完整训练结果，并判断 ARC-TopK 在当前 Muon DDP 训练中是否带来与论文 Table V 类似的 wall-clock 收益。比较时区分理论通信量、通信受限 microbenchmark 和端到端训练时间，避免将论文的最高降时比例直接外推到当前硬件与负载。

### 实验配置与对照口径

- ARC-TopK 实验：`CM001-m001-gpt160m-ddp-ws4-s42`，4 张 RTX 4090，GPT 162M，Muon + AdamW，global batch size 1024，device batch size 32，梯度累积 8，sequence length 1024，共 3000 steps。
- dense baseline：`trial-baseline-gpt160m-ddp-ws4-wandb`，W&B `hpo9y2w4`；除 ARC-TopK 专属设置和 optimizer-side DDP gradient sync 外，核心训练超参数相同。
- CM001 前 300 steps 使用 dense tracker 同步，第 301 个 optimizer step 开始 `ratio=0.2`、projection rank `r=4` 的 ARC-TopK。
- 论文 Table V 测量 4 张 A100 40 GB 上 LLaMA 60M、130M、350M 和 1B 的 50-step 平均时间；使用 Adam、local batch size 1、C4 sequence length 256、NCCL SHM transport，并明确禁用 NVLink P2P 来模拟低带宽多机环境。

### 验证、结果与观察

- CM001 于 2026-09-04 03:08 CST 正常结束，`exit_code=0`，完成全部 3000 steps。
- dense Muon baseline：平均 step time `2186.40 ms`，约 `479590 tokens/s`，训练计时 `6537.327 s`，最终 validation loss `3.3469`，峰值显存 `18033 MiB`。
- ARC-TopK Muon：平均 step time `2207.68 ms`，约 `474967 tokens/s`，训练计时 `6600.960 s`，最终 validation loss `3.6581`，峰值显存 `19023 MiB`。
- 在完整 3000 steps 口径下，ARC-TopK Muon 比 dense baseline 慢约 `0.97%`；只比较压缩稳定启用后的 step 500–3000，平均 step time 分别约为 `2204.24 ms` 和 `2185.52 ms`，ARC-TopK 仍慢约 `0.86%`。因此差异不能归因于前 300-step dense warmup。
- ARC-TopK 的峰值显存增加 `990 MiB`，约 `5.5%`；当前参数下 validation loss 比 baseline 高 `0.3112`。
- 论文 Table V 中，ARC-TopK 相对 dense 的单步降时随模型规模增长：60M 为 `27.8%`、130M 为 `44.6%`、350M 为 `57.3%`、1B 为 `60.7%`。该趋势符合其刻意构造的通信占主导条件：local batch 很小、模型逐渐增大、P2P 被禁用，dense 梯度同步占比随模型规模上升。
- 当前训练每卡每个 optimizer step 处理 `32 × 8 × 1024 = 262144` tokens，而论文 wall-clock 设置按 local batch 1、sequence length 256 仅处理约 256 tokens。当前每次梯度同步之前的计算量大约高三个数量级，通信成本被前向、反向和 Muon Newton–Schulz 正交化摊薄。
- 论文使用 DDP communication hook 处理 gradient bucket，而 M001 在 optimizer 路径中逐矩阵执行 projection、Top-K 和 collective。当前路径增加小 kernel 与 collective 调度开销，也较难利用标准 DDP bucket 合并和反向传播期间的通信计算重叠。
- 论文 Table V 是 50-step 平均单步 microbenchmark，不是包含长期训练、周期验证和达到相同质量所需时间的 time-to-quality 结果。论文总体实验称 1000 iterations 后开始压缩，但 Table V 只运行 50 iterations，公开仓库也未提供完整复现该表的独立命令，因此该表实际使用的 compression-start 覆盖设置存在报告缺口。

### 通信缩减没有转化为 wall-clock 收益的机制判断

1. **计算与通信比例不同。** 当前每卡每个 optimizer step 在一次梯度同步前处理 262144 tokens，前向、反向和 Muon Newton–Schulz 占据主要时间；论文 wall-clock 设置每卡仅处理约 256 tokens，且主动禁用 P2P，使 dense All-Reduce 成为主要瓶颈。当前即使缩短一部分梯度通信，对约 2.2 秒完整 step 的影响也有限。
2. **梯度累积降低了单位 token 的同步频率。** 当前梯度累积为 8，每卡完成 8 个 microbatches 后才同步一次；与 local batch 1、gradient accumulation 1 的通信受限设置相比，同样大小的梯度通信被更多计算摊薄。
3. **压缩降低 payload，但增加计算和调度。** 每个 ARC 矩阵需要生成投影、计算 sketch、选择 Top-K、更新 EF21M 状态，并通信 sketch 和 selected values。通信字节下降不代表这些额外工作为零；当 dense collective 已较快时，额外开销可以完全抵消带宽收益。
4. **collective 粒度和重叠方式发生变化。** dense DDP 可以将多个梯度放入 bucket，并在 backward 中异步启动 All-Reduce。M001 在 backward 和梯度累积完成后进入 optimizer，再逐矩阵执行多个 collective；这既增加 collective 启动次数和延迟，也减少通信与 backward 重叠。因而理论 payload ratio 不能直接换算为相同的 NCCL 时间比例。
5. **ARC 只覆盖部分通信对象。** M001 优化的是二维矩阵梯度同步；未压缩张量以及 Muon 正交化任务分配、结果聚合等其他通信不会按 `ratio=0.2` 同比例下降。Muon 的 Newton–Schulz 计算也不受梯度压缩影响。
6. **单机互联不属于论文强调的低带宽场景。** 当前 4 张 RTX 4090 使用机器的正常 NCCL 路径；论文 Table V 明确使用 SHM transport 并禁用 NVLink P2P。ARC-TopK 的收益依赖 dense 通信在关键路径中占有足够高的比例，而不是只依赖参数量或压缩率。
7. **端到端效率还受收敛影响。** CM001 最终 validation loss 高于 dense baseline。即使未来测得单步加速，也必须比较达到相同 validation loss 所需的 wall-clock；否则只能说明通信或吞吐改善，不能说明训练效率改善。

可用 Amdahl 近似解释预期上限。设 dense step 中可被 ARC 优化的梯度通信占比为 `f_comm`，ARC 对该部分的实际时间降幅为 `R_grad_comm`，额外压缩开销占完整 dense step 的比例为 `o_arc`，则：

```text
R_step ≈ f_comm × R_grad_comm - o_arc
```

即使乐观假设 `R_grad_comm=80%` 且 `o_arc=0`，要达到 `60%` 的端到端降时也要求 dense baseline 中可压缩通信至少占 `75%`。考虑 projection、Top-K、EF21M、更多 collective 和不可压缩通信后，所需通信占比还会更高。CM001 约 `0.97%` 的负收益说明当前 `f_comm × R_grad_comm` 不足以覆盖 `o_arc`，但仅凭端到端时间还不能确定各项的具体占比，需要 profiler 验证。

### 推荐给后续 agent 的测试设置

#### 研究问题和比较矩阵

目标是判断在论文式通信受限设置下，ARC-TopK+Muon 能否获得与 ARC-TopK+AdamW 相近的**梯度通信收益**，并进一步判断该收益能否转化为相近的**端到端收益**。至少完成以下 2×2 对照：

| 优化器 | Dense | ARC-TopK |
|---|---|---|
| AdamW | AdamW Dense | AdamW ARC-TopK |
| Muon | Muon Dense | Muon ARC-TopK |

四组必须固定模型结构、数据、dtype、随机种子、world size、GPU、NCCL transport、local batch、sequence length、gradient accumulation 和测量区间。ARC 两组还必须固定 compression ratio、projection rank、EF21M 参数、compression-start 语义及压缩张量集合。

若 AdamW 使用 DDP bucket comm hook、Muon 使用当前逐矩阵 optimizer 路径，结果只能解释为两套实际系统的比较，不能把差异全部归因于 optimizer。为了回答“同一 ARC 通信实现对两个 optimizer 是否有同等收益”，优先让两者共用同一压缩层和 collective 组织；若当前阶段做不到，必须同时记录 bucket 数量、逐矩阵 collective 数量和通信重叠差异，并明确结论边界。

#### 第一阶段：论文式通信受限短 benchmark

- 使用 4 张经检查为空闲的 GPU；模型先测约 60M、130M、350M，显存允许再测 1B。同一模型规模的四个实验必须使用完全相同的 GPU。
- `local_batch_size=1`、`sequence_length=256`、`gradient_accumulation=1`，使每次梯度同步前的计算量接近论文 wall-clock 设置。
- 设置 `NCCL_P2P_DISABLE=1`、确认 `NCCL_SHM_DISABLE=0`。首次运行可用 `NCCL_DEBUG=INFO` 保存 transport 证据，正式计时关闭冗余 debug 输出。
- dtype 在四组间必须一致并写入产物。论文 Table V 未清楚报告 dtype，因此当前硬件上的结果用于验证相对趋势，不宣称绝对复现其秒数。
- ARC 使用 `ratio=0.2`、projection rank `r=4`。短 benchmark 必须保证正式计时区间已经启用压缩；可设 compression start 为 0，但丢弃首步 dense 初始化和编译阶段。
- 至少进行 20 个不计时 warmup steps，再测量至少 100 个稳定 steps。每个配置使用独立进程重复至少 3 次，报告均值、标准差和变异系数；若变异系数超过 5%，继续排查资源竞争或增加重复次数。
- 测量区间关闭 validation、checkpoint 和其他周期任务；W&B 和日志不得把同步 I/O 放入计时区间。计时边界使用 CUDA event 或显式 `torch.cuda.synchronize()`，不能只依赖未同步的 CPU wall clock。
- 先在正常 NCCL transport 下跑一组，再在禁用 P2P 的论文式 transport 下跑一组，用于区分实际部署收益和人为通信受限收益。

#### 必须采集的指标

1. 每 step 理论/实测通信字节，分别列出 dense gradient、ARC sketch、selected values、未压缩张量和 Muon 其他通信。
2. collective 类型、次数、消息大小和 NCCL GPU kernel 总时间。
3. backward 中被覆盖的通信时间，以及真正暴露在 step 关键路径上的通信时间。
4. ARC projection、norm/Top-K、EF21M 状态更新和压缩/解压的 GPU 时间。
5. forward、backward、Muon Newton–Schulz、optimizer 和完整 step 时间。
6. tokens/s、峰值显存，并保存至少一个代表性 PyTorch Profiler 或 Nsight Systems trace。

对 AdamW 和 Muon 分别计算：

```text
R_bytes     = 1 - ARC通信字节 / Dense通信字节
R_grad_comm = 1 - ARC梯度同步时间 / Dense梯度同步时间
R_step      = 1 - ARC完整step时间 / Dense完整step时间
```

#### 预先约定的判定方式

- 若 Muon 与 AdamW 的 `R_bytes` 相差不超过 5 个百分点，可认为两者获得近似相同的理论梯度通信量缩减。
- 若两者 `R_grad_comm` 相差不超过 5 个百分点，且 3 次重复的误差范围不改变结论，可认为 ARC-TopK+Muon 获得与 AdamW 相近的实际梯度通信收益。
- 若 `R_grad_comm` 接近但 Muon 的 `R_step` 明显更低，应将差异归因到 Muon 非通信计算、不可压缩通信、ARC 实现开销或通信重叠，而不能得出“ARC 对 Muon 不压缩”的结论。
- 只有两者 `R_step` 也接近且均为正，才能声称 ARC-TopK+Muon 在该设置下获得与 AdamW 类似的 wall-clock 收益。
- 若目标是训练效率而非 microbenchmark，必须另做足够长的收敛实验，比较达到同一 validation loss 所需的总时间；短 benchmark 不用于证明模型质量。

#### 执行和产物要求

- 开始前重新阅读 `AGENTS.md` 和 `docs/compressed_muon/RESEARCH_GUIDE.md`，检查 GPU 与既有进程，不得干扰其他用户任务。
- 在 `docs/compressed_muon/EXPERIMENTS.md` 选择下一个未使用的 `CMxxx` 编号。同一 2×2 比较组可使用 `a`–`d` 后缀，并保证本地目录、W&B run name 与实验编号对应。
- 长任务使用 `tmux`；原始日志、配置、命令、环境信息、计时数据和 profiler trace 放入 `artifacts/compressed_muon/<experiment-id>/`。
- 可复用 benchmark/profiler 脚本放入 `benchmark/compressed_muon/`，不要放在仓库根目录；一次性 launcher 可放在对应 artifacts 目录。
- 先完成最小模型的 2×2 smoke/benchmark 并检查四组均走预期通信路径，再扩展模型规模。collective 调用顺序、张量大小和各 rank step 数必须一致。
- 实验结束后更新 `EXPERIMENTS.md`，并将成功、失败或结论不明确的结果都追加到本 worklog；只在证据具有正式比较价值后再整理到 `RESULTS.md` 或 `PAPER_NOTES.md`。

### 结论和下一步

CM001 没有观察到端到端加速；在当前单机 4 卡、162M 模型、大 batch、长序列和 Muon 额外正交化计算的组合下，梯度通信不是足够大的瓶颈，ARC-TopK 节省的通信时间不足以覆盖 sketch、Top-K、EF21M 状态和逐矩阵 collective 的额外成本。该结果不否定 ARC-TopK 在大模型、更多节点或低带宽环境中的潜在收益，但目前不能宣称 ARC-TopK+Muon 具有论文所报告的 wall-clock 加速。

下一步若继续性能研究，应先使用 profiler 分离前向、反向、dense gradient All-Reduce、ARC sketch/selected-values collective、Muon Newton–Schulz 和 Muon 结果聚合的时间，再决定是否开展更大模型、更多节点和通信受限环境下的公平对照。性能比较还应同时报告 optimizer-step 时间、端到端吞吐、通信时间、显存，以及达到相同 validation loss 所需的 wall-clock 时间。

### 关联位置

- ARC-TopK 论文：`https://arxiv.org/pdf/2510.26709`
- 官方实现：`https://github.com/pkumelon/ARC-TopK-release`
- ARC-TopK 配置：`configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`
- ARC-TopK 产物：`artifacts/compressed_muon/CM001-m001-gpt160m-ddp-ws4-s42/`
- dense baseline 产物：`artifacts/compressed_muon/trial-baseline-gpt160m-ddp-ws4-wandb/`
- W&B：ARC-TopK `21qkn1do`；dense baseline `hpo9y2w4`

## 2026-09-04：CM002/CM003 60M ARC-TopK AdamW/Muon formal benchmark

### 目的、假设与执行口径

本次实验比较相同 60M GPT DDP 工作负载下 AdamW/Muon 的 dense 与 ARC-TopK 路径，分别覆盖正常 NCCL 和 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0` 的通信受限环境。假设是：ARC-TopK 对矩阵梯度使用相同的 sketch、Top-K、EF21M 和 selected-values collective，因此 AdamW 与 Muon 应有近似的理论梯度载荷缩减；实际梯度通信时间和完整 step 是否相近则由 profiler/timing 直接测量。

代码与配置：benchmark metadata 的 commit 为 `78d36f0e3bee4c25dba0901c38ad86d717cad759`，核心 benchmark/optimizer code gate commit 为 `a55103d`；BF16，gpt60m（64,094,208 parameters），world size 4，local batch 1，sequence length 256，gradient accumulation 1，seed 42，ARC ratio 0.2、projection rank 4、eta 0.1、compression start 0。每次 timing 为 fresh process、20 warmup + 100 measured steps；CM002a–d 与 CM003a/c/d 各 3 次，CM003b AdamW ARC 因初始 CV 超过 5% 增加 fresh r4/r5，共 5 次。Profiler 与 timing 分离，每个 cell 独立执行 3 次严格 3 wait + 3 warmup + 5 active schedule，共 24 次；不将 profiler step 样本混入 timing 均值。

八个正式实验 ID 为：`CM002a-adamw-dense-gpt60m-ddp-ws4-s42`、`CM002b-m001-adamw-arc-gpt60m-ddp-ws4-s42`、`CM002c-muon-dense-gpt60m-ddp-ws4-s42`、`CM002d-m001-muon-arc-gpt60m-ddp-ws4-s42`、`CM003a-adamw-dense-gpt60m-ddp-ws4-s42`、`CM003b-m001-adamw-arc-gpt60m-ddp-ws4-s42`、`CM003c-muon-dense-gpt60m-ddp-ws4-s42`、`CM003d-m001-muon-arc-gpt60m-ddp-ws4-s42`。所有运行固定 GPU 2–5：GPU 2 `GPU-e6622753-895a-f8fc-6082-ac71bbfa0037`、GPU 3 `GPU-f7d3c0fb-aed7-235f-7332-68966b63e0c5`、GPU 4 `GPU-0bc7caf4-d72c-c6b9-a9c1-60d2ff5779c4`、GPU 5 `GPU-76169292-9c3b-c2e1-682b-bacd97a2f23c`；GPU 0/1 上的既有 PID 未触碰。CM002 timing/profiler 均 unset P2P/SHM overrides；CM003 每次均精确设置 `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0`，formal timing 未设置 `NCCL_DEBUG`。

### Raw 产物与 timing 观察

Timing raw JSON 分别位于以下八个目录的 `timing-r1.json`、`timing-r2.json`、`timing-r3.json`；CM003b 另有 `timing-r4.json`、`timing-r5.json`：

```text
artifacts/compressed_muon/CM002a-adamw-dense-gpt60m-ddp-ws4-s42/
artifacts/compressed_muon/CM002b-m001-adamw-arc-gpt60m-ddp-ws4-s42/
artifacts/compressed_muon/CM002c-muon-dense-gpt60m-ddp-ws4-s42/
artifacts/compressed_muon/CM002d-m001-muon-arc-gpt60m-ddp-ws4-s42/
artifacts/compressed_muon/CM003a-adamw-dense-gpt60m-ddp-ws4-s42/
artifacts/compressed_muon/CM003b-m001-adamw-arc-gpt60m-ddp-ws4-s42/
artifacts/compressed_muon/CM003c-muon-dense-gpt60m-ddp-ws4-s42/
artifacts/compressed_muon/CM003d-m001-muon-arc-gpt60m-ddp-ws4-s42/
```

下表的 fwd/bwd、optimizer、step、throughput、显存均来自 timing JSON；均值和 CV 是独立 process repeat 之间的统计。Profiler 的 NCCL 与 exposed 列见下一节，exposed 明确是 trace-derived estimate。

| cell | timing repeats | fwd/bwd ms | optimizer ms | full step ms（CV） | tokens/s | peak alloc/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| CM002a AdamW dense normal | 3 | 16.897965 | 3.001526 | 19.900334（3.4660%） | 51498.410969 | 884.261882 / 2946.000000 |
| CM002b AdamW ARC normal | 3 | 1.847274 | 17.084153 | 18.887576（3.9705%） | 54271.415678 | 965.814941 / 1192.000000 |
| CM002c Muon dense normal | 3 | 16.550148 | 2.906673 | 19.422119（3.9470%） | 52779.005873 | 858.805501 / 2957.333333 |
| CM002d Muon ARC normal | 3 | 1.860179 | 18.876050 | 20.703583（0.7448%） | 49461.871843 | 861.803223 / 1096.000000 |
| CM003a AdamW dense P2P-disabled | 3 | 16.663302 | 3.000105 | 19.663192（4.4407%） | 52147.293332 | 884.553548 / 2946.000000 |
| CM003b AdamW ARC P2P-disabled | 5 | 1.873306 | 15.746766 | 17.553639（**6.3068%**） | 58509.270036 | 965.814941 / 1192.000000 |
| CM003c Muon dense P2P-disabled | 3 | 17.955371 | 2.901864 | 20.822211（3.8273%） | 49225.346761 | 858.805501 / 2957.333333 |
| CM003d Muon ARC P2P-disabled | 3 | 1.864635 | 19.166942 | 21.001396（1.8032%） | 48769.164195 | 861.803223 / 1096.000000 |

所有 26 个 timing launch 均为 exit 0；finite loss/parameters、四 rank parameter checksum agreement、collective signature agreement 和相关 observer bytes 检查均通过。CM003b 的 5 次 step means 原始值为 `16.770048, 17.128663, 19.471452, 17.497559, 16.900475 ms`；追加 r4/r5 后仍超过预先约定的 5% CV gate，因此没有删除 outlier 或继续运行。

### 通信 bytes、collective 与 profiler 观察

Timing JSON 的 logical per-step communication bytes 在所有对应 cell 中一致：dense 为 `dense_gradient=128188416` bytes；ARC 为 `arc_seed=24`、`arc_sketch=147456`、`arc_selected_values=5054464`、`uncompressed=103022592` bytes。Profiler active 5 steps 的 observer aggregate 与 trace `message_bytes` 一致：dense `ddp_gradient` 为 `640942080` bytes、10 kernels；ARC `arc_seed` 为 `120` bytes/15 kernels，`arc_sketch` 为 `737280`/15，`arc_selected_values` 为 `25272320`/15，`arc_dense_uncompressed` 为 `515112960`/10；Muon ARC 另有 `muon_result` 为 `31457280`/15。对应的 collective message size/aggregate bytes 和 category 在 24 份 profiler summary 中全部匹配，未出现 `unattributed`。

Profiler summary 的 NCCL 总 kernel 时间、gradient subset 时间和 exposed 时间（每项为 3 次 profiler 的 mean，单位 ms）如下；gradient subset 对 ARC 只包括 seed/sketch/selected/dense-uncompressed，不把 Muon result collective 计入梯度通信：

| optimizer/transport | dense NCCL / gradient | ARC NCCL / gradient | dense exposed | ARC exposed |
|---|---:|---:|---:|---:|
| AdamW normal | 67.445109 / 67.445109 | 87.346332 / 87.346332 | 59.985469 | 82.194081 |
| Muon normal | 72.341463 / 72.341463 | 88.641753 / 72.741819 | 65.448999 | 80.635589 |
| AdamW P2P-disabled | 73.587885 / 73.587885 | 70.650401 / 70.650401 | 66.767080 | 67.238248 |
| Muon P2P-disabled | 72.957913 / 72.957913 | 73.929255 / 65.005935 | 65.971962 | 67.847936 |

`exposed` 不是独立端到端计时，而是 profiler trace 中 NCCL union 减去与非 NCCL compute union 的重叠所得估计。Profiler 的 trace raw paths 对每个 cell 均为 `artifacts/compressed_muon/<experiment-id>/profiler/profile-r1-rank0.json`、`profile-r2-rank0.json`、`profile-r3-rank0.json`，summary paths 为同目录的 `profile-r1-summary.json` 至 `profile-r3-summary.json`；24 个 summary 和 24 个 rank-0 full trace 均保留。一次性 profiler launcher 为 `artifacts/compressed_muon/formal_profile_launcher.sh`。

下表为 profiler active trace 中各 named range 的 3 次 trace aggregate mean（单位 ms；每个值是该 trace 的 5 active steps 合计，不应与 timing JSON 的未加 profiler fwd/bwd/optimizer 样本混同）：

| ARC cell | fwd/bwd | optimizer | projection | Top-K | selected-values range | EF21M | Newton–Schulz | Muon result range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CM002b AdamW normal | 72.630477 | 49.964263 | 1.815897 | 2.413674 | 3.264076 | 2.349721 | 不适用 | 不适用 |
| CM002d Muon normal | 73.843712 | 74.481387 | 1.825061 | 2.666384 | 3.162141 | 2.153707 | 21.624165 | 3.847616 |
| CM003b AdamW P2P-disabled | 74.402554 | 45.050390 | 1.872287 | 2.838119 | 3.291741 | 2.372291 | 不适用 | 不适用 |
| CM003d Muon P2P-disabled | 75.466137 | 72.105939 | 1.828947 | 2.900010 | 3.170042 | 2.145668 | 21.866812 | 3.986063 |

### Summary、observations 与 inference

四份机械 summary 为 `artifacts/compressed_muon/cm002-adamw-summary.json`、`cm002-muon-summary.json`、`cm003-adamw-summary.json`、`cm003-muon-summary.json`。其中 CM003 AdamW summary 使用 dense r1–r3 和 ARC r1–r5；其他 cell 使用各自 r1–r3。结果为：

- `R_bytes=0.1557386`，AdamW 与 Muon 相同。这是 logical gradient communication bytes 的观察，符合预注册的“理论 byte reduction 近似相同”规则。
- normal AdamW：`R_grad_comm=-0.2951`、`R_step=0.0509`；normal Muon：`R_grad_comm=-0.00553`、`R_step=-0.06598`。
- P2P-disabled AdamW：`R_grad_comm=0.03992`、`R_step=0.10728`，但 ARC timing CV 为 `6.3068%`，不能视作稳定 wall-clock 结果。P2P-disabled Muon：`R_grad_comm=0.10899`、`R_step=-0.00861`。
- P2P-disabled 下 AdamW/Muon 的 `R_grad_comm` 差异约 6.91 个百分点，超过预注册的 5 个百分点规则；加上 AdamW ARC timing 不稳定，不能认定两种 optimizer 获得相近的实际通信收益。
- wall-clock 观察也不支持两者都获得稳定正收益：Muon 的两个 transport `R_step` 均非正；AdamW normal 收益约 5.09%，P2P-disabled 的约 10.73% 受 CV gate 拒绝。上述是本 synthetic microbenchmark 的 observations，不是对优化器或方法的普遍结论。

`R_bytes` 与 profiler gradient/NCCL time 的对应关系不能直接推出完整 step 加速；ARC projection、Top-K、EF21M、更多 collective、Muon Newton–Schulz、不可压缩张量和 overlap 都可能改变 wall-clock。trace exposed time 只是重叠推断，不能替代独立端到端通信计时。该 synthetic 100-step benchmark 不证明 convergence、time-to-quality 或最终模型质量。

### Scale-up 决策与限制

按预注册 scale gate，只有八个 60M cells correctness/required metrics 齐全且所有 timing CV ≤5% 才能扩展到 130M；CM003b 即使追加两次 fresh repeat 仍为 6.3068%，所以本轮不扩展到 130M、350M 或 1B。`docs/compressed_muon/RESULTS.md` 未更新，等待稳定证据。正式实验登记已将 7 个稳定 cell 标为 `completed`，CM003b 标为 `stopped`（运行成功但 comparison gate failed）。

### 关联产物

- Task 6 执行报告：`.superpowers/sdd/2026-09-04-arc-topk-adamw-muon-benchmark/task-6-execution-report.md`
- Task 6/7 ledger：`.superpowers/sdd/2026-09-04-arc-topk-adamw-muon-benchmark/progress.md`
- 实验登记：`docs/compressed_muon/EXPERIMENTS.md`
- timing/profiler raw：`artifacts/compressed_muon/CM002a-adamw-dense-gpt60m-ddp-ws4-s42/` 至 `CM003d-m001-muon-arc-gpt60m-ddp-ws4-s42/`

## 2026-09-05：CM004a–CM009d serial scale-out 结果整理

### 目的和口径

本轮按 serial launcher 完成 GPT-130M、350M、1B 的 normal 与 `NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=0` 两种 transport。所有 formal cell 使用 4 卡 DDP、BF16、local batch 1、sequence length 256、gradient accumulation 1、20 warmup + 100 measured steps、seed 42；ARC 使用 `ratio=0.2`、`projection_rank=4`、`eta=0.1`、`start_compress_step=0`。原始 timing JSON、profiler summary/Chrome trace、环境文件、失败/OOM 日志、事件 manifest 和 per-model partial manifest 均保留在 `artifacts/compressed_muon/`，未改 benchmark 或 optimizer math。

### 24 个 formal cell 的最终分类

| 模型 / transport | AdamW dense | AdamW ARC | Muon dense | Muon ARC |
|---|---|---|---|---|
| GPT-130M normal（CM004a–d） | CM004a completed | CM004b completed | CM004c failed：三次 timing/profile 的 exact `parameter_checksum_agreement=false` | CM004d completed（但无有效 dense-Muon 对） |
| GPT-130M P2P-disabled（CM005a–d） | CM005a completed | CM005b completed | CM005c failed：三次 timing/profile 的 exact `parameter_checksum_agreement=false` | CM005d completed（但无有效 dense-Muon 对） |
| GPT-350M normal（CM006a–d） | CM006a completed | CM006b completed | CM006c completed | CM006d completed |
| GPT-350M P2P-disabled（CM007a–d） | CM007a completed | CM007b completed | CM007c completed | CM007d completed |
| GPT-1B normal（CM008a–d） | CM008a completed | CM008b stopped：4-rank AdamW ARC probe 在 optimizer-state prepopulation OOM；formal cell 未启动 | CM008c failed：timing r1–r3 exact `parameter_checksum_agreement=false`；profiler checksum field true 但 timing gate 失败 | CM008d completed（但无有效 dense-Muon 对） |
| GPT-1B P2P-disabled（CM009a–d） | CM009a completed | CM009b stopped：沿用同一 model-scoped AdamW ARC OOM gate；formal cell 未启动 | CM009c failed：timing r1–r3 exact `parameter_checksum_agreement=false`；profiler checksum field true 但 timing gate 失败 | CM009d completed（但无有效 dense-Muon 对） |

独立按 raw timing/profile evidence 统计为 18 个 completed、4 个 checksum-invalid failed、2 个 OOM-stopped。partial manifest 的 `status_counts` 当前分别为 `4/2/2`、`6/0/2`、`4/2/2`（completed/invalid/skipped）；其中 GPT-130M 的 CM004b/CM005b 和 GPT-350M 的 CM006b/CM007b 含有历史 skip 事件但最终 timing/profile 已 valid，故不能直接把该字段当作终态。登记表按逐 cell 的最终 raw evidence 把 `invalid` 写为 `failed`、对应 OOM gate 写为 `stopped`，并保留 manifest 历史事件。

### 六个完整成对 summary

六个已有 summary（GPT-130M AdamW 两种 transport；GPT-350M AdamW 两种 transport；GPT-350M Muon 两种 transport）用 `benchmark/compressed_muon/summarize_arc_2x2.py` 从各自 6 个 timing/profile 输入重算，JSON 与已有文件逐字段 exact match。下表仅列稳定、有效的 paired R；数值为 `mean±sample std (CV)`，step/throughput 来自 timing，gradient NCCL 来自 profiler 的相应类别，logical bytes 为每步通信字节。

| pair | step Dense → ARC (ms) | throughput Dense → ARC (tokens/s) | peak allocated / reserved Dense → ARC (MiB) | logical bytes Dense → ARC | gradient NCCL Dense → ARC (ms) | total NCCL Dense → ARC (ms) | R_bytes / R_grad_comm / R_step |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-130M AdamW normal | 39.992±1.540 (3.85%) → 29.401±1.130 (3.84%) | 25629.7±969.0 → 34862.9±1317.7 | 1835.8±0.5 / 5009.3±79.9 → 2277.8±0.0 / 2878.0±0.0 | 267780096 → 177672216 | 154.065±11.148 → 126.108±13.176 | 154.065±11.148 → 126.108±13.176 | 0.3365 / 0.1815 / 0.2648 |
| GPT-130M AdamW P2P-disabled | 39.924±1.788 (4.48%) → 28.745±0.316 (1.10%) | 25682.6±1135.5 → 35627.0±391.5 | 1835.3±0.0 / 5044.0±0.0 → 2277.8±0.0 / 2878.0±0.0 | 267780096 → 177672216 | 159.504±2.056 → 127.369±7.914 | 159.504±2.056 → 127.369±7.914 | 0.3365 / 0.2015 / 0.2800 |
| GPT-350M AdamW normal | 103.959±0.987 (0.95%) → 75.578±1.691 (2.24%) | 9850.6±94.0 → 13553.4±303.9 | 4756.1±0.0 / 11498.0±0.0 → 7691.1±0.0 / 10632.0±0.0 | 709361664 → 308281368 | 413.803±26.158 → 231.813±3.919 | 413.803±26.158 → 231.813±3.919 | 0.5654 / 0.4398 / 0.2730 |
| GPT-350M AdamW P2P-disabled | 106.521±3.330 (3.13%) → 76.587±1.214 (1.59%) | 9619.4±303.9 → 13372.7±210.1 | 4756.1±0.0 / 11498.0±0.0 → 7691.1±0.0 / 10632.0±0.0 | 709361664 → 308281368 | 418.947±27.947 → 234.583±12.846 | 418.947±27.947 → 234.583±12.846 | 0.5654 / 0.4401 / 0.2810 |
| GPT-350M Muon normal | 143.888±6.856 (4.76%) → 113.655±1.931 (1.70%) | 7127.2±330.9 → 9011.4±153.8 | 4396.6±0.0 / 10964.0±41.6 → 7211.0±0.0 / 9672.0±0.0 | 709361664 → 308281368 | 425.991±19.477 → 205.261±14.015 | 425.991±19.477 → 377.209±24.670 | 0.5654 / 0.5182 / 0.2101 |
| GPT-350M Muon P2P-disabled | 143.753±2.018 (1.40%) → 113.365±1.544 (1.36%) | 7124.3±100.7 → 9033.9±123.4 | 4396.6±0.0 / 10940.0±0.0 → 7211.0±0.0 / 9672.0±0.0 | 709361664 → 308281368 | 440.632±23.855 → 212.484±17.692 | 440.632±23.855 → 392.111±29.265 | 0.5654 / 0.5178 / 0.2114 |

这里 `R_x=1-ARC/Dense`。Muon ARC 的 total NCCL 包含 `muon_result`，而 gradient NCCL 只含 gradient communication 类别；因此二者必须分开解释。部分 profiler gradient/NCCL CV 超过 5%（最高约 10.45%），保留为 uncertainty；按用户要求未回溯拒绝 scale-out，正式成对 validity 以 checksum/finite/signature 和 timing/profile 完整性为准。

其余通过 validator 但没有有效 paired R 的 completed cell 也保留如下：CM004d step `37.875±1.114 ms`（CV 2.94%，27051.9±803.2 tokens/s，alloc/reserved `2167.1±0.4/2669.3±9.2 MiB`，logical bytes `177672216`，gradient/total NCCL `114.226±13.268/154.084±15.440 ms`）；CM005d step `35.985±0.567 ms`（1.57%，28461.1±452.3 tokens/s，`2166.9±0.0/2664.0±0.0 MiB`，`177672216` bytes，`110.682±11.335/150.658±15.300 ms`）。1B standalone cells：CM008a `299.496±5.549 ms`（1.85%，3419.8±62.7 tokens/s，`13532.5/23386.0 MiB`，`2007760896` bytes，`1213.642±17.805 ms`）；CM008d `313.485±0.097 ms`（0.03%，3266.5±1.0 tokens/s，`22457.3/23200.0 MiB`，`652732440` bytes，gradient/total `448.908±29.500/1069.532±66.971 ms`）；CM009a `306.819±12.349 ms`（4.02%，3341.2±137.6 tokens/s，`13532.5/23386.0 MiB`，`2007760896` bytes，`1246.154±34.759 ms`）；CM009d `311.104±9.427 ms`（3.03%，3293.6±101.6 tokens/s，`22457.3/23200.0 MiB`，`652732440` bytes，`483.374±20.446/1114.913±12.254 ms`）。这些 standalone 观测不用于推导任何 paired R。

### Invalid cell 的 exploratory raw timing（不得用于 paired R）

以下仅帮助审计原始运行，不是正式比较：CM004c `45.410±0.593 ms`（CV 1.31%，22552.8 tokens/s）、CM005c `44.862±1.988 ms`（4.43%，22855.0 tokens/s）、CM008c `463.131±7.855 ms`（1.70%，2211.5 tokens/s）、CM009c `471.954±6.564 ms`（1.39%，2170.0 tokens/s）。四个 cell 的 timing r1–r3 均有 finite loss/parameters 和 matching collective signatures，但 exact checksum agreement 为 false；130M profiler 也为 false，1B profiler checksum field 为 true，cell 仍因 timing gate 失败。因而四者不能与 ARC cell 组成有效 Muon paired R，也不能据此宣称 130M 或 1B Muon 收益。对应 rank-0 130M checksum 为 `-5961.451935559417`，1B 为 `-170494.38884379686`；raw evidence 未包含非零 rank 的 checksum 或参数差分幅度。

### OOM、checksum 和解释边界

- GPT-1B AdamW ARC 的 4-rank probe 在 `ArcTopKAdamW._prepopulate_group_state` 的 `torch.zeros_like(param)` 处 OOM，日志显示尝试额外分配 148 MiB、每卡约 23.39 GiB 已用；因此 CM008b/CM009b 是 model-scoped OOM skip，而不是 timing failure。保留 probe 与 formal OOM 日志，不改变 workload 以强行适配。
- GPT-130M 与 GPT-1B dense Muon 的 exact checksum divergence 在 normal/P2P-disabled 两种 transport 均出现；已有诊断支持“局部 Polar Express/Triton shape/backend 可能放大 rank-local 差异”的 inference（130M 的 768/3072 shape 是窄相关），但 root cause 未证明。不要把 checksum gate 放宽为 allclose，也不要将这些 raw timing 当作 valid comparison。
- 现有 launcher 仍有一个窄的 signal-vs-OOM 状态竞争窗口：controller signal 与子进程 OOM/终止若在同一边界发生，事件分类可能依赖到达顺序。该限制已记录、raw evidence 保留，本任务按用户要求不修复；不影响本次最终 partial-manifest 分类。
- 观察（observation）：在 130M/350M 有效 AdamW pairs 与 350M 有效 Muon pairs 中，ARC logical bytes 分别减少约 33.65%/56.54%，step R 为约 21.01–28.10%。推断（inference）：本机 workload 的 measured step 受 projection、Top-K、EF21M、collective 粒度和 Muon 正交化共同影响，不能由 bytes ratio 单独解释。
- 限制：这是 synthetic 100-step timing/profiler benchmark，不包含收敛、time-to-quality、跨节点网络或最终模型质量证据；不得报告任何 1B paired benefit。

### 验证和关联产物

- 对全部 18 个 completed cell 的 108 个 timing/profile JSON 运行 `validate_scale_to_1b.validate`：identity/configuration、100 个正 step samples、finite/checksum/signature、required collective category、trace JSON、observer/trace byte agreement 和 zero unattributed active NCCL 均通过。4 个 invalid cell 的 12 个 timing JSON 均因 exact checksum gate 被拒绝；130M 的 6 个 profiler JSON 也被拒绝，1B 的 6 个 profiler JSON 通过但不改变 timing-invalid 分类；2 个 stopped OOM cell 无 timing/profile JSON。
- 六个 summary 通过 `summarize_arc_2x2.py` fresh recompute，并与既有 summary 逐字段 exact match；`R_bytes/R_grad_comm/R_step`、CV 和 sample std 均由脚本重算。
- 原始 evidence：`artifacts/compressed_muon/CM004a...` 至 `CM009d...` 各 cell 目录、`gpt130m-partial.json`、`gpt350m-partial.json`、`gpt1b-partial.json`、`scale-to-1b-manifest.jsonl`、最新 status log、以及 `.superpowers/sdd/2026-09-04-arc-topk-adamw-muon-benchmark/muon-dense-checksum-diagnosis.md`。
- 稳定结果入口：`docs/compressed_muon/RESULTS.md`；24 行登记：`docs/compressed_muon/EXPERIMENTS.md`；本任务完整报告：`.superpowers/sdd/2026-09-04-arc-topk-adamw-muon-benchmark/task-9-scale-results-report.md`。

## 2026-09-05：GPT-1B AdamW ARC 显存缓解阶梯测试

### 目的与停止条件

在不改变 GPT-1B workload、ARC 数学语义和 4 卡 DDP 规模的前提下，依次测试低风险显存缓解方案。每一级使用同一个 `warmup_steps=1`、`measure_steps=1` smoke probe；任一级成功即停止，全部仍 OOM 则停止，不继续进入 ZeRO/FSDP 或 ARC 状态复用等架构改造。

### 阶梯结果

1. 仅设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：仍 OOM，但从原 probe 的 optimizer-state prepopulation（申请 148 MiB）推进到首个 ARC step；最终在 `arc_topk.scatter_rows` 创建 `local_compressed` 时申请 540 MiB 失败。每卡约 23.19 GiB 已用、322.62 MiB free、PyTorch allocated 22.45 GiB、reserved-but-unallocated 151.47 MiB。
2. 在第一级基础上为 benchmark DDP 加 `gradient_as_bucket_view=True`：仍在同一 540 MiB `scatter_rows` 分配处 OOM，显存数字与第一级一致。当前 ARC 路径在 `DDP.no_sync()` 下运行，未观察到 bucket view 带来可用显存收益；该试验性 benchmark 改动已回退，避免改变后续 benchmark 口径。
3. 再消除 optimizer/ARC 已存在状态的 eager `dict.setdefault(..., torch.zeros_like(param))` 默认值分配：聚焦红测确认旧实现会重复分配，改为显式 key 判断后测试通过；GPT-1B probe 仍在同一 540 MiB `scatter_rows` 处 OOM，显存数字不变。

三次运行均使用 GPU 2–5，启动前四卡空闲；日志分别保存在：

- `artifacts/compressed_muon/oom_recovery_gpt1b/allocator/`
- `artifacts/compressed_muon/oom_recovery_gpt1b/bucket_view/`
- `artifacts/compressed_muon/oom_recovery_gpt1b/state_init/`

### 保留修改与验证

- `dion/adamw_arctopk.py`：已存在 momentum/variance 时不再构造无用的同形默认 tensor。
- `dion/arc_topk_sync.py`：已存在三个 ARC tracker 时不再构造无用的同形默认 tensor。
- 新增相应回归测试；直接相关的 `arc_topk_sync`、`adamw_arctopk` 和 benchmark 测试共 `48 passed`；扩大到 M001 本地与两 rank Gloo 同步测试后为 `91 passed`。

### 结论

allocator 配置可以消除最早的初始化 OOM，但当前 24GB 卡仍不足以承受首个压缩 step 的 540 MiB full-batch scatter 临时张量。DDP bucket view 和消除 eager 默认 tensor 都没有改变该处的可用显存。按预定停止条件，本轮不再启动更多实验；若后续继续，应把 `scatter_rows` 的 full-size 临时张量复用/原位更新作为新的、有独立语义与峰值显存测试的实现任务，或将 ZeRO/FSDP 作为新的系统配置实验。

## 2026-09-05：GPT-1B Muon ARC 补充 smoke probe

### 目的与配置

验证同一 GPT-1B、4 卡 DDP、BF16、local batch 1、sequence length 256 workload 改用 Muon ARC 后是否仍 OOM。使用 `warmup_steps=1`、`measure_steps=1`、`ratio=0.2`、`projection_rank=4`、`eta=0.1`、`start_compress_step=0`，并保留 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。该运行是短 smoke probe，不是新的正式性能比较。

### 结果

运行成功，退出码为 0，没有 OOM。单个 measured step 为 `411.430 ms`；峰值 allocated `22450.44 MiB`、reserved `22852 MiB`。`finite_loss=true`、`finite_parameters=true`、`parameter_checksum_agreement=true`、collective signature 全 rank 一致；观察到 ARC seed/sketch/selected-values、dense-uncompressed 和 Muon result collective。

原始 JSON 与日志：`artifacts/compressed_muon/oom_recovery_gpt1b/muon_arc/`。

### 结论

在当前 4×RTX 4090 环境和 smoke workload 下，GPT-1B Muon ARC 可以运行而不 OOM，显存仍接近上限。该结果只证明短路径可执行；单样本 step time 不用于替代 CM008d 的正式重复结果，也不与 OOM 的 AdamW ARC 形成 optimizer 性能优劣结论。

## 2026-09-05：启动 CM010 GPT-1B Muon dense/ARC 配对重测

为补齐 1B Muon 的有效 paired 指标，新增串行 launcher `artifacts/compressed_muon/gpt1b_muon_pair_launcher.sh`，使用 GPU 2–5 交错运行 CM010a dense Muon 与 CM010b ARC Muon，各 3 次 timing（20+100 steps）和 3 次 profiler（3+3+5 schedule）。两侧统一使用 normal NCCL、BF16、local batch 1、sequence length 256、seed 42 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；ARC 参数保持 `ratio=0.2`、projection rank 4、eta 0.1、start step 0。

脚本逐 artifact 调用已有 validator；只有 12 个 artifact 全部通过 finite/checksum/signature/schema/trace gate 才生成 `CM010-muon-gpt1b-normal-summary.json`。若 dense Muon 重现历史 checksum divergence，脚本仍保留所有 raw timing/profile，但明确拒绝 paired summary。实验在 tmux 会话 `cm010_gpt1b_muon_pair` 中串行执行，不需要持续监控。

### CM010 完成结果

脚本于 12:14:33 完成，12 个 job 均产出 JSON/trace。CM010b ARC 的 3 timing/3 profiler 全部通过 validator；CM010a dense 的 3 profiler 通过，但 3 timing 均因 exact `parameter_checksum_agreement=false` 被拒绝，rank-0 checksum `-170494.38884379686` 与 CM008c 完全相同。两侧 loss/parameters 均 finite，collective signature 均全 rank 一致。dense step CV `5.44%` 也超过 5% 稳定性阈值；脚本正确写入 `PARTIAL` 并拒绝生成 paired summary。

探索性 raw 均值为：dense → ARC step `465.933±25.325 → 313.546±1.861 ms`（表面降低 32.71%），throughput `2202.1±120.5 → 3265.9±19.4 tokens/s`（提高 48.31%），logical bytes `2007760896 → 652732440`（降低 67.49%），profiler gradient NCCL `1361.856±9.801 → 502.375±33.305 ms`（降低 63.11%），total NCCL `1361.856±9.801 → 1144.152±80.410 ms`（降低 15.99%）。peak allocated `12586.2 → 22450.9 MiB`（增加 78.38%），reserved `20632 → 22852 MiB`（增加 10.76%）。这些数字复现旧 CM008 的方向，但不得作为正式 paired R 或算法质量结论。

## 2026-09-05：dense Muon checksum 五 cell 归因短跑

### 目的与配置

用 GPT-130M、4 卡 DDP、BF16、local batch 1、sequence length 256、seed 42 和 `2 warmup + 12 measured steps` 做单次低成本归因，逐项隔离 Muon `distributed_mesh`、benchmark 自定义 DDP hook 和 Triton backend。五个 cell 串行使用 GPU 2–5，均退出码 0、loss/参数 finite。工具与原始结果位于 `benchmark/compressed_muon/dense_muon_attribution.py`、`benchmark/compressed_muon/run_dense_muon_attribution.sh` 和 `artifacts/compressed_muon/dense_muon_attribution/`。

### 结果

| mode | Muon process group | DDP reducer | Triton | exact checksum agreement |
|---|---|---|---|---|
| `rank_local_custom_hook` | 无 | benchmark custom | 开 | false |
| `process_group_custom_hook` | DDP process group | benchmark custom | 开 | true |
| `rank_local_default_reducer` | 无 | PyTorch default | 开 | false |
| `rank_local_no_triton` | 无 | benchmark custom | 关 | false |
| `upstream_ddp` | DDP process group | PyTorch default | 开 | true |

三个 rank-local cell 都只有 rank 1 与 rank 0/2/3 不同。自定义 hook 和默认 reducer 的四个 checksum pair 逐值完全相同，对应 sum/squared-sum 的跨 rank range 为 `0.0131173/0.000596821`；关闭 Triton 后仍失败，range 为 `0.0375733/0.0228399`。两个 process-group cell 的四 rank pair 完全一致，range 均为 0。

### 结论与边界

当前自定义 DDP hook 不是主因，Triton 也不是必要条件；最强的决定因素是 dense benchmark 是否将 Muon 配置成 rank-local `distributed_mesh=None`。原仓库 `train.py` 支持的 DDP 配置会传入 process group，Muon 内部的 result all-gather 使四 rank 最终更新 exact 一致；它可能在同步结果的同时掩盖了更早的 rank-local 数值差异。所有 cell 均为 dense `Muon`，没有经过 ARC 同步层，故结果不支持“ARC 修改导致该问题”。

该结论来自单 seed、单模型、单次短跑；现有证据没有 post-DDP gradient/post-NS 边界值或参数 `max_abs_diff`/relative L2/unequal count，因此还不能确定 rank-local 差异首次出现在哪个算子，也不能判定它是否会影响长训练质量。
## 2026-09-05：启动 1B 原仓库设置正式训练（CM018）

- 目的：比较普通 Muon 与 M001 ARC-TopK+Muon 在 GPT-1B、4 卡 DDP 完整 3000-step 训练中的验证损失、吞吐和峰值显存。
- 设置：沿用 `train.py` / `train_arctopk.py` 及各自原仓库配置；仅覆盖模型为 dim 1536、30 layers、24 heads，数据为 FineWeb10B。保留 seq 1024、global batch 1024、BF16、compile、学习率调度、每 125 step 验证和 W&B 默认设置。
- OOM 防护：controller 先对 ARC 按 device batch 1、2、4、8 逐级执行 1-step 完整初始化/训练/验证 probe，选择首个 OOM 前的最大值，再用相同 device batch 验证 dense Muon；只有两侧 probe 都通过才启动正式训练。
- 执行：CM018a dense Muon 与 CM018b ARC-TopK+Muon 使用 GPU 2–5 串行运行；CM018b 仅在 CM018a 正常完成后启动。controller、probe 配置和状态日志保存在 `artifacts/compressed_muon/CM018-gpt1b-muon-formal-training-controller/`。
- Probe 结果：ARC 在最小 device batch 1 的首次 compiled forward OOM（每卡约 23.49 GiB 已用，仅余 20.56 MiB，Triton autotune 申请 72 MiB 失败）。启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后再次 OOM（约 23.50 GiB 已用，仅余 4.56 MiB，申请 18 MiB 失败），说明不是单纯 allocator 碎片；CM018b 按 gate 停止。dense Muon 相同模型、seq 1024、device batch 1 的 1-step probe 通过，峰值显存 20587 MiB，因此 CM018a 单独进入正式训练。
- CM018a 已于 2026-09-05 21:41（Asia/Shanghai）在 tmux `cm018a_dense_1b` 启动，W&B run `3blckr4g`；正式配置的 global batch 1024 对应 256 次梯度累积，日志与环境快照位于 `artifacts/compressed_muon/CM018a-muon-dense-gpt1b-train-ddp-ws4-s42-db1/`。
- CM018a 随后在梯度累积阶段 OOM：先前 probe 的 global batch 4 只有一次 micro-step，未覆盖“梯度 tensor 已常驻后再次 forward”的峰值；正式训练在下一次 compiled forward 申请 18 MiB buffer 时，每卡仅余约 10.56 MiB。该 probe 缺口在 CM019 中修正为至少两次梯度累积，ARC probe 还会强制执行一次压缩 optimizer step。

## 2026-09-05：350M 正式训练队列（CM019）

- 使用已有 benchmark GPT-350M preset：dim 1024、20 layers、16 heads，共 354,680,832 参数；保持 device batch 1、global batch 1024 和其余原仓库 Muon/M001 配置。
- 先按 seq 1024、512、256 从长到短测试 ARC；每个 probe 覆盖两次 micro-step、梯度常驻后的 forward、以及至少一次实际 ARC 压缩 step。找到最长安全 seq 后，用同一 seq 验证 dense，并串行运行 CM019a dense 与 CM019b ARC 正式 3000-step 训练。
- Probe 结果：无需缩短 seq。ARC 在 seq 1024 的 20-step debug probe 中实际启用压缩并通过，峰值显存 14406 MiB；dense 相同 seq/device batch/两次梯度累积 probe 通过，峰值 8760 MiB。controller 选择 seq 1024，并于 21:53 启动 CM019a（W&B `b7g8rfy6`）；CM019b 将在 CM019a 成功完成后自动接续。
