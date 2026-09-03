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
