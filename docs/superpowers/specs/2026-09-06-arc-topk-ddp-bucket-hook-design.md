# ARC-TopK + Muon DDP Bucket Hook 设计

状态：修订后待二次评审

日期：2026-09-06

关联方法：M001 ARC-TopK-EF21M-Muon

修订记录：2026-09-06 根据首次独立架构审阅，补充训练循环现状核验、EF21M full-support 语义、跨 bucket 全局 collective 顺序、rank-local DCP schema 和首版 unused-parameter 支持边界。

## 1. 背景与问题

当前 `ArcTopKMuon` 在 `optimizer.step()` 中执行 ARC-TopK 梯度同步。其预期训练语义是：为了避免 DDP 在 backward 中先做一次普通梯度同步，同一 optimizer step 的所有 micro-batch（包括最后一个）都应将 **forward 和 backward 一起**放入 `DDP.no_sync()`。在该预期语义下，执行顺序是：

```text
所有 backward 完成
        │
        ▼
optimizer.step()
        ├─ 按 shape/dtype 组织矩阵参数
        ├─ ARC-TopK / EF21M 梯度同步
        ├─ Muon 动量与正交化
        ├─ Muon 结果 all-gather
        └─ 参数更新
```

ARC-TopK 即使降低了通信字节或单个同步操作的时延，也只能在 backward 完成后开始，无法与较早层的反向计算重叠。这是“optimizer-side 同步失去 DDP overlap”的准确含义。

但是，当前正式训练循环在 `train.py` 中先执行 forward，随后才用 `no_sync()` 包住 backward。PyTorch DDP 要求 forward 也处于该 context，否则梯度仍会同步。因此，设计实施前必须先用两 rank hook 计数和 profiler 核实当前正式训练是否实际执行了：

```text
backward 中的 dense DDP all-reduce
                 ＋
optimizer.step() 中的 ARC-TopK 同步
```

若核实为双重同步，则当前长训练中的 ARC 输入已经不是预期的 rank-local 累积梯度，且其 wall-clock 不能用于评价纯 optimizer-side ARC。专用 benchmark 的 `sync_context` 已经把 forward/backward 一起放入 `no_sync()`，所以 benchmark 结果与正式训练入口不能在修复前直接类比。修复后的 optimizer-side 路线才是 DDP-hook 路线的有效对照基线；既有实验需标记其具体入口和同步行为。

目标路线是把“数据并行梯度同步”迁移到 DDP communication hook。某个 DDP bucket 就绪后，ARC-TopK 可以立即开始处理该 bucket；与此同时，autograd 可以继续计算更早层的梯度。

```text
backward: bucket 0 ready ── ARC sync bucket 0 ───────┐
          └─ continue backward: bucket 1 ready ─ ARC sync bucket 1 ─┤
                                                                   ▼
                                                backward complete
                                                       │
                                                       ▼
                                             ordinary Muon.step()
```

需要特别区分两类通信：

1. 数据并行梯度同步：当前由 `ArcTopKMuon` 执行，目标是迁移到 DDP hook。
2. Muon 分布式正交化结果的 all-gather：仍由 Muon optimizer 执行，不迁移到 DDP hook。

## 2. 目标

- 在最后一个 gradient-accumulation micro-batch 中启用 DDP reducer，使 bucket-ready 事件触发 ARC-TopK。
- hook 返回与原 bucket 等长、等 dtype、等 device 的完整同步梯度，backward 结束后普通 `Muon` 可直接消费 `.grad`。
- Muon 矩阵参数使用 ARC-TopK + EF21M；AdamW/Lion 等非 Muon 参数仍进行精确 dense average。
- 保留当前 optimizer-side `ArcTopKMuon`，用于回归、消融实验和结果对照。
- 让压缩器状态能以稳定参数名保存和恢复，不依赖 DDP bucket 编号。
- 最终实现真实的异步 collective/Future 链，使梯度通信具有与 backward 重叠的机会。

## 3. 非目标

- 不把 Muon 正交化或其结果 all-gather 移入 DDP hook。
- 不改变 ARC-TopK / EF21M 的数学定义。
- 不保证压缩率小于 1 时，DDP-hook 模式与旧 optimizer-side 模式逐 bit 相同；两者的投影分组和随机数消费顺序不同。
- 不在本改造中重新设计 DDP reducer、跨节点拓扑或通信后端。
- 不删除现有 shape/dtype megabatch 路线。

## 4. 方案比较

### 方案 A：直接照搬官方同步 hook

在 DDP hook 中逐参数执行 ARC-TopK，并在 hook 内调用阻塞式 collective。

优点是改动直观，便于快速验证数学正确性。缺点是“使用了 hook”不等于“获得了 overlap”：阻塞 collective 和 Python 控制流可能占住执行线程，削弱甚至消除 bucket 提前就绪的收益。

### 方案 B：分阶段异步 DDP hook，并保留旧 optimizer 路线（采用）

将算法拆成“本地准备—异步 sketch all-reduce—TopK 与打包—异步 selected-values all-reduce—重建 bucket”，用 `Future` 串联。非压缩参数单独打包做 dense all-reduce。不同 bucket 的完整通信链通过一个全局 tail Future 排序，首版不允许 callback 自由穿插 collective。旧 `ArcTopKMuon` 继续存在，训练配置显式选择同步模式。

该方案改动较大，但组件边界清楚，既能验证正确性，也能针对 wall-clock overlap 做优化。

### 方案 C：为每个参数注册 autograd hook

参数梯度一生成就立刻压缩和通信，不依赖 DDP bucket。

粒度最细，但需要自行处理通信顺序、unused parameter、梯度累积和大量小 collective；等同于部分重写 reducer，风险和维护成本最高，因此不采用。

## 5. 当前模块及合作关系

```text
train_arctopk.py
  ├─ 构造 ARC 配置与参数组
  └─ 创建 ArcTopKMuon
           │
train.py   │  预期所有 micro-batch 的 forward/backward 使用 DDP.no_sync()
           │  当前实现仅包 backward，必须先核实并修复
  └────────┘
           ▼
dion/muon_arctopk.py
  ArcTopKMuon
  ├─ 管理 h_local / g_local / g_global
  ├─ 调用 arc_topk_sync.py 进行梯度同步
  ├─ 执行 Muon 动量与正交化
  └─ 对非矩阵参数执行 dense optimizer-side all-reduce
           │
           ├──────────────► dion/arc_topk_sync.py
           │                 ├─ 按 shape/dtype 分组
           │                 ├─ 初始化压缩器状态
           │                 └─ 适配 ARC 同步任务
           │                              │
           │                              ▼
           │                    dion/arc_topk.py
           │                    投影、TopK、EF21M、collective
           │
           └──────────────► dion/megabatch_base.py
                              ├─ optimizer task 编排
                              └─ Muon 结果分片/all-gather
                                      │
                                      ▼
                                dion/opt_utils.py
                                AsyncTask / AsyncRuntime
```

当前“按 shape/dtype 组成 batch”是 optimizer megabatch 的实现选择：相同形状的矩阵可堆叠后批量投影和处理。它不是 DDP reducer 的 bucket 规则，也不是必须由官方 ARC-TopK 实现规定的算法语义。

## 6. 目标模块及合作关系

新增 `dion/arc_topk_ddp_hook.py`，同步所有权从 optimizer 转移给 DDP reducer：

```text
train_arctopk.py
  ├─ sync_mode=optimizer ─────────────► ArcTopKMuon（旧基线）
  │
  └─ sync_mode=ddp_hook
       ├─ 创建 ArcTopKDDPState
       ├─ DDP.register_comm_hook(state, arc_topk_ddp_hook)
       └─ 创建普通 Muon

train.py
  ├─ micro-batch 0 .. N-2: DDP.no_sync()
  └─ micro-batch N-1:      启用 reducer/hook
                                  │ bucket ready
                                  ▼
                       dion/arc_topk_ddp_hook.py
                       ├─ 识别 bucket 中每个参数的角色
                       ├─ Muon 矩阵参数：ARC-TopK/EF21M
                       ├─ 其他参数：dense all-reduce
                       ├─ 将结果写回 bucket.buffer()
                       └─ 返回 Future[Tensor]
                                  │
                                  ▼
                         backward 完成，.grad 已同步
                                  │
                                  ▼
                           普通 dion/muon.py
                           ├─ 动量/Nesterov
                           ├─ 分布式正交化
                           ├─ Muon 结果 all-gather
                           └─ 更新参数

dion/arc_topk.py
  ├─ 保留核心数学操作
  └─ 拆出可被两条路线复用的本地 prepare/finalize 原语

dion/arc_topk_sync.py
  └─ 保留旧 optimizer-side shape/dtype 分组与适配逻辑
```

`AsyncRuntime` 在 hook 模式下不再负责编排梯度同步；它仍可服务于 Muon optimizer 内部计算和正交化结果通信。

## 7. DDP Hook 数据流

### 7.1 Gradient accumulation

一个 optimizer step 含 N 个 micro-batch 时：

```text
micro 0     forward + backward under no_sync ─┐
micro 1     forward + backward under no_sync ─┼─ 本地累加到 .grad
...                                │
micro N-1   normal DDP forward + backward ────┘
                       │
                       └─ bucket ready 后调用 ARC hook
```

只有最后一个 micro-batch 触发 reducer。hook 看到的是该 optimizer step 的累积梯度，而不是只看到最后一个 micro-batch 的梯度。这一行为必须由真实两 rank hook 计数与数值测试确认；不能沿用当前仅包 backward 的 context 写法。

旧 `optimizer` 模式则让所有 micro-batch 的 forward/backward 都处于 `no_sync()`，确保 ARC 收到真正的 rank-local 累积梯度。修复该行为会改变当前正式训练入口的实际语义，因此修复前后的结果必须分开记录。

### 7.2 Bucket 分类与重建

hook 使用 `GradBucket.parameters()`、`gradients()` 和 buffer views 建立 bucket 内映射，不再要求参数 shape 相同。

每个参数属于以下一种角色：

- `arc_matrix`：由 Muon 更新且满足 ARC-TopK 支持条件的矩阵参数。
- `dense_aux`：由 AdamW/Lion 等分支更新的参数，或 ARC 不支持而回退到 dense 的参数。

首版不定义动态 unused parameter 的压缩语义：要求 `find_unused_parameters=False`，并要求注册到 optimizer 的训练参数在每个 optimizer step 静态参与反向传播。构造期拒绝已知冲突配置；运行期若发现更新集合不完整或 rank 间不一致则快速失败，而不是静默归入 `ignored`。`find_unused_parameters=True`、locally unused 和不同 micro-batch 使用不同参数留作后续扩展。

压缩阶段开启后，不允许为了方便而对整个 bucket 先做 dense all-reduce，再覆盖矩阵切片；那会保留主要通信量，失去压缩意义。正确做法是：

- 将所有 `dense_aux` 梯度打包成一个连续 buffer，执行一次 dense all-reduce，再散回对应 view。
- 对各 `arc_matrix` 参数独立维护 EF21M 状态；本地生成 sketch 后，将同一 bucket 的 sketch 扁平拼接并统一 all-reduce。
- 根据聚合 sketch 选择索引，拼接对应 values 后统一 all-reduce，再分别重建 `g_global`。
- 最终把两条路径的结果写回原始 `bucket.buffer()`，Future 返回该 buffer。

DDP bucket 通常按 dtype/device 构造，但 hook 仍应显式检查并基于实际 buffer dtype/device 工作，不把这一点当成未验证假设。

### 7.3 EF21M 初始化、warmup 与 dense fallback

ARC 参数不能把 warmup 或 `ratio=1` 简化成“直接平均原始梯度”，否则会改变 EF21M 数学语义。对每个 ARC 参数必须执行：

```text
step == 1:
    h_local = grad_local

step > 1:
    h_local = (1 - eta) * h_local + eta * grad_local

step == 1 或 step <= start_compress_step:
    g_local  = h_local
    g_global = mean_rank(h_local)       # dense/full-support 同步的是 tracker

压缩阶段:
    delta_local = h_local - g_local
    通过 ARC support 同步 delta
    g_local  += compressed(delta_local)
    g_global += mean_rank(compressed(delta_local))
```

`ratio=1` 在压缩阶段表示 full support；它使 `g_global` 等于跨 rank 的 tracker 平均，而一般不等于当前原始梯度平均。只有 `eta=1`（以及首步初始化）时，才可把 ARC 参数结果与标准 DDP 原始梯度平均直接对照。

如果整个 bucket 进入 warmup/full-support 路线，可以使用一次整桶 dense all-reduce，但在发起 collective 前必须先把 ARC slices 写成更新后的 tracker；`dense_aux` slices 仍写入原始梯度。完成后分别恢复 ARC state 并把结果散回。单 rank 仅省略 collective，不能省略上述 EF21M 状态转换。

进入稀疏压缩阶段后，仅 `dense_aux` 走原始梯度 dense 路线。任何运行时不支持情况应在所有 rank 上依据一致的静态元数据作出相同回退决定，避免 collective 顺序分叉。

## 8. 异步协议

最终 hook 不在主体中等待 collective 完成，而返回串联后的 Future。首版用 `ArcTopKDDPState.tail_future` 强制完整 bucket 链按 DDP bucket-ready 顺序发射：

```text
bucket 0 ready
  └─ local prepare
       ├─ async dense_aux all-reduce
       ├─ async ARC sketch all-reduce
       ├─ TopK + async selected-values all-reduce
       └─ EF21M finalize + scatter ───────────────► tail_future(0)

bucket 1 ready
  └─ wait tail_future(0)
       └─ local prepare → dense/sketch → values → finalize ─► tail_future(1)

bucket 2 ready
  └─ wait tail_future(1) → ...
```

同一 bucket 内允许在固定调用顺序下先后 enqueue dense buffer 和 sketch buffer，再等待二者完成；selected-values collective 只能在 sketch 完成和 TopK 后发起。下一 bucket 的任何 collective 必须等待上一 bucket 的 selected-values 和 scatter 都进入完成状态。这样会串行化 bucket 间通信，但仍允许当前 bucket 通信与后续 backward 计算重叠；只有在该版本正确且 profiler 证明串行链成为瓶颈后，才设计多 process-group 或显式阶段流水线。

实施顺序允许先构建阻塞式 correctness prototype，再替换为 Future 链；但阻塞式版本不满足最终性能验收标准，也不能作为“已经获得 overlap”的依据。

所有 rank 必须具有相同的 **全局 collective 发射序列**，而不仅是各 bucket 内局部顺序相同。callback 中的分支只能依赖各 rank 一致的元数据或 collective 结果，不能依赖 rank-local 的数值条件决定是否发起通信。

`Work.get_future()` 的值可能是 tensor list，而 DDP hook 最终要求单个 `Future[Tensor]`。`Future.then()` 的 callback 返回另一个 Future 时不会自动 flatten。实现必须使用显式的 CUDA-aware bridge/completion Future 或等价状态机，把内部 collective Future 的成功和异常传递到最终 Future；最终 completion 必须覆盖 scatter 所在 CUDA stream 的工作。

bucket views、projection、delta、local selected、averaged selected 和 packing buffers 必须由 bucket context 强引用到最终 Future 完成。`local_selected` 与参与 all-reduce 的 `averaged_selected` 必须分离，避免原地 collective 污染 `g_local` 所需的 rank-local delta。

## 9. 状态、随机性与生命周期

### 9.1 `ArcTopKDDPState`

新状态对象至少持有：

- data-parallel process group、world size 和 rank；
- ARC 配置、warmup 配置和当前 optimizer-step 编号；
- 参数对象到静态元数据的运行时映射；
- 每个 ARC 参数的 `h_local`、`g_local`、`g_global`；
- 稳定参数名、参数序号、角色、shape 和 dtype；
- 可选的 profiling/byte counters。
- 上一 bucket 完整链的 `tail_future` 和仍在 flight 的 bucket contexts。

运行时可按 parameter object identity 查找状态；持久化时必须使用 `model.named_parameters()` 的稳定名称。不能用 bucket index 作为长期身份，因为 DDP 可能 rebuild buckets。

### 9.2 Step 与 seed

训练循环在一个 optimizer step 的 backward 开始前，显式调用类似 `arc_state.begin_step(step)` 的接口。hook 调用时捕获该 step 的局部值；step 不按 bucket 自增，避免异步 callback 观察到下一步编号。

每个参数的投影 seed 由 `base_seed`、optimizer step 和稳定参数序号确定，各 rank 本地得到相同值，不额外广播 seed。bucket 重排不改变某个参数的 seed。

取消逐任务 seed broadcast 的前提是：hook 注册时对 process group、base seed、ARC 配置、稳定参数名/序号、角色和 optimizer parameter coverage 生成 fingerprint，并在所有 rank 上做一致性校验；恢复 checkpoint 时再次校验。不一致立即报错，不能继续训练。

这保证 hook 模式自身可重复，但不承诺与旧 shape-batched 路线在 `ratio < 1` 时使用完全相同的随机投影。

### 9.3 State checkpoint

checkpoint 中新增独立的 `arc_compressor` state，而不是把它伪装成普通 Muon optimizer state。参数名只解决 bucket rebuild 后的参数身份，不能区分各 rank 本来就不同的 `h_local/g_local`：

```text
checkpoint
  ├─ model
  ├─ optimizer       # 普通 Muon 状态
  ├─ scheduler
  └─ arc_compressor
       ├─ shared
       │    ├─ schema_version
       │    ├─ dp_world_size
       │    ├─ config_fingerprint
       │    ├─ ordered_parameter_table
       │    └─ g_global（保存一份或验证为 replicated）
       └─ rank_<global_rank>
            └─ <stable_parameter_name>
                 ├─ h_local
                 └─ g_local
```

当前 `CheckpointManager` 使用 DCP 默认 planner；若每个 rank 以相同 metadata key 提供普通 tensor，replicated-tensor 去重会丢失 rank-local 差异。因此 rank-local tensor 必须使用 rank namespace，或采用明确的 rank-sharded 表示，不能只把同名字典附加到公共 state dict。

加载时按 global rank 和稳定参数名匹配，并校验 schema version、DP world size、rank mapping、配置 fingerprint、shape/dtype/角色。首版不支持改变 DP world size 后精确续训。缺失、重复或不兼容状态默认报错；只有显式选择“重新初始化压缩器状态”时才允许丢弃。

## 10. 配置与兼容性

增加显式同步模式，例如：

```yaml
arc_sync_mode: optimizer  # 现有 ArcTopKMuon，默认兼容模式
# 或
arc_sync_mode: ddp_hook   # 新路线
```

选择 `ddp_hook` 时：

- 必须将模型包装为 DDP 并在 backward 前注册 hook。
- optimizer 必须是普通 `Muon`，不能再由 `ArcTopKMuon` 同步梯度，否则会发生双重同步。
- gradient accumulation 仅让非最后 micro-batch 的 forward 和 backward 使用 `no_sync()`。
- `replicate_mesh_grad_sync` 不能继续独立决定 DDP context；由 `arc_sync_mode` 派生唯一同步策略，冲突配置立即报错。
- 要求 `find_unused_parameters=False`，首版不支持动态 unused parameters。
- hook process group 必须与 Muon distributed process group 相同，每个 optimizer 参数必须恰有一个同步角色。
- 单 rank 训练可以省略 collective，但仍执行完整 EF21M 状态更新。

选择 `optimizer` 时，所有 micro-batch 的 forward/backward 都必须处于 `no_sync()`。这会修复当前正式训练入口仅包 backward 的问题；修复后的行为才作为兼容基线。除非新路线改变 ARC/EF21M 数学语义，否则仍归入 M001，作为新的同步调度实现，不创建新方法编号。

## 11. 正确性不变量

实现必须维持以下条件：

1. hook 返回 tensor 与输入 bucket 的 numel、shape、dtype、device 一致。
2. ARC 参数的首步、warmup 和 `ratio=1` 结果等于 EF21M tracker oracle；仅当 `eta=1` 时才要求等价于原始梯度的 DDP average。
3. 所有 rank 在每个 iteration 具有相同的全局 collective 发射序列，而不仅是每桶局部顺序一致。
4. 同步后 `g_global` 和最终参数在所有 rank 一致；`h_local`、`g_local` 可以因本地梯度不同而不同。
5. 同一 optimizer step 的所有 bucket 使用同一个 step 编号。
6. bucket rebuild 不会重置或串换逐参数 EF21M 状态。
7. hook 模式中 optimizer 不执行第二次数据并行梯度 all-reduce。
8. accumulation 的前 N-1 个 micro-batch 不触发 hook，最后一个触发且同步累积梯度。
9. dense auxiliary 参数的数值结果不受 ARC 压缩率影响。
10. rank-local `h_local/g_local` 通过 DCP round-trip 后仍保持各 rank 原值；不会被 replicated 去重。
11. 所有内部 Future 的异常传播到 DDP 返回 Future，且临时 tensor 生命周期覆盖最终 CUDA 工作。

## 12. 测试设计

### 12.1 单元测试

- 将 ARC 核心拆出的 prepare/finalize 原语与现有实现做固定输入对照。
- 验证不同 shape 的参数可在同一 bucket 中被正确打包、偏移和散回。
- 验证 mixed `arc_matrix` / `dense_aux` bucket。
- 验证 stable parameter name 与状态映射、保存和恢复。
- 验证 step/seed 在 bucket 重排前后保持参数级稳定。
- 验证不支持参数的确定性 dense fallback。
- 验证 callback 返回 Future 不会被误当成已 flatten 的 `Future[Tensor]`。
- 用手写多步 EF21M oracle 覆盖首步、warmup 最后一步、首个压缩步和 full-support。

### 12.2 两 rank 分布式测试

优先用 Gloo CPU 测试控制流和数值正确性：

- 首先对当前正式训练写法注册计数 hook，证明仅包 backward 时仍发生 DDP 同步；再验证修正后的 optimizer/hook 两种 context 策略。
- `eta=1, ratio=1` 对照标准 DDP all-reduce；一般 `eta` 对照多步 EF21M oracle。
- `ratio<1` 不仅验证 rank 一致，还逐项验证 `h_local/g_local/g_global` 和最终参数的 oracle。
- 两个以上 bucket，人为引入不同 rank 的 prepare/callback 延迟，验证全局 collective 标签、shape、顺序和完成 Future。
- gradient accumulation，验证前 N-1 次 forward/backward 不触发 hook，最后一次恰好触发并同步累积梯度。
- mixed optimizer roles、`gradient_as_bucket_view` 和 DDP bucket rebuild；首版对 unused 配置做 fail-fast 测试。
- 使用真实 `CheckpointManager`/DCP 做 checkpoint round-trip：保存前令各 rank 的本地状态明显不同，恢复后比较状态和后续多步轨迹。
- 旧 `ArcTopKMuon` 路线回归不受影响。

随后增加 NCCL 多 GPU 测试，覆盖 CUDA stream/Future 行为、allocator 压力、不同 bucket shape、非默认 stream、多步执行和无死锁退出。

### 12.3 性能验证

性能结论必须分别报告：

- 纯 hook/collective microbenchmark；
- 单步 latency（warmup 后稳定区间）；
- 完整训练 wall-clock / tokens per second；
- backward 与 ARC collective 的 profiler timeline；
- dense DDP、optimizer-side ARC、DDP-hook ARC 三方对照；
- 通信字节、bucket 数、bucket size、gradient accumulation 和压缩率。

三方对照中的 optimizer-side ARC 必须使用修复后的 forward+backward `no_sync()`；历史入口行为和修复后行为分开报告。正式训练还需记录 hook 调用次数和实际 collective signature，排除 dense DDP 与 ARC 双重同步。

只有 profiler 显示 ARC collective 位于 backward 区间并与后续反向计算重叠，才能声称恢复了 DDP overlap。只有端到端 wall-clock 改善，才能声称该路线带来训练加速。专门 benchmark 的单步改善不能自动外推到长训练；数据加载、验证、checkpoint、日志、不同 gradient accumulation/bucket layout 和系统抖动都需分别核对。

## 13. 分阶段交付

1. 为当前 `no_sync()` 写两 rank 回归测试，修复 shared training loop，并重新建立 optimizer-side ARC 基线。
2. 用 dummy payload 实现全局 tail Future、多 bucket、多 collective 的最小骨架；先验证 Gloo/NCCL 顺序、异常传播和 CUDA completion。
3. 抽离可复用的 ARC 本地数学原语，以多步 EF21M oracle 保持旧语义。
4. 实现 hook state、静态参数角色/fingerprint 和 fail-fast unused 支持边界。
5. 实现首步、warmup/full-support 和 mixed dense bucket hook。
6. 实现稀疏 ARC sketch/values 链，完成两 rank 数值测试并增加 NCCL 压力测试。
7. 将 rank-local compressor state 接入真实 DCP checkpoint 并验证恢复轨迹。
8. 接入 `train_arctopk.py` 配置，普通 Muon 与修复后的旧 `ArcTopKMuon` 双路线共存。
9. 运行 profiler 和 GPT 350M 对照实验；按研究规范记录 M001 worklog 和正式 CM 实验。

每个阶段都应可独立回归。阻塞 prototype 只作为短期数学/测试参考，不保留为第二套生产实现；异步骨架验证通过后，生产入口只保留 Future 路线。

## 14. 风险与缓解

- **hook 注册了但没有 overlap**：阻塞 collective 或 callback 调度仍可能串行化。用 profiler timeline 验证，不以 API 形态推断性能。
- **bucket 太小导致 launch/Python 开销上升**：记录 bucket 数和每 bucket payload；调优 `bucket_cap_mb`，并在 bucket 内合并 sketch/value collective。
- **bucket 太大导致就绪过晚**：同步启动接近 backward 尾部。对 bucket size 做扫描，而非固定使用默认值。
- **跨 bucket callback 穿插导致 collective 错配/死锁**：首版通过全局 tail Future 串接完整 bucket 链；测试人为制造 rank 延迟并核对全局发射序列。
- **异步状态竞争**：callback 捕获当前 step 和参数状态引用；不允许下一 optimizer step 在当前 backward Future 完成前开始。
- **checkpoint 与 bucket rebuild 错配**：状态按参数名持久化、按参数对象运行，不按 bucket index；rank-local 状态使用 rank namespace，拒绝 world-size 变化恢复。
- **双重同步**：配置构造阶段强制 `ddp_hook -> ordinary Muon`，发现 `ArcTopKMuon` 组合时立即报错。
- **算法轨迹与旧路线不同**：固定 hook 路线的参数级 seed；文档和实验中明确只要求统计/收敛可比，不要求压缩场景逐 bit 相同。
- **正式训练仍发生双重同步**：同步 context 由 mode 唯一派生，并用 hook count/collective signature 作为启动前 correctness gate。
- **CUDA Future 提前完成或临时 buffer 被释放**：显式 completion bridge、异常传播和 bucket context 强引用；用 allocator/stream 压力测试验证。

## 15. 验收标准

设计实现完成需同时满足：

- 所有新增单元测试、两 rank Gloo 测试和现有 ARC/Muon 测试通过。
- NCCL 多 bucket/stream 压力测试无死锁、无未完成 Future、rank 参数一致。
- `eta=1, ratio=1` 与标准 DDP 在规定容差内一致；一般 eta 的 full-support 与多步 EF21M oracle 一致。
- accumulation、mixed bucket、rank-local DCP round-trip、bucket rebuild 和跨 rank 延迟有自动化覆盖。
- profiler 与 hook count 证明 optimizer-side 基线没有 dense DDP 双重同步。
- profiler 证明 ARC 同步在 backward 的 bucket-ready 时刻启动，并至少与一部分后续反向计算重叠。
- 旧 optimizer-side 模式仍可运行，且配置不会造成双重同步。
- GPT 350M 正式实验完整记录环境、吞吐、step time、通信量和 timeline；不预设新路线一定改善 wall clock。

本文件只确定架构、状态所有权、数据流和验收边界。逐文件、逐测试的实施步骤将在本设计评审通过后另行写入 implementation plan。
