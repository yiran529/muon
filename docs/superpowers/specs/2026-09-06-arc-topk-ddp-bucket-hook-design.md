# ARC-TopK + Muon DDP Bucket Hook 设计

状态：待评审

日期：2026-09-06

关联方法：M001 ARC-TopK-EF21M-Muon

## 1. 背景与问题

当前 `ArcTopKMuon` 在 `optimizer.step()` 中执行 ARC-TopK 梯度同步。训练侧为了避免 DDP 在 backward 中先做一次普通梯度同步，会让同一 optimizer step 的所有 micro-batch（包括最后一个）都进入 `no_sync()`。因此执行顺序是：

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

将算法拆成“本地准备—异步 sketch all-reduce—TopK 与打包—异步 selected-values all-reduce—重建 bucket”，用 `Future` 串联。非压缩参数单独打包做 dense all-reduce。旧 `ArcTopKMuon` 继续存在，训练配置显式选择同步模式。

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
train.py   │  所有 micro-batch 使用 DDP.no_sync()
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
micro 0     backward under no_sync ─┐
micro 1     backward under no_sync ─┼─ 本地累加到 .grad
...                                │
micro N-1   normal DDP backward ───┘
                       │
                       └─ bucket ready 后调用 ARC hook
```

只有最后一个 micro-batch 触发 reducer。hook 看到的是该 optimizer step 的累积梯度，而不是只看到最后一个 micro-batch 的梯度。这一行为必须由分布式测试确认。

### 7.2 Bucket 分类与重建

hook 使用 `GradBucket.parameters()`、`gradients()` 和 buffer views 建立 bucket 内映射，不再要求参数 shape 相同。

每个参数属于以下一种角色：

- `arc_matrix`：由 Muon 更新且满足 ARC-TopK 支持条件的矩阵参数。
- `dense_aux`：由 AdamW/Lion 等分支更新的参数，或 ARC 不支持而回退到 dense 的参数。
- `ignored`：DDP 未提供有效梯度的参数；不发起与其他 rank 不一致的额外 collective。

压缩阶段开启后，不允许为了方便而对整个 bucket 先做 dense all-reduce，再覆盖矩阵切片；那会保留主要通信量，失去压缩意义。正确做法是：

- 将所有 `dense_aux` 梯度打包成一个连续 buffer，执行一次 dense all-reduce，再散回对应 view。
- 对各 `arc_matrix` 参数独立维护 EF21M 状态；本地生成 sketch 后，将同一 bucket 的 sketch 扁平拼接并统一 all-reduce。
- 根据聚合 sketch 选择索引，拼接对应 values 后统一 all-reduce，再分别重建 `g_global`。
- 最终把两条路径的结果写回原始 `bucket.buffer()`，Future 返回该 buffer。

DDP bucket 通常按 dtype/device 构造，但 hook 仍应显式检查并基于实际 buffer dtype/device 工作，不把这一点当成未验证假设。

### 7.3 Warmup 与 dense fallback

warmup step、`ratio=1` 或 bucket 中没有可压缩参数时，对整个 bucket 执行标准 dense all-reduce，并显式除以 data-parallel world size。PyTorch comm hook 不会自动完成除法。

进入压缩阶段后，仅 `dense_aux` 走 dense 路线。任何运行时不支持情况应在所有 rank 上依据一致的静态元数据作出相同回退决定，避免 collective 顺序分叉。

## 8. 异步协议

最终 hook 不在主体中等待 collective 完成，而返回串联后的 Future：

```text
local prepare
  ├─ pack dense_aux ───── async dense all-reduce ─────┐
  └─ pack ARC sketches ─ async sketch all-reduce ─┐  │
                                                   ▼  ▼
                                      wait/collect completed futures
                                                   │
                                      select TopK + pack values
                                                   │
                                      async values all-reduce
                                                   │
                                      EF21M finalize + scatter
                                                   │
                                      return bucket.buffer()
```

实施顺序允许先构建阻塞式 correctness prototype，再替换为 Future 链；但阻塞式版本不满足最终性能验收标准，也不能作为“已经获得 overlap”的依据。

所有 rank 必须以相同 bucket 顺序和每个 bucket 内相同 collective 顺序执行。callback 中的分支只能依赖各 rank 一致的元数据或 collective 结果，不能依赖 rank-local 的数值条件决定是否发起通信。

## 9. 状态、随机性与生命周期

### 9.1 `ArcTopKDDPState`

新状态对象至少持有：

- data-parallel process group、world size 和 rank；
- ARC 配置、warmup 配置和当前 optimizer-step 编号；
- 参数对象到静态元数据的运行时映射；
- 每个 ARC 参数的 `h_local`、`g_local`、`g_global`；
- 稳定参数名、参数序号、角色、shape 和 dtype；
- 可选的 profiling/byte counters。

运行时可按 parameter object identity 查找状态；持久化时必须使用 `model.named_parameters()` 的稳定名称。不能用 bucket index 作为长期身份，因为 DDP 可能 rebuild buckets。

### 9.2 Step 与 seed

训练循环在一个 optimizer step 的 backward 开始前，显式调用类似 `arc_state.begin_step(step)` 的接口。hook 调用时捕获该 step 的局部值；step 不按 bucket 自增，避免异步 callback 观察到下一步编号。

每个参数的投影 seed 由 `base_seed`、optimizer step 和稳定参数序号确定，各 rank 本地得到相同值，不额外广播 seed。bucket 重排不改变某个参数的 seed。

这保证 hook 模式自身可重复，但不承诺与旧 shape-batched 路线在 `ratio < 1` 时使用完全相同的随机投影。

### 9.3 State checkpoint

checkpoint 中新增独立的 `arc_compressor` state，而不是把它伪装成普通 Muon optimizer state：

```text
checkpoint
  ├─ model
  ├─ optimizer       # 普通 Muon 状态
  ├─ scheduler
  └─ arc_compressor  # hook 的 step、配置校验信息和逐参数 EF21M 状态
```

加载时按稳定参数名匹配，并校验 shape/dtype/角色。缺失、重复或不兼容状态默认报错；只有显式选择“重新初始化压缩器状态”时才允许丢弃。

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
- gradient accumulation 仅让非最后 micro-batch 使用 `no_sync()`。
- 单 rank 训练可以退化为本地 identity，但状态更新语义应与多 rank 一致。

选择 `optimizer` 时保持现有行为和配置兼容，便于复现实验。除非新路线改变 ARC/EF21M 数学语义，否则仍归入 M001，作为新的同步调度实现，不创建新方法编号。

## 11. 正确性不变量

实现必须维持以下条件：

1. hook 返回 tensor 与输入 bucket 的 numel、shape、dtype、device 一致。
2. `ratio=1` 且 warmup/dense 模式下，结果等价于 DDP SUM 后除以 world size。
3. 所有 rank 在每个 iteration 发起相同数量、相同顺序、兼容 shape 的 collective。
4. 同步后 `g_global` 和最终参数在所有 rank 一致；`h_local`、`g_local` 可以因本地梯度不同而不同。
5. 同一 optimizer step 的所有 bucket 使用同一个 step 编号。
6. bucket rebuild 不会重置或串换逐参数 EF21M 状态。
7. hook 模式中 optimizer 不执行第二次数据并行梯度 all-reduce。
8. accumulation 的前 N-1 个 micro-batch 不触发 hook，最后一个触发且同步累积梯度。
9. dense auxiliary 参数的数值结果不受 ARC 压缩率影响。

## 12. 测试设计

### 12.1 单元测试

- 将 ARC 核心拆出的 prepare/finalize 原语与现有实现做固定输入对照。
- 验证不同 shape 的参数可在同一 bucket 中被正确打包、偏移和散回。
- 验证 mixed `arc_matrix` / `dense_aux` bucket。
- 验证 stable parameter name 与状态映射、保存和恢复。
- 验证 step/seed 在 bucket 重排前后保持参数级稳定。
- 验证不支持参数的确定性 dense fallback。

### 12.2 两 rank 分布式测试

优先用 Gloo CPU 测试控制流和数值正确性：

- `ratio=1` 对照标准 DDP all-reduce。
- `ratio<1` 验证 rank 间 `g_global` 与参数一致。
- 两个以上 bucket，验证 collective 顺序和完成 Future。
- gradient accumulation，验证只在最后 micro-batch 触发同步。
- mixed optimizer roles、缺失梯度和 DDP bucket rebuild。
- checkpoint round-trip 后继续一步，与未中断运行对照。
- 旧 `ArcTopKMuon` 路线回归不受影响。

随后增加 NCCL 多 GPU smoke test，覆盖 CUDA stream/Future 行为和无死锁退出。

### 12.3 性能验证

性能结论必须分别报告：

- 纯 hook/collective microbenchmark；
- 单步 latency（warmup 后稳定区间）；
- 完整训练 wall-clock / tokens per second；
- backward 与 ARC collective 的 profiler timeline；
- dense DDP、optimizer-side ARC、DDP-hook ARC 三方对照；
- 通信字节、bucket 数、bucket size、gradient accumulation 和压缩率。

只有 profiler 显示 ARC collective 位于 backward 区间并与后续反向计算重叠，才能声称恢复了 DDP overlap。只有端到端 wall-clock 改善，才能声称该路线带来训练加速。

## 13. 分阶段交付

1. 抽离可复用的 ARC 本地数学原语，保持旧测试通过。
2. 实现 hook state、参数角色映射和 checkpoint schema。
3. 实现 dense/`ratio=1` hook，验证 DDP 与 accumulation 语义。
4. 实现阻塞式 ARC bucket correctness prototype，完成两 rank 数值测试。
5. 替换为 Future 异步链并增加 NCCL 测试。
6. 接入 `train_arctopk.py` 配置，普通 Muon 与旧 `ArcTopKMuon` 双路线共存。
7. 运行 profiler 和 GPT 350M 对照实验；按研究规范记录 M001 worklog 和正式 CM 实验。

每个阶段都应可独立回归；在异步版验证完成前，不移除阻塞 prototype 所提供的可诊断边界。

## 14. 风险与缓解

- **hook 注册了但没有 overlap**：阻塞 collective 或 callback 调度仍可能串行化。用 profiler timeline 验证，不以 API 形态推断性能。
- **bucket 太小导致 launch/Python 开销上升**：记录 bucket 数和每 bucket payload；调优 `bucket_cap_mb`，并在 bucket 内合并 sketch/value collective。
- **bucket 太大导致就绪过晚**：同步启动接近 backward 尾部。对 bucket size 做扫描，而非固定使用默认值。
- **rank 间 collective 分叉导致死锁**：角色和 fallback 只依赖一致静态元数据；测试多 bucket、unused 参数和异常路径。
- **异步状态竞争**：callback 捕获当前 step 和参数状态引用；不允许下一 optimizer step 在当前 backward Future 完成前开始。
- **checkpoint 与 bucket rebuild 错配**：状态按参数名持久化、按参数对象运行，不按 bucket index。
- **双重同步**：配置构造阶段强制 `ddp_hook -> ordinary Muon`，发现 `ArcTopKMuon` 组合时立即报错。
- **算法轨迹与旧路线不同**：固定 hook 路线的参数级 seed；文档和实验中明确只要求统计/收敛可比，不要求压缩场景逐 bit 相同。

## 15. 验收标准

设计实现完成需同时满足：

- 所有新增单元测试、两 rank Gloo 测试和现有 ARC/Muon 测试通过。
- NCCL smoke test 无死锁、无未完成 Future、rank 参数一致。
- `ratio=1` 与标准 DDP 在规定容差内一致。
- accumulation、mixed bucket、checkpoint round-trip 和 bucket rebuild 有自动化覆盖。
- profiler 证明 ARC 同步在 backward 的 bucket-ready 时刻启动，并至少与一部分后续反向计算重叠。
- 旧 optimizer-side 模式仍可运行，且配置不会造成双重同步。
- GPT 350M 正式实验完整记录环境、吞吐、step time、通信量和 timeline；不预设新路线一定改善 wall clock。

本文件只确定架构、状态所有权、数据流和验收边界。逐文件、逐测试的实施步骤将在本设计评审通过后另行写入 implementation plan。
