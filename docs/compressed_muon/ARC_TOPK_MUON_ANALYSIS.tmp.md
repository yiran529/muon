ARC-TopK 结合 Dion Muon 的最小改动分析（临时笔记）
=====================================================

论文：An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning
链接：https://arxiv.org/abs/2510.26709
分析对象：Dion 仓库中的 dion/muon.py 与现有训练路径


一、核心结论
------------

最合适的核心插入点是 dion/muon.py 的 muon_update_megabatch_async() 中，
在调用 muon_update_pre_orthogonalize()、更新 Muon momentum 之前。

当前路径：

    p.grad
      -> Muon momentum / Nesterov
      -> 分布式重组与正交化
      -> LR 缩放、weight decay、参数更新

推荐路径：

    各数据并行 rank 的本地 p.grad
      -> ARC-TopK sketch All-Reduce
      -> 得到所有 rank 一致的行索引
      -> selected rows All-Reduce
      -> 重建全局压缩梯度 g_hat
      -> 原有 Muon momentum / Nesterov
      -> 原有分布式正交化
      -> 原有参数更新

对应更新关系为：

    g_hat_t = ARC-TopK({g_t^(i)})
    m_t     = mu * m_(t-1) + g_hat_t
    x_(t+1) = x_t - lr * Ortho(m_t)

这样 ARC-TopK 只替换数据并行梯度聚合，Muon 的 momentum、Newton-Schulz、
学习率缩放、weight decay 和参数更新路径均可保持不变。


二、核心代码修改位置
--------------------

主要文件：dion/muon.py

1. Muon.__init__()

建议新增独立的数据并行通信配置，不能复用当前 distributed_mesh 的含义：

    replicate_mesh
    replicate_mesh_grad_sync
    arc_topk_enabled
    arc_topk_ratio
    arc_projection_rank
    arc_seed
    arc_error_feedback（可选，第一版建议关闭）

distributed_mesh 继续只负责现有正交化通信；replicate_mesh 专门负责数据并行梯度聚合。

DDP 中两者可以是同一个 ProcessGroup；HSDP/FSDP 中通常是不同的通信组。

2. Muon._create_ortho_tasks()

当前代码在这里取得：

    gradients = [p.grad for p in params]
    states = [self._get_or_initialize_state(p, "muon") for p in params]
    momentums = [s["momentum"] for s in states]

需要把 ARC 配置、replicate process group，以及可选的误差反馈状态传入
muon_update_megabatch_async()。

3. muon_update_megabatch_async()

核心插入位置是当前的 pre-orthogonalize 之前：

    if arc_topk_enabled and replicate_process_group is not None:
        G = yield from arc_topk_allreduce_async(
            G=to_local(G),
            ...
        )

    # 以下保持现有实现
    U = muon_update_pre_orthogonalize(
        G=G,
        M=to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

不要把 ARC-TopK 放在 muon_update_pre_orthogonalize() 之后。否则压缩对象会从梯度
变成各 rank 的本地 momentum/Nesterov 输出，本地 momentum 将发生分叉，算法、状态恢复
和同步语义都会更复杂。

4. 新增 arc_topk_allreduce_async()

第一版可以直接放在 dion/muon.py 中，避免改动共享的 megabatch_base.py。

对同形状矩阵进行 megabatch 处理：

    G.shape = [B, m, n]
    V.shape = [n, r]

    P_local  = G @ V / sqrt(r)
    P_global = AllReduce-AVG(P_local)

    scores  = sum(P_global**2, dim=-1)
    indices = topk(scores, K=ceil(ratio * m), dim=-1)

    values_local  = gather_rows(G, indices)       # [B, K, n]
    values_global = AllReduce-AVG(values_local)

    G_hat = zeros_like(G)
    scatter_rows(G_hat, indices, values_global)

因为所有 DP ranks 使用同一个投影并对 P_global 做 Top-K，它们会得到完全相同的
indices，因此第二次 All-Reduce 只传 values，不需要传 indices。

论文的通信量记法是：

    Dense All-Reduce: 2mn
    ARC-TopK:         2Kn + 2mr

其中 K = ceil(ratio * m)，r 是 projection rank。


三、不应该修改的位置
--------------------

第一版不建议改动：

    dion/megabatch_base.py::megabatch_orthogonalize_async
    dion/megabatch_base.py::muon_update_newton_schulz
    dion/muon.py::muon_update_post_orthogonalize

原因是 ARC-TopK 面向数据并行梯度聚合；megabatch_orthogonalize_async() 中的
All-to-All/All-Gather 负责 FSDP 矩阵重组及正交化计算任务分配，是另一类通信。

Muon 的正交化是非线性的：

    Ortho(Average(G)) != Average(Ortho(G))

因此不能把 ARC-TopK 简单放到正交化输出之后，也不能让各 rank 先对不同的局部
momentum 做正交化再聚合，否则不再是“压缩梯度同步后继续执行 Muon”。


四、严格只修改 muon.py 的限制
------------------------------

严格只修改 dion/muon.py，可以实现一个能被外部调用的 ARC-TopK-Muon 原型，
但当前 train.py 默认不会产生真实的数据并行通信收益。

原因：

1. DDP 默认在 backward 阶段已经完成 dense gradient All-Reduce。
2. FSDP 默认在 backward 阶段完成 Reduce-Scatter/数据并行同步。
3. Muon.step() 通常收到的已经是同步梯度。

对这些已经同步且各 rank 相同的梯度再次运行 ARC-TopK，只相当于对全局梯度做
一次稀疏化；sketch All-Reduce 和 selected-values All-Reduce 都是冗余通信。

仓库已有关闭框架同步、让 optimizer 自行同步的基础设施：

    train.py:799 附近：FSDP 的 dp_name 切换
    train.py:1023 附近：DDP 的 model.no_sync()

但 train.py:578 附近当前只允许 Dion/DionReference 使用 replicate_mesh_grad_sync，
显式拒绝 Muon。因此获得真实收益所需的最小仓库改动实际上是：

    1. dion/muon.py：ARC-TopK 和 optimizer-side gradient sync。
    2. train.py：允许 Muon 使用 replicate_mesh_grad_sync，并传入 replicate_mesh。

训练主循环中的 no_sync/FSDP 控制逻辑已经存在，不需要重写。


五、通信组必须分离
------------------

当前 Muon 只有 distributed_mesh：

    DDP：用于把不同矩阵的正交化计算分配给不同 rank，再 All-Gather。
    FSDP：通常是 outer_shard_mesh，用于重组分片矩阵并分配正交化任务。

ARC-TopK 需要的是数据并行副本组 replicate_mesh。

建议接口：

    Muon(
        ...,
        distributed_mesh=outer_shard_mesh,
        replicate_mesh=replicate_mesh,
        replicate_mesh_grad_sync=True,
        arc_topk_enabled=True,
        arc_topk_ratio=0.2,
        arc_projection_rank=4,
    )

对于纯 DDP：

    distributed_mesh == replicate_mesh

对于 HSDP/FSDP：

    distributed_mesh 通常是 fs group
    replicate_mesh   是 dp group

不能在 HSDP 情况下用一个 process group 同时承担这两种语义。


六、误差反馈的选择
------------------

推荐第一版仅实现论文 Algorithm 1 的 ARC-TopK compressor，把其输出作为原始 Muon
的输入梯度。这样最容易定位通信收益和数值影响。

论文与 EF21M 结合时使用：

    h_t^(i) = (1-eta) h_(t-1)^(i) + eta * grad_t^(i)
    g_t^(i) = g_(t-1)^(i) + C_local(h_t^(i) - g_(t-1)^(i))
    x_(t+1) = x_t - gamma * Average_i(g_t^(i))

如果完整移植到 Muon，需要额外维护：

    h_local     # 本地 gradient tracker
    g_local     # 本地已传输估计
    g_global    # 所有 DP ranks 一致的全局估计
    momentum    # 原有 Muon momentum

可能的实现：

    h_local.lerp_(local_grad, eta)
    delta = h_local - g_local

    compressed_delta, avg_delta = arc_topk(delta)

    g_local.add_(compressed_delta)
    g_global.add_(avg_delta)

    # 把 g_global 传给原有 Muon momentum

之所以需要 g_global，是因为 g_local 中未被本轮选择的行会保留历史值；不能只聚合
本轮 selected rows 后就把其他行清零。g_global 可以通过每步累加 avg_delta 维护，
从而避免对完整的 g_local 做 dense All-Reduce。

但这会形成“EF21M tracker + Muon momentum”的双动量方法，已经不是论文分析的
EF21M-SGD，也不是原始 Muon。论文收敛定理不能直接套用，应作为新的优化器变体说明。

另一种较省状态的方案是标准 residual error feedback：

    z_i = local_grad_i + residual_i
    c_i = ARC-TopK(z_i)
    residual_i = z_i - c_i
    g_hat = Average_i(c_i)

它只增加一个 residual buffer，但不等同于论文公式 (11a)-(11c)。


七、保证正常执行路径所需的关键条件
------------------------------------

1. Baseline 可切换

    arc_topk_enabled=False

必须完全绕过 ARC 状态和 collectives，保持当前 Muon baseline。ratio=1 也应该提供
一个 dense-average 验证模式，但不一定能做到与 baseline bitwise 相同，因为 collective
的运算顺序可能不同。

2. 各 rank 的任务集合必须相同

当前 Muon 使用：

    group_params = [p for p in group["params"] if p.grad is not None]

当 optimizer 自己执行 collective 时，如果某些 rank 的 p.grad 为 None、另一些 rank
不为 None，会造成 collective 数量或张量大小不一致，从而死锁。

ARC 开启后应以完整、稳定的参数顺序建立任务，并把本地缺失梯度视作零；必要时先
同步 has_grad mask，只在所有 ranks 一致认定全局未使用时跳过参数。

3. Scalar optimizer 参数也要同步

train.py 的 DDP no_sync() 是全模型级别的。关闭框架梯度同步后，不只是 Muon 矩阵，
embedding、lm_head 等 AdamW/Lion 参数也不再自动同步。

因此 Muon 的 AdamW/Lion task 必须在更新前做普通 dense All-Reduce。否则矩阵参数可能
一致，而 scalar optimizer 参数会在各 rank 间分叉。

可参考 dion/dion.py 中已有的：

    adamw_update_allreduce_grad
    lion_update_allreduce_grad
    all_reduce_replicate_mesh

但应注意当前 DistributedOrthoBase 的 scalar path 使用了新的 capturable state_steps；
直接导入旧 helper 可能损失 CUDA Graph 兼容性。更稳妥的是在 Muon 中添加对应的异步
dense gradient sync，再调用当前 megabatch_base 的 scalar update 路径。

4. 保持平均语义

sketch 和 selected values 都必须使用 AVG，或者 SUM 后显式除以 DP world size。
不能把当前平均梯度无意改成梯度和。

5. 随机投影同步

所有同一 DP group 中对应参数必须生成同一个 V。建议 seed 从以下信息稳定派生：

    base_seed
    optimizer step
    参数稳定序号或 shape-group 内序号

不要使用 Python object id，也不要依赖各 rank 可能不同的全局 RNG 消耗顺序。

如果要求论文严格语义，应每一步重新采样投影；如果需要 CUDA Graph，则随机数状态、
buffer 地址和捕获行为需要单独验证。

6. 固定通信形状

    K = ceil(ratio * m)

K 应只依赖静态 shape 和配置。indices 可以变化，但通信 tensor shape 不能变化，
这样更容易保证 AsyncRuntime 和 CUDA Graph 的正常路径。

7. FSDP/HSDP 语义

对 FSDP 参数，应先在每个本地参数 shard 对应的 DP replica group 内做 ARC，然后将
同步后的压缩 shard 送入现有 outer-shard All-to-All 正交化路径。

这种实现严格来说是 shard-wise ARC-TopK，而不是先构造完整矩阵再做全局行 Top-K。
它不增加 FSDP unshard 通信，而且每个 shard 的压缩比可控，适合作为第一版。

如果参数按列分片，不同 FS shards 可能选择不同的行集合；这对各自 DP All-Reduce
仍然合法，但最终完整矩阵不具有统一的全局 row support。论文描述和实验报告中应明确
这是 shard-wise compressor。

8. Checkpoint

不带误差反馈时没有新的 rank-local optimizer state，checkpoint 最简单。

加入 residual 或 EF21M 后，residual/h_local/g_local 会在 DP ranks 间不同。当前分布式
checkpoint 通常假设 replicated optimizer state 在副本间一致，因此必须验证保存恢复。

可以参考 Dion.synchronize_for_checkpoint() 的处理方式，但简单平均或清零 rank-local
EF 状态会导致 checkpoint 前后轨迹不同。若要求严格的 uninterrupted-vs-resume 一致性，
需要把这些状态作为真正的 rank-local checkpoint state 保存。

9. CUDA Graph

现有 CudaGraphOptimizer 要求固定 shape、固定 grad buffer、无 host sync。ARC 的固定 K
和 Top-K indices 本身可以满足固定 shape，但以下部分需要专项验证：

    每步随机 Gaussian projection
    新增 NCCL collectives 的 capture
    ARC/EF buffer 是否预分配
    是否存在 .item()、CPU seed 或动态 Python 分支

因此第一版宜先保证 eager/torch.compile 和多 GPU 正常，再单独声明 CUDA Graph 支持。

10. 日志中的 grad_norm

train.py 当前在 optimizer.step() 之前统计 p.grad norm。当框架梯度同步关闭后，这个值
变成本地未同步梯度范数，不再是全局平均梯度范数。它不影响参数更新，但实验解读时
需要注明；若要保持指标定义，则需要额外处理日志，而这已经超出“只改 Muon”的范围。


八、建议的最小实现结构
----------------------

建议把第一版限制为两个文件：

    dion/muon.py
        - 新增 ARC 配置
        - 保存 replicate mesh/group
        - 新增 arc_topk_allreduce_async
        - 在 pre-orthogonalize 前调用 ARC
        - scalar 参数执行 dense DP sync
        - 保持 baseline 分支

    train.py
        - Muon 构造时传 replicate_mesh
        - 允许 Muon 使用 replicate_mesh_grad_sync
        - 现有 no_sync/FSDP 开关逻辑继续复用

不需要修改：

    dion/megabatch_base.py
    Newton-Schulz/polar express 实现
    模型 forward/backward
    参数更新公式


九、推荐的实现顺序
------------------

第一阶段：仅 ARC compressor

    local raw gradient
      -> ARC-TopK 两次 All-Reduce
      -> existing Muon momentum
      -> existing orthogonalization/update

目标是先验证 collective、安全性和通信收益，不引入新的持久状态。

第二阶段：加入一种误差反馈

    A. 标准 residual EF：状态少、实现简单，但不是论文 EF21M。
    B. 精确 EF21M tracker：更贴近论文，但状态和算法语义明显更复杂。

两个版本应通过配置分开，避免把 compressor 效果和 error-feedback 效果混在一起。


十、最低验证集合
----------------

1. 单 GPU、ARC disabled：与当前 Muon 多步更新一致。
2. 两 GPU、人工构造不同本地梯度：
   - sketch All-Reduce 后的 scores/indices 在各 rank 一致；
   - reconstructed g_hat 在各 rank 一致；
   - optimizer step 后参数和 momentum 一致。
3. ratio=1：ARC 路径等价于 dense averaged gradient 后执行 Muon。
4. 非整除 K、宽矩阵、长矩阵、batch 中参数数目不能整除 world size。
5. 某 rank 的局部 grad=None：不死锁，且行为符合全局 has_grad 语义。
6. DDP 小模型 smoke test：包含 Muon、AdamW/Lion 参数组。
7. HSDP/FSDP smoke test：确认 DP ARC collective 与 FS All-to-All 顺序不冲突。
8. checkpoint save/resume：特别是启用 error feedback 后。
9. 通信 profiler：分别统计 DP ARC 和 Muon 正交化通信，不能把局部收益直接表述为
   整个 optimizer 或端到端训练的同比收益。


最终建议
--------

如果目标是“最小改动、保留正常 Muon 路径”，第一版应把 ARC-TopK 定义为原始梯度到
Muon momentum 之间的可选数据并行聚合层。核心代码只放在 muon.py；为了让现有训练
真正提供未同步的本地梯度，只对 train.py 做最小 wiring 修改。不要修改
megabatch_orthogonalize_async 和 Newton-Schulz。

如果绝对不允许修改 train.py，则只能提供一个供外部训练框架手动关闭 DDP/FSDP 梯度
同步后调用的 Muon API；使用当前 train.py 运行时不会获得论文针对的 DP 通信收益。


十一、为什么可能不会像论文场景一样得到通信收益
------------------------------------------

这里的“可能没有通信收益”有两层含义：第一层是 ARC-TopK 是否真正替换了原来的
dense gradient All-Reduce；第二层是即使替换成功，端到端训练是否真的更快。

### 11.1 当前 Muon.step() 看到的通常已经是同步后的梯度

论文的通信场景是：每个数据并行节点先得到不同的本地梯度 G_i，然后 ARC-TopK 直接
替换原来的 dense gradient All-Reduce：

    原论文 dense 路径：
        local G_i -> dense All-Reduce -> global average G

    原论文 ARC 路径：
        local G_i
          -> sketch All-Reduce
          -> 共同 Top-K support
          -> selected values All-Reduce
          -> compressed global average G_hat

ARC-TopK 的通信节省来自“不再执行 dense All-Reduce”。

但是 Dion 当前 Muon 的通常调用顺序是：

    backward
      -> DDP All-Reduce 或 FSDP Reduce-Scatter
      -> optimizer.step()
      -> muon_update_megabatch_async()

也就是说，如果只在 muon_update_megabatch_async() 中加入 ARC，输入的 p.grad 已经是
同步后的 global gradient。此时实际路径会变成：

    local G_i
      -> 原有 dense All-Reduce
      -> global G
      -> ARC sketch All-Reduce
      -> ARC selected-values All-Reduce

这没有替换原通信，反而在 dense All-Reduce 后新增了两次 collective。所有 rank 上的
global G 已经相同，所以 ARC 的两次 All-Reduce 在信息上也是冗余的。

因此，只有让 DDP/FSDP 不再执行数据并行 dense gradient sync，并把尚未同步的本地
梯度交给 Muon 内部的 ARC-TopK，论文中的通信量比较才成立。

### 11.2 Muon 本身还有论文没有覆盖的正交化通信

论文比较的是数据并行梯度聚合：

    Dense:    2mn
    ARC-TopK: 2Kn + 2mr

但 Dion Muon 在梯度聚合之外，还有正交化相关通信：

    DDP/replicated 参数：
        各 rank 分担一部分矩阵的 Newton-Schulz，然后 All-Gather 结果

    FSDP sharded 参数：
        All-to-All 重组矩阵
        -> Newton-Schulz
        -> All-to-All 返回更新 shard

ARC-TopK 只减少数据并行 gradient sync，不减少这些 All-Gather/All-to-All。因此总时间是：

    T_total = T_forward_backward
            + T_DP_gradient_sync
            + T_ARC_compute
            + T_ARC_collectives
            + T_Muon_orthogonalization
            + T_Muon_orthogonalization_comm
            + T_parameter_update

ARC 只可能降低其中的 T_DP_gradient_sync。如果当前训练主要受 Newton-Schulz、Muon
All-to-All、模型计算或内存带宽限制，即使 DP 通信明显减少，端到端吞吐提升也可能很小。

例如，假设 optimizer step 中：

    dense DP gradient sync = 20 ms
    Muon 正交化及通信     = 60 ms
    其他开销              = 20 ms

即使 ARC 把 DP sync 从 20 ms 降到 5 ms，总时间也只是从 100 ms 降到 85 ms，而不是
按照 gradient payload 的压缩比例获得 4 倍或 5 倍端到端加速。

### 11.3 ARC-TopK 是“两次较小 collective”，不一定总比一次 dense collective 快

ARC 每个压缩单元至少需要：

    1. sketch All-Reduce，大小约为 m*r
    2. selected-values All-Reduce，大小约为 K*n

它还需要额外执行：

    Gaussian projection G @ V
    row score
    Top-K selection
    gather/scatter
    压缩 buffer 的构造或清零

在跨节点、低带宽环境中，减少字节数通常很有价值。但在以下场景中，额外成本可能抵消收益：

    - 单节点 NVLink/NVSwitch，dense NCCL All-Reduce 已经很快；
    - 矩阵较小，通信主要由 collective latency 而不是 payload 决定；
    - 每个 shape group 的矩阵很少，产生大量小 collective；
    - projection GEMM、Top-K 或 gather/scatter 的 kernel launch 开销较高；
    - 压缩比例不够高，K*n 仍接近 m*n；
    - r 较大，m*r sketch 成本不可忽略。

论文中的通信公式主要计算传输的 scalar entries，并不自动等价于实际 wall-clock 时间。
真实时间还取决于 latency、collective 数量、消息大小、网络拓扑、NCCL 算法和压缩算子开销。

### 11.4 Dion 的 megabatch 方式可能让 ARC 的粒度不同于论文实现

当前 Muon 按 shape、sharding 和 dtype 将多个参数组织成 megabatch。若 ARC 按每个参数
分别执行两次 All-Reduce，会产生大量小通信，延迟很可能成为主要开销。

要接近论文中的收益，应尽量做到：

    - 对同 shape 参数 stack 后批量计算 sketch；
    - 将一个 shape group 的 sketches 合并成一次 collective；
    - 将 selected values 打包成一个连续 buffer 再做一次 collective；
    - 尽量与其他 shape group 或正交化计算重叠。

但不同 shape 的矩阵通常具有不同的 m、n、K，难以全部合并为一个 buffer；因此模型中
shape 种类越多，collective 数量越多，实际收益越不稳定。

### 11.5 标量参数仍需要 dense gradient sync

训练入口关闭 DDP/FSDP 的数据并行同步后，所有参数都不再自动同步。ARC-TopK 通常只
应用于 Muon 管理的二维矩阵，而 embedding、lm_head、bias、norm 等参数仍由 AdamW
或 Lion 更新。

这些 scalar-optimizer 参数必须在 optimizer 内执行普通 dense All-Reduce。于是实际
节省的是：

    原全部梯度通信
      - Muon 二维矩阵的 dense gradient sync
      + Muon 二维矩阵的 ARC 通信
      + scalar 参数的 dense gradient sync

如果 embedding 或 lm_head 很大，例如语言模型中 vocab projection 占据较大参数比例，
剩余 dense 通信仍可能占据显著时间。

### 11.6 FSDP 下首先得到的可能是 shard-wise ARC，而不是论文的完整矩阵 ARC

论文把一个完整梯度 view 成 m x n 矩阵，然后从全局矩阵中选择 K 行。

FSDP 中 optimizer 通常只持有本地 shard。为了不先 unshard 完整梯度，最自然的实现是：

    每个 FSDP shard
      -> 在对应 DP replica group 内独立执行 ARC
      -> 得到同步的压缩 shard
      -> 进入现有 Muon All-to-All 重组

这减少的是 DP replicas 之间对应 shard 的同步，但它属于 shard-wise ARC：

    - row-sharded 时，每个 shard 独立选择局部 K 行；
    - column-sharded 时，不同 column shards 甚至可能选择不同的全局行。

系统上仍然可以节省 DP 通信，但其 support 和论文“完整矩阵统一选择 K 行”的定义不完全
相同。若强制完整矩阵统一 support，就可能需要先跨 FSDP group 聚合 row sketch，增加新的
通信；这又会削弱或改变整体收益。

### 11.7 Muon 的非线性会放大“压缩误差”和“时间收益”之间的权衡

ARC-TopK 保证的是压缩后的全局梯度具有 contractive error；论文随后分析的是 EF21M
更新。Muon 会继续对压缩结果做非线性正交化：

    Ortho(G_hat) 与 Ortho(G)

之间的差异不与 ||G_hat-G|| 简单成比例。尤其当稀疏化改变矩阵秩、奇异值分布或弱方向时，
Newton-Schulz 的输出可能发生明显变化。

因此即使单步通信更快，也可能需要：

    - 更低压缩率；
    - 更大 projection rank；
    - error feedback；
    - 更长训练步数才能达到相同 loss。

最终应比较 time-to-quality，而不只是单步通信时间。如果每步快 15%，但达到同等验证
loss 需要多 20% 的 step，总训练时间可能没有收益。

### 11.8 论文实验条件不一定等同于当前环境

论文报告的优势是在其模型、tensor grouping、节点规模、网络环境和 compressor 实现下
得到的。论文实验常用 ratio=0.2、projection rank r=4，并在部分实验中经过 warmup 后
才开启压缩。

当前 Dion Muon 的差异包括：

    - optimizer 是 Muon 而不是论文分析的 EF21M-SGD；
    - 存在 Newton-Schulz/Polar Express 计算；
    - 存在矩阵任务分配的 All-Gather/All-to-All；
    - 可能运行在单节点高速互联而不是跨节点网络；
    - FSDP 下压缩对象是 parameter shard；
    - 参数按 shape megabatch，而不是论文实现中的原始 tensor grouping。

所以可以预期“DP gradient payload 减少”，但不能在实测前直接预期论文中类似的
wall-clock 百分比。


十二、怎样判断当前运行是否真的有收益
----------------------------------

至少需要分别测量下面三层指标：

1. 理论通信量

    dense DP bytes
    ARC sketch bytes
    ARC selected-values bytes
    scalar optimizer dense bytes
    Muon orthogonalization communication bytes

2. 分段 wall-clock

    backward（关闭和开启框架 grad sync 分开测）
    ARC projection/Top-K 时间
    ARC 两次 collective 时间
    Muon orthogonalization communication 时间
    完整 optimizer.step 时间
    端到端 iteration 时间

3. 收敛或质量

    相同步数的 train/validation loss
    达到同一 loss 的训练时间
    ARC ratio、projection rank、warmup 和 error feedback 消融

最关键的对照不是“ARC optimizer.step 对当前 Muon optimizer.step”，而是：

    Baseline：backward 中 dense DP sync + 原 Muon step

    ARC：backward 不做 DP sync
         + optimizer 内 ARC matrix sync
         + optimizer 内 scalar dense sync
         + 原 Muon 后续路径

只有这个对照才能确认 ARC 确实替换了原 dense gradient sync，而不是在其后增加了额外操作。
