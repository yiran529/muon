# M001：ARC-TopK-EF21M-Muon 设计

## 状态

第一版实现完成，M001 聚焦测试与相关基线回归测试通过。未启动正式训练、收敛实验或性能实验。

## 目标

在 DDP 场景中，用论文 ARC-TopK Algorithm 1 替换 backward 阶段的 dense gradient All-Reduce，并实现论文公式 11a–11c 的 EF21M tracker/error-feedback。EF21M 生成的全局梯度估计随后进入 Dion 现有 Muon momentum、正交化和参数更新路径。

原始 `train.py` 保持默认行为，原始 `dion/muon.py` 保持 baseline 行为。新增薄入口 `train_arctopk.py`，通过对 `train.py` 做通用依赖注入重构来复用训练主体。

## 非目标

- 第一版不支持 FSDP、HSDP 或 TP。
- 第一版不保证 CUDA Graph capture。
- 第一版不执行正式收敛实验、吞吐 benchmark 或 profiler。
- 第一版不为论文针对 EF21M-SGD 的收敛结论新增 Muon 理论证明。
- 第一版不压缩 AdamW/Lion 参数的梯度；这些参数继续使用 dense All-Reduce。

## 方法语义

对于第 `i` 个 DDP rank 的本地随机梯度，EF21M 状态更新为：

```text
h_0^(i) = grad_0^(i)
g_0^(i) = h_0^(i)
g_global_0 = Average_i(h_0^(i))

h_t^(i) = (1 - eta) h_(t-1)^(i) + eta grad_t^(i)
delta_t^(i) = h_t^(i) - g_(t-1)^(i)
```

首步使用 dense All-Reduce 初始化，使论文证明中的 `g_0 = h_0`、`V_0 = 0` 成立。默认前 1000 个 optimizer steps 保持 dense；warmup 期间继续更新 tracker，并令 `g_t^(i) = h_t^(i)`。第 1001 步开始执行 ARC-TopK。

对所有 `delta_t^(i)` 执行完整 ARC-TopK：

```text
P_t^(i) = delta_t^(i) V / sqrt(r)
P_t = Average_i(P_t^(i))                       # sketch All-Reduce
I_t = TopK_rows(diag(P_t P_t^T), K)
C_local(delta_t^(i)) = [delta_t^(i)]_(I_t, :)
C_global(delta_t) = Average_i(C_local(delta_t^(i)))
                                                    # selected-values All-Reduce
```

随后更新本地估计与全局估计：

```text
g_t^(i) = g_(t-1)^(i) + C_local(delta_t^(i))
g_global_t = g_global_(t-1) + C_global(delta_t)
```

`g_global_t` 在所有 DDP ranks 上一致，并作为当前 Muon 的输入梯度：

```text
g_global_t
  -> Muon momentum
  -> Nesterov
  -> Newton-Schulz / Polar Express
  -> NS result All-Gather
  -> weight decay and parameter update
```

这保留完整 ARC-TopK 与 EF21M，但在 EF21M tracker 之后仍保留 Muon 自身 momentum，因此算法是“双动量”的 ARC-TopK-EF21M-Muon，而不是论文分析的 EF21M-SGD。

## 文件结构

### `train.py`

只进行通用入口重构，不添加 ARC-TopK 专属分支：

```python
def parse_cli_args(configure_parser=None): ...

def main(
    hyperparameters_factory=Hyperparameters,
    optimizer_factory=init_optimizer,
    configure_parser=None,
): ...
```

`configure_parser` 在基础参数注册完成后添加额外 CLI 参数；`hyperparameters_factory` 创建方法专属配置对象；`optimizer_factory` 替换当前硬编码的 `init_optimizer()` 调用。

直接执行 `python train.py` 时，所有默认参数仍指向现有对象，因此行为不变。

### `train_arctopk.py`

作为薄入口：

- 定义继承自基础配置的 `ArcTopKHyperparameters`。
- 增加 `arc_topk_ratio`、`arc_projection_rank`、`arc_eta` 和 `arc_seed` 参数。
- 增加 `arc_start_compress_step`，默认值为论文实验设置的 `1000`。
- 默认设置 `optimizer="arc_topk_muon"` 和 `replicate_mesh_grad_sync=True`。
- optimizer factory 只接受纯 DDP，即 `device_mesh is None`。
- 复用基础训练循环、DDP `no_sync()`、数据加载、日志和 checkpoint 管理。
- 调用注入后的 `train.main(...)`。

### `dion/arc_topk.py`

包含与 Muon 解耦、可独立测试的功能：

- Gaussian projection 生成。
- batched row sketch。
- row score 和 Top-K support。
- batched gather/scatter。
- sketch 与 selected values 的异步 All-Reduce。
- EF21M 本地与全局状态更新。

第一版对同 shape 参数构造 `[B, m, n]` 张量。一个 shape group 使用一次同步 seed 生成 `[B, n, r]` 的独立 Gaussian projections，从而避免每个参数单独发起 seed collective。

### `dion/muon_arctopk.py`

新增 `ArcTopKMuon`，复用现有 Muon 的正交化与更新函数，但拥有独立 task creation 和状态初始化：

```text
momentum
arc_h_local
arc_g_local
arc_g_global
```

状态在 optimizer 构造阶段预创建，保持各 rank 的 state key 和 shape 一致。ARC 输出在调用现有 `muon_update_pre_orthogonalize()` 前生成。

AdamW/Lion 参数在更新前沿 DDP process group 执行普通 dense `AllReduce-AVG`。

### `dion/__init__.py`

导出 `ArcTopKMuon`，不修改现有 `Muon` 导出。

### 配置与测试

```text
configs/compressed_muon/m001_arc_topk_muon_ddp.yaml
tests/test_arc_topk.py
tests/test_muon_arctopk.py
tests/test_muon_arctopk_distributed.py
```

## 完整 ARC-TopK 细节

### 随机 seed

每个 optimizer task 在 rank 0 生成 seed，并广播给同一 DDP process group。所有 rank 使用独立的 `torch.Generator` 和相同 seed 生成相同的 Gaussian `V`。不依赖各 rank 的默认全局 RNG 消耗顺序。

第一版为 eager-only，允许将设备 seed 转成 Python 整数；CUDA Graph 对 RNG 和 host synchronization 的限制留待后续处理。

### 投影与选择

对于每个二维矩阵 `G`，保持其自然 `(m, n)` 布局。配置必须满足：

```text
0 < arc_topk_ratio <= 1
projection_rank >= 1
K = ceil(arc_topk_ratio * m)
```

采用论文公式中的 `1 / sqrt(r)` 缩放。因为该缩放不改变 Top-K 排序，但保留它能使实现与论文定义一致。projection 和 sketch 跟随梯度/EF21M 状态 dtype，避免 BF16 训练时将 sketch 固定提升到 FP32 而增加通信字节。为了获得投影压缩收益，实际配置通常应使 `projection_rank < n`；这属于效率建议，而不是算法有效性的输入约束。

### 通信

每个 shape group 在压缩阶段的正常顺序固定为：

```text
seed broadcast
sketch AllReduce-AVG
selected-values AllReduce-AVG
```

首步和 warmup 阶段不生成 seed 或 sketch，只对完整 tracker batch 执行一次 dense AllReduce-AVG。`arc_start_compress_step=0` 会关闭额外 warmup，但首步仍执行理论要求的 dense 初始化。

所有 ranks 必须按相同参数顺序创建 task。ARC 模式不按本地 `grad is not None` 独立过滤参数；本地缺失梯度按零梯度参与 EF21M 递推，避免 collective 顺序不一致。即使所有 ranks 的当前梯度都缺失，历史 tracker 仍按公式继续演化，因此不额外引入 active-mask collective。

### `ratio=1`

当 `K=m` 时仍执行完整 ARC 通信路径。重建结果必须等于本地梯度的 dense DDP 平均值，用于验证 compressor、EF21M 和通信平均语义。

## 训练数据流

`replicate_mesh_grad_sync=True` 使基础训练循环在所有梯度累积 microsteps 中使用 DDP `no_sync()`。最后一个 microstep 后，`.grad` 保存本 rank 的累积本地梯度。

矩阵参数由 `ArcTopKMuon` 执行 ARC-TopK-EF21M。标量优化器参数由 `ArcTopKMuon` 在 optimizer step 内执行 dense All-Reduce。由此，optimizer step 结束后所有 DDP ranks 的模型参数保持一致。

基础 `grad_norm` 日志仍发生在 optimizer-side sync 之前，因此记录的是本地梯度范数。第一版接受这一指标差异，不增加额外 dense 通信。

## 错误处理

- 非 DDP 启动时立即报错，不静默退化到 FSDP。
- process group 缺失且 world size 大于 1 时立即报错。
- 非法 ratio、projection rank 或 eta 在构造时报错。
- 非法 `arc_start_compress_step`（负数、布尔值或非整数）在构造时报错。
- 第一版拒绝 `flatten=True`、`num_heads>1` 和 `split_sizes`，避免 ARC 行定义与 Muon 子矩阵语义混杂。
- collective task 的参数集合必须 rank-symmetric；测试覆盖某 rank 本地梯度缺失的情况。

## Checkpoint

普通单进程 `state_dict()` 必须包含并恢复全部 ARC/EF21M 状态。

加入 warmup 配置之前创建的 M001 checkpoint 没有 `arc_start_compress_step`。加载时将该字段迁移为 `0`，保留旧 checkpoint 恢复后立即压缩的行为，避免重新执行默认 1000 步 warmup。

分布式 checkpoint 的 rank-local `h_local` 和 `g_local` 在不同 DDP ranks 上合法地不同，不能在保存前简单平均。第一版不承诺分布式 checkpoint 的不间断轨迹等价；专用训练配置默认 `checkpoint_freq=0`。后续若需要正式长训练，应将 rank-local compressor state 设计为显式 rank-local checkpoint 数据。

## 自动化测试完成标准

第一版只要求自动化测试通过，不要求训练性能结论。测试至少覆盖：

- ARC 参数校验。
- 固定 seed 下 Gaussian projection 和 support 可复现。
- 两个 rank 获得相同 support。
- selected-values All-Reduce 等于手工计算的平均值。
- `ratio=1` 时 ARC 输出等于 dense average。
- EF21M 的 `h_local`、`g_local`、`g_global` 多步递推符合公式 11a–11c。
- 首步 dense 初始化满足 `g_0 = h_0`，warmup 截止步仍走 dense tracker 同步。
- BF16 输入产生 BF16 projection/sketch，不额外扩大 sketch 通信字节。
- 两个 DDP ranks 使用不同本地矩阵梯度后，Muon momentum 和参数保持一致。
- AdamW/Lion 参数通过 dense All-Reduce 保持一致。
- 某个 rank 的局部梯度缺失时 collective 不死锁。
- `train.py` 原入口的默认 factory 和参数解析行为不变。
- `train_arctopk.py` 能导入、能解析 ARC 参数，并拒绝非 DDP 配置。
- optimizer `state_dict()` 单进程保存恢复 ARC 状态。

## 最低验证 prompt

实现完成后另行生成一个独立 Markdown 文件，内容是可直接交给其他 agent 的验证 prompt。该 agent 只执行和审查验证，不修改实现；prompt 将包含精确测试命令、分布式运行要求、需要核查的状态一致性、失败证据格式和最终报告模板。
