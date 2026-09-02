# Compressed Muon 研究指南

本指南承载 Muon 通信优化研究的详细约定。它服务于方法设计、实验管理和论文整理，不要求普通代码维护任务每次都完整执行其中所有建议。

## 1. 研究定位

本项目尝试将团队在 Adam 和分布式梯度压缩中的经验应用到 Muon，主要参考：

1. [ARC-TopK: An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning](https://arxiv.org/abs/2510.26709)：使用轻量 sketch 对齐各节点的稀疏位置，使 Top-K 压缩兼容无需传输索引的 All-Reduce。
2. [GreedyLore: Greedy Low-Rank Gradient Compression for Distributed Learning with Convergence Guarantees](https://arxiv.org/abs/2507.08784)：结合贪心低秩压缩、误差反馈和半惰性子空间更新，并研究其在 Adam 等优化器中的收敛性质。

候选方向包括但不限于：

- All-Reduce 兼容的 Muon Top-K 压缩；
- Muon 梯度、动量或更新的贪心低秩压缩；
- Top-K 与低秩压缩的组合；
- 适配 Muon 的误差反馈或动量误差修正；
- 减少矩阵重组、正交化任务分配或结果回传的通信；
- 通信压缩与计算/通信重叠；
- 对新方法的误差、稳定性或收敛分析。

早期阶段可以优先探索可行性；当方案表现出稳定价值后，再逐步补充理论、完整消融和更大规模实验。

## 2. 明确优化的通信对象

Muon 分布式训练中的通信可能来自不同阶段：

1. DDP/HSDP 的数据并行梯度同步；
2. FSDP 分片矩阵在正交化前的重组；
3. 跨 rank 分配正交化计算；
4. 正交化结果的回传或聚合；
5. 新方法引入的 sketch、索引、低秩因子或误差状态同步。

方法设计应明确优化的是哪一部分，不把局部通信收益直接表述为整个 Muon 的通信收益。

由于正交化是非线性的，压缩、聚合和正交化的顺序会改变算法。方法记录中建议给出更新流程或公式，并说明：

- 与原始 Muon 相比哪些状态和步骤不变；
- 哪些步骤是近似的；
- 各 rank 最终是否得到相同更新；
- 是否仍可使用标准 All-Reduce；
- 误差反馈作用于哪个量。

## 3. 目录规范

研究内容使用以下目录：

```text
dion/
├── muon.py
├── megabatch_base.py
└── muon_<method>.py              # 相对独立的方法，可选

configs/
└── compressed_muon/
    ├── baseline/
    └── <method>/

benchmark/
└── compressed_muon/
    ├── benchmark_collective.py
    ├── benchmark_optimizer.py
    └── <method>/

tests/
├── test_muon_<method>.py
└── test_muon_distributed_<method>.py

docs/
└── compressed_muon/
    ├── RESEARCH_GUIDE.md
    ├── RESEARCH.md
    ├── METHOD_INDEX.md
    ├── EXPERIMENTS.md
    ├── RESULTS.md
    ├── PAPER_NOTES.md
    ├── methods/
    └── experiments/

artifacts/
└── compressed_muon/
    └── <experiment-id>/
```

使用原则：

- 现有 Muon 路径的小改动可以直接进入 `dion/muon.py` 或 `dion/megabatch_base.py`。
- 相对独立的方法可以使用 `dion/muon_<method>.py`，但不要求每个想法都新建实现文件。
- 正式配置放在 `configs/compressed_muon/`。
- 可复用的性能测试与 profiler 脚本放在 `benchmark/compressed_muon/`。
- 测试沿用仓库现有的 `tests/test_*.py` 风格。
- 临时脚本可以留在对应实验产物目录；具有复用价值后再整理进 `benchmark/compressed_muon/`。
- 大日志、checkpoint、trace 和原始结果放在 `artifacts/`，通常不提交到 Git。
- 目录约定确有不适用之处时可以调整，同时更新 `METHOD_INDEX.md` 或相关说明。

## 4. 方法编号

当一个方法开始产生正式代码、配置或比较实验时，为其分配稳定编号：

```text
M001, M002, M003, ...
```

推荐名称：

```text
M001-arc-topk-muon
M002-greedy-lowrank-muon
M003-topk-lowrank-hybrid
```

早期讨论、一次性尝试和很小的实现变体不强制单独编号。一个方法的超参数变化通常属于消融实验，而不是新方法。

方法登记在 `docs/compressed_muon/METHOD_INDEX.md`：

```markdown
| 方法编号 | 名称 | 核心思路 | 主要文件 | 状态 |
|---|---|---|---|---|
```

可使用以下状态：

```text
idea / implementing / testing / active / paused / rejected
```

主要方法可以建立独立文档：

```text
docs/compressed_muon/methods/M001_<method>.md
```

小方案也可以直接记录在 `RESEARCH.md`，无需为了形式创建空文档。

## 5. 实验编号

用于正式比较、图表、论文判断或长期训练的实验使用：

```text
CM<三位编号>-<方法>-<模型>-<并行方式>-ws<world-size>-s<seed>
```

例如：

```text
CM001-baseline-gpt160m-ddp-ws8-s1
CM002-m001-gpt160m-ddp-ws8-s1
CM003-m001-gpt160m-fsdp-ws8-s1
```

同一实验的轻微变体可以增加字母后缀：

```text
CM003a-m001-gpt160m-fsdp-ws8-s1
CM003b-m001-gpt160m-fsdp-ws8-s1
```

单元测试、短 smoke test、排查错误的临时运行和未确定配置的探测不强制编号。一旦其结果被用于正式判断，可以补充编号并整理必要记录。

实验登记在 `docs/compressed_muon/EXPERIMENTS.md`：

```markdown
| 实验编号 | 方法 | 配置 | 目的 | 状态 | 结果位置 |
|---|---|---|---|---|---|
```

状态可以使用：

```text
planned / running / completed / failed / stopped
```

本地实验目录、主要日志和 W&B run name 应使用同一实验编号。实验的轻微重跑是否复用编号，可根据配置和研究目的是否实质变化判断，并在登记表中注明。

## 6. 研究文档职责

### `RESEARCH.md`

维护研究背景、Muon 当前通信路径、主要假设、候选路线、当前判断和待解决问题。它描述研究全局，不堆放每次运行的详细日志。

### `METHOD_INDEX.md`

维护方法编号、实现位置和当前状态，作为所有研究方法的入口。

### `EXPERIMENTS.md`

登记正式实验及其目的、状态和结果位置。临时调试运行无需登记。

### `RESULTS.md`

整理具有比较价值的 baseline、主要结果、消融、性能与收敛观察。不要求每个实验都进入该文件。

### `PAPER_NOTES.md`

维护可能的贡献点、方法命名、理论问题、图表计划、related work、证据缺口和写作素材。

### 方法与实验文档

- 主要方法：`docs/compressed_muon/methods/M001_<method>.md`
- 有长期参考价值的实验：`docs/compressed_muon/experiments/CM001.md`

简单方法或实验可以只在索引中登记，不强制创建独立文档。

## 7. 方法记录建议

根据方法成熟度，逐步记录以下内容：

- 研究动机和假设；
- 压缩对象及其在 Muon 中的位置；
- 聚合和通信方式；
- 是否兼容 All-Reduce；
- 是否传输索引、sketch、低秩因子或其他辅助信息；
- 误差反馈和持久状态的定义；
- 与原始 Muon 的关系；
- 通信量、计算量和显存开销；
- 已知适用条件和未覆盖场景；
- 相关代码、配置和实验编号。

原型阶段不要求一次写全。影响实现语义或实验解释的内容应优先记录。

## 8. 验证建议

验证应与修改范围和当前阶段相称，可以从以下项目中选择：

- 语法、导入和相关单元测试；
- 单 GPU 更新检查；
- 多 GPU collective smoke test；
- 与原始 Muon 或参考实现的数值对比；
- 非整除 shard、空 shard、不同矩阵方向与形状；
- 不同 world size、DDP 或 FSDP2 路径；
- 多步动量和误差反馈状态；
- checkpoint 保存与恢复；
- 小模型短训练。

保持原始 Muon 语义的通信重排适合进行数值一致性检查。有损压缩或新优化器不要求逐元素一致，更应关注误差、稳定性和训练表现。

原型阶段可以只验证主要路径，并记录暂未覆盖的场景。若运行环境不具备所需 GPU，应说明未执行的验证，而不是为了满足形式启动不合适的任务。

## 9. 实验设计与论文证据

实验可以逐步扩展，不要求每个想法立即覆盖所有模型、硬件和并行方式。常见阶段包括：

- collective 或算子 microbenchmark；
- optimizer step 性能测试；
- 小模型短训练；
- baseline 与新方法比较；
- 关键参数和组件消融；
- 更大模型、更多 GPU 或多节点实验；
- 多随机种子和收敛实验。

根据研究目的选择指标，例如：

- 理论通信量和 collective 次数；
- 实际通信时间；
- optimizer step 时间；
- 端到端吞吐；
- 额外计算和显存；
- 训练 loss、验证 loss；
- 达到目标质量所需时间。

探索性实验只需记录最相关的信息。准备形成论文结论时，再补充公平 baseline、必要消融、重复实验和环境信息。

描述结果时区分理论通信量、microbenchmark 和端到端训练表现。只有初步证据时使用“观察到”“初步表明”等措辞，避免过早泛化到未测试的模型、规模或网络环境。

对论文核心结论，应尽量保留代码版本、配置、日志、原始数据和绘图方式之间的对应关系。临时调试和普通失败运行无需完整归档，但具有启发性的失败原因值得记录。

## 10. 实验产物

正式实验目录建议为：

```text
artifacts/compressed_muon/<experiment-id>/
├── config.yaml
├── command.txt
├── stdout.log
├── metrics.jsonl
├── environment.txt
├── profiler/
└── checkpoints/
```

按实验需要保存即可：

- benchmark 可以只保存配置、命令和结果；
- 短训练通常不需要 checkpoint；
- profiler 仅在分析通信或性能瓶颈时开启；
- 正式对比和论文实验再补充环境与版本信息；
- 图表使用的数据和绘图脚本应尽量保留，避免只留下最终图片。

注意控制 checkpoint、trace 和日志体积，不保存无研究价值的大文件。

## 11. 工作阶段与更新

研究工作可以处于以下阶段：

```text
idea -> prototype -> validated -> experimenting -> paper-ready
```

这些状态用于表达进展，不是统一的完成门槛。方法可以因结果不佳、资源不足或研究方向变化而暂停，并保留已有判断。

每次重要工作后，根据实际影响更新相应位置：

- 研究方向或判断变化：`RESEARCH.md`
- 新方法或状态变化：`METHOD_INDEX.md`
- 正式实验启动或结束：`EXPERIMENTS.md`
- 出现有比较价值的结果：`RESULTS.md`
- 影响论文叙事或证据：`PAPER_NOTES.md`

不记录无关紧要的小修改。任务汇报应说明已完成内容、验证情况、主要观察和下一步建议。
