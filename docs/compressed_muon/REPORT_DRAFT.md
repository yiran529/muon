# 压缩优化器预训练质量与 Timing 对比（粗稿）

本文比较 ARC-TopK、GreedyLore 和 PowerSGD，以及它们对应的 Dense、Rand-K 和 Top-K baseline。质量指标主要使用最终 validation loss 和 PPL；性能指标主要使用平均 step time 和吞吐率。除特别说明外，step average 是完整训练过程的平均值，包含压缩启用前的 Dense 阶段。

## 1. ARC-TopK 及其 baseline 的质量对比

ARC-TopK 的质量结果采用 CM038 和 CM040 系列。两组实验分别对应 GPT-60M 和 GPT-130M，并在相同训练配置下比较 Dense、ARC-TopK、Rand-K 和 Top-K。

实验统一使用 4 张 RTX 4090、BF16、compile、FineWeb10B、sequence length 256、global batch 512、device batch 128、gradient accumulation 1 和 seed 42。60M 模型训练 8,393 updates，约 1.1B tokens；130M 模型训练 16,785 updates，约 2.2B tokens。所有压缩方法都使用 all-2D DDP hook，压缩比例为 0.2，projection rank 为 4，step 1000 后开始压缩，bucket cap 为 160 MiB。

ARC-TopK 使用 EF14 error feedback。Rand-K 和 Top-K 也使用 EF14，并保持相同的压缩比例、rank、训练预算和压缩起始时间。Rand-K 使用由 seed 和参数 ID 确定的共享随机 support；Top-K 则由每个 rank 独立选择 support，并通过 All-Gather 传输 values 和 indices。

| 模型 | 方法 | Final val loss | PPL |
|---|---|---:|---:|
| GPT-60M | Dense | 3.9832 | 53.69 |
| GPT-60M | ARC-TopK/EF14 | 4.0032 | 54.77 |
| GPT-60M | Rand-K/EF14 | 4.1018 | 60.45 |
| GPT-60M | Top-K/EF14 | 3.9920 | 54.16 |
| GPT-130M | Dense | 3.5804 | 35.89 |
| GPT-130M | ARC-TopK/EF14 | 3.5949 | 36.41 |
| GPT-130M | Rand-K/EF14 | 3.6723 | 39.34 |
| GPT-130M | Top-K/EF14 | 3.5881 | 36.17 |

在 GPT-60M 上，ARC-TopK 的最终 loss 比 Dense 高 0.0200，PPL 高 2.02%；在 GPT-130M 上，loss 只高 0.0145，PPL 高 1.46%。因此，ARC-TopK 在两个模型规模上都基本保持了 Dense 的预训练质量。

Rand-K 的质量明显较差。60M 上其 loss 比 Dense 高 0.1186，PPL 高 12.59%；130M 上 loss 高 0.0919，PPL 高 9.63%。Top-K 的质量则非常接近 Dense，60M 和 130M 的 PPL 分别只高 0.88% 和 0.77%。这说明在当前配置下，确定性的 Top-K support 对训练质量更有利，而随机 support 会产生较大的优化误差。

CM037 还比较了早期 EF21M 版本。60M 上 EF21M 的 PPL 为 108.29，相比 Dense 高 101.70%，而切换到 EF14 后下降到 54.77。因此，ARC-TopK 的最终质量结果应采用 EF14 版本；EF21M 只适合作为算法演进或 error-feedback 消融结果，不应作为最终 ARC-TopK 主结果。

需要注意的是，所有质量实验都只有 seed 42，因此这些结果证明的是当前配置下的质量趋势，而不是多 seed 意义上的统计显著性。

## 2. ARC-TopK 及其 baseline 的 timing 对比

ARC-TopK 的 timing 也采用 CM038 和 CM040 系列。每个完整训练实验均包含 20 个 warmup steps 和 measured training steps，step average 是整个训练过程的平均值，包括 step 1000 之前未压缩的 Dense 阶段。因此，这些数字反映的是端到端训练平均速度，而不是压缩阶段的纯 steady-state 速度。

| 模型 | 方法 | Step average | Tokens/s |
|---|---|---:|---:|
| GPT-60M | Dense | 99.03 ms | 1,323,559 |
| GPT-60M | ARC-TopK/EF14 | 96.35 ms | 1,360,374 |
| GPT-60M | Rand-K/EF14 | 114.86 ms | 1,141,146 |
| GPT-60M | Top-K/EF14 | 182.30 ms | 718,991 |
| GPT-130M | Dense | 209.89 ms | 624,479 |
| GPT-130M | ARC-TopK/EF14 | 211.05 ms | 621,047 |
| GPT-130M | Rand-K/EF14 | 250.07 ms | 524,141 |
| GPT-130M | Top-K/EF14 | 424.46 ms | 308,797 |

60M 上，ARC-TopK 比 Dense 快 2.71%，吞吐提高 2.78%；130M 上，ARC-TopK 比 Dense 慢 0.55%，两者基本持平。由于实验包含前 1000 个未压缩 step，ARC-TopK 在压缩阶段的实际收益可能略大于全程平均值，但当前结果仍不足以声称稳定加速。

Rand-K 在两个规模上都比 ARC-TopK 慢：60M 慢 19.21%，130M 慢 18.49%。这主要与当前 Rand-K 参考实现的逐 tensor support 生成、packing 和 collective 路径有关。Top-K 的 timing 最差，60M 比 Dense 慢 84.09%，130M 比 Dense 慢 102.23%。虽然 Top-K 的质量接近 Dense，但每个 rank 独立选择 support，并且需要 All-Gather values 和 indices，带来了较大的选择、重建和通信开销。

因此，ARC-TopK 在质量、速度和显存之间取得了最好的综合折中。Top-K 可以作为“质量接近 Dense、但实现速度较慢”的 baseline；Rand-K 则同时存在质量和速度劣势。需要注明的是，Rand-K 和 Top-K 当前仍是功能正确的参考实现，没有使用 fused selection、跨 tensor batching 或专用 packing kernel，所以这些结果不代表它们的理论性能上限。

## 3. GreedyLore 及其 baseline 的质量对比

GreedyLore 的主要质量结果采用 CM070，并加入最终优化版 PowerSGD 的 CM096 和 CM097 进行比较。

CM070 使用 4 张 RTX 4090、BF16、FineWeb10B、sequence length 256、global/device batch 为 512/128、gradient accumulation 1、bucket cap 160 MiB、seed 1234。60M 训练 10,000 updates，包含 1,000-step warmup；130M 训练 20,000 updates，包含 2,000-step warmup。训练使用 cosine decay 和 global gradient clipping，GreedyLore 从 step 1000 开始压缩，update interval 为 200，使用 local-SVD。GreedyLore rank32 和 high-rank 配置分别为 60M rank128、130M rank256。

CM096/CM097 使用与 CM093/CM094 相同的完整预训练配置，但采用了优化后的 PowerSGD 实现，包括 warm-start Q 复用和同 shape batched Gram–Schmidt。PowerSGD 使用 rank32、EF14 和 step 1000 后压缩。

| 模型 | 方法 | Final val loss | PPL | 相对 Dense 的质量 |
|---|---|---:|---:|---:|
| GPT-60M | Dense | 4.1079 | 60.82 | baseline |
| GPT-60M | GreedyLore rank32 | 4.2119 | 67.48 | PPL +10.96% |
| GPT-60M | GreedyLore rank128 | 4.1230 | — | PPL +1.52% |
| GPT-60M | 优化后 PowerSGD rank32 | 4.1485 | 63.34 | loss +0.0406 |
| GPT-130M | Dense | 3.6530 | — | baseline |
| GPT-130M | GreedyLore rank32 | 3.7645 | — | PPL +11.80% |
| GPT-130M | GreedyLore rank256 | 3.6618 | — | PPL +0.88% |
| GPT-130M | 优化后 PowerSGD rank32 | 3.7255 | 41.49 | loss +0.0725 |

GreedyLore rank32 在两个规模上都出现约 11% 的 PPL 退化。提高 rank 后，质量明显恢复：60M rank128 的 PPL gap 降至 1.52%，130M rank256 的 PPL gap 降至 0.88%。这说明 rank32 是 GreedyLore 质量损失的重要来源。

优化后的 PowerSGD rank32 质量优于 GreedyLore rank32，但仍不如 Dense。60M 上 PowerSGD 的 loss 比 GreedyLore rank32 低 0.0634；130M 上低 0.0390。与 Dense 相比，PowerSGD 的 loss 分别高 0.0406 和 0.0725。

当前质量排序大致为：

> Dense ≈ GreedyLore high-rank > 优化后 PowerSGD rank32 > GreedyLore rank32。

这里的 PowerSGD 与 Dense、GreedyLore 的比较来自不同实验批次。虽然模型、数据、训练预算和主要超参数保持一致，但 GPU 组合和代码版本存在差异，因此只能用来比较质量趋势，不能把小幅差异解释为严格的 paired statistical result。

## 4. GreedyLore 及其 baseline 的 timing 对比

GreedyLore 的 timing 需要结合两类实验理解。第一类是 CM045–CM051、CM067 和 CM071，它们比较了 GreedyLore 与 Dense 在不同模型规模和 physical batch 下的速度。第二类是 CM098–CM101，它们比较了最终优化版 PowerSGD、历史 Dense 和历史 GreedyLore。

CM067 使用 BF16 整模型参数、gradient 和 DDP bucket，sequence length 256、global/device batch 512/128、GA1、bucket80 MiB、rank32、interval200。60M 和 130M 都完成了 3 组 rotated pairing。

| 模型 | Dense | GreedyLore | GreedyLore 相对 Dense |
|---|---:|---:|---:|
| GPT-60M | 92.043 ms | 94.343 ms | 慢 2.50% |
| GPT-130M | 199.543 ms | 204.203 ms | 慢 2.34% |

CM071 将 physical batch 改为 micro-batch32、GA4，同时保持 effective batch 为 128。结果仍然没有显示 GreedyLore 加速：60M 慢 3.94%，130M 慢 1.41%，350M 慢 5.92%。

但是，在 CM051 中，350M 使用 global/device batch32/8，GreedyLore 的 step 为 195.567 ms，而 Dense 为 232.880 ms，GreedyLore 反而快 15.97%。因此，GreedyLore 的 timing 对 physical batch 和模型几何非常敏感。

最终优化版 PowerSGD 的 timing 来自 CM098–CM101。实验使用 rank32、EF14、warm-start 和同 shape batched Gram–Schmidt；每个 cell 为 profiler-off 单次 timing，前 20 个 PowerSGD updates 用作 warmup。由于 Dense 和 GreedyLore 是历史结果，以下属于跨实验单样本比较。

| 实验配置 | Dense | GreedyLore | 优化后 PowerSGD |
|---|---:|---:|---:|
| 350M BF16，batch72 | 370.31 ms | 408.73 ms | 384.06 ms |
| 1B BF16，batch24 | 518.72 ms | 632.97 ms | 694.61 ms |
| 350M FP32，batch64 | 392.65 ms | 432.13 ms | 460.53 ms |
| 350M FP32，batch8，bucket160 | 239.11 ms | 201.66 ms | 239.26 ms |
| 350M FP32，batch8，bucket80 | 252.70 ms | 215.93 ms | 363.22 ms |
| 720M FP32，batch12 | 515.79 ms | 449.70 ms | 723.26 ms |

最终优化版 PowerSGD 在 350M BF16、batch72 上只比 Dense 慢 3.71%，说明优化后的实现已经消除了原始 PowerSGD 约 2.9 倍的严重性能退化。但在 1B BF16 和较大 FP32 模型上，PowerSGD 仍然明显慢于 Dense。CM100 还显示出很强的 bucket sensitivity：350M FP32、batch8 下，bucket160 几乎与 Dense 持平，但 bucket80 却慢 43.74%。该结果只有单次测量，因此只能说明实现对 bucket geometry 敏感，不能作为稳定的 bucket 优势结论。

GreedyLore 在大 batch 下较慢、小 batch 下可能变快，主要有以下原因。

首先，Dense 每个 data-parallel step 都需要同步完整梯度，而 gradient communication 的规模主要由模型参数量决定，并不会随 batch 等比例增加。相反，forward/backward 的计算量会随 physical batch 增大。因此在大 batch 下，Dense 的主要时间更多由计算组成，GreedyLore 能节省的通信时间占比变小。

其次，GreedyLore 每个 compressed bucket 仍然需要执行 score、Top-r、factor、error-feedback 和 reconstruction 等本地操作，并需要多个 collective。此外，每 200 个 update 还会执行一次 corrected-gradient dense All-Reduce 和 local SVD refresh。这些都是相对固定的压缩开销。在大 batch 下，通信节省不足以抵消这些本地计算、callback、Future sequencing 和 refresh 成本，所以 GreedyLore 可能比 Dense 慢 2%–5%。

当 physical batch 较小时，forward/backward 计算时间下降，通信占整个 step 的比例上升。此时，压缩梯度同步所节省的通信时间更容易抵消 GreedyLore 的本地压缩开销。CM051 的 350M、device batch8 结果正体现了这一点：GreedyLore 比 Dense 快 15.97%。

不过，这个现象不能简单总结为“batch 越小，GreedyLore 一定越快”。CM051 同时改变了模型规模、batch 和实验配置；CM067/CM071 使用的是 BF16 和不同的 batch/GA 几何。因此更准确的解释是：GreedyLore 的收益取决于通信在总 step 中的占比、可压缩矩阵梯度的比例、bucket 划分方式、refresh 间隔以及模型规模。小 batch 只是更容易让通信成为瓶颈，并不保证一定产生端到端加速。

综合来看，GreedyLore rank32 的质量和 timing 都不是最优折中；提高 rank 可以恢复质量，但增加计算和显存。最终优化版 PowerSGD 在质量上优于 GreedyLore rank32，并在部分 BF16 小规模配置下接近 Dense timing，但在更大模型和 FP32 配置下仍有明显的本地投影、正交化和 collective 扩展开销。
