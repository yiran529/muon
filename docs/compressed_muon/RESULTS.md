# M001 ARC-TopK-EF21M-Muon：scale-out 正式结果（2026-09-05）

本节只纳入具有完整 3 次 timing、3 次 profiler、有限值、collective signature 和 exact parameter checksum agreement 的成对 cell。共同配置为 BF16、4 卡 DDP、local batch 1、sequence length 256、gradient accumulation 1、20 warmup + 100 measured steps、seed 42、ARC `ratio=0.2`、`projection_rank=4`、`eta=0.1`、`start_compress_step=0`。Muon 的 dense 与 ARC 两侧均使用 DDP process group 执行 result sharding/all-gather。均值后的 `mean±std (CV)` 为三次独立 process repeat 的样本统计；step 和 throughput 来自 timing JSON，NCCL 来自 profiler summary。

| 成对证据 | Dense step → ARC step (ms) | Dense → ARC throughput (tokens/s) | Dense → ARC peak allocated / reserved (MiB) | logical bytes Dense → ARC | profiler gradient NCCL Dense → ARC (ms) | profiler total NCCL Dense → ARC (ms) | R_bytes / R_grad_comm / R_step |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-130M AdamW normal | 39.992±1.540 (3.85%) → 29.401±1.130 (3.84%) | 25629.7±969.0 → 34862.9±1317.7 | 1835.8±0.5 / 5009.3±79.9 → 2277.8±0.0 / 2878.0±0.0 | 267780096 → 177672216 | 154.065±11.148 → 126.108±13.176 | 154.065±11.148 → 126.108±13.176 | 0.3365 / 0.1815 / 0.2648 |
| GPT-130M AdamW P2P-disabled | 39.924±1.788 (4.48%) → 28.745±0.316 (1.10%) | 25682.6±1135.5 → 35627.0±391.5 | 1835.3±0.0 / 5044.0±0.0 → 2277.8±0.0 / 2878.0±0.0 | 267780096 → 177672216 | 159.504±2.056 → 127.369±7.914 | 159.504±2.056 → 127.369±7.914 | 0.3365 / 0.2015 / 0.2800 |
| GPT-350M AdamW normal | 103.959±0.987 (0.95%) → 75.578±1.691 (2.24%) | 9850.6±94.0 → 13553.4±303.9 | 4756.1±0.0 / 11498.0±0.0 → 7691.1±0.0 / 10632.0±0.0 | 709361664 → 308281368 | 413.803±26.158 → 231.813±3.919 | 413.803±26.158 → 231.813±3.919 | 0.5654 / 0.4398 / 0.2730 |
| GPT-350M AdamW P2P-disabled | 106.521±3.330 (3.13%) → 76.587±1.214 (1.59%) | 9619.4±303.9 → 13372.7±210.1 | 4756.1±0.0 / 11498.0±0.0 → 7691.1±0.0 / 10632.0±0.0 | 709361664 → 308281368 | 418.947±27.947 → 234.583±12.846 | 418.947±27.947 → 234.583±12.846 | 0.5654 / 0.4401 / 0.2810 |
| GPT-130M Muon normal | 44.919±3.517 (7.83%) → 39.111±0.269 (0.69%) | 22894.5±1874.0 → 26183.0±181.0 | 1696.6±0.0 / 4454.0±0.0 → 2151.6±0.0 / 2274.0±0.0 | 267780096 → 177672216 | 163.096±8.760 (5.37%) → 108.072±14.492 (13.41%) | 205.027±10.193 (4.97%) → 146.732±17.940 (12.23%) | 0.3365 / 0.3374 / 0.1293 |
| GPT-130M Muon P2P-disabled | 44.848±0.571 (1.27%) → 37.662±1.797 (4.77%) | 22835.3±292.0 → 27232.0±1335.6 | 1696.6±0.0 / 4454.0±0.0 → 2151.6±0.0 / 2274.0±0.0 | 267780096 → 177672216 | 150.866±13.611 (9.02%) → 113.992±1.831 (1.61%) | 190.654±16.594 (8.70%) → 154.824±2.666 (1.72%) | 0.3365 / 0.2444 / 0.1602 |
| GPT-350M Muon normal | 135.272±8.520 (6.30%) → 111.308±1.319 (1.18%) | 7590.8±495.8 → 9200.5±108.8 | 4516.3±0.0 / 9408.0±220.0 → 7211.0±0.0 / 7428.0±0.0 | 709361664 → 308281368 | 448.660±9.835 (2.19%) → 210.305±18.596 (8.84%) | 625.096±14.858 (2.38%) → 391.314±32.242 (8.24%) | 0.5654 / 0.5313 / 0.1772 |
| GPT-350M Muon P2P-disabled | 143.083±0.783 (0.55%) → 112.636±1.972 (1.75%) | 7156.8±39.2 → 9093.0±157.7 | 4516.3±0.0 / 9434.7±273.0 → 7211.0±0.0 / 7428.0±0.0 | 709361664 → 308281368 | 461.161±8.479 (1.84%) → 227.190±7.526 (3.31%) | 641.800±11.199 (1.74%) → 416.532±7.571 (1.82%) | 0.5654 / 0.5074 / 0.2128 |
| GPT-1B Muon normal | 420.583±17.135 (4.07%) → 300.741±16.595 (5.52%) | 2437.4±100.4 → 3412.0±193.5 | 13099.2±0.0 / 21812.0±0.0 → 22450.9±0.0 / 22852.0±0.0 | 2007760896 → 652732440 | 1349.389±17.455 (1.29%) → 480.672±19.867 (4.13%) | 2009.977±18.492 (0.92%) → 1142.696±37.758 (3.30%) | 0.6749 / 0.6438 / 0.2849 |
| GPT-1B Muon P2P-disabled | 430.026±12.398 (2.88%) → 284.279±1.700 (0.60%) | 2382.6±67.9 → 3602.2±21.6 | 13099.2±0.0 / 21812.0±0.0 → 22450.9±0.0 / 22852.0±0.0 | 2007760896 → 652732440 | 1158.050±76.302 (6.59%) → 475.051±13.389 (2.82%) | 1731.601±113.667 (6.56%) → 1140.318±32.511 (2.85%) | 0.6749 / 0.5898 / 0.3389 |

`R_x = 1 - ARC/Dense`。`R_bytes` 和 `R_grad_comm` 只比较梯度同步：dense 侧为 DDP gradient，ARC 侧为 seed/sketch/selected-values/dense-uncompressed，不包含两侧 Muon 共有的 `muon_result`；total NCCL 则包含 result all-gather。不能把 `R_grad_comm` 解释成完整 NCCL reduction。所有正式 artifact 均通过 correctness gate，但 normal Muon 的 130M dense、350M dense、1B ARC step CV 分别为 7.83%、6.30%、5.52%，超过预设的 5% 报告阈值；profiler gradient CV 最高为 13.41%。这些 cell 保留为正式正确性证据，但性能均值需连同波动性解读。

## Muon 补充 timing 与 profiler 指标

下表只补充主表未单列的 Muon 指标。fwd/bwd 包含 dense DDP 梯度通信，ARC 则在 `no_sync` 下把压缩同步移到 optimizer，因此两个分项不能分别解释成纯计算加速或减速。`exposed NCCL` 是 trace 区间并集扣除与计算重叠后的估计值，不是独立墙钟计时。

| Muon 配对 | fwd/bwd Dense → ARC (ms) | optimizer Dense → ARC (ms) | exposed NCCL Dense → ARC (ms) | compute overlap Dense → ARC (ms) |
|---|---:|---:|---:|---:|
| GPT-130M normal | 34.931±2.891 (8.28%) → 3.710±0.005 (0.14%) | 10.128±0.638 (6.30%) → 35.462±0.276 (0.78%) | 160.846±11.277 (7.01%) → 115.820±15.838 (13.67%) | 44.181±1.726 (3.91%) → 30.912±2.334 (7.55%) |
| GPT-130M P2P-disabled | 34.855±0.483 (1.39%) → 3.707±0.007 (0.18%) | 10.153±0.088 (0.86%) → 34.017±1.804 (5.30%) | 148.762±19.031 (12.79%) → 124.574±0.495 (0.40%) | 41.892±4.129 (9.86%) → 30.250±2.172 (7.18%) |
| GPT-350M normal | 91.483±6.044 (6.61%) → 17.967±0.782 (4.35%) | 44.125±2.537 (5.75%) → 94.911±1.504 (1.58%) | 511.615±12.950 (2.53%) → 267.310±29.781 (11.14%) | 113.481±1.941 (1.71%) → 124.004±2.473 (1.99%) |
| GPT-350M P2P-disabled | 97.241±0.586 (0.60%) → 17.404±0.798 (4.58%) | 46.148±0.194 (0.42%) → 96.113±1.235 (1.28%) | 532.851±6.624 (1.24%) → 289.200±5.744 (1.99%) | 108.949±4.575 (4.20%) → 127.332±3.507 (2.75%) |
| GPT-1B normal | 260.935±11.975 (4.59%) → 35.569±2.349 (6.60%) | 159.716±5.182 (3.24%) → 268.372±16.626 (6.20%) | 1708.356±16.454 (0.96%) → 706.378±32.051 (4.54%) | 301.620±6.878 (2.28%) → 436.317±5.897 (1.35%) |
| GPT-1B P2P-disabled | 267.954±8.366 (3.12%) → 34.515±1.154 (3.34%) | 162.141±4.061 (2.50%) → 251.837±1.416 (0.56%) | 1419.783±115.261 (8.12%) → 707.302±33.905 (4.79%) | 311.818±9.069 (2.91%) → 433.016±1.401 (0.32%) |

本轮 Muon 共生成 36 个 timing 与 36 个 profiler artifact，全部满足有限值、collective signature 和 exact parameter checksum agreement；无 OOM。旧 CM004–CM010 中 `distributed_mesh=None` 的 dense Muon 数据因使用了不受支持的 rank-local benchmark 路径而废弃，不再作为结果或探索性结论展示。当前结果仍是本机 synthetic short benchmark，不包含长训练收敛、validation perplexity、time-to-quality 或能耗证据。

## CM024：本地 seed 与异步 DDP bucket hook（2026-09-08）

CM024 Task 11 在 compatible-shape batching 之前的 CPU correctness gate 为 `192 passed, 0 failed, 0 skipped`；两卡 NCCL hook stress 为 `2 passed`，三模式正式入口 smoke 为 `1 passed`。三卡 GPT-350M、BF16、compile、seq 1024、GA256 的 5/25/50 MiB attribution scan 均为正常 NCCL、每 cell 三份 rank trace、unattributed fraction 0，且没有 seed collective：

| bucket | hook ARC/backward GPU overlap | hook exposed gradient tail | optimizer ARC tail | hook profile window |
|---:|---:|---:|---:|---:|
| 5 MiB | 18.920 ms | 24.505 ms | 77.618 ms | 318.117 ms |
| 25 MiB | 17.874 ms | 0.130 ms | 73.719 ms | 246.960 ms |
| 50 MiB | 6.373 ms | 24.713 ms | 83.197 ms | 215.822 ms |

该证据满足“真实 ARC NCCL kernel 与后续 genuine backward compute kernel 有正交集”的 overlap gate。25 MiB 按预登记规则以最小 exposed tail 进入四卡 GA256 正式比较。正式运行完成了 r1/r2 后，用户明确要求停止并改做低 GA sensitivity probe；r3 在 optimizer cell 中途终止，因此该 artifact 不满足三完整 paired blocks 的接受门槛。两个完整 block 中 hook 分别比 optimizer ARC 慢 `0.83%` 和 `1.39%`，也分别比 dense 慢 `0.16%` 和 `0.68%`；这些数字只记录中止边界，不构成正式 wall-clock 结论。

随后按用户要求运行 GPT-350M、4 GPU、BF16、compile、seq 512、global batch 16、device batch 1、GA4 的探索性实验。25 MiB 三组结果为 dense `361.74±8.86 ms (CV 2.45%)`、optimizer ARC `296.94±11.13 ms (3.75%)`、hook ARC `314.02±1.46 ms (0.46%)`。hook 相对 dense 的平均 paired improvement 为 `13.16%`，但相对 optimizer ARC 平均慢 `5.86%`。

50/100/200 MiB 单次扫描后以正 overlap 且最小 hook/optimizer step ratio 选择 100 MiB。100 MiB 三组确认如下：

| 模式 | step mean±sd (CV) | throughput mean | trace profile window | ARC/backward overlap | exposed gradient tail |
|---|---:|---:|---:|---:|---:|
| dense | 344.11±7.61 ms (2.21%) | 23.8k token/s | 262.088 ms | 0 | 164.381 ms |
| optimizer ARC | 262.68±4.71 ms (1.79%) | 31.2k token/s | 189.572 ms | 0 | 88.409 ms |
| DDP-hook ARC | 271.38±3.48 ms (1.28%) | 30.2k token/s | 205.512 ms | 14.717 ms | 0.586 ms |

以完整 paired block 为统计单位，hook 相对 dense 平均快 `21.12%`，三样本穷举 paired bootstrap 区间为 `[20.47%, 21.99%]`；hook 相对 optimizer ARC 平均慢 `3.32%`，区间为慢 `[2.61%, 3.96%]`。因此 CM024 支持“hook 恢复局部 overlap 并几乎消除同步尾部”，但不支持“hook 比保留的 optimizer-side ARC 更快”。GA256 中局部几十毫秒收益被约 9.2 秒完整 step 稀释；GA4 下 ARC 相对 dense 的收益显现，但 batching 前第一版 hook 的 per-parameter dispatch、更多 collective launch 与串行 bucket chain 仍有成本。seq 与 batch 同时变化，所以 GA4 结果是 workload sensitivity，不是单变量 GA 消融，也不用于训练质量结论。

作为后续优化，sparse hook 在 bucket 内按兼容 shape/dtype 批处理，同时保留每参数 stable seed、EF21M state、collective 顺序、Future chain 和 checkpoint schema。两卡同拓扑短 A/B 中，本地 ARC kernel 数从约 `2151` 降至 `803`；singleton 单次 hook/optimizer 劣势为 `12.34%`，batched 三组为平均慢 `6.87%`。但 batched stack/state-copy 使本地 ARC GPU interval 从约 18 ms 增至约 42 ms，故该结果只证明 dispatch 数下降和相对差距缩小，不证明已消除 hook overhead。下一研究点是持久化 grouped state/workspace 或减少 stack/copy，而不是重新引入 seed collective。

Task 12 在 batching 后的最终 HEAD 上运行方案指定的完整 repository-relevant suite，结果为 `178 passed, 0 failed, 0 skipped`（15 warnings，含两项审查修复的新增回归）。独立审查未发现 hook 的 Critical/Important 正确性问题；审查发现的两项 profiler evidence 问题已修正：跨 rank 校验现在比较按 launch 时间排序的 `(category, operation, bytes)` 序列，而非仅比较类别聚合值；backward compute 只接受关联到 `final_backward` 内 CPU op 的 GPU kernel，未关联 kernel 视为 unknown。用严格 parser 重新解析 5/25/50 MiB scan、四卡 100 MiB confirmation 和两卡 batched confirmation 后，所有有序 signature 均一致，上表及 batched run 的 overlap/tail 数值不变。最终 HEAD 的两卡 NCCL hook stress 与三模式正式入口 smoke 另为 `3 passed`（14 warnings）。

## CM025：四卡 batched hook bucket-cap sensitivity（2026-09-08）

保持 GPT-350M、4 GPU、BF16、compile、seq 512、global batch 16、device batch 1、GA4 和 ARC 数学配置不变，对 64/80/120/160/256/384 MiB 各运行单个 20 warmup + 50 measured cell，并串行运行同期 dense/optimizer ARC 各一次。全部 cell exit 0、各有四份 rank trace，有序 collective signature 一致且没有 seed collective。同期 dense 为 `316.69 ms`，optimizer ARC 为 `267.58 ms`。

| bucket cap | DDP buckets | sketch / selected launches | step average | ARC/backward overlap | exposed gradient tail | profile window |
|---:|---:|---:|---:|---:|---:|---:|
| 64 MiB | 17 | 15 / 15 | 282.80 ms | 12.094 ms | 26.522 ms | 192.405 ms |
| 80 MiB | 14 | 12 / 12 | 280.57 ms | 16.466 ms | 35.638 ms | 194.242 ms |
| 120 MiB | 9 | 8 / 8 | 271.91 ms | 13.353 ms | 29.755 ms | 188.165 ms |
| 160 MiB | 8 | 6 / 6 | **269.78 ms** | 15.634 ms | 34.390 ms | 197.803 ms |
| 256 MiB | 5 | 5 / 5 | 280.14 ms | 14.064 ms | 36.754 ms | 206.210 ms |
| 384 MiB | 4 | 4 / 4 | 273.72 ms | 13.507 ms | 0.149 ms | 198.862 ms |

160 MiB 是本次单次扫描的最优点，相对同期 dense 快 `14.81%`，相对 optimizer ARC 慢 `0.82%`。结果不是随 bucket 增大单调改善：更大的 bucket 继续减少 launch 和本地 prepare/Top-K/finalize，但会改变 ready 时机、collective duration 和 overlap；384 MiB 即使只有 4/4 次压缩 collective 且 trace tail 接近零，完整 step 仍慢于 160 MiB。因此 exposed tail 或 collective 次数都不能单独预测 wall-clock。每个 cell 仅一次，本节只用于选择候选参数，不作稳定反超或统计显著性结论。

## CM026：四卡 gradient-accumulation sensitivity（2026-09-08）

固定 GPT-350M、4 GPU、BF16、compile、seq 512、device batch 1、160 MiB bucket cap 和 ARC 数学配置，对 GA 1/2/4/8/16/32 的 dense、optimizer ARC、DDP-hook ARC 各运行单个 20 warmup + 50 measured cell。ARC pair 顺序交替；dense 由紧随其后的 CM026d supplement 在同一 GPU topology 上补齐。所有 cell exit 0、各有四份 rank trace并通过 summary gate。

| GA | dense | optimizer ARC | DDP-hook ARC | hook 快于 dense | hook 慢于 optimizer |
|---:|---:|---:|---:|---:|---:|
| 1 | 232.95 ms | 173.94 ms | 195.62 ms | 16.02% | 12.46% |
| 2 | 276.52 ms | 207.20 ms | 212.72 ms | 23.07% | 2.66% |
| 4 | 305.71 ms | 250.43 ms | 276.28 ms | 9.63% | 10.32% |
| 8 | 437.81 ms | 375.40 ms | 383.70 ms | 12.36% | 2.21% |
| 16 | 629.55 ms | 578.09 ms | 588.78 ms | 6.48% | 1.85% |
| 32 | 1057.19 ms | 976.88 ms | 1007.51 ms | 4.70% | 3.14% |

两个 ARC 路径在所有 GA 均快于 dense，optimizer ARC 在所有 GA 均最快。hook 的局部 trace overlap 约为 `11.4–15.8 ms`、exposed gradient tail 约为 `25.9–28.9 ms`，基本不随 GA 增大；完整 step 中累积计算增加后，这部分固定通信收益相对 dense 总体被稀释。hook/optimizer 差值明显非单调，尤其 GA4 是单次异常高点；每个模式/GA 只有一次且 dense 在补充队列中运行，因此该表支持 Amdahl 趋势观察，不支持对某个 GA 作稳定最优或显著性结论。

## CM027/CM028：论文规模 60M/130M 单 seed 质量实验（2026-09-09）

使用 4×RTX 4090、BF16、compile、FineWeb10B、seq 256、device batch 128、GA1、effective local batch 128/global batch 512 和 seed 42。GPT-60M/130M 分别训练约 1.1B/2.2B tokens；ARC optimizer 配置为 ratio 0.2、projection rank 4、eta 1，并在 step 1000 后压缩。该实验仿照论文的模型量级、序列长度、batch、token 预算和压缩配置，但使用仓库 GPT + Muon 而非 LLaMA + Adam，数据为 FineWeb10B 而非 C4。

| 模型 | dense val loss | ARC val loss | Δloss | dense/ARC step | ARC wall-clock | dense/ARC peak memory |
|---|---:|---:|---:|---:|---:|---:|
| GPT-60M | 3.8904 | 4.3355 | +0.4451 | 101.60 / 111.13 ms | 慢 9.38% | 6834 / 6977 MiB |
| GPT-130M | 3.5648 | 4.0944 | +0.5296 | 213.98 / 238.51 ms | 慢 11.46% | 12760 / 13422 MiB |

四个 cell 和 controller 均 exit 0。两个模型上 ARC 均未通过预设 `Δloss≤0.02` 的质量非劣门槛，也未产生完整 step 加速。尽管只有一个 seed，差距方向跨规模一致且远大于门槛，因此不优先重复完全相同的配置；下一步应先改变算法/压缩配置，而不是追加相同 cell 的统计重复。

## CM029/CM030：论文 batch 下 hook vs optimizer 单次短测（2026-09-09）

保持 CM027/CM028 的 seq 256、effective local batch 128/global batch 512、ratio 0.2、rank 4、eta 1，但从 step 0 开始压缩；每个 cell 为 20 warmup + 200 measured。60M/130M 均在 device batch 128/GA1 通过，无 OOM。GPU 4–7 同时存在外部进程，因此以下只作为 shared-GPU exploratory evidence。

| 模型 | optimizer ARC | DDP-hook ARC | hook vs optimizer | optimizer/hook peak memory |
|---|---:|---:|---:|---:|
| GPT-60M | 131.01 ms | 133.29 ms | 慢 1.74% | 7009 / 7430 MiB |
| GPT-130M | 290.86 ms | 284.64 ms | 快 2.14% | 13709 / 14226 MiB |

两个规模方向相反且差距约 2%，不支持稳定胜负，只支持“大 physical batch 下两条 ARC 路径基本持平”。hook 的额外峰值显存约为 421/517 MiB。

## CM031：论文 batch 下 dense vs optimizer vs hook targeted profiler（2026-09-09）

保持 CM029/CM030 的 GPT-60M/130M、4 GPU、device batch 128/GA1、seq 256、ARC 参数和 160 MiB bucket cap；每个模型/模式只采集一个稳定 final-microstep + optimizer trace。先完成 optimizer/hook 四个 cell，随后在同一 CM031 artifact 中补充两个 dense cell。6 个 cell 均 exit 0、各有 4 份 rank trace，unattributed NCCL fraction 为 0。GPU 4–7 与外部进程共享，以下数字只用于关键路径归因；profile window 也不能替代无 profiler 的稳定 wall-clock。

### 完整阶段与关键路径

下表每行均取四个 rank 的最大值；不同列的最大值不保证来自同一 rank。forward、backward 和 optimizer 是 CPU annotation wall duration，包含其中的调度、hook callback 和可能的等待，不是纯 GPU compute。NCCL/backward overlap 则是 NCCL GPU kernel 与由 `final_backward` 内 CPU op 发射、且排除了 hook-local kernel 的 genuine backward GPU compute 的区间交集。

| 模型 | 模式 | profile window | forward CPU | backward CPU | optimizer CPU | NCCL union | NCCL/backward overlap | exposed NCCL | gradient-sync tail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-60M | dense | 166.280 ms | 7.192 ms | 8.020 ms | 8.591 ms | 87.510 ms | 0.000 ms | 87.510 ms | 81.042 ms |
| GPT-60M | optimizer ARC | 158.670 ms | 5.712 ms | 9.040 ms | 18.714 ms | 78.095 ms | 0.000 ms | 78.095 ms | 73.380 ms |
| GPT-60M | DDP-hook ARC | 162.164 ms | 5.946 ms | 14.201 ms | 29.174 ms | 71.929 ms | 0.000 ms | 71.929 ms | 1.189 ms |
| GPT-130M | dense | 261.345 ms | 16.116 ms | 13.828 ms | 9.824 ms | 79.545 ms | 23.609 ms | 56.470 ms | 45.735 ms |
| GPT-130M | optimizer ARC | 274.757 ms | 13.201 ms | 11.406 ms | 24.504 ms | 70.398 ms | 0.000 ms | 70.398 ms | 60.023 ms |
| GPT-130M | DDP-hook ARC | 258.927 ms | 9.729 ms | 33.865 ms | 13.436 ms | 58.918 ms | 18.888 ms | 40.346 ms | 1.154 ms |

130M dense 的 GPU overlap 大于 CPU backward annotation 并不矛盾：compute kernel 按其关联的 backward CPU op 归类，异步发射的 GPU kernel 可以在 CPU annotation 退出后继续执行。60M hook 虽没有严格的 kernel 时间交集，但 tail 已降至 1.189 ms，表示大部分梯度通信在最后一个 backward compute kernel 之前完成、落在 compute kernel 之间的空隙，而不是与 compute 同时执行。

### Collective 分类

以下仍是四 rank 最大 GPU kernel duration。ARC 并不压缩所有参数；不适合 ARC 的梯度继续走 dense all-reduce。hook 的 bucket 统计显示，60M 的 `244.5 MiB` 梯度中有 `196.5 MiB` 仍为 dense，130M 的 `510.75 MiB` 中有 `294.75 MiB` 仍为 dense。

| 模型 | 模式 | dense gradient | ARC sketch | ARC selected values | Muon result | NCCL union |
|---|---|---:|---:|---:|---:|---:|
| GPT-60M | dense | 81.042 ms | — | — | 6.467 ms | 87.510 ms |
| GPT-60M | optimizer ARC | 41.822 ms | 27.850 ms | 3.899 ms | 4.715 ms | 78.095 ms |
| GPT-60M | DDP-hook ARC | 68.309 ms | 0.317 ms | 1.189 ms | 2.604 ms | 71.929 ms |
| GPT-130M | dense | 70.981 ms | — | — | 8.563 ms | 79.545 ms |
| GPT-130M | optimizer ARC | 34.915 ms | 18.176 ms | 6.977 ms | 10.967 ms | 70.398 ms |
| GPT-130M | DDP-hook ARC | 44.435 ms | 0.766 ms | 5.222 ms | 8.624 ms | 58.918 ms |

optimizer ARC 的核心问题不是 NCCL 总量没有下降，而是压缩和通信位于 backward 之后：它没有任何 NCCL/backward overlap，optimizer CPU range 相对 dense 增加 `10.123/14.680 ms`，130M 的 exposed NCCL 和 gradient tail 甚至分别比 dense 高 `13.928/14.288 ms`。因此通信字节减少并不自动转化为关键路径缩短。这与 CM027/CM028 长训练中的 wall-clock 劣势方向一致；CM031 的 60M 单 trace window 反而较低，属于 shared-GPU 单样本波动，不能推翻长训练 timing。

DDP-hook ARC 把梯度同步移入 backward，代价是 backward CPU range 相对 dense 增加 `6.181/20.037 ms`，其中包含 bucket callback、local prepare、Top-K、EF21M state 和 finalize。收益是 NCCL union 相对 dense 降低 `17.80%/25.93%`，gradient tail 降低 `98.53%/97.48%`；130M 还获得 `18.888 ms` genuine overlap。最终 profile window 相对 dense 低 `2.48%/0.93%`，但 60M 相对 optimizer ARC仍高 `2.20%`。60M hook 的 optimizer CPU range `29.174 ms` 中，`muon_result` annotation 单独达到异常的 `22.087 ms`，而对应 NCCL kernel 仅 `2.604 ms`，进一步显示共享 GPU/host 调度噪声，不应把该单点直接解释为算法固有成本。

综合来看，optimizer ARC 慢在 backward 后串行暴露的压缩与多阶段通信；hook 已基本解决同步尾部，但剩余瓶颈是大量未压缩 dense gradient、bucket 内本地压缩/state 搬运，以及较小模型上难以形成有效 GPU overlap 的固定调度成本。CM031 支持这一机制归因，但由于 shared-GPU、每模式单 trace 和 profiler 扰动，只能与 CM029/CM030 的无 profiler timing 联合解读，不能单独声明稳定性能胜负。

## CM032：all-2D hook 短程 wall-clock（2026-09-09）

将 DDP hook 的压缩资格从 Transformer block 扩展到所有二维参数后，在 CM029/CM030 相同的 GPT-60M/130M、4 GPU、seq256、global batch512、device batch128/GA1、ratio0.2/rank4/eta1/start0 和 160 MiB bucket cap 下，分别串行运行一个 dense、既有 optimizer ARC 与 all-2D hook cell。每个 cell 为 20 warmup + 200 measured；6 个 cell 和 controller 均 exit 0。GPU 4–7 与外部进程共享，且每个设置只有一次，因此本节是探索性同期比较，不是稳定性能主结果。

| 模型 | dense | optimizer ARC | all-2D hook | hook vs dense | hook vs optimizer | hook peak memory |
|---|---:|---:|---:|---:|---:|---:|
| GPT-60M | 143.37 ms | 143.77 ms | **130.22 ms** | 快 9.17% | 快 9.42% | 7844 MiB |
| GPT-130M | 298.78 ms | 291.97 ms | **274.13 ms** | 快 8.25% | 快 6.11% | 15115 MiB |

相对同期 dense，all-2D hook 的峰值显存增加 `979/2067 MiB`；相对 optimizer ARC 增加 `835/1406 MiB`。与 CM029/CM030 的旧 hook 数字跨运行比较时，60M 从 `133.29` 降至 `130.22 ms`、130M 从 `284.64` 降至 `274.13 ms`，方向与“减少未压缩 dense payload”一致，但 shared-GPU 负载不同，不能把这两个跨运行差值当作干净的 all-2D A/B 因果估计。本轮可信度更高的是同一队列内 hook 相对两个同期基线均快 `6%–9%`。

速度收益伴随明显的短程质量风险。220 步最终 validation loss 在 60M 上为 dense/optimizer/hook `5.2222/5.5972/6.2157`，130M 为 `5.2599/5.6035/6.5219`。该实验从 step 0 压缩、步数很短且目标是 wall-clock，不足以判断最终收敛；但结果明确禁止把吞吐收益直接表述成 time-to-quality 收益。下一步应先调高 ratio 或恢复延迟压缩，并用较短质量筛选找出 loss 可接受点，再对候选配置做更长训练。

## CM033：all-2D hook GPT-60M 1.1B-token 训练（2026-09-09）

在 CM027 的 GPT-60M 规模与训练设置下，只运行 all-2D DDP hook：4 GPU、FineWeb10B、seq256、global batch512、device batch128/GA1、8393 updates（约 1.1B tokens）、ratio0.2/rank4/eta1、step1000 后压缩、160 MiB bucket cap、seed42。probe、formal 和 controller 均 exit 0。

| 模式 | final validation loss | step average | peak memory |
|---|---:|---:|---:|
| CM027 dense | **3.8904** | 101.60 ms | 6834 MiB |
| CM027 optimizer ARC | 4.3355 | 111.13 ms | 6977 MiB |
| CM033 all-2D hook | 4.6718 | **100.43 ms** | 7844 MiB |

all-2D hook 相对 dense 单步快 `1.15%`，相对 optimizer ARC 快 `9.63%`，说明完整异步 hook 加上更大压缩覆盖范围已经消除 optimizer-side 串行路径的 wall-clock 劣势。但它相对 dense 的 validation loss 高 `0.7814`，相对 optimizer ARC 也高 `0.3363`；峰值显存分别多 `1010/867 MiB`。因此该配置只通过速度目标，没有通过质量目标，不能声称 time-to-quality 改善，也不值得原样追加 130M。后续质量筛选应优先降低 embedding/lm_head 的压缩强度或采用分角色 ratio。

## CM034：AdamW all-2D hook GPT-60M 1.1B-token 训练（2026-09-09）

使用与 CM033 相同的 4 GPU、FineWeb10B、seq256、global batch512、device batch128/GA1、8393 updates（约 1.1B tokens）、seed42、ratio0.2/rank4/eta1、step1000 后压缩和 160 MiB bucket cap，串行比较标准 dense AdamW 与 all-2D ARC DDP-hook AdamW。两种模式先在共同 device batch 上通过 probe，两个 formal cell 和 controller 均 exit 0。

| 模式 | final validation loss | PPL | step average | tokens/s | peak memory |
|---|---:|---:|---:|---:|---:|
| dense AdamW | **4.0959** | **60.09** | **100.15 ms** | **1,308,756.86** | **6882 MiB** |
| all-2D ARC-hook AdamW | 4.7810 | 119.22 | 101.97 ms | 1,285,397.67 | 7893 MiB |

all-2D ARC-hook AdamW 相对 dense 的 validation loss 高 `0.6851`，PPL 高约 `98.40%`，step average 慢 `1.82%`，吞吐低 `1.78%`，峰值显存多 `1011 MiB`（`14.69%`）。PPL 为 `exp(loss)`，采用当前 FineWeb10B/GPT-2 tokenizer validation 口径，只用于仓库内部对比，不能与官方 ARC-TopK 的 C4/T5-base tokenizer 数字直接比较。

因此 CM034 同时未通过质量和 wall-clock 目标。它说明 CM033 的严重质量退化并非只与 Muon 更新规则有关：标准 AdamW 使用同一 all-2D、ratio0.2 压缩边界时也出现接近翻倍的 PPL，而且没有速度收益。当前配置不应原样扩展；后续应优先让 embedding/lm_head 恢复 dense，或采用更高、分角色的 ratio，再做短质量筛选。
