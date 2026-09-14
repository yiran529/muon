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

## CM035/CM036：论文 AdamW 配方下的 EF21M 与 EF14 对照（2026-09-10）

保持 GPT-60M、4 GPU、FineWeb10B、seq256、global batch512、device
batch128/GA1、8393 updates（约 1.1B tokens）、seed42、all-2D、ratio0.2、
projection rank4、step1000 后压缩和 160 MiB bucket cap 不变。AdamW 两侧统一
使用 LR `0.001`、betas `(0.9, 0.999)`、epsilon `1e-8`、global grad-norm clip
`1.0`、1000-step linear warmup 后 cosine decay，以及所有参数 weight decay `0`。
CM035 串行运行 dense 与 EF21M；CM036 只补充 EF14 cell，并复用 CM035 选出的
device batch/GA。三个 formal cell 和两个 controller 均 exit 0。

| 模式 | final validation loss | PPL | step average | tokens/s | peak memory |
|---|---:|---:|---:|---:|---:|
| dense AdamW | **4.1903** | **66.04** | 127.64 ms | 1,026,888 | **6882 MiB** |
| EF21M all-2D hook AdamW | 4.7176 | 111.90 | 127.80 ms | 1,025,603 | 7893 MiB |
| EF14 all-2D hook AdamW | **4.3159** | **74.88** | **123.25 ms** | **1,063,465** | 7404 MiB |

EF21M 相对 dense 的 loss 高 `0.5273`、PPL 高 `69.44%`；EF14 相对 dense 的
loss 只高 `0.1256`、PPL 高 `13.38%`。在压缩开启前，step1000 的
dense/EF21M/EF14 validation loss 分别为 `5.2528/5.2566/5.2559`。到 step1500，
三者分别为 `4.8878/5.7735/5.2531`：EF21M 开启压缩后出现明显 loss 反弹，EF14
没有出现同类跳升。最终 EF14 相对 EF21M 将 loss 降低 `0.4017`，PPL 降低
`33.08%`。

该对照把 AdamW 配方、压缩范围和 ARC 参数固定，仅改变 error-feedback 语义，因此
强烈支持“此前大部分质量退化来自 EF21M 与 AdamW 的交互，而不是仅由 all-2D
embedding/lm_head 压缩造成”。EF14 剩余 `0.1256` loss gap 与论文 Table IV 的约
`0.1004` 已处于接近量级，但本实验仍使用 GPT/FineWeb/GPT-2 tokenizer，不能视为
LLaMA/C4/T5-base 的逐项复现，也不能据此断言剩余差距的唯一来源。

EF14 的 step average 相对 dense 低 `3.44%`、吞吐高 `3.56%`，相对 EF21M 的
吞吐高 `3.69%`；峰值显存相对 dense 多 `522 MiB`，相对 EF21M 少 `489 MiB`。
CM035 与 CM036 是先后运行的独立 controller，且 GPU 4–7 存在共享负载，因此这些
约 3% 的跨 controller 性能差异只作参考；同 seed、完整 token 预算下的质量改善是
本轮更有诊断价值的证据。

原始产物位于：

- `artifacts/compressed_muon/CM035-m001-adamw-paper-recipe-all2d-hook-gpt60m-train-ddp-ws4-s42/`
- `artifacts/compressed_muon/CM036-m001-ef14-all2d-hook-adamw-paper-recipe-gpt60m-ddp-ws4-s42/`

## CM037–CM041：独立 scalar AdamW、EF14 与 Sparse-K 基线（2026-09-10–13）

本轮继续使用 4×RTX 4090、BF16、compile、FineWeb10B、seq256、global batch512、
device batch128/GA1、seed42 和严格 token-weighted validation。GPT-60M 训练 8393
updates（`1,100,087,296` tokens），GPT-130M 训练 16785 updates
（`2,200,043,520` tokens）。所有 ARC cell 均为 all-2D DDP hook：压缩全部
`ndim == 2` 梯度，包括 Transformer block、embedding 和 lm_head；非二维梯度保持
dense。ARC 固定 EF14（CM037 的诊断对照除外）、ratio0.2、projection rank4、eta1、
seed42、step1000 后压缩和 160 MiB bucket cap。

同期的 CM038c/CM038d 与 CM040c/CM040d 严格复用 CM038/CM040 的模型、
token budget、schedule、scalar AdamW、EF14、ratio0.2、step1000 压缩起点
和 bucket cap，只将 ARC 替换为 M003 Rand-K 或 M004 Top-K all-2D DDP
hook。Rand-K 使用由 seed 和稳定参数 ID 推导的共享随机 support，只
All-Reduce selected values；Top-K 由每个 rank 独立选择 support，All-Gather
values 和 int32 indices 后重建。

Muon 矩阵继续使用 LR `0.02`、momentum `0.95`、Nesterov、
`adjust_lr=spectral_norm` 和 weight decay `0.01`。embedding/lm_head 的 AdamW
首次独立配置为 LR `0.001`、betas `(0.9, 0.999)`、epsilon `1e-8` 和 weight
decay `0`。实现同时修正旧参数组写入未被 Muon scalar kernel 消费的 `betas` 键：
现在显式写入实际读取的 `beta1`、`beta2` 和 `epsilon`。CM037/CM038/CM040 无
warmup、最后 20% 线性衰减且不裁剪；CM039/CM041 使用 1000-step linear warmup、
随后 cosine decay 到 0 和 global grad-norm clip `1.0`。

### GPT-60M：CM037–CM039

| 实验 | 模式 | final val loss | PPL | step average | tokens/s | peak memory |
|---|---|---:|---:|---:|---:|---:|
| CM037 | dense | 3.9832 | 53.69 | 98.97 ms | 1,324,361 | 6835 MiB |
| CM037 | EF21M all-2D | 4.6848 | 108.29 | 99.89 ms | 1,312,163 | 7844 MiB |
| CM038 | dense repeat | 3.9832 | 53.69 | 99.03 ms | 1,323,559 | 6835 MiB |
| CM038 | EF14 all-2D | **4.0032** | **54.77** | **96.35 ms** | **1,360,374** | 7355 MiB |
| CM038c | M003 Rand-K-EF14 all-2D | 4.1018 | 60.45 | 114.86 ms | 1,141,146 | 7843 MiB |
| CM038d | M004 Top-K-EF14 all-2D | 3.9920 | 54.16 | 182.30 ms | 718,991 | 7946 MiB |
| CM039 | dense + warmup/cosine/clip | 4.0274 | 56.11 | 99.41 ms | 1,318,499 | 6835 MiB |
| CM039 | EF14 all-2D + warmup/cosine/clip | 4.0494 | 57.36 | 96.59 ms | 1,356,993 | 7355 MiB |

CM037 中 EF21M 相对同期 dense 高 `0.7016` loss、PPL 高 `101.70%`，且单步慢
`0.93%`；仅降低并修正 scalar AdamW 超参数没有解决 EF21M 的质量问题。CM038 只将
ARC 的误差反馈换成 EF14，并重复相同 dense：loss gap 缩至 `0.0200`、PPL gap
`2.02%`，ARC 单步快 `2.71%`、吞吐高 `2.78%`，峰值显存多 `520 MiB`。CM039
加入 warmup/cosine/clip 后，dense/ARC 相对 CM038 分别高 `0.0442/0.0462` loss，
配方未改善最终质量；其 ARC 相对同期 dense 高 `0.0220` loss，单步仍快 `2.84%`。

在 CM038 同配方对照中，Rand-K 相对 dense 的 loss 高 `0.1186`、PPL 高
`12.59%`，step 慢 `15.99%`；Top-K 的 loss 反而低 `0.0088`、PPL 只高
`0.88%`，但 step 慢 `84.09%`。相对 ARC-EF14，Rand-K 的 PPL/step
分别高 `10.36%/19.21%`；Top-K 的 PPL 低 `1.11%`，但 step 慢
`89.21%`。因此 60M 的 Top-K 是“质量接近 dense、性能明显负向”，
Rand-K 则质量和性能均负向。

### GPT-130M：CM040–CM041

| 实验 | 模式 | final val loss | PPL | step average | tokens/s | peak memory |
|---|---|---:|---:|---:|---:|---:|
| CM040 | dense | 3.5804 | 35.89 | **209.89 ms** | **624,479** | 12760 MiB |
| CM040 | EF14 all-2D | **3.5949** | **36.41** | 211.05 ms | 621,047 | 14075 MiB |
| CM040d r3 | M004 Top-K-EF14 all-2D | 3.5881 | 36.17 | 424.46 ms | 308,797 | 15095 MiB |
| CM040c r3 | M003 Rand-K-EF14 all-2D | 3.6723 | 39.34 | 250.07 ms | 524,141 | 15095 MiB |
| CM041 | dense + warmup/cosine/clip | 3.5851 | 36.06 | 213.20 ms | 614,784 | 12760 MiB |
| CM041 | EF14 all-2D + warmup/cosine/clip | 3.6002 | 36.61 | **212.78 ms** | **615,998** | 14075 MiB |

CM040 的 EF14 相对 dense 只高 `0.0145` loss、PPL 高 `1.46%`，但单步慢
`0.55%`；CM041 对应 gap 为 `0.0151`/`1.52%`，ARC 单步快 `0.20%`。后者已经小到
单次运行噪声范围，不应解释为稳定加速。warmup/cosine/clip 相对 CM040 使 dense/ARC
分别高 `0.0047/0.0053` loss，也没有显示最终质量收益。EF14 在两个模型规模、两种
schedule 下均将 all-2D 压缩的质量差距控制在约 `0.015–0.022 loss`；这是当前最强的
质量证据，但所有设置仍只有 seed42。

在 CM040 同配方对照中，Top-K 相对 dense 的 loss/PPL 只高
`0.0077/0.77%`，单次结果也比 ARC-EF14 低 `0.0068 loss`/`0.68% PPL`，
但 step 相对 dense/ARC 分别慢 `102.23%/101.12%`，吞吐减少约一半。
Rand-K 相对 dense 的 loss/PPL 高 `0.0919/9.63%`，step 慢 `19.14%`；
相对 ARC-EF14 的 PPL/step 分别高 `8.05%/18.49%`。130M 因此复现
60M 的定性结论：Top-K 接近 dense 质量但当前实现过慢，Rand-K 在质量
和性能上都不及 dense/ARC。

### Sparse-K 速度解释限制

M003/M004 当前的目标是建立功能正确、同配方的 Rand-K/Top-K 质量
对照，**尚未对两条路径做专门性能优化**。现有实现使用逐 tensor
选择与重建及通用 PyTorch collective，没有专用 fused selection/
packing/scatter kernel、跨 tensor batching/packing，也没有针对 Rand-K 或
Top-K 通信路径单独调优 overlap。因此上述 step time 是当前参考实现
的端到端结果，不是这两种算法在优化实现下的性能上限。但 Top-K
在两个规模上均慢 `84%–102%`，也说明当前每 rank 独立 support 与
values/indices All-Gather 路径需要显著优化，才值得重新评估 wall-clock
竞争力。在现有证据下，ARC-EF14 在质量、速度和显存之间的综合折中
明显优于当前 Rand-K/Top-K 基线。

### 为什么正式长程 wall-clock 没有复现 CM032 的 6%–9%

本轮并非完全“没有优势”：60M 的两个 EF14 cell 相对同期 dense 快约
`2.7%–2.8%`，只是 130M 为 `-0.55%/+0.20%`，整体只能总结为小幅或基本持平。
CM032 的 60M/130M all-2D hook 曾相对同期 dense 快 `9.17%/8.25%`，但不能把该
比例直接外推到本轮，原因如下：

1. CM032 是 shared-GPU、每模式单次、20 warmup + 200 measured 的探索性短测；其
   dense 绝对时间为 `143.37/298.78 ms`，明显慢于本轮的约 `99/210–213 ms`，说明
   外部负载和短窗口基线占了相当比例。CM032 自己已预登记为非稳定性能主结果。
2. CM032 从 step0 压缩；本轮前 1000 steps 使用 dense，同一平均数中约有 60M
   `11.9%`、130M `6.0%` 的更新不享受压缩收益，因此会摊薄压缩阶段的潜在降时。
3. CM032 使用 EF21M，本轮使用 EF14；两者的 tracker/residual 更新和显存流量不同，
   所以不是只改变训练长度的同实现复测。
4. 单机 4×4090、device batch128/GA1 下，forward/backward/Muon 正交化占据大量关键
   路径，dense DDP 通信也可与 backward 重叠。all-2D hook 减少通信字节和同步尾部，
   但 projection、Top-K、residual 更新、bucket callback 及额外内存流量仍需付费；
   当 dense 基线从受扰动的 `299 ms` 回到约 `210 ms` 时，这些固定成本足以抵消大部分
   通信收益。

因此 CM032 仍是“all-2D hook 在特定短测负载下存在加速机会”的有效探索证据，而
CM037–CM041 更适合描述当前质量配方和长程平均 step time：EF14 已解决主要质量问题，
但当前单机大 physical batch 下只有 60M 小幅加速，130M 尚无稳定 wall-clock 优势。
若继续验证性能，应在独占 GPU 上做多次、随机化顺序的纯 steady-state paired timing，
并单独报告 step1000 后的压缩区间；更低带宽或跨节点环境更可能放大通信缩减收益。

原始产物：

- `artifacts/compressed_muon/CM037-CM039-m001-staged-scalar-adamw-ef14-muon-gpt60m-ws4-s42/`
- `artifacts/compressed_muon/CM040-CM041-m001-ef14-muon-gpt130m-ws4-s42/`
- `artifacts/compressed_muon/CM038cd-CM040cd-m003-m004-sparse-k-ef14-muon-ws4-s42/`
- `artifacts/compressed_muon/CM040cd-r2-m003-m004-sparse-k-ef14-muon-ws4-s42/`（按用户指定切换 GPU 后停止，不纳入结果）
- `artifacts/compressed_muon/CM040cd-r3-m003-m004-sparse-k-ef14-muon-ws4-s42/`

## CM044：M002 GreedyLore-Muon 最小两卡 evidence gate（2026-09-11）

### 口径与完整性

在独占 GPU 2/3（RTX 4090）上比较 dense Muon、M002 local-SVD 和 M002 broadcast。三者固定 dim64/2 layers/4 heads、FP32 参数/BF16 autocast、seq32、global/device batch4/2、GA1、bucket1 MiB、seed42 和同一 FineWeb10B loader；M002 使用 rank2、update interval2、warmup2。每个模式以 rotated order 完成 3 个 profiler-off timing block，每 block 的 measured window 恰好覆盖 2 个 update，即一个完整 period；另为每 block/mode 分离采集 refresh 与 compressed trace。

27/27 cell exit `0`，18 个 profiler cell 各有 2 份 rank trace（共 36），9 个 timing cell 都有 final timing，日志无 OOM/timeout/traceback marker。summary 的 hook signature rank-exact，所有 trace unattributed NCCL fraction 为 0。第一次 launcher attempt 因 worktree 缺少 dataset path 在 cell 前 fail closed，建立指向已验证共享数据的 symlink 后从空 artifact root 完整重跑；失败 attempt 另存，不混入统计。

### Logical gradient payload

trace 的实际 message size 显示 dense gradient 为 `26,148,864 B/update`。M002 local-SVD 的 refresh 为 `26,148,864 B`，compressed 为 `25,771,008 B`，完整 interval 平均 `25,959,936 B/update`，仅减少 `0.7225%`。broadcast refresh 另有 `196,608 B` full-basis payload，因此 period 平均 `26,058,240 B/update`，相对 dense 只减少 `0.3466%`。Muon result communication 保持独立、未被 M002 压缩。

压缩比例很小不是算法绕过：该 tiny GPT 的 `26,148,864 B` bucket 中只有 `393,216 B` 属于 shared Muon matrix，`25,755,648 B` 是 M002 有意保持 dense 的 embedding/lm-head 等 auxiliary 参数。它说明方法范围与模型几何强烈决定总 payload，不能只用 matrix factor rank 推导整体通信降幅。

### Refresh/compressed trace 归因

以下为 3 个 repeat 的 rank-max 平均，单位 ms；overlap 只与 genuine backward GPU kernels 相交，本 tiny workload 的所有模式均为 `0`。

| 模式/phase | NCCL union | 梯度 collectives | Muon result | SVD/score/factor/error/reconstruct GPU | collective tail | compressor critical-path tail |
|---|---:|---|---:|---|---:|---:|
| dense refresh | 4.287 | DDP gradient 2.925 | 1.362 | — | 1.382 | 0.000（parser 输出） |
| dense compressed | 5.109 | DDP gradient 3.211 | 1.897 | — | 1.529 | 0.000（parser 输出） |
| local-SVD refresh | 6.929 | corrected dense 2.943 | 4.095 | SVD 10.847 | 1.616 | 212.564 |
| local-SVD compressed | 8.133 | dense 1.580；score AR 1.643；factor AR 1.452 | 3.582 | score 0.183；factor 0.045；error 0.069；reconstruct 0.054（Top-r 0.110） | 0.364 | 1.808 |
| broadcast refresh | 223.444 | corrected dense 2.870；basis broadcast 213.579 | 6.996 | SVD 10.484 | 211.489 | 211.499 |
| broadcast compressed | 145.529 | dense 1.660；score AR 55.680；factor AR 80.475 | 7.717 | score 0.178；factor 0.044；error 0.067；reconstruct 0.051（Top-r 0.106） | 0.000 | 1.343 |

local-SVD refresh 的完整 critical-path tail 远大于 SVD GPU kernel sum，表明 per-parameter launch/host scheduling/Future sequencing 在极小 workload 上主导；这与尚未进行 same-shape batching 的已知限制一致，但 trace 不能把全部差额唯一归因于 batching。broadcast 的 full-basis collectives在 refresh 明确暴露。broadcast compressed 虽无 basis broadcast，collective 时间仍显著高于 local-SVD；本实验没有足够规模/重复去解释该差异为算法固有值。

### Complete-period wall-clock 与显存

| 模式 | repeats step ms | mean / median | sample SD / CV | throughput | peak allocated |
|---|---|---:|---:|---:|---:|
| dense | 8.54 / 8.86 / 9.73 | 9.043 / 8.860 ms | 0.616 ms / 6.81% | 14,197 tok/s | 193 MiB |
| local-SVD | 155.95 / 154.00 / 155.36 | 155.103 / 155.360 ms | 1.000 ms / 0.64% | 825 tok/s | 225 MiB |
| broadcast | 207.08 / 211.15 / 207.90 | 208.710 / 207.900 ms | 2.153 ms / 1.03% | 613 tok/s | 225 MiB |

按相同 repeat 配对，local-SVD 相对 dense 平均增加 `146.06 ms`/`1620.3%`，paired bootstrap mean-difference 95% interval 为 `[145.14, 147.41] ms`；broadcast 相对 dense增加 `199.67 ms`/`2214.9%`，interval `[198.17, 202.29] ms`。local-SVD 相对 broadcast 少 `53.61 ms`/`25.68%`，interval `[-57.15,-51.13] ms`。两种 M002 的 peak allocated 都比 dense 多 `32 MiB`（`16.58%`）。区间不跨 0 且差异远大于 2%，故按预登记规则不扩到 10 blocks；dense CV 6.81% 仍提示 tiny denominator 易受固定开销/抖动影响。

### 结论边界

CM044 分类为 **negative**：在该 tiny 单机负载，M002 只有 `0.72%`（broadcast `0.35%`）logical gradient payload reduction，却显著增加 measured collective/exposed work、complete-period wall-clock 和 peak memory。该结果不证明较大 shared-Muon-matrix 比例或不同网络下也为负，也不证明 GreedyLore-Muon 等价于 dense Muon；它只足以拒绝在当前证据上宣称加速，并阻止条件性的 10,000-update paper-oriented run。

原始产物：`artifacts/compressed_muon/CM044-m002-greedylore-muon-tiny-ddp-ws2-s42/`，包括 plan、每 cell command/controller environment/stdout/stderr/exit、36 traces、`summary.json`、`timing-summary.json` 与 timing parser。注意 `environment.txt` 是 controller snapshot；精确 per-command CUDA/PYTHONPATH/NCCL override 以各 cell `command.txt` 为准。

## CM045–CM051：M002 interval-200 规模化 timing（2026-09-11）

在 update interval 调整到原 GreedyLore 的 `200` 后，使用 profiler-off、每个 repeat
恰好覆盖一个完整 200-update period 的配对 timing 比较 dense 与 local-SVD。每个模型
均完成 3 个 rotated-order repeat；下表中的变化率以相同 repeat 配对计算，负数表示
local-SVD 更快。

| 模型/实验 | dense mean | local-SVD mean | 配对变化 | 95% mean-difference interval | dense / local peak memory | 判断 |
|---|---:|---:|---:|---:|---:|---|
| tiny / CM045 | 7.980 ms | 16.643 ms | +108.58% | [8.39, 8.92] ms | 193 / 251 MiB | negative |
| 60M / CM047 | 113.167 ms | 113.360 ms | +0.17% | [-2.32, 1.79] ms | 6865 / 7347 MiB | null |
| 130M / CM049 | 239.217 ms | 234.770 ms | -1.86% | [-5.74, -3.14] ms | 13048 / 13889 MiB | preliminary-positive |
| 350M / CM051 | 232.880 ms | 195.567 ms | **-15.97%** | [-45.33, -31.43] ms | 7062 / 9731 MiB | positive |

CM051 的三个配对 repeat 均为 local-SVD 更快（`-35.18/-45.33/-31.43 ms`），平均
吞吐从 `35,200` 增至 `41,893 tokens/s`；代价是 peak allocated 增加 `2669 MiB`
（`37.8%`）。该实验采用 global/device batch `32/8`，而 60M/130M 采用
`512/128`，因此不能跨模型比较绝对 step time 或 throughput。CM049 的优势小于预登记
的 `2%` 实用阈值，当前只视作初步正向信号；CM047 未显示差异。

这些实验只有 3 个性能 repeat，并且是短程 timing，不提供收敛、最终质量或
time-to-quality 证据。CM051 使用 timing-only runner，6/6 cell exit `0`，不生成新
trace；保留的 trace 来自单独的 CM050 preflight，不能用来替代 CM051 的性能统计。

原始产物：

- `artifacts/compressed_muon/CM045-m002-greedylore-muon-tiny-interval200-ddp-ws2-s42/`
- `artifacts/compressed_muon/CM047-m002-gpt60m-interval200-ddp-ws4-s42/`
- `artifacts/compressed_muon/CM049-m002-gpt130m-interval200-ddp-ws4-s42/`
- `artifacts/compressed_muon/CM051-m002-gpt350m-interval200-ddp-ws4-s42/`

## CM052–CM053：M002 paper-aligned 完整训练（2026-09-12）

在同一台 4×RTX 4090 节点上，以 seed1234 串行完成 GPT-60M/130M 的 dense Muon 与 M002 GreedyLore-Muon local-SVD 配对训练。公共设置为 FineWeb10B、BF16 compile、seq256、global/device batch512/128、cosine decay to 10%、clip1；60M 使用 10,000 updates/1.31072B tokens 和 1,000-step warmup，130M 使用 20,000 updates/2.62144B tokens 和 2,000-step warmup。M002 使用 rank32、interval200、step1000 后压缩、error feedback，并只压缩 Muon matrix group。

| 模型 | dense / M002 val loss | dense / M002 ppl | ppl 变化 | dense / M002 step | step 变化 | dense / M002 peak | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| 60M / CM052 | 4.0003 / 4.0837 | 54.61 / 59.36 | **+8.70%** | 112.61 / 114.25 ms | +1.46% | 6865 / 7347 MiB | quality-negative；performance null/negative |
| 130M / CM053 | 3.5749 / 3.6582 | 35.69 / 38.79 | **+8.69%** | 241.22 / 238.01 ms | -1.33% | 13048 / 13889 MiB | quality-negative；preliminary performance-positive |

两组的 best validation loss 都出现在最终 step；M002 在既定 token budget 内均未达到对应 dense 的最终质量，因此没有可报告的 dense-final-quality time-to-quality。60M 的 perplexity ratio 为 `1.08698`，通过预登记的 `1.10` 继续门禁，但该门禁只控制是否启动 130M，不代表质量等价。130M 的 `1.33%` step 加速与 CM049 的 `1.86%` 短程信号方向一致，但单次完整训练不足以建立稳定性能结论，且远不足以抵消约 `8.69%` 的 perplexity 恶化。

结论分类为 **quality-negative**：当前 rank32/interval200 的 M002 配方在 60M 和 130M 上都稳定跑完，但相对 dense Muon 出现几乎相同的约 `8.7%` perplexity 退化，并增加约 `6.4%–7.0%` peak memory。现有证据不支持宣称保持 dense-Muon 质量，也不建议立即扩展多 seed；下一步应优先做低成本 rank/压缩起始时刻或 all-2D 独立消融，再决定是否投入新的完整训练。

该实验是 paper-aligned M002-on-Muon controlled study，不是 GreedyLore 论文的 C4/AdamW 严格复现。原始命令、日志、checkpoint、W&B 对应关系和自动质量门禁位于：

- `artifacts/compressed_muon/CM052-CM053-m002-paper-aligned-quality-controller/`
- `artifacts/compressed_muon/CM052a-dense-muon-gpt60m-paper-aligned-ddp-ws4-s1234/`
- `artifacts/compressed_muon/CM052b-m002-greedylore-muon-gpt60m-paper-aligned-ddp-ws4-s1234/`
- `artifacts/compressed_muon/CM053a-dense-muon-gpt130m-paper-aligned-ddp-ws4-s1234/`
- `artifacts/compressed_muon/CM053b-m002-greedylore-muon-gpt130m-paper-aligned-ddp-ws4-s1234/`

## CM052c–CM053c：M002 high-rank 质量消融（2026-09-13）

针对 CM052/CM053 中 rank32 的约 `8.7%` perplexity 退化，严格复用同一
paper-aligned 配方和 seed1234，只将 60M 的 GreedyLore rank 从 32 提高到
128、130M 的 rank 提高到 256。本次不重跑 dense，与 CM052a/CM053a
的历史 dense 基线及 CM052b/CM053b 的 rank32 结果对照。两个强制
压缩 probe 和两组正式训练均 exit `0`。

| 模型/实验 | dense / high-rank val loss | dense / high-rank ppl | high-rank 相对 dense ppl | high-rank 相对 rank32 ppl | dense / high-rank step | dense / high-rank peak | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| 60M / CM052c rank128 | 4.0003 / 4.0140 | 54.61 / 55.37 | **+1.38%** | **-6.73%** | 112.61 / 117.52 ms (`+4.36%`) | 6865 / 7363 MiB | quality substantially recovered；performance-negative |
| 130M / CM053c rank256 | 3.5749 / 3.5821 | 35.69 / 35.95 | **+0.72%** | **-7.33%** | 241.22 / 249.68 ms (`+3.51%`) | 13048 / 13889 MiB | quality substantially recovered；performance-negative |

提高 rank 将 60M/130M 相对 dense 的 perplexity 差距从
`+8.70%/+8.69%` 缩小到 `+1.38%/+0.72%`，说明 rank32 是先前质量损失
的重要因素。代价是相对 dense 的平均 step 变慢 `4.36%/3.51%`，peak
memory 增加 `7.25%/6.45%`。相对各自 rank32 运行，high-rank step 也分别
变慢 `2.86%/4.90%`；因此这两个配方不再保留 CM053 中的小幅性能正向
信号。

结论分类为 **quality substantially recovered, not equivalence established**：高 rank
在单一 seed 上大幅缓解了质量退化，但两组最终质量仍低于 dense，
且没有多 seed 置信区间，不能据此宣称质量等价。这一结果也表明，
后续若追求接近 dense 的质量，需要将 rank 提升带来的额外计算/通信开销
与质量收益一并评估，而不能沿用 rank32 的性能结论。

原始产物：

- `artifacts/compressed_muon/CM052c-CM053c-m002-rank-scaling-quality-controller/`
- `artifacts/compressed_muon/CM052c-m002-greedylore-muon-gpt60m-paper-aligned-r128-ddp-ws4-s1234/`
- `artifacts/compressed_muon/CM053c-m002-greedylore-muon-gpt130m-paper-aligned-r256-ddp-ws4-s1234/`

## CM054：M002 ordinary-step batching 60M timing（2026-09-12）

在 M002 普通 compressed step 加入 canonical same-shape batching，并让 batched factor 直接写入一次分配的 packed collective buffer 后，严格复用 CM047 的 60M timing 几何和 3 次 rotated-order pairing；本实验只运行 profiler-off timing，不生成 trace。

| 模式 | repeats step ms | mean / median | sample SD / CV | throughput | peak allocated |
|---|---|---:|---:|---:|---:|
| dense | 109.86 / 112.36 / 110.70 | 110.973 / 110.700 ms | 1.272 ms / 1.15% | 1,181,216 tok/s | 6865 MiB |
| batched local-SVD | 110.52 / 110.54 / 113.08 | 111.380 / 110.540 ms | 1.472 ms / 1.32% | 1,176,936 tok/s | 7345 MiB |

按相同 repeat 配对，batched local-SVD 相对 dense 为 `+0.66/-1.82/+2.38 ms`，平均 `+0.407 ms` / `+0.377%`，paired bootstrap mean-difference 95% interval 为 `[-1.82,2.38] ms`，分类为 **null**。相对历史 CM047，dense/local-SVD 的绝对 step 分别下降 `1.94%/1.75%`，两者同步变化；local-SVD 相对 dense 的配对结果则从 CM047 的 `+0.17%` 变为 CM054 的 `+0.38%`，均接近零。因此没有证据表明 batching/direct-write 在 60M 上产生了可测的完整周期加速，也不能把跨实验的绝对下降归因于实现改动。peak allocated 相对 CM047 仅少 `2 MiB`，视为无实用差异。

6/6 cells exit `0`，所有末尾 timing marker 均为有限正值，日志无 OOM/timeout/traceback；step0 的 `nan` 只出现在 dense 计时尚未开始的初始化 marker。实验使用动态空闲门禁选出的 GPU `2,4,6,7`，timing-only artifact 不含 profiler trace。该短程实验不提供新的质量或 time-to-quality 结论。

原始产物：`artifacts/compressed_muon/CM054-m002-gpt60m-batched-interval200-ddp-ws4-s42/`。

## CM058–CM059：仅保留 factor direct-write 的 60M/130M timing（2026-09-12）

撤销 ordinary-step same-shape batching、恢复逐矩阵 score/Top-r/factor/error/reconstruction 后，仅保留 factor collective buffer 一次分配和 `mm(..., out=view)` direct-write。两组严格复用各自历史实验的 4-GPU、seq256、global/device batch512/128、GA1、rank32、interval200、bucket160 MiB、seed42 口径，均完成 3 次 rotated pairing；每个 cell 为 20 warmup + 200 measured updates，timing-only、不生成 trace。

| 模型/实验 | dense repeats / mean | local-SVD repeats / mean | paired 变化 | 95% mean-difference interval | dense / local peak | 判断 |
|---|---:|---:|---:|---:|---:|---|
| 60M / CM058 | 109.17/113.76/113.59 / 112.173 ms | 113.81/114.62/114.26 / 114.230 ms | +1.87% | [0.67, 4.64] ms | 6865 / 7347 MiB | slight-negative |
| 130M / CM059 | 239.78/240.07/238.73 / 239.527 ms | 234.79/236.64/235.53 / 235.653 ms | -1.62% | [-4.99, -3.20] ms | 13048 / 13889 MiB | preliminary-positive；低于 2% |

相对最初逐矩阵实现 CM047/CM049，direct-write 的 local-SVD 绝对 step 分别慢 `0.77%/0.38%`；结合 350M CM057 的 `+0.98%`，三个规模都没有显示 direct-write 的 wall-clock 收益。跨实验差值不能单独证明 direct-write 导致回退，但足以拒绝保留该优化的性能主张。60M/130M 的 M002 相对 dense 结论仍分别为小幅负向与低于 2% 的初步正向；这些短程 timing 不提供新的训练质量结论。

CM058/CM059 共 12/12 cells exit `0`，所有末尾 timing marker 均为有限正值，无 OOM/timeout/traceback，trace 数为 0；dense step0 的 `nan` 是计时开始前的预期初始化 marker。原始产物：

- `artifacts/compressed_muon/CM058-m002-gpt60m-factor-direct-interval200-ddp-ws4-s42/`
- `artifacts/compressed_muon/CM059-m002-gpt130m-factor-direct-interval200-ddp-ws4-s42/`

## CM060–CM062：M002 1B / global batch 512 四卡显存预检（2026-09-12）

恢复 batching 前的原始 GreedyLore 实现后，固定 GPT-1B（dim1536、30 layers、24 heads）、4×RTX 4090、seq256、global batch512、device batch1 / GA128、rank32、bucket160 MiB、seed42。device batch 已是可用下限；预检临时使用 interval2、1 warmup + 2 measured update，仅为同时触发 refresh/compressed 路径并检查容量，**不是 interval200 性能结果**。

| 实验 | 执行模式 | dense | M002 local-SVD | 判定 |
|---|---|---|---|---|
| CM060 | BF16 compile | 通过；peak 20219 MiB | 首次 compiled forward OOM；申请 148 MiB 时仅余 16.56 MiB | 容量不足 |
| CM061 | compile + expandable segments | 通过；peak 20217 MiB | OOM；申请 144 MiB 时仅余 110.56 MiB；未分配 reserve 74.21 MiB | allocator 设置不足以解决 |
| CM062 | no-compile + expandable segments | 通过；peak 21108 MiB | 首次 backward OOM；申请 296 MiB 时仅余 174.62 MiB | no-compile 仍不可行 |

三次均为 dense cell exit `0`、M002 cell exit `1`；因此没有启动正式 20 warmup + 200 measured timing，也不报告 1B GreedyLore wall-clock 加速比。在 global batch512、四卡、seq256 约束下，device batch1 已将梯度累积提高到 128，继续增加 GA 无法降低单次 micro-batch 显存。后续若坚持 24 GiB 四卡，需要改变显存策略（例如 activation checkpointing）并重新建立 dense/M002 配对基线；否则需增加 GPU 数或单卡显存。原始产物：

- `artifacts/compressed_muon/CM060-m002-gpt1b-global512-preflight-ddp-ws4-s42/`
- `artifacts/compressed_muon/CM061-m002-gpt1b-global512-expandable-preflight-ddp-ws4-s42/`
- `artifacts/compressed_muon/CM062-m002-gpt1b-global512-nocompile-preflight-ddp-ws4-s42/`

## CM063：130M refresh/ordinary 低开销分相 timing（2026-09-12）

为避免 Kineto 和逐步 `cudaSynchronize` 改变执行路径，CM063 在恢复后的原始 M002 上使用 differential timing：固定 CM049 的 GPT-130M、4卡、seq256、global/device batch512/128、rank32、bucket160 MiB 和 seed42，分别运行 interval100×2 periods 与 interval200×1 period。两侧都恰好覆盖 200 个 profiler-off measured update，但前者包含两个 refresh、后者包含一个；完成3组交替顺序配对。

若 interval100/200 的平均 step 分别为 `A/B`，则 `ordinary = 2B - A`、`refresh = 199A - 198B`。结果如下：

| repeat | interval100 A | interval200 B | ordinary 估计 | refresh 估计 | refresh−ordinary / 200 |
|---|---:|---:|---:|---:|---:|
| 1 | 239.06 ms | 232.15 ms | 225.24 ms | 1607.24 ms | 6.91 ms/update |
| 2 | 239.51 ms | 232.99 ms | 226.47 ms | 1530.47 ms | 6.52 ms/update |
| 3 | 244.66 ms | 231.25 ms | 217.84 ms | 2899.84 ms | 13.41 ms/update |
| mean | 241.077 ms | 232.130 ms | 223.183 ms | 2012.517 ms | 8.947 ms/update |

6/6 cells exit `0`、0 trace，无 OOM/timeout/traceback。三组都显示 refresh 明显慢于 ordinary，确认完整 full-basis SVD refresh 是130M interval200 的主要摊销项；但差分会把两个独立 process 的环境波动放大，第三组 refresh 估计明显偏高，因此 `8.95 ms/update` 只作量级诊断，不作精确分解或性能 claim。下一步对 refresh 优化的优先级高于继续修改 factor packing。

同期修复 profiler local GPU 归因：GPU kernel 不再仅因执行时间落入 hook CPU annotation 就被归为 local work，而必须关联到同 pid/tid 且嵌套在该 annotation 中的 CPU launch/op。新反例覆盖并发 backward kernel 落入 score 时间窗的情形；CM049 完整历史 trace 的修正版汇总仍作为后台派生产物，不用未完成数据更新本节数值。

原始产物：`artifacts/compressed_muon/CM063-*-m002-gpt130m-phase-timing-*/`。

## CM063b–CM063c：bucket 80 MiB 的 60M / 350M timing（2026-09-13）

固定原始 M002 local-SVD、4×RTX 4090、seq256、rank32、interval200 和 seed42，将 DDP bucket 从 CM049 系列的 160 MiB 降至 80 MiB；60M 使用 global/device batch 512/128，350M 使用 32/8。每个 cell 运行 20 warmup + 200 profiler-off measured update，dense 与 M002 各完成 3 次交替顺序配对。12/12 cells exit `0`，无 OOM、timeout 或 traceback。

| 模型 / repeat | dense step | M002 step | M002 相对 dense | dense / M002 val loss |
|---|---:|---:|---:|---:|
| 60M / r1 | 99.54 ms | 102.17 ms | +2.64% | 5.4918 / 5.7988 |
| 60M / r2 | 101.22 ms | 102.64 ms | +1.40% | 5.4914 / 5.7872 |
| 60M / r3 | 100.07 ms | 103.13 ms | +3.06% | 5.4927 / 5.7934 |
| **60M / mean** | **100.277 ms** | **102.647 ms** | **+2.37%** | **5.4920 / 5.7931** |
| 350M / r1 | 222.44 ms | 208.07 ms | -6.46% | 6.3974 / 6.7630 |
| 350M / r2 | 220.45 ms | 206.90 ms | -6.15% | 6.3968 / 6.7768 |
| 350M / r3 | 254.69 ms | 207.57 ms | -18.50% | 6.3961 / 6.7731 |
| **350M / mean** | **232.527 ms** | **207.513 ms** | **-10.37%** | **6.3968 / 6.7710** |

60M 的三组配对均为负向：M002 平均慢 `2.37 ms/update`（`+2.37%`），吞吐从 1.307M 降至 1.277M tokens/s，peak allocated 从 6865 增至 7182 MiB。因此 bucket80 在 60M 上没有产生 wall-clock 加速。

350M 的 M002 三次结果稳定在 `206.90–208.07 ms`，但 dense r3 从前两次的 `220.45–222.44 ms` 抬升到 `254.69 ms`，使 dense CV 达 `8.27%`，并将均值加速放大到 `10.37%`。前两组配对加速为 `6.46%/6.15%`，中位数比较为 `6.69%`；因此只将其记为 **约 6%–7% 的稳定加速信号**，不将 `10.37%` 视为稳健 claim。代价是 peak allocated 从 7062 增至 9707 MiB（+2645 MiB）。

本次是仅 220 update 的短程 timing，最终 val loss 只用于运行健康检查，不支持最终收敛质量结论。原始产物：

- `artifacts/compressed_muon/CM063b-m002-gpt60m-bucket80-interval200-ddp-ws4-s42/`
- `artifacts/compressed_muon/CM063c-m002-gpt350m-bucket80-interval200-ddp-ws4-s42/`

## CM064：130M bucket-cap 粗扫与 refresh 分相（2026-09-13）

固定 CM049/CM063 的 GPT-130M、4 卡、seq256、global/device batch512/128、rank32、seed42 和恢复后的原始 M002 local-SVD，只扫描 bucket cap `80/160/256/384 MiB`。每个 cap 各运行一次 interval100 和 interval200，两个窗口都覆盖 200 个 profiler-off measured update；为减轻固定顺序偏差，四组交替采用 `100→200` 与 `200→100`。8/8 cells exit `0`，无 OOM、timeout 或 traceback。

| bucket cap | interval100 A | interval200 B | ordinary 估计 `2B-A` | refresh 估计 `199A-198B` | refresh 摊销 `(A-B)` | peak allocated |
|---:|---:|---:|---:|---:|---:|---:|
| 80 MiB | 233.97 ms | **228.86 ms** | 223.75 ms | 1245.75 ms | **5.11 ms/update** | 13841 MiB |
| 160 MiB | 243.21 ms | 233.85 ms | 224.49 ms | 2096.49 ms | 9.36 ms/update | 13889 MiB |
| 256 MiB | 257.37 ms | 248.16 ms | 238.95 ms | 2080.95 ms | 9.21 ms/update | 13937 MiB |
| 384 MiB | 257.31 ms | 250.02 ms | 242.73 ms | 1700.73 ms | 7.29 ms/update | 13937 MiB |

本次单次粗扫中，80 MiB 的 interval200 完整周期比 160 MiB 快 `4.99 ms/update`（`2.13%`），ordinary 估计也最低；因此 130M 下一轮候选范围收窄到约 `64–128 MiB`，没有证据支持继续测试 `>=256 MiB`。不过每个点只有一次运行，差分 refresh 数字会放大跨进程波动，只能用于选区间，不能作为精确分解或正式加速 claim。

bucket 最优值不能跨规模直接复用：CM063b 的 60M bucket80 相对 dense 慢 `2.37%`；CM063c 的 350M bucket80 稳健信号约为快 `6%–7%`，弱于 bucket160 的 CM051 `15.97%`。因此当前结论是 **130M 值得围绕 80 MiB 复测，60M/350M 不应据此改默认值**。

原始产物：`artifacts/compressed_muon/CM064-m002-gpt130m-bucket-cap-sweep-ws4-s42/`。

## CM049 ordinary trace 修正版聚焦重解析（2026-09-13）

在修复 profiler local GPU 归因后，从 CM049 历史 trace 中各取 12 份 dense ordinary 与 GreedyLore ordinary trace 做聚焦重解析。该样本不是尚未完成的 48-trace 全量汇总，也没有生成新的 artifact；用途是获得比旧时间窗归因更可信的热点量级。

| 指标（12 traces mean） | dense ordinary | GreedyLore ordinary |
|---|---:|---:|
| profile window | 240.719 ms | 283.761 ms |
| NCCL union | 82.443 ms | 54.046 ms |
| compute overlap | 21.320 ms | 0.154 ms |
| exposed NCCL | 61.123 ms | 53.892 ms |
| exposed gradient-sync tail | 52.003 ms | 0 ms |
| Muon result collective | 8.834 ms | 8.694 ms |

GreedyLore ordinary 的修正版 local GPU 均值为：score `4.675 ms`、Top-r `0.925 ms`、factor `0.578 ms`、error `0.960 ms`、reconstruction `0.603 ms`，逐项合计约 `7.741 ms`。collective 均值为 score+dense-aux All-Reduce `41.535 ms`、factor All-Reduce `3.817 ms`；CM049 记录的逻辑 payload 为 score+dense-aux `294.891 MiB`、factor `9 MiB`，相对 dense gradient payload 减少约 `40.5%`。

证据边界如下：

- profiler-off 的 CM049 paired timing（dense/M002 `239.217/234.770 ms`）仍是 wall-clock 主证据；Kineto profile window 反而把 M002 显著拉慢，不能用其绝对窗口复算端到端收益。
- 修正版 local range 可用于热点排序；score 明显是 ordinary local 算术的最大单项。通信暴露量与 local work 都约为数毫秒级差异，但存在并发，不能把两者直接相减当作因果分解。
- 当前更精细的结论是：ordinary 并非没有优化空间，但此前尝试的 same-shape batching 和 factor direct-write 已在 CM054/CM058–CM059 显示无稳定收益；若继续优化 ordinary，应优先收集 score 的 allocator/kernel-launch 证据，再考虑融合或 workspace。refresh 在 cap80 下的平均摊销粗估降至 `5.11 ms/update`，理论上限约占该点完整周期的 `2.2%`，所以 refresh 优化仍低风险，但已不是可期待大幅加速的方向。

## CM065：130M score+dense_aux BF16 通信 timing（2026-09-13）

固定 CM064 最优粗扫点的 GPT-130M、4×RTX 4090、seq256、global/device batch512/128、rank32、interval200、bucket80 MiB 和 seed42，只改变普通 compressed step 中 packed `score+dense_aux` All-Reduce 的显式通信 dtype。FP32/BF16 各运行 20 warmup + 200 measured update，采用 `FP32→BF16 / BF16→FP32 / FP32→BF16` 的交替顺序。模型参数与 DDP gradient bucket 仍为 FP32，前后向采用 BF16 autocast；factor、refresh basis 与 dense-only bucket 路径不变。

| repeat | FP32 packed buffer | BF16 packed buffer | BF16−FP32 | FP32 / BF16 val loss |
|---|---:|---:|---:|---:|
| r1 | 226.72 ms | 221.08 ms | -5.64 ms (-2.49%) | 5.6823 / 5.6723 |
| r2 | 228.49 ms | 221.07 ms | -7.42 ms (-3.25%) | 5.6715 / 5.6727 |
| r3 | 230.06 ms | 218.76 ms | -11.30 ms (-4.91%) | 5.6749 / 5.6731 |
| **mean** | **228.423 ms** | **220.303 ms** | **-8.120 ms (-3.55%)** | **5.6762 / 5.6727** |

前两组为同一串行 controller 内的严格相邻配对，均值为 FP32/BF16 `227.605/221.075 ms`，BF16 快 `6.530 ms`（`2.87%`）。r3 的 FP32 cell 完成后，controller 因运行期间 launcher 被编辑而在子进程汇总阶段遇到 shell parse error；FP32 原始日志仍有 step220 最终 marker 且训练进程 exit `0`，BF16 r3 随后按相同配置单独补跑。因此六个训练 cell 都完成且没有 OOM/训练 traceback，但 r3 不是严格相邻配对，只作为较弱的补充证据。最初的 attempt1 还曾因 wrapper 局部变量初始化错误在启动训练前退出，不计入结果。

`score+dense_aux` 的逻辑 payload 从 FP32 的 `294.891 MiB` 减半到约 `147.446 MiB`，但完整 interval200 周期只改善 `3.55%`；严格相邻证据为 `2.87%`。峰值 allocated 两种 dtype 都是 `13841 MiB`，没有可测显存收益。短程 val loss 只作运行健康检查，不能支持收敛质量结论。本次没有 Kineto trace，因此不能把端到端差值直接解释为 collective duration 的等量下降。

结论分类为 **preliminary-positive but below practical threshold**：结果方向一致，且 FP32 mean 与 CM064 bucket80 的 `228.86 ms` 相符，但总体收益低于预先关注的 `>=5%`（约 `11.44 ms/update`）门槛。BF16 packed communication 可保留为显式选项，不应仅凭本结果改默认值，也不值得作为单独优化方向继续投入；若后续推进，应与能产生十余毫秒级收益的跨 bucket Future/collective overlap 一起评估。

原始产物：`artifacts/compressed_muon/CM065-m002-gpt130m-bf16-dense-aux-bucket80-ddp-ws4-s42-attempt2/`。首次未进入训练的失败产物保留在同名前缀的 attempt1 目录。

## CM067：bucket-native BF16 参数的 60M / 130M / 350M timing（2026-09-13）

固定 4×RTX 4090、BF16 整模型参数/gradient/DDP bucket、BF16 autocast、seq256、global/device batch512/128、bucket80 MiB、rank32、interval200、local-SVD 和 seed42。每个成功 cell 运行 20 warmup + 800 profiler-off measured updates，即四个完整 interval；60M/130M 各完成 3 组 dense/GreedyLore rotated pairing。

| 模型 / repeat | dense step | M002 step | M002−dense | dense / M002 val loss |
|---|---:|---:|---:|---:|
| 60M / r1 | 92.06 ms | 94.33 ms | +2.27 ms (+2.47%) | 4.6618 / 4.9670 |
| 60M / r2 | 92.04 ms | 94.49 ms | +2.45 ms (+2.66%) | 4.6618 / 4.9670 |
| 60M / r3 | 92.03 ms | 94.21 ms | +2.18 ms (+2.37%) | 4.6618 / 4.9670 |
| **60M / mean** | **92.043 ms** | **94.343 ms** | **+2.300 ms (+2.50%)** | **4.6618 / 4.9670** |
| 130M / r1 | 199.25 ms | 203.94 ms | +4.69 ms (+2.35%) | 4.3641 / 4.7894 |
| 130M / r2 | 199.70 ms | 204.22 ms | +4.52 ms (+2.26%) | 4.3641 / 4.7894 |
| 130M / r3 | 199.68 ms | 204.45 ms | +4.77 ms (+2.39%) | 4.3641 / 4.7894 |
| **130M / mean** | **199.543 ms** | **204.203 ms** | **+4.660 ms (+2.34%)** | **4.3641 / 4.7894** |

60M dense/M002 CV 为 `0.017%/0.149%`，paired bootstrap mean-difference interval 为 `[2.18,2.45] ms`；吞吐从 `1.424M` 降至 `1.389M tokens/s`（约 `-2.44%`），peak allocated 从 `6058` 增至 `6303 MiB`（+245 MiB）。130M CV 为 `0.127%/0.125%`，paired interval `[4.52,4.77] ms`；吞吐从 `656.9K` 降至 `641.9K tokens/s`（约 `-2.28%`），peak 从 `11208` 增至 `11633 MiB`（+425 MiB）。结果波动很低且三组方向一致，分类为 **negative**：bucket-native BF16 下当前 GreedyLore 在两个可运行规模均稳定慢约 2%–3%。

350M 严格保持 `512/128` 后两模式均 OOM：dense 在训练分配 256 MiB 时每卡只余约 62.56 MiB；GreedyLore fallback 分配 64 MiB 时只余约 40.56 MiB。两者都没有产生有效 timing，按预登记策略不调整 batch 或重试。

本次短程 val loss 只用于运行健康检查，不能支持最终收敛质量结论。原始产物：`artifacts/compressed_muon/CM067-m002-bucket-native-bf16-scale-timing-ws4-s42/`。

## CM068：130M bucket-native BF16 bucket-cap 粗扫（2026-09-14）

为检查 CM067 直接沿用 80 MiB 是否因 BF16 每元素字节减半而改变有效 bucket 几何，固定 GPT-130M、4卡、BF16 参数/gradient/bucket、seq256、global/device batch512/128、rank32、interval200 和 seed42，串行扫描 `24/32/40/48/64/80 MiB`。每点运行一次 dense 和 local-SVD GreedyLore，采用交替顺序、20 warmup + 200 profiler-off measured update。12/12 cells 与 controller 均 exit `0`。

| bucket cap | dense step | GreedyLore step | GreedyLore−dense | dense / GreedyLore peak |
|---:|---:|---:|---:|---:|
| 24 MiB | **188.84 ms** | 201.23 ms | +12.39 ms (+6.56%) | 11207 / 11633 MiB |
| 32 MiB | 194.15 ms | 203.10 ms | +8.95 ms (+4.61%) | 11207 / 11632 MiB |
| 40 MiB | 192.42 ms | **200.99 ms** | +8.57 ms (+4.45%) | 11206 / 11632 MiB |
| 48 MiB | 193.98 ms | 205.11 ms | +11.13 ms (+5.74%) | 11207 / 11632 MiB |
| 64 MiB | 194.20 ms | 203.02 ms | +8.82 ms (+4.54%) | 11206 / 11632 MiB |
| 80 MiB | 200.28 ms | 204.57 ms | **+4.29 ms (+2.14%)** | 11207 / 11632 MiB |

40 MiB 使 GreedyLore 相对 80 MiB 快 `3.58 ms`，说明按元素规模匹配 FP32 bucket80 有局部收益；但 dense 对较小 bucket 的收益更大，所有点上的 GreedyLore 都慢于同 cap dense。80 MiB 与 CM067 的 `199.543/204.203 ms` 基本复现。每点仅一次，尤其 24 MiB dense 低值不能作为稳定最优 claim；本次足以否定“CM067 负结果主要由 bucket80 过大造成”的单一解释。

原始产物：`artifacts/compressed_muon/CM068-m002-gpt130m-bf16-bucket-cap-sweep-ws4-s42/`。

## CM069：130M bucket-native BF16 bucket40/80 targeted profile（2026-09-14）

对 CM068 的 GreedyLore 绝对最优点 40 MiB 和相对差距最小点 80 MiB，各采一个 dense/GreedyLore refresh 与 ordinary targeted step；共 8 cells、32 份 rank trace，全部 exit `0` 并完成 summary。以下采用每个 cell 的 rank-max；Kineto 显著放大 GreedyLore CPU callback range，因此 profile window 不替代 CM067/CM068 的 profiler-off wall-clock。

| bucket | 模式 | NCCL union | exposed NCCL | exposed grad tail | 主要 collective |
|---:|---|---:|---:|---:|---|
| 40 MiB | dense ordinary | 48.117 ms | 20.701 ms | 13.091 ms | gradient 40.727；Muon 7.405 ms |
| 40 MiB | GL ordinary | 48.760 ms | 29.400 ms | 0 ms | dense-only 14.151；score+aux 18.060；factor 8.456；Muon 8.173 ms |
| 80 MiB | dense ordinary | 50.377 ms | 32.525 ms | 24.635 ms | gradient 42.641；Muon 7.986 ms |
| 80 MiB | GL ordinary | 35.728 ms | 35.043 ms | 0 ms | score+aux 21.609；factor 6.934；Muon 7.311 ms |

GreedyLore ordinary local GPU 在 40/80 MiB 分别为：score `3.020/2.293 ms`、Top-r `1.032/0.763 ms`、factor `0.389/0.344 ms`、error `0.813/0.764 ms`、reconstruction `0.608/0.505 ms`，合计 `5.862/4.669 ms`。refresh local-SVD GPU 为 `1508.717/1503.761 ms`，按 interval200 的简单摊销约 `7.54/7.52 ms/update`。

诊断显示较小 bucket 并未让 GreedyLore 的暴露通信低于 dense：40 MiB 多 `8.699 ms`，且因 bucket composition 出现单独的 `14.151 ms` dense-only collective；80 MiB 仍多 `2.518 ms`。parser 的 GreedyLore targeted-collective/backward overlap 指标在两点均为 `0`，而 ordinary 本地算术和 refresh 成本继续存在。这支持 CM067/CM068 的结构性解释：BF16 已使 dense gradient payload 减半，当前两阶段串行 Future 没有把压缩 collective 有效藏入 backward，剩余通信收益不足以覆盖本地压缩与 refresh。各项存在并发，不能把 exposed、local 和 refresh 摊销直接相加复算 step 差值。

原始产物：`artifacts/compressed_muon/CM069-m002-gpt130m-bf16-bucket{40,80}-targeted-profile-ws4-s42/`。

## CM070：CM052/CM053/CM052c/CM053c 的 BF16 参数完整训练（2026-09-14）

六个 cell 除整模型 `model_dtype=bfloat16` 和实验/W&B 名称外，分别复用 CM052a/b、CM053a/b、CM052c、CM053c 的完整 YAML：4卡、FineWeb10B、seq256、global/device batch512/128、bucket160 MiB、seed1234、cosine to 10%、clip1；60M 运行 10,000 updates/1,000 warmup，130M 运行 20,000/2,000 warmup。GreedyLore 从 step1000 开始、interval200、local-SVD；rank32 与 high-rank 分别为 60M rank128、130M rank256。六个 formal cell 和 best-effort controller 均 exit `0`。

| 模型 / 方法 | val loss | 相对同 dtype dense perplexity | step average | 相对同 dtype dense | peak allocated |
|---|---:|---:|---:|---:|---:|
| 60M BF16 dense | 4.1079 | baseline | 93.05 ms | baseline | 6058 MiB |
| 60M BF16 GL rank32 | 4.2119 | +10.96% | 94.82 ms | +1.77 ms (+1.90%) | 6303 MiB |
| 60M BF16 GL rank128 | 4.1230 | +1.52% | 95.53 ms | +2.48 ms (+2.67%) | 6311 MiB |
| 130M BF16 dense | 3.6530 | baseline | 210.56 ms | baseline | 11208 MiB |
| 130M BF16 GL rank32 | 3.7645 | +11.80% | 215.04 ms | +4.48 ms (+2.13%) | 11633 MiB |
| 130M BF16 GL rank256 | 3.6618 | +0.88% | 219.71 ms | +9.15 ms (+4.35%) | 11633 MiB |

提高 rank 基本消除了相对同 dtype dense 的质量差距，但扩大了性能负担；因此当前不存在同时满足质量和 wall-clock 的 BF16 GreedyLore 点。与历史 FP32 参数的 CM052a/CM053a 相比，BF16 dense 的 perplexity 分别高约 `11.36%/8.12%`；该跨 dtype 比较只用于判断无 FP32 master weights 的整模型 BF16 数值代价，不能与同 dtype 配对混为一谈。rank32 BF16 相对对应历史 FP32 rank32 的 perplexity 也分别高约 `13.68%/11.22%`。

结论分类为 **quality/performance trade-off negative**：rank32 慢约 2% 且质量差约 11%；high-rank 质量接近 BF16 dense，但慢约 3%–4%。整模型 BF16 明显降低绝对 step time 和显存，却不应替代当前 FP32 参数的 paper-aligned 主配方。W&B run IDs 依次为 `1sg0ag69/rlrmb6ug/zsv0lnte/jfbm84g5/3e63ukpn/31vbzafp`。

原始产物：`artifacts/compressed_muon/CM070{a,b,c,d,e,f}-*/`；controller：`artifacts/compressed_muon/CM069-CM070-m002-bf16-profile-and-quality-controller/`。

## CM071：论文 micro-batch `32 × GA4` 的 60M / 130M / 350M timing（2026-09-14）

为修正 CM067 将每卡 effective batch128 直接作为 physical batch 的口径差异，固定 4×RTX 4090、BF16 整模型参数/gradient/DDP bucket、BF16 autocast、seq256、micro-batch/device `32`、GA`4`、effective/device `128`、global batch `512`、bucket80 MiB、rank32、interval200、local-SVD 和 seed42。每个 cell 运行 20 warmup + 800 profiler-off measured optimizer updates，即四个完整 interval；三个规模各完成 3 组 dense/GreedyLore rotated pairing。运行使用当时空闲的 GPU `2,3,6,7`，controller 和 18/18 cells 均 exit `0`，350M 不再 OOM。

| 模型 / repeat | dense step | M002 step | M002−dense | dense / M002 val loss |
|---|---:|---:|---:|---:|
| 60M / r1 | 92.27 ms | 94.80 ms | +2.53 ms (+2.74%) | 4.6650 / 4.9749 |
| 60M / r2 | 92.24 ms | 95.79 ms | +3.55 ms (+3.85%) | 4.6650 / 4.9749 |
| 60M / r3 | 91.64 ms | 96.42 ms | +4.78 ms (+5.22%) | 4.6650 / 4.9749 |
| **60M / mean** | **92.050 ms** | **95.670 ms** | **+3.620 ms (+3.94%)** | **4.6650 / 4.9749** |
| 130M / r1 | 214.69 ms | 221.64 ms | +6.95 ms (+3.24%) | 4.3780 / 4.8242 |
| 130M / r2 | 217.03 ms | 217.34 ms | +0.31 ms (+0.14%) | 4.3780 / 4.8242 |
| 130M / r3 | 216.59 ms | 218.46 ms | +1.87 ms (+0.86%) | 4.3780 / 4.8242 |
| **130M / mean** | **216.103 ms** | **219.147 ms** | **+3.043 ms (+1.41%)** | **4.3780 / 4.8242** |
| 350M / r1 | 592.20 ms | 629.27 ms | +37.07 ms (+6.26%) | 4.1758 / 4.7278 |
| 350M / r2 | 597.34 ms | 639.36 ms | +42.02 ms (+7.03%) | 4.1758 / 4.7278 |
| 350M / r3 | 602.03 ms | 628.93 ms | +26.90 ms (+4.47%) | 4.1758 / 4.7278 |
| **350M / mean** | **597.190 ms** | **632.520 ms** | **+35.330 ms (+5.92%)** | **4.1758 / 4.7278** |

60M dense/M002 CV 为 `0.386%/0.854%`，paired bootstrap mean-difference interval 为 `[2.53,4.78] ms`；吞吐从 `1.424M` 降至 `1.370M tokens/s`（约 `-3.78%`），peak allocated 从 `2036` 增至 `2281 MiB`（+245 MiB）。130M CV 为 `0.575%/1.018%`，paired interval `[0.31,6.95] ms`；吞吐从 `606.5K` 降至 `598.1K tokens/s`（约 `-1.38%`），peak 从 `3827` 增至 `4250 MiB`（+423 MiB）。350M CV 为 `0.823%/0.937%`，paired interval `[26.90,42.02] ms`；吞吐从 `219.5K` 降至 `207.2K tokens/s`（约 `-5.59%`），peak 从 `9627` 增至 `11023 MiB`（+1396 MiB）。三个规模的每组配对方向均为负，分类为 **negative**；其中 130M 差值波动较大，但没有 GreedyLore 加速样本。

与 CM067 的 physical batch128/GA1 相比，GA4 的 60M dense 几乎不变（`92.043→92.050 ms`），130M dense 变慢约 `8.3%`（`199.543→216.103 ms`）；60M/130M peak 显存则显著降低，并使 350M global batch512 可运行。由于 CM067 使用 GPU `2–5`，CM071 使用 GPU `2,3,6,7`，两次绝对时间比较同时包含卡组/系统负载差异，不能作为纯 GA 因果估计。尽管如此，改成论文的 micro-batch/GA 口径没有产生数量级的 step-time 增长，因此不能解释本地结果与论文 Table V 接近十倍的绝对时间差异。

短程 val loss 只用于运行健康检查，不能支持最终收敛质量结论。原始产物：`artifacts/compressed_muon/CM071-m002-paper-microbatch-ga4-bf16-scale-timing-ws4-s42/`。

## CM072：M002 ordinary bucket 第一阶段流水化 timing（2026-09-14）

在 ordinary compressed hook 中拆分 collective ordering tail 与 DDP completion tail：bucket ready 后可提前在 preparation stream 计算 corrected gradient、score 和 packing；collective 仍严格按 `score(A)→factor(A)→score(B)→factor(B)` 排序，但 A 的 factor 完成后，A 的 reconstruction 可与 B 的 score All-Reduce 并行。warmup、refresh 和 dense fallback 保持原串行路径。

测速严格复用 CM067 的 GPT-130M BF16/GA1 口径：4-rank DDP、seq256、global/device batch512/128、bucket80 MiB、rank32、interval200、local-SVD、seed42；每 cell 为 20 warmup + 800 profiler-off measured updates，共四个完整 interval。使用 GPU `2,3,6,7`，完成 3 组 rotated dense/M002 pairing，6/6 cells exit `0`。

| repeat | dense step | 流水化 M002 step | M002−dense | dense / M002 val loss |
|---|---:|---:|---:|---:|
| r1 | 197.20 ms | 203.18 ms | +5.98 ms (+3.03%) | 4.3641 / 4.7894 |
| r2 | 203.37 ms | 203.38 ms | +0.01 ms (+0.00%) | 4.3641 / 4.7894 |
| r3 | 201.71 ms | 203.40 ms | +1.69 ms (+0.84%) | 4.3641 / 4.7894 |
| **mean** | **200.760 ms** | **203.320 ms** | **+2.560 ms (+1.29%)** | **4.3641 / 4.7894** |

M002 的 CV 为 `0.060%`，三次稳定在 `203.18–203.40 ms`；dense CV 为 `1.590%`，导致 paired bootstrap mean-difference interval 较宽，为 `[+0.01,+5.98] ms`。吞吐为 dense/M002 `653.0K/644.7K tokens/s`，peak allocated 为 `11207/11632 MiB`。因此本轮仍分类为 **negative**：流水化 M002 没有超过同轮 dense，平均慢 `1.29%`。

与旧 CM067 同配置但不同 GPU 组合的历史结果相比，M002 绝对均值从 `204.203` 降至 `203.320 ms`（`-0.883 ms`, `-0.43%`），M002−dense gap 从 `+4.660` 缩至 `+2.560 ms`。这可记为小幅改善信号，但 CM067 使用 GPU `2,3,4,5`，CM072 使用 `2,3,6,7`，且 CM072 dense 波动明显较高；因此跨实验差值不能作为流水化的严格因果收益。要确认暴露通信是否下降，需要在同环境保留旧实现作 A/B，或补 targeted profiler。

原始产物：`artifacts/compressed_muon/CM072-m002-gpt130m-bf16-ordinary-pipeline-timing-ws4-s42/`。

## CM073：M002 ordinary bucket 流水化 targeted profile（2026-09-14）

为检查 CM072 的调度改动是否真正形成 GPU overlap，在同一 GPT-130M BF16/GA1、bucket80 MiB、rank32、interval200 配置上，以 GPU `2,3,6,7` 串行采集 dense/M002 的 refresh 与 ordinary targeted step。4/4 cells 均成功，共生成 16 份 rank trace；Kineto profile 只用于时间线诊断，不替代 CM072 profiler-off timing。

| 模式 | profile window | NCCL union | exposed NCCL | exposed gradient tail |
|---|---:|---:|---:|---:|
| dense ordinary | 194.772 ms | 45.731 ms | 29.234 ms | 21.648 ms |
| 流水化 M002 ordinary | 268.732 ms | 49.103 ms | 47.766 ms | 0 ms |

流水化 M002 ordinary 的 score+dense_aux/factor/Muon-result collective 分别为 `31.422/9.244/8.709 ms`；本地 GPU score/Top-r/factor/error/reconstruction 分别为 `2.385/0.766/0.348/0.777/0.454 ms`。标准 parser 的 targeted-collective/backward overlap 仍为 `0`。

进一步直接检查四份 ordinary trace 的 GPU event 区间：每个 rank 的 reconstruction 与任何 BF16 AllReduce kernel 的交集均为 `0 ms`，因此本次没有实际形成预期的 `reconstruct(A) ∥ score-AllReduce(B)`。局部 score preparation 与 reconstruction 只在 rank0/rank2 分别重叠约 `0.549/0.457 ms`，rank1/rank3 为 `0`，不足以构成稳定的 rank-wide critical-path 收益。时间线上主要原因是 reconstruction 仅约亚毫秒：有的后续 bucket 尚未 ready；即使已开始准备下一 bucket，score preparation 结束并提交 AllReduce 时，前一 bucket reconstruction 已完成。

CM073 的 M002 exposed NCCL 比 CM069 旧实现的 bucket80 profile `35.043 ms` 更高，但两次使用不同 GPU 组合（CM069 为 `2,3,4,5`），且 score/factor collective 本身也从 `21.609/6.934` 波动到 `31.422/9.244 ms`；这项跨实验差异不能解释成流水化导致通信回退。可确定的结论是：**第一阶段放宽了依赖，但在当前 bucket 几何下没有把 reconstruction 与下一 bucket collective 实际叠起来**，与 CM072 只有约 `0.43%` 历史绝对改善信号相符。

原始产物：`artifacts/compressed_muon/CM073-m002-gpt130m-bf16-ordinary-pipeline-targeted-profile-ws4-s42/`。
