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
