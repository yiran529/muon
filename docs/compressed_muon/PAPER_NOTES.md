# M002 GreedyLore-Muon 证据与写作边界（2026-09-11）

- 方法位置必须写成 DDP gradient-input compression before unchanged ordinary Muon；它不是对 Muon result communication 的优化，也不是 dense Muon 的等价实现。
- GreedyLore 论文的 MSGD/Adam 收敛结果不能直接转移到动量后执行非线性正交化的 Muon。除非另有专门证明，不能写“theoretically convergent from GreedyLore”。
- `local_svd` 省掉完整 basis 通信，但 repeated singular-value subspace 的旋转不唯一仍是同构环境假设；`broadcast` 提供强一致性 baseline，并必须单列 `a*a/update_interval` 的 refresh cost。
- CM044 的 logical gradient payload reduction 只有 local-SVD `0.7225%`、broadcast `0.3466%`，因为 tiny GPT 中绝大部分 payload 属于 M002 有意不压缩的 dense embedding/lm-head auxiliary。logical reduction、collective time、tail、throughput、memory 与 quality 必须分别呈现。
- CM044 complete-period local-SVD/dense 为 `155.103/9.043 ms`，paired mean difference `+146.06 ms`，95% bootstrap interval `[145.14,147.41] ms`；peak `225/193 MiB`。这是 tiny negative evidence，不能外推到大模型/跨节点，但足以阻止当前加速 claim。
- 三 seed、12-update synthetic regression preflight 无 NaN/Inf，local-SVD validation loss delta 为 `+0.00518/+0.00414/+0.00443`，mean `+0.00458`，residual norm max `0.6303`。它只说明短 synthetic stability，不是 GPT/C4 convergence 或 time-to-quality evidence。
- 条件性的 10,000-update 四卡 recipe 未启动：短 synthetic quality 不是代表性语言模型质量证据，且 CM044 性能为负。未满足的 paper gate 包括真实 GPT-60M 多 seed short quality、独占 4-GPU representative timing、validation loss/time-to-quality 与完整 gradient/residual telemetry。
- 若未来启动首个 paper-oriented recipe：GPT-60M，global/device batch `512/128`、4 ranks、seq256、10,000 updates、1,000 warmup updates、rank32、interval200、Adam-style scalar settings；采用 Dion 已有 cosine-to-zero。它只是 paper-oriented comparison，不是 exact reproduction。
- 必须显式记录 recipe deltas：作者 snapshot launcher 设置 `grad_clipping=0`，当前 CM039 staged Muon recipe 是 `grad_clip_norm=1.0`，因此 clipping 应做 paired ablation；snapshot scheduler warmup 是 200 steps，而 paper Table VII 报告 1,000。snapshot 又缺少所引用的 C4 training source，无法本地验证 cosine endpoint。
- `/home/wyr/greedy_lore` 是 read-only/non-oracle snapshot：真实 greedy hook score helper 不完整，fake hook 通信 dense tensor。只能借鉴 dimensionally valid orientation、same-shape batching 与 cadence/packing 结构。

## M002 / GreedyLore 的 dtype 口径（2026-09-13）

- 必须分别报告 parameter storage dtype、forward/backward autocast dtype、gradient/DDP bucket dtype、显式 communication-buffer dtype 和 optimizer-state dtype；“BF16 training”本身不能推出 gradient communication 是 BF16。
- 参考 snapshot 的 `subspace_hook.py` 以 `bucket.buffer()` 的 dtype 构造并 All-Reduce dense auxiliary 和低秩工作区，所以可验证的实现策略是 **通信跟随 bucket dtype**。launcher 虽设置 `dtype=bfloat16`，但 snapshot 缺少其引用的 C4 training source，不能据此断言作者实验的参数或实际通信一定为 BF16。
- Dion 当前 M002 路径使用 FP32 参数与 FP32 DDP gradient bucket，BF16 仅用于 autocast 计算；因此 `bucket` 默认值产生 FP32 通信，并与参考 hook 的 dtype 策略一致。显式 `bfloat16` 是额外的通信压缩 ablation，不应称为“恢复作者 BF16 默认逻辑”。
- CM065 只将普通 compressed step 的 packed `score+dense_aux` All-Reduce 从 FP32 改为 BF16；Top-r 前 score 转回 FP32，factor、refresh basis、dense-only bucket 和 Muon 语义不变。其 130M/bucket80 三次 mean 从 `228.423` 降至 `220.303 ms`，改善 `8.120 ms`（`3.55%`）；严格相邻 r1/r2 改善 `6.530 ms`（`2.87%`），低于项目关注的 5% 实用门槛。短程 val loss 仅作健康检查，不能给出最终质量结论。
- 论文写作中可将 BF16 packed communication 报告为有效但幅度有限的系统 ablation；不能声称 payload 减半会等比例转化为 wall-clock，也不能在没有 profiler 的情况下把完整 step 差值直接归因给 NCCL collective。
