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
