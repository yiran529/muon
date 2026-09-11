# M002：GreedyLore-Muon

## 定位与边界

M002 在 PyTorch DDP 的梯度同步阶段压缩 Muon 矩阵参数的 rank-local gradient input。所有 rank 先重建相同的近似全局梯度，再交给未修改的 ordinary `dion.Muon` 执行动量、Nesterov、非线性正交化、Muon result communication 与参数更新：

```text
rank-local accumulated gradient
  -> GreedyLore DDP hook
  -> reconstructed approximate global gradient
  -> unchanged ordinary Muon
  -> parameter update
```

因此 M002 是新的近似 Muon 变体，不是 dense Muon 的等价通信实现。GreedyLore 论文对 MSGD/Adam 的收敛分析不会自动迁移到 Muon：Muon 在动量之后应用非线性正交化，通常有 `Ortho(Average(G)) != Average(Ortho(G))`。在完成专门理论之前，不宣称 M002 具有论文给出的收敛保证。

首版只支持固定 process group 的 DDP、`find_unused_parameters=False`、FP32 Muon 矩阵 bucket 和同构软件/硬件栈。FSDP/HSDP、DDP join、unused parameters、动态参数组、world-size-changing resume、三维 matrix batch、AMP GradScaler skip/retry 与异构 accelerator stack 不在范围内。矩阵资格由 `train.build_muon_param_groups()` 的第一个 Muon group 决定；二维 embedding/lm-head 与其他参数保持 exact dense auxiliary sync。

## 算法状态与通信

原矩阵 `m x n` 统一定向为 `a x b`，其中 `a=min(m,n)`、`b=max(m,n)`；`m>n` 时转置。每个矩阵以稳定参数名/ID 持有 FP32 error、完整 `a x a` basis 和最近 support。生命周期为 one-based compressor step：`step <= start_compress_step` 走 dense warmup；之后第一个 step 及每隔 `update_interval` 个 step refresh，其余 step 执行 greedy score、Top-r、factor、error feedback 和重建。

refresh 对 `orient(g_i)+E_i` 做平均，输出该 corrected gradient 的 dense average、刷新完整左奇异向量 basis，并按论文主算法清零 error。compressed step 先对 signed score 向量做 All-Reduce/average、再平方和稳定 Top-r；error 必须在独立 factor buffer All-Reduce 覆盖前由 local factor 计算；重建使用维度正确的 `P @ R`。

每个 compressed bucket 的 collective 顺序是：

```text
dense auxiliary values + signed score vectors All-Reduce
  -> local Top-r/factor/error
matrix factors All-Reduce
  -> reconstruction/scatter
```

Muon 自身的 result communication 未修改，并与 GreedyLore gradient collectives 分开归因。

## local-SVD 与 broadcast

- `local_svd`（默认）：每个 rank 对相同的 all-reduced corrected gradient 独立执行 SVD，并对列符号做 canonicalization。它省去完整 basis 通信，但 exactly repeated singular-value subspace 内仍可能存在旋转不唯一；这只是经记录同构环境验证的实验假设，不是普遍一致性保证。
- `broadcast`：process-group rank 0 计算/canonicalize 完整 basis，再逐矩阵广播 `a x a` basis。它是强一致性基线，但 refresh 周期额外支付完整 basis payload，不能把这部分成本隐藏在 paper-faithful local-SVD 载荷中。

单个 `a x b` 矩阵在 local-SVD 下，每个 period step 的平均 logical element payload 为：

```text
ab / update_interval
  + (1 - 1 / update_interval) * (a + rank * b)
```

broadcast 另加 `a*a/update_interval`；dense auxiliary payload 单独增加。该公式只描述 logical payload，不是 collective 时间、exposed tail 或端到端加速声明。

## 本地作者代码快照

只读参考路径为 `/home/wyr/greedy_lore`，没有 `.git`，无法用 upstream commit 标识。2026-09-10/11 检查文件的 SHA-256：

```text
comm_hooks/lore_hook.py                 3f4b9fdb0a46537c46cf6d62fd6ca168c7489a11041fc6d8c61aab226cc57b85
comm_hooks/subspace_hook.py             beb7c30037904928c1eedfdf92342938ec62b4ef6f6469f6f1ddbadd50783545
comm_hooks/fake_subspace_hook.py        094fe77718247d93da2b9f023a87848a2b0d572bae31b3d192f1589fd555d2be
comm_hooks/utils.py                     501861b91065d6e551f50a578f3dc33368a5091a1ea619158423f14aeb863d35
run_c4_llama60m_lore_fp32.slurm         ec344141e2c08c28978e923e8d498ad38ee2a6820e496b3ef5fc67764165fced
run_c4_llama60m_lore_bf16.slurm         9691a53244b94fb2106a4e51e655765eaa6935234f3d30f41118eb93964a4de1
```

该快照可复用的设计线索是左右 SVD orientation、按同 shape batching、只压缩 Transformer 矩阵、warmup/refresh cadence、factor packing 与 reconstruction structure。它是只读参考而不是可执行 oracle：真实 greedy `subspace_hook.py` 的非随机 score helper 未实现，`fake_subspace_hook.py` 则用完整 dense tensor 通信模拟。

M002 相对快照的有意差异如下：

1. 持久状态按参数 identity/稳定名与 ID 组织，不按可能 rebuild/reorder 的 DDP bucket index；step 由训练循环 lifecycle 驱动，不靠 `bucket.is_last()`。
2. hook 异步链不调用 `Work.wait()`、`torch.cuda.synchronize()`、tensor `.item()` 或 device-to-host polling，而使用一条显式、非嵌套 Future tail 和 stream event visibility。
3. score 随机向量采用独立 standard normal；先全局平均 signed lambda，再逐元素平方，而非 `rand` 后取绝对值。
4. refresh 输入为 local gradient 加旧 error，先同步 corrected gradient，随后按主 Algorithm 3 清零新 error；不在 reduce 前提前丢弃 residual。
5. refresh 不计算没有消费者的 `P.T @ H`；重建采用维度有效的 `P @ R`，并修正 Appendix/Algorithm 4 中不匹配输出 shape 的 transpose 分支。
6. 不使用只计算 factor、忽略 score 与摊销 refresh payload 的 legacy `min_compression_rate` gate；按完整 logical payload 报告。
7. 第一个版本只覆盖 shared Muon group 的二维矩阵，不继承 ARC 后续 all-2D 范围；embedding/lm-head 仍是 dense auxiliary。
8. compressor/state/低秩计算与矩阵通信固定 FP32，并提供 local-SVD 诊断与 full-basis broadcast fallback。

## 实现与验证入口

- 设计与计划：`docs/superpowers/specs/2026-09-09-greedylore-muon-design.md`、`docs/superpowers/plans/2026-09-09-greedylore-muon-ddp-hook.md`
- 算法/状态/hook：`dion/greedy_lore.py`、`dion/greedy_lore_layout.py`、`dion/greedy_lore_ddp_hook.py`
- unchanged Muon integration：`train_greedylore.py`、`train.py`、`dion/muon.py`
- 默认配置：`configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml`
- profiler/period launcher：`benchmark/compressed_muon/run_greedy_lore_profiler.sh`
- 测试：`tests/test_greedy_lore*.py`、`tests/test_train_greedylore.py`
- append-only 记录：`docs/worklog/M002-greedy-lore-muon.md`

## 当前状态与已知限制

状态为 `testing`。Task 10 已在本机同构 2×RTX 4090 环境通过完整 CPU gate、NCCL stress、rank-consistency preflight 与 dense/local-SVD/broadcast 真实训练入口 smoke。CM044 的 tiny profiler/timing 结果为负：该配置中绝大多数 payload 是不压缩的 dense auxiliary 参数，local-SVD 与 broadcast 的完整 period logical gradient payload 只比 dense 少 `0.72%` 与 `0.35%`，而平均 step time 分别是 dense 的约 `17.2×` 与 `23.1×`。三 seed、12-update 合成回归 gate 无 NaN/Inf，但不能替代真实语言模型质量实验。因此不启动 10,000-update paper-oriented recipe；跨模型/网络的正式性能和训练质量仍为 pending，不作一般速度、显存、训练质量或理论收敛声明。

当前合并阻塞项是 timing fail-closed 的一个尾部边界：若同一日志先出现合法 `step_avg`，随后以 malformed、负数、`NaN` 或 `Inf` 的 `step_avg` 结束，parser 会忽略坏的末标记并接受较早值。CM044 的 9 个 timing 日志均已人工核验为合法，因此现有 tiny negative 结论不受影响；但修复并增加回归测试前，不能把 evidence launcher 视为完全 fail-closed，也不建议合并本分支。

已登记但本任务不重构的工程限制：local matrix math 尚未对同 compressed shape 参数 batching；直接 `load_state_dict()` 在存储损坏 carve-out 下的 late validation 可能发生非事务性部分写入；新生成的 `timing-summary.json` 缺少旧 parser 的 per-cell `throughput_tokens_per_second` 字段（aggregate throughput 保留）；`environment.txt` 是 controller snapshot，不等于每条命令的精确 CUDA/PYTHONPATH/NCCL 环境（精确覆盖项仍记录于 `command.txt`）。完整阻塞、优化优先级与 CM044 性能根因记录在 `docs/worklog/M002-greedy-lore-muon.md` 的“最终审查状态与性能诊断”。
