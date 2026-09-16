# M002 GreedyLore-Muon 工作日志

## 2026-09-11：方法登记与 Task 10 evidence gate 启动

### 目的与假设

在 formal experiment 前登记 M002，并依次完成 CPU correctness、两卡 NCCL/真实 `train.main` smoke、refresh/compressed trace、完整 period wall-clock/memory 与多 seed 短质量 gate。假设仅限于：GreedyLore 可在 DDP gradient-input 层减少 logical payload，同时让 unchanged ordinary Muon 消费 rank-consistent reconstructed gradient；MSGD/Adam 理论不会自动迁移到 Muon，logical payload reduction 也不会自动变成 collective 或端到端加速。

### 实现与版本

- 分支起点：`17cfb077f29a985193d25f95843a5520e1185b64`。
- Task 1–9 当前 HEAD：`4b21445acf60c323b2bfd9741c4f8aff5afef755`。
- 实现提交：`2612480` tensor foundations；`9636b12` recurrence；`714f6cb` state/layout；`504e4b2` Future sequencing；`6380152`/`63b1f51` dense/refresh；`5887b91` compressed sync；`a28336e` Muon integration/config；`20570ef` checkpoint；`654f818`/`4b21445` profiler/launcher hardening。
- 设计：`docs/superpowers/specs/2026-09-09-greedylore-muon-design.md`。
- 计划：`docs/superpowers/plans/2026-09-09-greedylore-muon-ddp-hook.md`。
- 方法：`docs/compressed_muon/methods/M002_greedy_lore_muon.md`。
- 配置：`configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml`。
- 入口与实现：`train_greedylore.py`、`dion/greedy_lore.py`、`dion/greedy_lore_layout.py`、`dion/greedy_lore_ddp_hook.py`。
- profiler launcher：`benchmark/compressed_muon/run_greedy_lore_profiler.sh`。

### 作者快照与边界

只读作者快照位于 `/home/wyr/greedy_lore`，不是 runtime dependency，也不是 executable oracle。固定 SHA-256、可复用 orientation/同 shape batching/cadence/packing/reconstruction 线索，以及所有 intentional divergence 统一记录在方法文档。特别是：真实 greedy hook 的 score helper 未完成，fake hook 通信 dense tensor；M002 不以其运行结果作为正确性判据。

### 当前证据状态

- METHOD_INDEX 的 Task 7 gap 已关闭，状态 `testing`。
- CPU correctness：待运行。
- NCCL stress 与真实 dense/local-SVD/broadcast `train.main`：待 CPU gate 通过后运行。
- 正式 experiment ID、trace 与 complete-period timing：待 correctness gate 后预登记/执行。
- 多 seed 短质量与 10,000-update paper-oriented recipe：待性能/稳定性 gate；后者是条件项，不满足条件不启动。

### 已知限制/延期项

- Task 6：local per-parameter math 缺少 same-shape batching，可能增加 kernel launch/local cost，但不改变已测试数学语义。
- Task 8：storage corruption carve-out 下，直接 load 的 late checks 非事务性；metadata incompatibility 仍在 DCP mutation 前拒绝。
- Task 9：`--summarize-only` 不验证全部 timing cell/final-timing artifact；使用者必须结合每 cell `exit_code.txt` 与 raw logs 审核。
- Task 9：`environment.txt` 捕获 controller environment，而精确 per-command `CUDA_VISIBLE_DEVICES`、`PYTHONPATH`、NCCL unset/override 在 `command.txt`；不能把前者描述为完整命令环境。

### 下一步

使用 `PYTHONPATH=.` 运行 Task 10 完整 CPU gate并记录精确 pass/skip；只有零失败且零非 `multi_gpu` skip 时才检查 GPU ownership 并进入 NCCL gate。

## 2026-09-11：完整 CPU correctness gate

### 命令与产物

精确命令（仅增加 worktree 必需的 `PYTHONPATH=.`）：

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest \
  tests/test_greedy_lore.py \
  tests/test_greedy_lore_oracle.py \
  tests/test_greedy_lore_layout.py \
  tests/test_greedy_lore_layout_distributed.py \
  tests/test_greedy_lore_ddp_state.py \
  tests/test_greedy_lore_ddp_future.py \
  tests/test_greedy_lore_ddp_future_distributed.py \
  tests/test_greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook_distributed.py \
  tests/test_greedy_lore_ddp_checkpoint.py \
  tests/test_train_greedylore.py \
  tests/test_train_ddp_sync.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_greedy_lore_profiler_trace.py \
  tests/test_greedy_lore_profiler_launcher.py \
  tests/test_training_profiler_trace.py \
  tests/test_configs.py -v
```

- 结果：`196 passed, 0 failed, 0 skipped, 14 warnings in 197.80s`，exit `0`。
- warnings：14 条既有 `torch.jit.script_method` deprecation warning。
- raw command/log/exit/time：`artifacts/compressed_muon/M002-task10-correctness-20260911/cpu-gate.command.txt`、`cpu-gate.log`、`cpu-gate.exit`、`cpu-gate.started`、`cpu-gate.finished`。

### 判断

CPU gate 通过且没有 skip；不存在被误计为 pass 的 GreedyLore/training integration skip。允许进入 GPU ownership 检查和 NCCL correctness gate；该结果本身不支持性能或训练质量声明。

## 2026-09-11：两卡 NCCL Future/allocator stress

### GPU 分配证据

启动前 `nvidia-smi` 显示 GPU 2/3（RTX 4090，UUID 分别为 `GPU-e6622753-895a-f8fc-6082-ac71bbfa0037`、`GPU-f7d3c0fb-aed7-235f-7332-68966b63e0c5`）各使用 3 MiB、utilization 0%，compute-app 列表无对应进程。GPU 0/1 上 PID `2994719`，GPU 7 上 PID `3063114`；均未触碰。原始快照：`artifacts/compressed_muon/M002-task10-correctness-20260911/nccl-gpu-before.csv`、`nccl-processes-before.csv`、`nccl-process-owners-before.txt`。

### 命令与结果

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. uv run --frozen --extra train --extra dev pytest \
  tests/test_greedy_lore_ddp_hook_nccl.py -v
```

结果：`3 passed, 0 failed, 0 skipped, 14 warnings in 23.52s`，exit `0`。覆盖 returned Future 的最终 compressor-stream visibility、broadcast refresh visibility，以及 compressed hook 在 bucket rebuild、gradient accumulation delay、rank-skewed delay 与 allocator churn 下的 real NCCL completion/signature。raw：同一目录下 `nccl-stress.command.txt`、`nccl-stress.log`、`nccl-stress.exit`、时间戳文件。

该 stress 不单独证明 local-SVD basis、Muon momentum/parameter tolerance 或真实 `train.main`；这些由后续 preflight/smoke gate 单独记录。

## 2026-09-11：NCCL rank-consistency preflight 与真实训练入口 smoke

### Preflight

启动前 GPU 2/3 仍各使用 3 MiB、无 compute app；原始快照为 `preflight-attempt2-gpu-before.csv` 与 `preflight-attempt2-processes-before.csv`。使用 artifact 内一次性 `nccl_preflight.py` 分别运行 `local_svd` 和 `broadcast`，两者串行使用：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. .venv/bin/python \
  artifacts/compressed_muon/M002-task10-correctness-20260911/nccl_preflight.py \
  --basis-sync local_svd \
  --output-dir artifacts/compressed_muon/M002-task10-correctness-20260911
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. .venv/bin/python \
  artifacts/compressed_muon/M002-task10-correctness-20260911/nccl_preflight.py \
  --basis-sync broadcast \
  --output-dir artifacts/compressed_muon/M002-task10-correctness-20260911
```

在 FP32 两步（refresh、compressed）上，两种模式均得到：basis rank-max max-abs diff `0`（门槛 atol `1e-6`/rtol `1e-5`）、support exact、reconstructed-gradient max-abs diff `0`（门槛 atol `1e-5`/rtol `1e-4`）、Muon momentum/parameter max-abs diff `0`（FP32 atol/rtol `1e-5`），collective signature exact，loss/grad norm finite。第二步启用 clip `0.5`，clip 前后 compressor error exact unchanged，覆盖 `finish -> reconstructed gradient clip -> ordinary Muon -> commit`。

local-SVD refresh 后在没有下一 compressor step 的 committed boundary 立即调用 `validate_replicated_basis_across_ranks(atol=1e-6, rtol=1e-5)`。第一次 harness attempt 错放在 commit 前，按 API contract 以 `GreedyLoreStateError: ... requires a committed step boundary` 失败；未修改 production，修正 harness 调用顺序后通过。失败尝试保留为 `preflight-attempt1-*`。成功 raw JSON 为 `local_svd.json`、`broadcast.json`；命令/日志/exit/time 与 GPU 快照均在 correctness artifact 目录。

该通过只验证本机两张 RTX 4090 与当前 PyTorch/CUDA/NCCL 同构环境；repeated singular-value subspace 的 local-SVD 旋转不唯一风险仍存在，broadcast 仍是强一致性 fallback。

### 真实 `train.main` smoke

再次确认 GPU 2/3 空闲后，用 `.venv/bin/torchrun --standalone --nproc_per_node=2` 串行启动：ordinary dense `train.py`、`train_greedylore.py` local-SVD、以及 `grad_clip_norm=0.5` 的 broadcast。共同几何为 dim64/2 layers/4 heads、seq32、global/device batch4/2、3 updates、rank2/interval2/start0、真实 FineWeb10B loader、no W&B/no compile/no Triton。精确三条命令在 `train-main.command.txt`。

三模式均 exit `0`，完成 step0–2 更新与 step3 validation；最终 val loss 分别 `11.0376`、`10.9663`、`10.8555`，peak memory 分别 `213/184/184 MiB`。不同进程未固定 initialization seed，因此这些 loss 只证明 entry 执行、不能用于模式质量比较或速度比较。第一次 attempt 因传入不存在的 CLI flag `--val_loss_every` 在 argparse 阶段 exit `1`，没有训练；失败 raw 保留为 `train-main-attempt1-*`，移除该 flag 后通过。

### Correctness gate 判断

CPU、NCCL stress、rank-consistency/tolerance 和三种真实入口均通过。允许启动预登记的最小 profiler/complete-period experiment；结果仍必须按 logical bytes、collective time、tail、throughput、memory 与 quality 分栏，不得把 correctness 通过写成性能结论。

## 2026-09-11：CM044 profiler 与完整 period timing

### 配置、GPU 与启动

预登记 ID：`CM044-m002-greedylore-muon-tiny-ddp-ws2-s42`。共同配置为 GPU2/3、dim64/2 layers/4 heads、FP32 params/BF16 autocast、seq32、global/device batch4/2、GA1、bucket1 MiB、seed42；M002 rank2/interval2/start2。每个模式 3 个 rotated blocks；每 block 单独 profile refresh/compressed，并用 profiler-off 运行 2 warmup + 2 measured updates（一个完整 interval）。

启动前 GPU2/3 各 3 MiB、0% utilization、无 compute app；GPU0/1 的外部进程未触碰，并按任务边界避开 GPU7。证据：correctness artifact 的 `cm044-attempt2-gpu-before.csv`、`cm044-attempt2-processes-before.csv`。精确 preregistered plan：`cm044-preregistered-plan.json`。

第一次 controller 在 cell 前因 worktree 缺少 dataset path 而 fail closed（`BLOCKED missing=.../data/fineweb10B`）；失败目录保留为 `CM044-...-attempt1-missing-data/`。建立指向已验证 `/home/wyr/dion/data/fineweb10B` 的 worktree-local symlink 后，从空正式 artifact root 完整重跑。没有删除或覆盖失败产物。

### 完整性

- controller exit `0`；27/27 cell exit `0`；36 rank traces；9/9 timing logs 有 final `step_avg`；无 OOM/timeout/traceback marker。
- `summary.json` 使用 `--require-plan` 完成 profiler cell/rank/signature/log gate；hook collective signature rank-identical，unattributed NCCL fraction `0`。
- 由于已知 `--summarize-only` 不验证 timing artifacts，另逐一检查 9 个 timing `exit_code.txt` 与 final timing，并用保留的 `summarize_timing.py` 生成 `timing-summary.json`。
- 每 cell `environment.txt` 仅为 controller snapshot；精确 `CUDA_VISIBLE_DEVICES=2,3`、`PYTHONPATH` 与 NCCL unset 记录在对应 `command.txt`，不将 environment snapshot 误称为 exact command env。

### Logical communication 与 profiler 结果

- dense gradient：`26,148,864 B/update`。
- local-SVD：refresh `26,148,864 B`、compressed `25,771,008 B`，period avg `25,959,936 B`，logical reduction `0.7225%`。
- broadcast：refresh `26,345,472 B`（含 `196,608 B` basis）、compressed `25,771,008 B`，period avg `26,058,240 B`，reduction `0.3466%`。
- tiny 模型的 matrix/dense auxiliary bytes 为 `393,216/25,755,648`，所以覆盖范围导致整体 reduction 很小；Muon result communication 单列且 unchanged。
- genuine-backward overlap：所有 tiny cell 为 `0 ms`。
- local-SVD refresh：NCCL union `6.929 ms`，SVD GPU `10.847 ms`，collective tail `1.616 ms`，complete compressor tail `212.564 ms`。
- local-SVD compressed：NCCL union `8.133 ms`；dense/score/factor collective `1.580/1.643/1.452 ms`；score/factor/error/reconstruct GPU `0.183/0.045/0.069/0.054 ms`；collective/complete tail `0.364/1.808 ms`。
- broadcast refresh：NCCL union `223.444 ms`，其中 basis broadcast `213.579 ms`；SVD GPU `10.484 ms`；collective/complete tail `211.489/211.499 ms`。
- broadcast compressed：NCCL union `145.529 ms`；score/factor AR `55.680/80.475 ms`；complete tail `1.343 ms`。该异常高于同 payload local-SVD，当前 tiny 数据不足以归因或泛化。

### Complete-period wall-clock/memory

- dense：`8.54/8.86/9.73 ms`，mean/median `9.043/8.860`，sample SD `0.616`，CV `6.81%`，mean throughput `14,197 tok/s`，peak `193 MiB`。
- local-SVD：`155.95/154.00/155.36 ms`，mean/median `155.103/155.360`，SD `1.000`，CV `0.64%`，`825 tok/s`，peak `225 MiB`。
- broadcast：`207.08/211.15/207.90 ms`，mean/median `208.710/207.900`，SD `2.153`，CV `1.03%`，`613 tok/s`，peak `225 MiB`。
- paired local-SVD minus dense：mean `+146.06 ms`、mean ratio `17.203`（`+1620.3%`），bootstrap mean-difference 95% interval `[145.14,147.41] ms`。
- paired broadcast minus dense：`+199.67 ms`/`+2214.9%`，interval `[198.17,202.29] ms`。
- paired local-SVD minus broadcast：`-53.61 ms`/`-25.68%`，interval `[-57.15,-51.13] ms`。
- M002 peak 比 dense 多 `32 MiB`/`16.58%`。区间不跨 0 且差异远大于 2%，按规则不延长到 10 blocks；dense CV 偏高使 tiny absolute ratio 更不适合泛化，但不会改变本负结果方向。

CM044 总体分类 **negative**：logical payload 有极小正向缩减，但 collective/critical path、complete-period throughput 与 peak memory 均为负。不作加速 claim，也不将 tiny 结果外推到代表性模型或网络。

### Artifact

正式目录：`artifacts/compressed_muon/CM044-m002-greedylore-muon-tiny-ddp-ws2-s42/`。包含 `plan.json`、每 cell command/environment/stdout/stderr/exit/time、36 traces、`summary.json`、`timing-summary.json`/parser。controller exact command/log/exit/time 和 prelaunch GPU snapshot 在 `artifacts/compressed_muon/M002-task10-correctness-20260911/cm044-controller-attempt2.*` 与 `cm044-attempt2-*before.csv`。

## 2026-09-11：三 seed 短质量/稳定性 gate 与长程决策

### 命令与配置

GPU2/3 启动前各 3 MiB、0% utilization、无 compute app；快照为 `short-quality/gpu-before.csv` 与 `processes-before.csv`。artifact 内 `short_quality.py` 使用 two-rank NCCL、FP32 16x16 两层 synthetic regression、ordinary Muon、12 updates；dense 与 local-SVD（rank4/interval4/start0）对每个 seed `7/42/123` 复用完全相同 initialization 与 rank-local data order。精确循环命令为：

```bash
for seed in 7 42 123; do
  for mode in dense local_svd; do
    CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. .venv/bin/python \
      artifacts/compressed_muon/M002-task10-correctness-20260911/short_quality.py \
      --mode "$mode" --seed "$seed" \
      --output-dir artifacts/compressed_muon/M002-task10-correctness-20260911/short-quality
  done
done
```

6/6 cases exit `0`，参数 rank-consistent；所有 loss、gradient、residual、parameter 均无 NaN/Inf。seed7/42/123 的 dense→local final train loss 分别为 `0.49735→0.50120`、`0.49636→0.50132`、`0.49308→0.49745`；validation loss delta 为 `+0.00518/+0.00414/+0.00443`，mean `+0.00458`。dense grad norm range 为 `0.2646–0.4171`，local-SVD 为 `0.1521–0.7397`；local residual norm range为 `0–0.6303`，每个 refresh reset 到 0。raw per-seed JSON/log、command、GPU snapshot、exit/time 与 `summary.json` 位于 `short-quality/`。

这是短 synthetic stability preflight，不是 GPT/FineWeb/C4 training-quality gate，因而不能支持 convergence/time-to-quality claim。结合 CM044 的明确 negative performance，条件性的四卡 10,000-update recipe **未启动**。

若未来补齐 paper-oriented gate，预定为 current GPT-60M、global/device batch512/128、4 ranks、seq256、10,000 updates、1,000 warmup、rank32、interval200、Adam-style scalar settings，并显式使用 Dion cosine-to-zero。作者 launcher 的 `grad_clipping=0` 与 CM039 的 `grad_clip_norm=1.0` 必须作为 paired ablation；作者 launcher scheduler warmup 为200而 Table VII 报告1000；snapshot 缺 C4 training source，cosine endpoint 无法本地核验。因此只能称 paper-oriented comparison，不能称 exact reproduction。

待满足 gate：代表性 GPT-60M 多 seed short quality（train/val/grad/residual telemetry）、独占四卡 representative complete-period timing，以及可接受的 validation loss/time-to-quality。方法状态保持 `testing`。

## 2026-09-11：最终相关回归

按 Task 10 Step 8 命令并增加 worktree 必需的 `PYTHONPATH=.`，运行 GreedyLore tensor/oracle/layout/state/Future/hook/checkpoint、训练集成、ARC hook、optimizer、training profiler 与 config 测试。结果为 `278 passed, 0 failed, 0 skipped, 14 warnings in 282.99s`，exit `0`；warnings 仍是既有 `torch.jit.script_method` deprecation。精确命令、raw log、exit 与时间戳位于 `artifacts/compressed_muon/M002-task10-final-20260911/final-regression.*`。

## 2026-09-11：最终审查状态与性能诊断

### 合并阻塞项

- **P0 / evidence fail-closed**：timing parser 会从日志中选择最后一个“能被当前正数正则识别”的 `step_avg`。若日志先有合法值，随后以 malformed、负数、`NaN` 或 `Inf` 的 `step_avg` 结束，坏的末标记会被忽略，较早合法值仍可使 `--summarize-only` 成功。最终 scoped re-review 将此判为未完全修复的 Important finding。必须先让 parser 识别所有 `step_avg` 标记并显式拒绝非法末值，再补 malformed/negative/non-finite terminal-marker 回归，分支才可视为 merge-ready。
- 该缺陷不改变 CM044 数字：现有 9/9 timing cell 的末标记均已逐一核验为有限正数，27/27 cell exit 0，原始 trace/log 未被最终修复重写。

### 非阻塞待优化/工程债

- **P1 / 性能**：把相同 compressed shape 的 score、factor、error 与 reconstruction 做真正的 batch，并优先复用持久 packing/workspace，减少当前 14 个矩阵逐参数的 Python、kernel-launch 和临时分配开销。
- **P1 / 测量**：在独占四卡、代表性 GPT-60M、默认量级 `update_interval=200` 上重跑完整周期；先确认 shared Muon matrix 占总梯度 payload 的比例足够大，再判断优化后的 hook 是否值得做长质量实验。
- **P1 / 独立消融**：若要压缩二维 embedding/lm-head，必须注册为 all-2D GreedyLore 消融；它会改变通信量、SVD 成本、optimizer coverage 与方法含义，不能静默改变 M002 的 transformer-Muon-matrix-only 边界。
- **P2 / 并发**：首版为安全性使用一条全局 cross-bucket Future tail。只有在 batched local math 稳定后，才评估按 bucket/shape 放宽串行化；必须继续保证所有 rank collective 顺序一致。
- **P2 / checkpoint**：直接 `load_state_dict()` 的 late committed-step/replicated-state validation 可能在损坏 payload 下留下部分本地写入；这符合当前 DCP corruption carve-out，但不是事务式恢复。
- **P2 / evidence schema**：新 `timing-summary.json` 保留 aggregate throughput，但缺旧 parser 的 per-cell `throughput_tokens_per_second`；恢复字段以避免下游 schema drift。
- **P2 / environment**：`environment.txt` 只是 controller snapshot；精确子进程 `CUDA_VISIBLE_DEVICES`、`PYTHONPATH` 与 NCCL override 仍以每个 cell 的 `command.txt` 为准。

### 为什么 CM044 慢

以下先区分可直接从 artifact 读出的事实与尚待代表性实验验证的推断。

**已观测事实：**

1. **几乎没有可省的总通信。** dense gradient payload 为 `26,148,864 B/update`，其中只有 `393,216 B`（约 `1.50%`）属于 M002 可压缩的 14 个 Muon 矩阵，`25,755,648 B`（约 `98.50%`）是仍需 exact dense sync 的 embedding/lm-head 等 auxiliary。compressed phase 把矩阵部分降到约 `15,360 B`，但完整 interval 的总 payload 只降低 `0.7225%`；broadcast 因 refresh basis payload 只降低 `0.3466%`。因此带宽收益从一开始就不足以覆盖任何明显固定开销。
2. **测试把 refresh 设得极频繁。** CM044 使用 `update_interval=2`，每两个 update 就有一次 dense corrected-gradient refresh 和 SVD；正式默认是 `200`。这个设置适合在最短时间同时覆盖 refresh/compressed trace，却会把 refresh 成本以 `50%` 权重计入 wall clock，不是性能友好的生产 cadence。
3. **refresh 的固定调度成本远大于矩阵计算本身。** local-SVD refresh 的 SVD GPU kernel 合计约 `10.847 ms`，但 compressor critical-path tail 为 `212.564 ms`。当前实现逐参数处理 14 个矩阵，并通过一条全局 Future tail 串行保护 collective 顺序；大量小 kernel、Python callback、Future/stream 交接及临时 packing 在 dim64 tiny workload 上压过实际算术。
4. **压缩路径增加了小 collective 数量。** dense DDP 可用少量大 All-Reduce；mixed compressed bucket 则需要 `score_plus_aux_allreduce -> factor_allreduce`，另有 dense-only bucket。score/factor payload 很小，处于 latency-bound 区域，无法有效利用 4090/NCCL 带宽。broadcast refresh 还按稳定参数顺序执行多个小 basis broadcast，观测 basis collective 合计约 `213.579 ms`。
5. **Muon 自己的 result communication 完全不变。** M002 只压缩 DDP gradient input；ordinary Muon 的 result communication 仍存在。因此即使矩阵梯度压缩理想化为免费，也只能优化 step 的一部分。
6. **额外状态与 workspace 在 tiny 模型上比例显眼。** basis、error、score/factor packing 和异步保活使 peak allocated 从 dense `193 MiB` 增至 `225 MiB`。绝对值只有 `32 MiB`，但 tiny baseline 下是 `16.58%`。

**最可能的解释：**CM044 的 `17.2×/23.1×` slowdown 不是单一 SVD kernel 导致，而是“`98.5%` payload 不可压缩 + interval2 高频 refresh + 逐参数小算子/小 collective + 全局安全串行链 + Muon result communication 不变”的叠加。local-SVD 比 broadcast 快约 `25.7%`，符合省掉 basis broadcast 的方向；但 broadcast compressed trace 中异常大的 score/factor collective 时间无法由当前 tiny 三重复唯一归因，不能据此声称算法固有的 NCCL 成本。

**尚待验证：**把 interval 改为 `200` 会显著摊薄 refresh，但在当前 tiny 参数边界下，总 payload 理论收益仍受约 `1.5%` 可压缩占比限制。是否能在 GPT-60M 等 transformer matrix 占比更高的模型上转正，必须先完成 same-shape batching，再用代表性几何、独占四卡和完整周期实测；不能由 CM044 外推。

## 2026-09-11：后续性能优化队列与首批执行

按“预期收益大、修改量小、尽量保持作者 GreedyLore 结构”的原则登记以下顺序：

1. **P0 / completed**：修复 timing parser 的非法末尾 `step_avg` fail-closed 边界，补 malformed、负数与非有限值回归。
2. **P1 / completed/negative**：CM045 复用 CM044 tiny paired 几何，将 `update_interval` 从 2 改为官方默认量级 200，量化 refresh 摊销；结果仍受 1.50% 可压缩 payload 上限约束。
3. **P1 / TODO**：参考 ARC 的 canonical shape/device/dtype 分组和作者仓库的 same-shape batching，先 batch 每个普通 compressed step 的 score、Top-r、factor、error 与 reconstruction；保持当前 score+dense_aux/factor 两阶段 packed All-Reduce 和全局 Future 顺序不变。
4. **P1 / TODO**：为 factor 一次性分配 packed buffer，并让 batched factor 直接写入 view，移除逐参数 `.clone()` 后再 `cat()`；仅在 profiler 证明必要时增加可安全处理 bucket rebuild 的持久 workspace。
5. **P2 / TODO**：普通步优化验证后再评估 batched refresh SVD；`update_interval=200` 时 refresh 仅占 0.5%，其平均收益优先级低于普通步 batching。
6. **P2 / TODO**：仅针对强一致性 fallback 评估按 bucket 打包 basis broadcast；跨 bucket Future 并发、CUDA Graph 与 all-2D GreedyLore 均推迟。all-2D 若开展必须单独登记消融，不能改变 M002 默认 transformer-Muon-matrix-only 边界。

首批 parser TDD 的 RED 已确认：在一个合法 timing 后追加 `invalid/-1/nan/inf/1e999` 末标记时，旧 parser 的 5 个 case 均错误成功。最小修复改为先识别末个 token，再显式执行 float、finite 和正值校验；相关 parser/launcher 测试随后为 `21 passed`。完整回归与 CM045 结果在本节后续追加。

### CM045 interval200 结果

controller 于 `2026-09-11T14:14:06+08:00` 至 `14:27:33+08:00` 在 GPU2/3 完成；27/27 cells exit `0`、18 个 profiler cells 共 36/36 rank traces，日志无 OOM/timeout/traceback marker。profiler-off timing 每个 cell 为 2 warmup + 200 measured updates，恰好覆盖一个 interval200 完整周期，并使用 3 个 rotated paired blocks。

- dense：`7.86/8.06/8.02 ms`，mean `7.980 ms`，CV `1.33%`，peak `193 MiB`。
- local-SVD：`16.54/16.45/16.94 ms`，mean `16.643 ms`，CV `1.57%`，peak `251 MiB`。
- broadcast：`16.64/16.67/16.72 ms`，mean `16.677 ms`，CV `0.24%`，peak `276 MiB`。
- paired local-SVD minus dense：`+8.663 ms`，mean ratio `2.0858`（`+108.58%`），bootstrap mean-difference 95% interval `[8.39, 8.92] ms`。
- paired broadcast minus dense：`+8.697 ms`/`+109.00%`；local-SVD minus broadcast 为 `-0.033 ms`，95% interval `[-0.22, 0.22] ms`，本实验无法区分两者完整周期性能。

与 CM044 interval2 的 `155.103/208.710 ms` 相比，local-SVD/broadcast 平均 step 分别下降 `89.27%/92.01%`；这确认高频 refresh 是此前极端 slowdown 的重要组成。结论仍为 **negative**：默认 cadence 将 slowdown 从约 17.2× 缩小到约 2.09×，但 tiny 模型仅 1.50% gradient payload 可压缩，普通 compressed step 的逐参数调度、两阶段 latency-bound collective、额外状态和 unchanged Muon result communication 仍足以压过带宽收益。CM045 只隔离 cadence，不支持代表性模型速度或训练质量结论；下一步保持队列中的 ordinary-step same-shape batching 优先级。

artifact：`artifacts/compressed_muon/CM045-m002-greedylore-muon-tiny-interval200-ddp-ws2-s42/`，包含 plan、每 cell command/environment/log/exit/time、36 traces、`summary.json` 与 `timing-summary.json`。

## 2026-09-11：代表性模型 staged timing 计划

用户批准把 interval200 测速扩展到 GPT-60M/130M/350M。为避免 broadcast fallback 和高风险 350M 直接消耗完整 timing 预算，登记 CM046–CM051 并串行设置 gate：60M preflight → 130M preflight → 60M/130M formal timing → 350M preflight → 350M formal timing。preflight 只运行 dense/local-SVD/broadcast 各一个 refresh/compressed profiler cell；formal 只运行 dense/local-SVD 的 3 rotated profiler/timing repeats，timing 为 20 warmup + 200 measured updates，覆盖一个完整 interval200 周期。

60M/130M 复用已有成功的 seq256、global/device batch512/128、GA1 几何；350M 为控制 activation 与完整 basis/error 状态叠加后的 OOM 风险，使用 seq256、global/device batch32/8、GA1。因此只在每个模型内部做 paired ratio，不横向比较三种模型的绝对吞吐。统一使用 4×RTX 4090、rank32、bucket160 MiB、seed42 和 local-SVD 主路径；broadcast 只作为单轮强一致性/通信诊断。launcher 新增独立 `--profile-modes`/`--timing-modes` 过滤，默认行为保持不变，相关 launcher/parser 测试为 `27 passed`。

GPU2–5 启动前均为 `3 MiB`、`0% utilization`，无 compute process；`m002-scale-timing` tmux controller 已按上述 gate 顺序启动。每个 launcher 自行保存 plan、cell command/environment/log/exit/time 和 summary；任一阶段非零退出将通过 `&&` 阻止后续阶段启动，不会在 preflight 失败后继续消耗正式 timing 预算。

### 2026-09-11：350M 切换为 timing-only

CM050 的 6/6 profiler cells 均 exit `0`，原始 24/24 rank traces 共 `2,338,797,659 B`（约 2.18 GiB），dense/GreedyLore peak allocated 为 `7062/9731 MiB`，因此 350M 显存与执行 preflight 已通过。trace summarizer 单核 99.9% CPU 连续运行超过 3.5 小时仍未完成；按用户决定停止该 parser 与旧串行 controller，不把 trace summary 作为正式 wall-clock 的前置 gate。

保留 `CM050-.../greedylore_local_svd-refresh-r1/profiler/rank-0.json`（约 424 MiB）供以后诊断，删除 CM050 其余 23 份 trace；删除不可从仓库恢复，CM046–CM049 历史 trace 未改动。launcher/summarizer 增加 `profile_modes=none` 的 timing-only 合法路径：仍要求 plan、command、exit、finished timestamp 和有限正值末尾 timing，正常生成 `summary.json`（空 profile cells）与 `timing-summary.json`。对应 launcher/parser 测试为 `29 passed`。CM051 随后只运行 dense/local-SVD 的 3 rotated、20 warmup + 200 measured update timing，不再生成 trace。

### CM051 350M timing-only 结果

CM051 于 `21:12:26–21:23:27+08:00` 完成，6/6 timing cells exit `0`，日志无 OOM/timeout/traceback，trace 文件数严格为 `0`。每个 cell 为 20 warmup + 200 measured updates，三轮使用 dense/local-SVD rotated pairing。

- dense：`228.14/241.35/229.15 ms`，mean `232.880 ms`，CV `3.16%`，throughput `35,200 tok/s`，peak `7062 MiB`。
- local-SVD：`192.96/196.02/197.72 ms`，mean `195.567 ms`，CV `1.23%`，throughput `41,893 tok/s`，peak `9731 MiB`。
- paired local-SVD minus dense：`-35.18/-45.33/-31.43 ms`，mean `-37.313 ms`，mean ratio `0.8403`（`-15.97%`），bootstrap mean-difference 95% interval `[-45.33,-31.43] ms`。

在该 350M、global/device batch32/8 几何下结果分类为 **positive**：三轮同方向，完整 interval200 周期 step 降低约 16%，但 peak allocated 增加 `2669 MiB`（约 `37.8%`）。尺度趋势为 60M null（`+0.17%`）、130M preliminary-positive（`-1.86%`）、350M positive（`-15.97%`），与更大模型中 transformer Muon matrix 占总 payload 比例上升的方向一致。三种模型 batch geometry 不同，不能横向比较绝对吞吐；三轮 timing 也不能替代多 seed 训练质量与 time-to-quality，尤其不能据此声称 GreedyLore-Muon 已保持 dense Muon 收敛性质。

artifact：`artifacts/compressed_muon/CM051-m002-gpt350m-interval200-ddp-ws4-s42/`，包含 `plan.json`、6 个 timing cell 的 command/environment/log/exit/time、空 profile `summary.json` 和 `timing-summary.json`。

## 2026-09-12：CM052/CM053 paper-aligned 实际训练计划

- 目的：先在 GPT-60M 和 GPT-130M 上比较 dense Muon 与 M002 local-SVD 的完整训练质量、稳定性和 time-to-quality，不用短 timing 结果替代收敛证据。
- 公共设置：4×RTX 4090 DDP、BF16 compile、FineWeb10B、seq256、global/device batch512/128、seed1234、validation 每 500 step；Muon matrix 使用 `lr=0.02`、momentum `0.95`、weight decay `0.01`，auxiliary scalar AdamW 使用 `lr=0.001`、betas `(0.9, 0.999)`、eps `1e-8`、weight decay `0`。
- 论文对齐项：60M/130M 分别使用 10,000/20,000 updates、1,000/2,000 warmup、cosine decay to 10%、clip1；M002 使用 rank32、interval200、step1000 开始压缩、error feedback、`local_svd`。
- 解释边界：数据仍是 Dion 现有 FineWeb10B，模型仍是 Dion GPT/LLaMA-like 架构，因此这是 paper-aligned M002-on-Muon controlled study，不是论文 C4/AdamW 严格复现；M002 仍只压缩 Muon matrix group，未静默扩展为 all-2D。
- 执行：`benchmark/compressed_muon/run_cm052_cm053_m002_quality.sh` 自动等待任意 4 张至少有 18 GiB 空闲显存的 GPU，并按 CM052a→CM052b→CM053a→CM053b 串行运行；正式训练不开 Kineto trace。CM052b perplexity 相对 CM052a 若超过 1.10，controller 自动停止，不启动 CM053。
- 启动状态：2026-09-12 01:32（Asia/Shanghai）在 tmux `cm052_cm053_m002_quality` 启动；当时仅 GPU 7 满足空闲门槛，controller 已进入自动等待。状态与计划位于 `artifacts/compressed_muon/CM052-CM053-m002-paper-aligned-quality-controller/`。

### 完成结果

controller 于 01:55 获得 GPU 2–5，4/4 probes 与 4/4 formal cells 均 exit `0`，并于 05:27 完成，controller exit `0`。60M dense/M002 的 final val loss 为 `4.0003/4.0837`，perplexity 为 `54.61/59.36`（M002 `+8.70%`），step 为 `112.61/114.25 ms`（`+1.46%`），peak 为 `6865/7347 MiB`。60M perplexity ratio `1.08698` 通过预登记的 1.10 继续门禁。

130M dense/M002 的 final val loss 为 `3.5749/3.6582`，perplexity 为 `35.69/38.79`（M002 `+8.69%`），step 为 `241.22/238.01 ms`（`-1.33%`），peak 为 `13048/13889 MiB`。两组 best val loss 都是 final loss，M002 均未在当前预算内达到 dense final quality，故没有 dense-final-quality time-to-quality。

结论为 **quality-negative**：130M 的小幅 step 收益延续 CM049 方向，但单次差异小于 2%，且不能补偿约 8.7% perplexity 退化；60M 同时没有性能收益。暂不扩展多 seed，优先评估修改量小的 rank/start-step 敏感性，all-2D 仍须作为单独方法范围消融登记。W&B IDs：CM052a `80b662fz`、CM052b `ktl0jkfd`、CM053a `wda2zc6n`、CM053b `9rugk05n`。

## 2026-09-12：普通 compressed step same-shape batching

为减少普通 compressed step 的逐参数 Python 与小 kernel launch 开销，按稳定参数顺序将矩阵以 canonical oriented shape、gradient device/dtype、error dtype 和 basis dtype 分组。每组使用 batched score、稳定 Top-r gather、factor、error 与 reconstruction；相反原始方向但 canonical shape 相同的矩阵进入同一组。现有 `score+dense_aux -> factor` 两阶段 All-Reduce、每参数稳定 seed 和全局 cross-bucket Future 顺序保持不变。

factor collective buffer 改为预先计算总大小并一次分配；每组 `torch.bmm(..., out=view)` 直接写入连续 packed view，移除原先逐参数 `.clone()` 后再 `torch.cat()` 的路径。首版未引入持久 workspace、refresh SVD batching、跨 bucket 并发或 collective 变更。

TDD 新增 operator-count 与 storage-sharing 回归：同 canonical shape 的 `(3, 2)`/`(2, 3)` 矩阵按一个 batch 执行，local factor 的 storage pointer、offset 和连续性与 packed collective buffer 一致。canonical grouping mutation 会令 `bmm` 次数由 8 增至 12 并触发测试失败。相关 hook/Future/Gloo 聚焦回归为 `22 passed`；完整 GreedyLore CPU gate 为 `208 passed`；在用户允许与现有任务共享且显存充足的 GPU4/5 上，两卡 NCCL gate 为 `3 passed`。静态 `ruff` 工具在冻结环境中不存在，未修改环境安装；`py_compile` 与 `git diff --check` 通过。

当前只完成实现与正确性验证，尚未产生正式性能数据，不更新 RESULTS 或作 wall-clock 改善声明。下一步是在相同 interval-200 paired timing 几何下先测 60M/130M，记录 compressed-step launch 数、完整周期 step time 与 peak allocated，再决定是否增加持久 workspace 或 batched refresh SVD。

## 2026-09-12：CM054 batched 60M paired timing 计划

CM054 严格复用 CM047 的 60M 几何与计时口径：dim512/4 layers/8 heads、4-rank DDP、BF16 compile、FineWeb10B、seq256、global/device batch512/128、GA1、rank32、interval200、bucket160 MiB、seed42；每个 cell 为 20 warmup + 200 measured updates，dense/local-SVD 做 3 次 rotated-order pairing。唯一方法实现差异是本日志上一节的 ordinary-step same-shape batching 与 factor direct-write。

为避免现有共享任务污染 wall-clock，复用 `run_greedy_lore_profiler.sh` 的动态 GPU gate，等待任意四张 `memory.used < 1024 MiB` 的卡，并在每个 cell 前重新检查。正式实验设 `profile_modes=none`，只生成 6 个 timing cell，不采集 Kineto trace；完成后再依据 paired repeats 比较 CM054 内 dense/local-SVD，并把两者分别与 CM047 的同期旧实现结果作历史参考。跨实验差值只解释为实现优化信号，不替代 CM054 内配对统计。

12:32 在 tmux `cm054_m002_60m_batched` 启动 controller；启动后 artifact plan 已核验为上述 6 个 rotated timing cells，首次 gate 记录 `idle_count=1 required=4`，当前处于自动等待。启动时 `dion/greedy_lore_ddp_hook.py` SHA-256 为 `8d4f9ac73bd9c40e4030617d19b2b0107d64159c2b852cbb6e0c8bd593f150ee`；状态日志为 `artifacts/compressed_muon/CM054-m002-gpt60m-batched-interval200-ddp-ws4-s42/status.log`。

### CM054 完成结果

controller 于 13:29 获得 GPU `2,4,6,7`，并于 13:35 完成。6/6 timing cells exit `0`，每个日志的末尾 `step_avg` 均为有限正值，只有 dense 日志 step0 的计时初始化 marker 为预期 `nan`；stderr 仅含 torchrun 的 OMP 提示和 tqdm 输出，无 OOM、timeout 或 traceback。`summary.json` 的 profile cells 为空，artifact 内 trace 数为 0。

- dense：`109.86/112.36/110.70 ms`，mean `110.973 ms`，CV `1.15%`，peak `6865 MiB`。
- batched local-SVD：`110.52/110.54/113.08 ms`，mean `111.380 ms`，CV `1.32%`，peak `7345 MiB`。
- paired local-SVD minus dense：`+0.66/-1.82/+2.38 ms`，mean `+0.407 ms` / `+0.377%`，bootstrap mean-difference 95% interval `[-1.82,2.38] ms`。

结果分类为 **null**。相对历史 CM047，CM054 dense 从 `113.167` 降至 `110.973 ms`（`-1.94%`），local-SVD 从 `113.360` 降至 `111.380 ms`（`-1.75%`）；两模式同向下降，且 CM054 内 local-SVD 相对 dense 的差值仍为零附近，因此不能把绝对下降归因于 batching。local-SVD peak 从历史 `7347` 变为 `7345 MiB`，2 MiB 差异无实用意义。当前证据表明 same-shape batching/direct-write 在 60M 上正确但没有可测的完整周期 wall-clock 收益；若继续性能诊断，应先采集单个普通 compressed step 的轻量算子/allocator 证据，确认 `stack` 临时量与 launch reduction 的实际抵消关系，再决定是否投入持久 workspace，或转向更可能受益的 130M/350M。

artifact：`artifacts/compressed_muon/CM054-m002-gpt60m-batched-interval200-ddp-ws4-s42/`。

## 2026-09-12：CM058/CM059 factor direct-write 复测

撤销 ordinary-step same-shape batching 后，仅保留逐矩阵 factor 直接写入一次分配的 packed collective buffer。CM058/CM059 分别严格复用 CM047/CM049 的 60M/130M 几何：4-rank DDP、BF16 compile、FineWeb10B、seq256、global/device batch512/128、GA1、rank32、interval200、bucket160 MiB、seed42；每组 3 次 rotated pairing、20 warmup + 200 measured updates、timing-only。

- CM058 60M：dense `109.17/113.76/113.59 ms`，mean `112.173 ms`；local-SVD `113.81/114.62/114.26 ms`，mean `114.230 ms`。paired difference `+4.64/+0.86/+0.67 ms`，mean `+2.057 ms` / `+1.865%`，interval `[0.67,4.64] ms`；peak `6865/7347 MiB`。
- CM059 130M：dense `239.78/240.07/238.73 ms`，mean `239.527 ms`；local-SVD `234.79/236.64/235.53 ms`，mean `235.653 ms`。paired difference `-4.99/-3.43/-3.20 ms`，mean `-3.873 ms` / `-1.617%`，interval `[-4.99,-3.20] ms`；peak `13048/13889 MiB`。

两组共 12/12 cells exit `0`、0 traces，日志无 OOM/timeout/traceback；dense step0 的 `nan` 仅为计时初始化 marker。相对原始逐矩阵 CM047/CM049，direct-write local-SVD 分别慢 `0.77%/0.38%`；连同 CM057 350M 的 `+0.98%`，没有证据支持 direct-write 带来收益。跨实验变化不能确诊其本身造成回退，但下一版恢复 batching 前的完整原始 factor clone+cat 路径，不再保留该优化。

artifacts：`artifacts/compressed_muon/CM058-m002-gpt60m-factor-direct-interval200-ddp-ws4-s42/`、`artifacts/compressed_muon/CM059-m002-gpt130m-factor-direct-interval200-ddp-ws4-s42/`。

按三种规模均无 direct-write 收益证据的结果，随后完整撤销 factor direct-write；`dion/greedy_lore_ddp_hook.py` 与 `7834661^` 的差异为空，即 ordinary compressed step 已恢复 batching 前的逐矩阵 factor 分配、clone 和最终 `cat`。TDD storage 回归先在 direct-write 版本上按预期失败，再在恢复后通过；完整 GreedyLore CPU/Future/Gloo gate 为 `186 passed`，两卡 NCCL gate 为 `3 passed`。

## 2026-09-12：CM060–CM062 GPT-1B / global batch512 四卡预检

固定 dim1536/30 layers/24 heads、4×RTX 4090、BF16、seq256、global/device batch512/1（GA128）、rank32、bucket160 MiB、seed42。由于 device batch1 已是下限，先用 interval2、1 warmup + 2 measured update 的无 trace 预检覆盖 refresh/compressed 路径；该窗口只用于显存 gate，不用于 wall-clock 结论。

- CM060 compile：dense exit0、peak 20219 MiB；M002 在首次 compiled forward OOM，申请 148 MiB 时仅余 16.56 MiB。
- CM061 compile + `expandable_segments:True`：dense exit0、peak 20217 MiB；M002 申请 144 MiB 时仅余 110.56 MiB，未分配 reserve 仅 74.21 MiB，说明并非只靠 allocator fragmentation 即可解决。
- CM062 no-compile + expandable segments：dense exit0、peak 21108 MiB；M002 在首次 backward OOM，申请 296 MiB 时仅余 174.62 MiB。

结论：恢复后的原始 M002 在 24 GiB 四卡、global batch512、seq256 下无法进入正式 1B timing；关闭 compile 也不能容纳。未启动 interval200 正式 run，后续若保持四卡需先引入 activation checkpointing 等显存策略并对 dense/M002 同时重建基线。

## 2026-09-12：P0 profiler 归因修复与 CM063 分相 timing

P0 发现 `profiler_trace.py` 会把仅在时间上落入 GreedyLore local CPU range 的并发 backward GPU kernel误归到 score等local类别，并进一步低估backward overlap、污染tail。TDD先构造“backward kernel执行于score时间窗、但launch关联到另一个CPU op/thread”的反例，旧实现按预期将 score GPU 从20 us误报为50 us且 overlap 为0；最小修复删除GPU时间窗local fallback，只接受关联到同pid/tid、且嵌套在local annotation内的CPU launch/op。新旧相关测试为 `21 passed`。CM049完整trace使用修正版的后台重解析单核耗时较长，其未完成结果不写入正式结论。

P1 不在训练循环插入每step同步或Kineto，而以两个等长profiler-off窗口差分：interval100的200 measured updates含2次refresh，interval200含1次，因而总时间差等于 `refresh - ordinary`。3组交替配对均exit0；A/B分别为 `239.06/232.15`、`239.51/232.99`、`244.66/231.25 ms`，推得ordinary `225.24/226.47/217.84 ms`、refresh `1607.24/1530.47/2899.84 ms`，均值 `223.18/2012.52 ms`。refresh相对ordinary按interval200摊销为 `6.91/6.52/13.41 ms/update`，mean `8.95 ms/update`。第三组差分波动较大，数字只用于确认refresh是主要量级瓶颈，不作精确wall-clock分解。

artifact：`artifacts/compressed_muon/CM063-*-m002-gpt130m-phase-timing-*/`。

## 2026-09-13：CM052c/CM053c high-rank 完整训练计划

为检查 rank 是否是 CM052b/CM053b 约 8.7% perplexity 退化的主要来源，只运行两个 GreedyLore-Muon cell，不重复 dense。CM052c 严格复制 CM052b 并仅把 rank32 改为 rank128；CM053c 严格复制 CM053b 并仅把 rank32 改为 rank256。两者继续使用 local-SVD、interval200、step1000 开始压缩、error feedback、bucket160 MiB、seed1234，以及各自原有的 10,000/20,000 updates 和 1,000/2,000 warmup。

`benchmark/compressed_muon/run_cm052c_cm053c_m002_rank_scaling_quality.sh` 先对两个模型执行覆盖 refresh/compressed 路径的 3-step probe，再按 CM052c→CM053c 串行正式训练；任一 probe 或正式 cell 失败即停止。controller 动态等待任意 4 张至少有 18 GiB 空闲显存的 GPU，不启用 Kineto trace，不设置基于历史 dense 的自动 quality gate。结果分别与已有 CM052a/CM053a dense 和 CM052b/CM053b rank32 进行历史对比。

延时会话于 2026-09-13 00:40:25（Asia/Shanghai）创建为 tmux `cm052c_cm053c_high_rank`，目标启动时间为 01:32:01，与本次请求首次记录时间相隔一小时。创建时 W&B 已认证，数据、torchrun 和三个目标 artifact 路径均通过预检；GPU 2–7 各约有 24.1 GiB 空闲，GPU 0–1 上已有进程且不会被选择。到点后的实际资源仍由 controller 重新检查，不满足门槛时自动等待。

## 2026-09-13：CM063b/CM063c bucket80 跨规模复测

在原始逐矩阵 M002 local-SVD 上固定 rank32、interval200 和 seed42，以 80 MiB bucket 对 60M/350M 各做 3 组 dense/M002 rotated pairing。60M dense/M002 mean 为 `100.277/102.647 ms`，M002 慢 `2.37%`，peak `6865/7182 MiB`；350M M002 稳定在 `206.90–208.07 ms`，前两组相对 dense 快 `6.46%/6.15%`，第三组 dense 异常偏慢，因此只保留约 `6%–7%` 的稳健加速信号，不采用 `10.37%` 均值作 claim。80 MiB 不是跨规模通用默认值：60M 为负向，350M 也弱于 CM051 的 bucket160 结果。

## 2026-09-13：CM064 130M bucket-cap 粗扫

串行扫描 `80/160/256/384 MiB`，每点各跑一次 interval100/200、各覆盖 200 measured update，并交替两种 interval 顺序。8/8 cells exit0，无 OOM。interval200 分别为 `228.86/233.85/248.16/250.02 ms`；对应差分 ordinary 估计为 `223.75/224.49/238.95/242.73 ms`，refresh 摊销为 `5.11/9.36/9.21/7.29 ms/update`。单次扫描中 80 MiB 比 160 MiB 快 `4.99 ms`（`2.13%`），据此只把后续候选范围收窄到 `64–128 MiB`；不为单点粗扫赋予正式性能 claim，也不把该结论外推到 60M/350M。

artifact：`artifacts/compressed_muon/CM064-m002-gpt130m-bucket-cap-sweep-ws4-s42/`。

## 2026-09-13：CM049 ordinary trace 修正版聚焦重解析

P0 parser 修复后，聚焦重解析 CM049 的 12 份 dense ordinary 与 12 份 GreedyLore ordinary trace。GreedyLore local GPU mean 为 score `4.675 ms`、Top-r `0.925 ms`、factor `0.578 ms`、error `0.960 ms`、reconstruction `0.603 ms`，合计约 `7.741 ms`；score 是 ordinary local 算术最大单项。dense/GreedyLore 的 exposed NCCL mean 为 `61.123/53.892 ms`，Muon-result collective 为 `8.834/8.694 ms`；GreedyLore score+dense-aux/factor collective 为 `41.535/3.817 ms`。

这只是 12+12 trace 的内存聚焦样本，不冒充未完成的 48-trace 全量汇总。Kineto profile window 对 M002 扰动很大（dense/GreedyLore `240.719/283.761 ms`），所以端到端结论继续以 profiler-off CM049 timing 为准；修正版 trace 只用于热点排序和通信量级判断。结合 CM054/CM058–CM059，ordinary 的 batching/factor packing 已无收益证据；下一项 ordinary 优化若继续，应先围绕 score 收集 allocator 与 kernel-launch 证据。CM064 的 cap80 refresh 摊销约 `5.11 ms/update`，说明 refresh 优化仍是较低风险方向，但完整消除的理论上限也只有该点周期时间约 `2.2%`。

## 2026-09-13：通信 dtype 审计与 CM065

对照 `/home/wyr/greedy_lore/comm_hooks/subspace_hook.py` 后确认：参考 hook 直接以 `bucket.buffer()` 的 dtype 构造并 All-Reduce dense auxiliary 与低秩工作区，即策略是“跟随 DDP bucket dtype”，而不是无条件 BF16。参考 launcher 虽写有 `dtype=bfloat16`，但本地 snapshot 缺失它引用的 C4 training source，无法仅凭 launcher 判断参数存储 dtype 或实际 bucket dtype。Dion 当前是 FP32 参数/gradient bucket 加 BF16 autocast，所以此前默认通信实际为 FP32。

实现新增 `greedy_lore_dense_aux_communication_dtype={bucket,float32,bfloat16}`，默认 `bucket` 保持参考 hook 的 dtype 策略与现有行为；显式 `bfloat16` 只把普通 compressed step 的 packed `score+dense_aux` buffer 在 All-Reduce 前降为 BF16。score 在 Top-r 前转回原 FP32 dtype，dense auxiliary 写回 gradient view 时转回 bucket dtype；factor、refresh 和 dense-only 路径不变。训练入口、YAML 与通用 profiler launcher 都记录该选项，并新增 CM065 串行 timing wrapper。

CM065 固定 GPT-130M、4卡、seq256、global/device batch512/128、rank32、interval200、bucket80 MiB，FP32/BF16 各三次。step time 分别为 FP32 `226.72/228.49/230.06 ms`、BF16 `221.08/221.07/218.76 ms`；全样本 mean 改善 `8.12 ms`（`3.55%`），严格相邻的 r1/r2 mean 改善 `6.53 ms`（`2.87%`），peak allocated 均为 `13841 MiB`。r3 因 controller 在 FP32 完成后遇到 launcher 编辑导致的 parse interruption，BF16 是同配置补跑，证据等级低于前两组。六个训练 cell 都 exit0，无 OOM/训练 traceback；短 val loss 仅为健康检查。

该结果没有达到预设 `>=5%`（约 `11.44 ms`）实用门槛，因此保留 BF16 为可组合的显式实验选项，但不将其设为默认或作为独立高优先级方向。下一项若要满足十余毫秒收益目标，应把重点放在全局 Future/跨 bucket pipeline 与 exposed communication 的重叠，BF16 只作为组合变量；在没有新 profile 前不把 `8.12 ms` 端到端差值等同于 NCCL 时间下降。

## 2026-09-13：bucket-native 模型参数 dtype 与 CM066 计划

按 `/home/wyr/greedy_lore` 的 bucket-dtype 语义补齐整模型参数 dtype 开关。训练入口新增 `model_dtype={float32,bfloat16}`，默认 `float32`，因此普通 dense、其他压缩方法和既有配置均保持原行为；当前 BF16 参数模式只允许 DDP，不扩展到 DTensor/FSDP。模型在初始化权重前整体转换 dtype，覆盖 embedding、lm_head 和全部 Transformer 参数；由此 gradient 与 DDP bucket 自然同 dtype。Muon matrix state、Lion/AdamW scalar state 延续参数 dtype，学习率与 step 控制量保持 FP32。

GreedyLore 的 error、basis、score、dense_aux 与 factor 默认全部跟随 bucket dtype；refresh SVD、符号规范化等数值敏感步骤临时升到 FP32，结果再转换回 bucket dtype。显式 `greedy_lore_dense_aux_communication_dtype` override 继续保留，默认 `bucket`。checkpoint metadata 记录模型 dtype，同 dtype continuation 可恢复，跨 dtype resume 在复制 payload 前明确失败，避免静默转换。

TDD/验证覆盖整模型 materialization、BF16 Muon/Lion/AdamW state、FP32/BF16 GreedyLore core/hook/profiler、两 rank Gloo collective 与 checkpoint roundtrip、两卡 NCCL BF16 payload。分项结果分别为训练入口 `29 passed`、CUDA optimizer `109 passed, 17 skipped`、GreedyLore core/hook `70 passed`、distributed/checkpoint `40 passed`、NCCL BF16 `1 passed, 3 deselected`；最终覆盖训练入口、Muon、GreedyLore、layout、profiler、distributed、checkpoint 与 CM066 launcher 的 fresh 综合 gate 为 `307 passed`。按用户要求暂不执行真实训练 smoke 或正式实验。

CM066 登记四个全新 GPT-130M/seed1234/bucket80 MiB cell：CM066a/b 是 FP32 dense/GreedyLore，CM066c/d 是 BF16 dense/GreedyLore。四者统一 20,000 updates、warmup2,000、rank32、interval200、step1000 开始压缩；controller 串行 fail-fast，先为每个 cell 跑覆盖 refresh/compressed 的 3-step probe。分析只允许 CM066a↔CM066b、CM066c↔CM066d 的同 dtype 配对，绝不把 BF16 参数结果与旧 FP32 dense 基线拼接。当前仅登记为 planned，尚无质量或性能结果，因此不更新 `RESULTS.md`。

## 2026-09-13：CM067 bucket-native BF16 跨规模 timing 计划

在 CM066 完整质量矩阵之前先做 timing-only 检查。CM067 显式使用 BF16 整模型参数，因此 gradient、DDP bucket 以及默认 `bucket` 模式下的 GreedyLore dense_aux/score/factor/error/basis 均为 BF16。固定 4-rank DDP、seq256、global/device batch512/128、bucket80 MiB、rank32、interval200、seed42；每个 timing cell 为 20 warmup + 800 measured updates，覆盖四个完整 interval，并做 3 组 dense/GreedyLore rotated pairing。

顺序为 60M（dim512/4 layers/8 heads）、130M（dim768/8 layers/12 heads），最后以完全相同 batch 尝试 350M（dim1024/20 layers/16 heads）。60M/130M 任一失败即停止；350M 的 OOM 或其他失败只记录，不令 controller 失败，并在主序列尚未触达 GreedyLore 时补一次 GreedyLore-only 尝试。通用 profiler launcher 新增默认 `float32` 的 `--model-dtype` 参数，既有调用保持不变；CM067 wrapper 才显式选择 `bfloat16`。运行前 shell/plan 与既有 launcher 回归为 `13 passed`。

22:03 在 GPU 2–5 启动 tmux `cm067_bf16_scale_timing`，启动时 commit 为 `79a3f32`；首个 60M dense cell 已进入执行。artifact 根目录为 `artifacts/compressed_muon/CM067-m002-bucket-native-bf16-scale-timing-ws4-s42/`。

### CM067 完成结果

Controller 于 22:38 完成并 exit `0`。60M/130M 共 12/12 timing cells exit `0`；dense 与 GreedyLore 的 CV 均低于 `0.15%`。60M dense/M002 为 `92.043/94.343 ms`，逐组差 `+2.27/+2.45/+2.18 ms`，M002 mean 慢 `2.300 ms`（`+2.50%`），paired bootstrap mean-difference interval `[2.18,2.45] ms`；peak allocated `6058/6303 MiB`。130M dense/M002 为 `199.543/204.203 ms`，逐组差 `+4.69/+4.52/+4.77 ms`，M002 mean 慢 `4.660 ms`（`+2.34%`），paired interval `[4.52,4.77] ms`；peak `11208/11633 MiB`。

350M 在同一 `512/128` batch 下两模式均 OOM。dense 完成初始 validation 后在训练分配 256 MiB 时每卡仅余约 62.56 MiB；GreedyLore fallback 在首次训练前分配 64 MiB 时仅余约 40.56 MiB。按预登记策略不继续重试，也不改变 batch。结论为 **negative**：整模型 BF16 明显降低绝对 step time 和显存，但在严格同 dtype 配对中，当前 bucket80/rank32/interval200 的 GreedyLore 在 60M/130M 都稳定慢约 2%–3%，没有速度收益。820-step 的 val loss 仅作健康检查，不作质量结论。

汇总时发现 `timing-summary.json` 的 scope 文本把 800 updates 硬编码描述成 one period；根因是 summarizer 未根据 `measured_updates / interval` 生成周期数。新增多周期回归并修正为 `4 complete interval-200 periods`，相关 summarizer/launcher gate 为 `30 passed`，不改变任何原始 timing 数值。

## 2026-09-14：CM068 BF16 bucket 粗扫

在 GPT-130M bucket-native BF16、4卡、seq256、global/device batch512/128、rank32、interval200、seed42 下，以单次交替顺序扫描 `24/32/40/48/64/80 MiB`；每个 cap 各跑 dense/M002 的 20 warmup + 200 measured update。12/12 cells exit0。dense 为 `188.84/194.15/192.42/193.98/194.20/200.28 ms`，M002 为 `201.23/203.10/200.99/205.11/203.02/204.57 ms`，对应 M002 相对差为 `+6.56/+4.61/+4.45/+5.74/+4.54/+2.14%`。

40 MiB 是 M002 绝对最快点，比 80 MiB 快 `3.58 ms`，但所有 cap 下 M002 仍慢于同 cap dense；80 MiB 又复现 CM067。由此排除“只需把 BF16 bucket80 按元素数折半至40即可恢复加速”的解释。每点只有一次，24 MiB dense 低值仅作粗扫信号。

## 2026-09-14：CM069 BF16 targeted profile

对 bucket40/80 各采 dense/M002 refresh 与 ordinary，共 8 cells、32 traces，全部成功。ordinary dense/M002 exposed NCCL 分别为 `20.701/29.400 ms`（40 MiB）和 `32.525/35.043 ms`（80 MiB）；40 MiB M002 另有 `14.151 ms` dense-only collective。M002 ordinary local GPU 合计为 `5.862/4.669 ms`，其中 score `3.020/2.293 ms`；refresh SVD GPU 为 `1508.717/1503.761 ms`，interval200 简单摊销约 `7.5 ms/update`。M002 targeted-collective/backward overlap 指标两点均为0。

这解释了 BF16 的相对回退：dense payload 已减半且仍能利用 backward overlap，M002 两阶段串行 Future 没有把剩余 collective 有效隐藏，暴露通信没有低于 dense，同时继续支付 ordinary local 与 refresh 成本。Kineto 对 M002 callback CPU range 扰动显著，profile 仅用于上述热点/重叠诊断；端到端仍以 CM067/CM068 profiler-off timing 为准，各分量不可直接相加。

## 2026-09-14：CM070 BF16 参数 paper-aligned 完整训练

单一 best-effort controller 先完成 CM069，再依次运行 CM052a/b、CM053a/b、CM052c、CM053c 的 BF16 参数副本。无 probe、无60M自动 quality gate；任一 cell 失败不阻断后续。实际六个 formal cell 均 exit0。除 `model_dtype=bfloat16` 与 artifact/W&B 名称外，训练参数读取原 YAML，继续使用 bucket160 MiB。

- 60M：dense `loss4.1079, 93.05 ms, 6058 MiB`；rank32 `4.2119, 94.82 ms, 6303 MiB`，相对 dense perplexity `+10.96%`、step `+1.90%`；rank128 `4.1230, 95.53 ms, 6311 MiB`，perplexity `+1.52%`、step `+2.67%`。
- 130M：dense `loss3.6530, 210.56 ms, 11208 MiB`；rank32 `3.7645, 215.04 ms, 11633 MiB`，相对 dense perplexity `+11.80%`、step `+2.13%`；rank256 `3.6618, 219.71 ms, 11633 MiB`，perplexity `+0.88%`、step `+4.35%`。

high-rank 基本恢复相对 BF16 dense 的质量，却进一步牺牲速度。BF16 dense 相对历史 FP32 CM052a/CM053a 的 perplexity 也高约 `11.36%/8.12%`，表明没有 FP32 master weights 的整模型 BF16 本身存在不可忽略的质量代价。当前 BF16 路径只适合作为 dtype ablation，不替代 FP32 paper-aligned 主结果。

artifacts：`CM068-m002-gpt130m-bf16-bucket-cap-sweep-ws4-s42/`、`CM069-m002-gpt130m-bf16-bucket{40,80}-targeted-profile-ws4-s42/`、`CM070{a,b,c,d,e,f}-*/`。

## 2026-09-14：CM071 论文 micro-batch/GA 口径跨规模 timing

为消除 CM067 将每卡 effective batch128 直接作为单次 physical batch 的口径差异，CM071 固定 micro-batch/device `32`、gradient accumulation `4`，从而保持 effective/device `128`、4卡 global batch `512`。其余复用 CM067：BF16 参数/gradient/DDP bucket、seq256、bucket80 MiB、rank32、interval200、local-SVD、seed42；60M/130M/350M 各做 3 组 dense/GreedyLore rotated pairing，每个 cell 为 20 warmup + 800 measured optimizer updates。

当前空闲 GPU 为 2,3,6,7，0/1 与 4/5 上有其他用户进程，因此 controller 显式选择 `2,3,6,7`。60M/130M fail-fast，350M best-effort；不启用 profiler/W&B，不运行额外脚本测试，先用 `--print-plan` 核对批量口径后在 tmux `cm071_paper_ga4_timing` 启动。

### CM071 完成结果

Controller 于 13:53 完成并 exit `0`，18/18 timing cells exit `0`，350M 在 micro-batch32/GA4 下不再 OOM。60M dense/M002 mean 为 `92.050/95.670 ms`，M002 慢 `3.620 ms`（`+3.94%`）；130M 为 `216.103/219.147 ms`，慢 `3.043 ms`（`+1.41%`）；350M 为 `597.190/632.520 ms`，慢 `35.330 ms`（`+5.92%`）。三个规模的逐组配对差值均为正，结论为 negative。

GA4 将 60M dense/M002 peak 降至 `2036/2281 MiB`、130M 降至 `3827/4250 MiB`，350M 为 `9627/11023 MiB`。相对 CM067 GA1，60M dense step 几乎不变，130M dense 慢约 `8.3%`；但两次实验卡组不同（CM067 GPU2–5，CM071 GPU2,3,6,7），不把差值解释为纯 GA 效应。核心结论不依赖这项跨实验比较：按论文 micro-batch `32 × GA4` 运行后，本地 step time 仍远低于论文 Table V，batch 口径不足以解释接近十倍的绝对差异，GreedyLore 也仍未快于 dense。

artifact：`artifacts/compressed_muon/CM071-m002-paper-microbatch-ga4-bf16-scale-timing-ws4-s42/`。

## 2026-09-14：ordinary bucket 第一阶段流水化实现

针对 CM069 中 ordinary compressed bucket 的两阶段 collective 暴露问题，实现了不改变数学结果与跨 rank collective 顺序的第一阶段流水线。长期存在的 `GreedyLoreDDPState` 现在分别维护 collective 提交尾部与全部 bucket 完成尾部；每次 hook 调用的 `BucketContext` 则快照本 bucket 的 ready event、独立准备结果、collective completion 和 DDP completion。ordinary bucket 在 ready 后立即于 preparation stream 计算 corrected gradient、score 与 packing，collective 仍严格按 `score(A) → factor(A) → score(B) → factor(B)` 排序，但 A 的 factor collective 完成后即可在 reconstruction stream 重建，同时通信路径开始 B 的 score All-Reduce。warmup、refresh 与 dense fallback 保留原串行路径。

实现同时补齐了跨 stream tensor 生命周期与失败语义：准备、通信、重建之间使用 CUDA event/stream wait，不引入设备级 synchronize；相关 tensor 通过 `record_stream` 防止 allocator 提前复用；collective 失败会毒化后续 collective 链并完成对应 DDP Future，而重建启动或异步重建失败只令该 bucket 的 DDP Future 失败，不阻断后续 rank 一致的 collective 提交。step 结束等待的是所有 bucket completion 的聚合 Future，而不是仅等待最后一个 bucket。

正确性回归覆盖 eager preparation、A 重建与 B score overlap、collective/reconstruction 失败传播、stream lifetime、collective 签名及原有 checkpoint/distributed 路径。CPU/Gloo 综合测试为 `148 passed`，两卡 NCCL smoke 为 `4 passed`；代码审查发现的同步重建异常导致 collective 链中断与缺少 `record_stream` 两项问题均已修复。按本阶段要求未运行正式 timing 或 profiler，因此当前只确认调度与正确性，不对 step time、暴露通信或峰值显存作性能 claim；正式对照实验留待后续。

### CM072 第一阶段流水化 timing

随后以 CM067 的 GPT-130M BF16/GA1 口径直接运行 profiler-off timing：4-rank DDP、GPU 2,3,6,7、seq256、global/device batch512/128、bucket80 MiB、rank32、interval200、seed42；dense/M002 做 3 组 rotated pairing，每 cell 为 20 warmup + 800 measured updates。6/6 cells exit0。dense 为 `197.20/203.37/201.71 ms`，mean `200.76 ms`、CV `1.59%`；流水化 M002 为 `203.18/203.38/203.40 ms`，mean `203.32 ms`、CV `0.06%`。逐组 M002-dense 差为 `+5.98/+0.01/+1.69 ms`，paired mean 为 `+2.56 ms`（`+1.29%`），bootstrap mean-difference interval `[+0.01,+5.98] ms`。peak allocated 为 dense/M002 `11207/11632 MiB`。

当前 M002 仍未快于同轮 dense。与旧 CM067 同配置的历史值相比，M002 绝对均值从 `204.20` 降至 `203.32 ms`（约 `0.88 ms`, `0.43%`），dense gap 从 `4.66` 缩至 `2.56 ms`（约 `2.10 ms`）；但 CM072 dense 自身波动明显高于 CM067，且 GPU 组合不同，因此这里只记为“小幅改善信号”，不能把历史差分全部归因于流水化。若要判断 exposed communication 是否真正下降，需要另做 targeted profiler 或同环境 A/B 旧实现复测。

artifact：`artifacts/compressed_muon/CM072-m002-gpt130m-bf16-ordinary-pipeline-timing-ws4-s42/`。

### CM073 第一阶段流水化 targeted profile

在 CM072 同一 130M BF16/GA1、bucket80 MiB 配置上，以 GPU 2,3,6,7 串行采集 dense/M002 refresh 与 ordinary，共 4 cells、16 traces，全部成功。流水化 M002 ordinary 的 profile window/NCCL union/exposed NCCL 为 `268.732/49.103/47.766 ms`，dense ordinary 为 `194.772/45.731/29.234 ms`；标准 targeted-collective/backward overlap 仍为0。M002 score+dense_aux/factor collective 为 `31.422/9.244 ms`，本地 reconstruction 为 `0.454 ms`。

直接检查四个 rank 的 GPU event 区间后，reconstruction 与任何 BF16 AllReduce 的实际交集均为 `0 ms`；score preparation 与 reconstruction 仅在 rank0/rank2 有 `0.549/0.457 ms` 交集，rank1/rank3 为0。说明实现虽然解除了“下一 bucket collective 必须等待前一 bucket reconstruction”的依赖，但当前 bucket readiness、约2 ms的 score preparation 与亚毫秒 reconstruction 时长没有给下一次 score AllReduce 留出实际重叠窗口。CM073 与 CM069 卡组不同且 collective 时长波动较大，不能用两次 exposed NCCL 的绝对差异归因流水化回退；本轮主要结论是目标 collective overlap 未发生。

artifact：`artifacts/compressed_muon/CM073-m002-gpt130m-bf16-ordinary-pipeline-targeted-profile-ws4-s42/`。

#### mixed bucket 与 embedding/head 压缩评估

进一步审计 CM073 bucket metadata：三个 bucket 中 A/C 为 mixed，分别含 `77.27 MiB dense_aux + 7.08 MiB matrix` 与 `77.27 MiB dense_aux + 21.23 MiB matrix`；B 为 `84.93 MiB` matrix-only。mixed bucket 占总 gradient bytes `68.28%`，dense_aux 占 `57.71%`。两个77.27 MiB张量对应 `transformer.wte.weight` 与 `lm_head.weight`，因当前通信 role 直接取自 Muon param group，它们由 AdamW 更新并走 dense AllReduce。DDP 不理解优化器/压缩角色，只按注册顺序和 soft cap 聚合不可切分的参数，因此形成 mixed bucket。cap40 历史 trace 能隔离其中一个大参数，但另一个仍与28.31 MiB matrix组成105.58 MiB mixed bucket，说明只调 cap 不足以稳定控制 composition。

记录的后续优先级为：先用参数名审计和注册顺序配合40/64 MiB cap，把两个大参数尽量变成 dense-only bucket；hook 内拆分 score/dense_aux 在单 communicator 下只能隐藏本地 factor preparation，不能消除 dense_aux 对 collective/Future 尾部的阻塞；跨 bucket score/factor 交错只在所有 rank 的下一 bucket 都足够早 ready 后再实现。

embedding/head 也进行 GreedyLore 通信压缩在接口上可行：解耦 optimizer role 与 communication role，继续用 AdamW 更新，只让 hook 压缩其二维梯度。两者为约 `50304×768`，rank32 ordinary factor 合计约 `6.44 MB`，相对当前 `154.53 MB` dense payload 有很大通信缩减潜力；代价是约147 MiB额外 full-size error state、宽矩阵 corrected/factor 临时量，以及每次refresh新增两次 `768×50304` FP32 SVD。该路径还直接扰动 embedding/logit 梯度，质量风险高；实际实现按后续决策只提供同时控制两者的 combined 开关，并将宽矩阵 refresh 改为只求左子空间的 Gram-eigh。

## 2026-09-14：embedding 与 lm-head 同时压缩实现及 CM074/CM075 启动准备

按后续决策只提供一个 `greedy_lore_compress_embedding_lm_head` 开关，不提供独立 embedding/head 开关。默认值为 false；启用时仅把 `transformer.wte.weight` 与 `lm_head.weight` 加入 GreedyLore 通信角色集合，原 Muon/AdamW optimizer param groups 完全不变。现有 hook 的 matrix state、ordinary score/factor/error/reconstruction 和 checkpoint schema 因此可直接覆盖两个新增二维参数，旧布局 checkpoint 继续通过 fingerprint fail closed。

为避免 refresh 对 canonical `512/768 × 50304` 宽矩阵生成巨大 FP32 `Vh`，当 columns 大于 rows 的4倍时，basis refresh 改为 FP32 `X X^T` 的 `eigh`，按特征值降序排列并沿用确定性符号规范；其他矩阵保留原 SVD。相关 CPU/checkpoint 回归为 `125 passed`，两卡 BF16 NCCL smoke 为 `1 passed`，controller 与通用 launcher 通过 `bash -n` 和 print-plan 检查。CM074 将在130M BF16/bucket80上做3组 dense/both rotated timing；成功后同一controller运行CM075的60M BF16、10,000-step完整训练。

### CM074/CM075 完成结果

串行 controller 使用 GPU 4–7 完成 CM074，随后在当时空闲的 GPU 2–5 完成 CM075；controller 与两个实验均 exit `0`。CM074 的三组 dense/both 配对为 `200.25/203.63`、`195.48/201.94`、`196.84/202.09 ms`，dense/both mean 为 `197.523/202.553 ms`，both 平均慢 `5.030 ms (+2.55%)`，paired interval 为 `[3.38,6.46] ms`。dense/both peak 为 `11207/11782 MiB`，吞吐为 `663.6K/647.1K tokens/s`。因此，即使理论 ordinary payload 从约159 MB降到约11 MB，扩大压缩范围仍未在130M端到端 timing 中超过 dense；结合宽矩阵新增的 full-basis score、factor、error 与 reconstruction GEMM，本地计算是后续 profiler 的首要怀疑对象，但 CM074 本身没有 trace，不能作严格归因。

CM075 在与 CM070a/b 相同的 GPU 2–5、BF16 GPT-60M paper-aligned 配方上完成10,000步，最终 val loss `4.2622`、ppl `70.97`、step `90.27 ms`、peak `6318 MiB`，W&B run 为 `ueijo55s`。历史同配方 BF16 dense/仅block压缩的 loss 为 `4.1079/4.2119`、ppl为 `60.82/67.48`；both 相对两者 ppl 分别高 `16.68%/5.16%`。压缩刚开始后 loss 在 step1000/1500 从 `4.8892` 升到 `5.1938`，此后恢复下降，说明训练未发散，但最终质量进一步劣化。CM075 的绝对 step 比历史 dense/block-only 快约 `2.99%/4.80%`，但这是跨日期、跨代码版本的非同期比较，与 CM074 的同期负结果方向相反，不能作为速度收益结论。

结论：rank32 下同时压缩 embedding/lm-head 为 **quality-negative 且没有同期 wall-clock 收益**，不应原样进入更大规模训练。artifact：`artifacts/compressed_muon/CM074-m002-greedylore-both-gpt130m-bf16-timing-ws4-s42/`、`artifacts/compressed_muon/CM075-m002-greedylore-both-gpt60m-bf16-ddp-ws4-s1234/`。

## 2026-09-15：dense auxiliary role-aligned DDP bucket 实现

按 CM073 mixed-bucket 审计后的第一优先级，实现默认关闭的 `greedy_lore_isolate_dense_aux_buckets`。该选项不改变 GPT module tree、参数名、forward、Muon/AdamW 参数组或 GreedyLore 数学路径，而是在 DDP 构造前生成 `bucket_cap_mb_list`：按 steady-state backward ready 顺序，首尾分别使用 lm-head/embedding 的精确参数字节数，中间按普通 `bucket_cap_mb` 将反向遍历的完整 Transformer block 聚合成精确边界。当前 GPT 参数注册顺序若不再是 embedding、blocks、lm-head，则直接拒绝启动；该选项也与 `greedy_lore_compress_embedding_lm_head` 互斥。

共享 `train.main` 新增 entry-specific DDP kwargs factory seam，其他训练入口默认行为不变。GreedyLore profiler 的 bucket metadata 现在额外记录稳定参数名和 matrix/dense_aux role，通用 launcher 也能通过 `--greedy-lore-isolate-dense-aux-buckets` 生成后续 profile/timing 命令。该可选路径依赖 PyTorch 2.11 的 `bucket_cap_mb_list`；项目整体版本下限保持不变，旧版运行时只有启用该开关才会以明确错误 fail closed。真实两 rank CPU/Gloo 回归覆盖 DDP rebuild，并加入非均匀 block 大小以防 cap 顺序被对称模型掩盖；连续两轮 rebuild 后的 bucket 均保持 `lm_head-only -> whole-block-only -> embedding-only`，两个 rank 完全一致。按本轮要求未启动 GPU profiler、timing 或训练，因此当前只证明布局机制和 rebuild 行为，不声称 NCCL overlap 或 wall-clock 改善。

### CM076 dense auxiliary bucket 隔离 timing

随后复用 CM074 的 GPT-130M BF16/GA1 timing 配方，在 GPU 2–5 启用 `greedy_lore_isolate_dense_aux_buckets`，不启用 embedding/lm-head 压缩。前两组 rotated dense/M002 配对均完成：r1 为 `199.03/203.77 ms`，r2 为 `200.36/203.08 ms`；有效配对均值为 `199.695/203.425 ms`，隔离版 M002 慢 `3.730 ms (+1.87%)`。第三个 dense cell 完成于 `200.17 ms`，但对应 M002 r3 在首次 compiled forward 申请 `3.07 GiB` 时因 GPU 2 仅余 `2.14 GiB` 而 OOM；失败后 GPU 2–4 可见其他任务占用，且前两个同配置 M002 cell 已成功，因此记录为外部资源争用。

实验最终为5/6 cells成功，没有正式 `summary.json`。结论按两组有效同期配对记为 `partial/negative`：物理隔离 dense auxiliary bucket 没有显现端到端收益，阶段性慢 `1.87%`；不使用未配对的 r3 dense 计算均值或区间。artifact：`artifacts/compressed_muon/CM076-m002-greedylore-dense-aux-isolation-gpt130m-bf16-timing-ws4-s42/`。

### CM077 targeted profile：实际仅完成 partial isolation

在 GPU 2–5 复用 CM076 配置，串行采集 dense/M002 refresh 与 ordinary；4/4 cells exit0，共生成16份 rank trace。profiler 新增逐 bucket 的 gradient-ready、collective-unblocked 和 Future-complete 时间线，以实际 reducer bucket 而非预期 cap 推断布局。

trace 显示 M002 的四个 rank、refresh/ordinary 均只有3桶：`73.6875 MiB` 的 `lm_head.weight` dense-only、`69.75 MiB` matrix-only，以及 `111.9375 MiB` 的末桶（`38.25 MiB` matrix + `73.6875 MiB` `transformer.wte.weight`）。原 cap 序列 `[73.6875, 67.5, 40.5, 73.6875] MiB` 没有产生预期四桶：按完整 block 估计的边界与真实 backward-ready 顺序不匹配，DDP 又把 cap 当作 soft threshold并按实际顺序装入不可切分参数；matrix-only 桶在 `69.75 MiB` 才越过 `67.5 MiB`，余下 matrix 的 `38.25 MiB` 未达到下一个 `40.5 MiB` cap，因而继续并入 embedding。由此追溯修正 CM076：它测得的是当前代码的 partial isolation，而不是两个 auxiliary 参数的完整物理隔离。

ordinary rank-max 的 dense/M002 exposed NCCL 为 `28.899/26.943 ms`，M002 只节省约 `1.956 ms`。M002 本地 score/Top-r/factor/error/reconstruction 分别为 `2.244/0.753/0.341/0.767/0.682 ms`，各类别 rank-max 合计约 `4.788 ms`；collective rank-max 的 dense/factor/score+aux/Muon-result 分别为 `12.958/4.854/15.899/7.029 ms`。有限的 exposed NCCL 改善与不可忽略的本地工作共同提示压缩开销可能抵消通信收益，方向上与 CM076 profiler-off 的 `+3.730 ms (+1.87%)` 一致；但它们不是同一 rank 上互斥的关键路径分量，不能机械相加或据此完成定量归因。Kineto 还将 M002 ordinary profile window 放大到 `269.879 ms`（dense 为 `195.788 ms`），该差值不能当作真实 step regression。

M002 refresh 的 local-SVD GPU 为 `1508.984 ms`，interval200 简单摊销约 `7.54 ms/update`；critical-path tail 为 `641.749 ms`。dense refresh 单样本的 NCCL union `163.156 ms` 明显偏离 ordinary 的 `45.774 ms`，不用于精确 refresh 对照。末尾 embedding mixed bucket 在 backward 后段 ready，Future 约在 backward CPU range 结束前 `1.1 ms` 完成，仍是晚期路径。

采样点还有首调用偏差：`start_compress_step=20`，而 refresh/ordinary profile step 正是 `20/21`，分别为第一次 refresh 和第一次 ordinary compressed 分支；rank0 首个 score/Top-r CPU range 约为 `87.8/49.0 ms`，显示初始化或 profiler 冷启动污染。因而本轮 bucket composition 是可靠结构证据，时间分解则只作定性诊断。

结论：先把采样移至第二个压缩周期（step220/221，或等价的分支预热后采样），并增加 prepare begin/end、chain-wait begin 标记，以拆分本地准备与真实依赖等待；再依据实际 reducer 顺序修正或替代 cap 构造，并在 compiled GPU trace 中验证真实四桶布局。最终用同批次 rotated dense/current-partial/fixed-full 三臂 profiler-off timing 做因果消融，同时分开统计 ordinary 分布与 refresh spike。在布局和稳态时间验证前，不把当前结果外推为“完整隔离无效”。artifact：`artifacts/compressed_muon/CM077-m002-gpt130m-bf16-dense-aux-isolation-targeted-profile-ws4-s42/`。

## 2026-09-15：稳态 profile、calibrated full isolation 与 CM078 启动

针对 CM077 的两个证据缺口完成可选诊断路径。GreedyLore hook 新增 per-rank rebuilt bucket layout JSON、指定 active step capture、role-aligned fail-closed 验证和预期 bucket count 验证；默认路径不启用这些诊断。calibration helper 将所有 rank 一致的真实 reducer bucket 顺序展开，按 dense auxiliary singleton 和不超过目标 cap 的连续 matrix 组重新生成精确 cap，避免硬编码 CM077 的 MiB 数字。正式 full-isolation cell 在第二个 optimizer step 起检查实际 bucket 数、matrix/dense_aux 不混桶，以及每个 dense auxiliary 均为 singleton；不满足即停止，不生成性能结论。

profiler marker 新增 per-bucket `prepare_begin/end` 与 `chain_wait_begin/end`，offline summary 记录 prepare/chain-wait 相对 ready 的时延，并在每个 rank 内计算 GreedyLore local GPU union、与 backward compute overlap 及 exposed local GPU，再在 cell 层取 rank-max。通用 launcher 新增 profile period 参数；CM078 使用第二周期 step220 refresh 与 step221 ordinary，避免 CM077 step20/21 的压缩分支首调用污染。

CM078 controller 于2026-09-15在 tmux `CM078-m002-full-isolation` 启动。它自行等待4张至少 `23500 MiB` 空闲显存的 GPU，先运行 compiled step2 calibration，再串行运行 dense/current-partial/calibrated-full 三臂的3组 rotated profiler-off timing（20 warmup + 800 measured），最后采集三臂稳态 refresh/ordinary profile。按用户要求不运行测试脚本；启动前仅完成 Python `py_compile`、两个 launcher 的 `bash -n`、profile step220/221 print-plan、`git diff --check` 和一次 GPU/磁盘检查，启动后不由 Agent 轮询。artifact：`artifacts/compressed_muon/CM078-m002-gpt130m-bf16-dense-aux-full-isolation-timing-profile-ws4-s42/`。

### CM078 timing 完成（当时 profile summary 尚在运行）

calibration 在 GPU 2–5 成功捕获四个 rank 一致的 compiled step2 reducer 顺序，并导出 `[73.6875,78.75,29.25,73.6875] MiB` caps。四个目标 bucket 为 lm-head singleton dense-only、`78.75/29.25 MiB` 两个 matrix-only、embedding singleton dense-only；正式 full-isolation timing/profile cells 的 fail-closed 检查均通过。

三臂 rotated timing 9/9 cells exit0。dense 三组为 `197.93/198.16/199.79 ms`，mean `198.627 ms`；current partial 为 `203.97/203.36/202.87 ms`，mean `203.400 ms`，相对 dense `+4.773 ms (+2.41%)`；calibrated full 为 `202.89/204.05/204.37 ms`，mean `203.770 ms`，相对 dense `+5.143 ms (+2.59%)`。full−partial 三组为 `-1.08/+0.69/+1.50 ms`，mean `+0.370 ms (+0.18%)`、区间跨零。full isolation 因此没有显示端到端收益，且不能把 CM076 的负结果主要归因于 embedding mixed bucket。dense/partial/full peak 为 `11207/11475/11447 MiB`。

step220/221 的6个 profile cells 也全部 exit0并生成24份 trace，但总量约 `1.02 GiB`，offline summarizer 截至本次记录仍以单核运行，尚无正式 `profile/summary.json`。暂不从单份 trace 抽取时间归因，实验状态记为 `timing-completed/profile-summary-pending`；待汇总完成后再追加稳态 prepare、chain wait、exposed local GPU、NCCL 与 refresh 结论。

## 2026-09-15：CM079/CM080 最大可行 device batch timing 启动

为检验模型增大后 M002 的通信收益边界，登记 CM079（GPT-350M）与 CM080（GPT-1B），统一使用4卡、BF16参数、seq256、GA1、bucket80 MiB、rank32、interval200、local-SVD、seed42。每个模型先让显存更严格的 M002 以 interval2、1 warmup + 2 measured 的短进程覆盖 refresh/ordinary 分支，按预设离散候选从大到小粗探测：350M 为 `128/96/72/64/48/32/24/16/8/4/2/1`，1B 为 `32/24/16/12/8/4/2/1`；首个可行值再用 dense 复核。该值是候选集中的最大共同可行 batch，不追求逐整数显存极限。global batch 随之设为 `4×device batch`，不以 GA 保持512。

正式 timing 对每个可行模型运行3组 dense/M002 rotated pairing，每 cell 为20 warmup + 800 measured updates；CM079结束后才开始CM080。controller 自行等待4张至少23500 MiB空闲显存的GPU，并在选中卡被外部任务占用时继续等待。按用户要求不运行脚本正确性测试、不生成 profiler trace，也不由 Agent 轮询。controller：`benchmark/compressed_muon/run_cm079_cm080_max_device_batch_timing.sh`；artifacts：`artifacts/compressed_muon/CM079-m002-gpt350m-bf16-max-device-batch-timing-ws4-s42/`、`artifacts/compressed_muon/CM080-m002-gpt1b-bf16-max-device-batch-timing-ws4-s42/`。

### CM078 稳态 trace 汇总完成

offline summarizer 于16:53完成，正式 `profile/summary.json` 约213 KiB。ordinary 的 dense/partial/full exposed NCCL 为 `32.931/20.850/23.593 ms`，partial/full local GPU union 为 `6.868/6.139 ms`、与 backward overlap 为 `2.756/2.273 ms`、exposed local 为 `4.262/3.915 ms`；dense gradient-sync tail 为 `25.125 ms`，两种M002均为0。partial/full各rank跨bucket prepare sum约`9.689–10.651/9.755–9.986 ms`，chain-wait sum仅`0.111–0.130/0.145–0.154 ms`，说明score preparation而非上一bucket chain wait是ordinary本地等待主项。

稳态 refresh 的 partial/full local-SVD GPU 为 `1569.157/1565.804 ms`，均无 backward GPU overlap，按interval200粗摊为 `7.846/7.829 ms/update`。ordinary exposed local与refresh摊销量级足以消耗通信节省，与profiler-off负结果方向一致；但rank-max不作机械加减，单样本Kineto window也不替代三重复timing。CM078最终状态更新为`completed/negative+diagnostic`：full isolation没有降低partial的端到端或通信时间，后续不再优先优化bucket chain，转向score与refresh复杂度。

### CM079/CM080 完成结果

controller 在GPU 2–5串行完成并exit `0`。CM079中350M的M002在device batch128/96 OOM，72通过且dense复核通过；正式6/6 cells exit0，dense/M002 mean为`370.310/408.727 ms`，逐组差值`+36.09/+42.30/+36.86 ms`，M002平均慢`38.417 ms (+10.38%)`，peak为`17218/18613 MiB`。CM080中1B的batch32实际为Triton CUDA OOM（controller状态文本因匹配式未覆盖该写法而记作`PROBE_FAILED`），batch24两模式通过；正式6/6 cells exit0，dense/M002为`518.720/632.973 ms`，逐组差值`+106.61/+127.13/+109.02 ms`，平均慢`114.253 ms (+22.04%)`，peak为`16603/20948 MiB`。

两组均为明确negative。跨130M/350M/1B的相对回退约为`2.6%/10.4%/22.0%`，但batch分别为128/72/24，不能解释成纯参数量效应。现实现ordinary score的完整`basis.T @ corrected`与refresh FP32 SVD/eigh近似按`layers×hidden³`增长，快于按参数量增长的通信节省；error/factor/reconstruction还遍历完整矩阵。模型越大时M002额外状态又迫使physical batch降低，使这些与batch无关的成本占比进一步上升。此前bucket isolation、流水化与buffer改进没有改变这两个主导复杂度，因此难以扭转趋势。下一步优先做共同device batch24的受控尺度比较、interval分相timing，以及固定support/projector诊断；算法方向转向避免每步扫描完整basis和降低/错峰refresh。
## 2026-09-15：CM081 shared-vector score 与 NCCL transport/bandwidth

- 目的：检验 ordinary step 的完整 independent-vector score 是否为 GPT-1B timing 退化的主要来源。
- 变体：增加 `score_randomization=shared`，以一个共享 Gaussian vector 计算所有 basis score；该变体不与论文 Algorithm 2 等价，仅作为性能消融。
- 实验：GPT-1B BF16、4 GPU、seq256、global/device batch96/24、GA1、bucket80 MiB、rank32、interval200；按用户缩减后的范围只正式运行 shared 三次，每 cell 为20 warmup + 800 measured updates，并探索性对照 CM080 历史 dense/independent。
- 通信诊断：正式 timing 前串行捕获 NCCL transport、PCIe/P2P topology，以及 BF16 1/32/80 MiB All-Reduce bandwidth；诊断不与 timing 并发。
- 结果：controller、NCCL诊断及3/3 shared cells均exit `0`。shared step为`599.02/599.33/608.39 ms`，mean `602.247 ms`、CV `0.884%`、throughput `40.81K tokens/s`、peak `20948 MiB`。相对CM080历史independent的`632.973 ms`快`30.727 ms (4.85%)`，但相对历史dense的`518.720 ms`仍慢`83.527 ms (16.10%)`；非同期配对，不计算区间。短窗口validation loss mean为`5.4894`，不作质量等价结论。
- NCCL：实际两个channel均为`SHM/direct/direct`。BF16 1/32/80 MiB All-Reduce分别为`0.190/3.698/9.132 ms`，algorithm bandwidth为`5.51/9.07/9.19 GB/s`，ring-equivalent bus bandwidth为`8.26/13.61/13.78 GB/s`。产物JSON字段后缀`gbps`为误标，数值实际单位是GB/s。
- 结论：shared score证明完整independent-vector score是1B额外开销之一，但只收回历史dense差距的约`26.9%`，没有扭转端到端负结果；SHM transport大类已与论文描述对齐。下一步若继续，优先做shared ordinary/refresh分相，再决定是否进行质量实验。
- 产物：`artifacts/compressed_muon/CM081-m002-shared-score-gpt1b-nccl-timing-ws4-s42/`。

## 2026-09-15：CM082 shared phase timing 与 gradient-sync oracle

- 目的：用 profiler-off interval差分拆分1B shared路径的ordinary与refresh摊销，并测量当前配置下DDP gradient sync可消除时间的上限。
- 配置：GPT-1B BF16、4 GPU、seq256、global/device batch96/24、bucket80 MiB、rank32；只新增shared interval800与no-gradient-sync oracle各1个20 warmup + 800 measured cell，dense与shared interval200分别复用CM080/CM081历史summary，由单一controller串行执行。
- 方法：假设`T(I)=T_ordinary+R_refresh/I`，由interval200/800解出ordinary和单次refresh额外时间。oracle通过benchmark-only DDP identity hook跳过gradient All-Reduce，但保留Muon optimizer/result communication；允许rank参数分叉，只作timing上限。
- 进展：首次controller在选卡后、任何cell启动前因shell同一`local`声明引用未绑定`label`而exit1，失败根目录移为`-attempt1-controller-bug`。修复后shared interval800成功，step为`519.53 ms`、peak为`20948 MiB`；结合CM081 interval200的`602.247 ms`，二点模型估计shared ordinary为`491.96 ms`、单次refresh额外约`22.06 s`，即interval200/800分别摊销`110.29/27.57 ms/update`。interval800仅比CM080历史dense `518.720 ms`慢`0.81 ms (0.16%)`，单样本/跨实验不作等效声明，但`82.72 ms`的interval差异远大于CM081约5.3 ms样本标准差，refresh主导方向明确。
- Oracle：首次在DDP hook注册时因bucket注解为`Any`而失败；改为PyTorch要求的`dist.GradBucket`后重跑成功。no-gradient-sync step为`434.24 ms`、peak为`16603 MiB`，相对CM080历史dense低`84.48 ms`，作为当前配置下可消除gradient-sync时间的timing上限。该oracle允许各rank参数分叉，只用于性能归因。
- 结论：completed/diagnostic。shared估计ordinary为`491.958 ms`，比历史dense快约`5.16%`，但单次refresh额外约`22.06 s`，在interval200下完全抵消ordinary收益；interval800仅比历史dense慢`0.81 ms (0.16%)`。所有对照均为跨实验单样本，不计算区间或宣称等效。
- 产物：`artifacts/compressed_muon/CM082-m002-shared-phase-gradient-sync-oracle-gpt1b-ws4-s42/`。

## 2026-09-15：CM083 GPT-1B FP32 shared score timing

- 目的：检查参数、gradient、DDP bucket与GreedyLore状态均为FP32时，CM081/CM082观察到的shared score改善和refresh主导趋势是否延续，并判断增加dense通信字节后interval800能否超过dense。
- dtype口径：`model_dtype=float32`，前向计算仍使用BF16 autocast；DDP bucket随参数/gradient为FP32；error/basis随参数为FP32，score/factor随corrected/bucket为FP32，`dense_aux_communication_dtype=bucket`保证packed communication不另行降精度。正式日志必须同时出现FP32模型、bucket通信dtype和shared score标记，否则controller失败关闭。
- 实验：device batch按`24/16/12/8/4/2/1`递减探测shared；找到首个可行点后运行dense、shared interval200、shared interval800各1个20 warmup + 800 measured cell，4 GPU、seq256、bucket80 MiB、rank32、seed42。
- 状态：completed/capacity-blocked；shared探测从device batch24递减到1均OOM，device batch1时每卡约占`23.50/23.52 GiB`，申请18 MiB BF16 activation失败。未进入dense、independent或shared正式timing，无summary；四rank一致且启动时满足空闲阈值，归为确定性容量不足。
- 产物：`artifacts/compressed_muon/CM083-m002-shared-score-gpt1b-fp32-max-batch-timing-ws4-s42/`。

## 2026-09-16：CM084 GPT-1B FP32 reduced-sequence timing

- 目的：降低sequence length以绕过CM083的FP32容量边界，同时保持1B模型、FP32参数/gradient/bucket/GreedyLore状态和BF16 autocast口径。
- 实验：先在seq128按device batch`8/4/2/1`递减探测shared；若全部OOM，再在seq64重复。首个可行组合串行运行dense、independent interval200、shared interval200、shared interval800，各1个20 warmup + 800 measured cell；4 GPU、bucket80 MiB、rank32、seed42。
- 状态：completed/capacity-blocked；seq128与seq64下device batch`8/4/2/1`均OOM，未进入正式timing。降低activation规模仍无法容纳1B FP32 M002，说明静态参数、gradient与压缩状态占用主导。
- 产物：`artifacts/compressed_muon/CM084-m002-shared-score-gpt1b-fp32-reduced-seq-timing-ws4-s42/`。

## 2026-09-16：CM085 GPT-350M FP32 score timing

- 目的：在CM083/CM084证明1B FP32受静态显存阻塞后，改用350M检验FP32 gradient/bucket通信量增加时，independent/shared score与refresh interval的端到端表现。
- 实验：GPT-350M（dim1024/20 layers/16 heads，354.7M参数）、seq256、4 GPU、bucket80 MiB、rank32、seed42；shared按device batch`64/48/32/24/16/8/4/2/1`递减探测，首个可行点串行运行dense、independent interval200、shared interval200、shared interval800，各1个20 warmup + 800 measured cell。
- 结果：修正后的350M在device/global batch64/256下完成4/4 timing cells。dense/independent200/shared200/shared800分别为`392.65/432.13/422.79/395.20 ms`，相对dense为`+10.05%/+7.68%/+0.65%`；peak为dense `19293 MiB`、三种M002 `22086 MiB`。shared差分估计ordinary `386.003 ms`、单次refresh额外`7357.333 ms`，即interval200/800摊销`36.787/9.197 ms/update`。
- 状态：completed/diagnostic-negative；每配置单样本，不计算区间。attempt1误设24层、实际405.0M，在batch probe阶段停止，保留为`-attempt1-wrong-depth`并排除出全部结论。
- 产物：`artifacts/compressed_muon/CM085-m002-score-gpt350m-fp32-max-batch-timing-ws4-s42/`。

## 2026-09-16：CM086 GPT-720M FP32 score timing

- 目的：在350M和1B之间增加规模点；选择dim1280/30 layers/20 heads（718.6M参数），与1B同为30层，以减少深度变化对趋势解释的干扰。
- 实验：seq256、4 GPU、FP32参数/gradient/bucket/GreedyLore状态、BF16 autocast、bucket80 MiB、rank32、seed42；shared按device batch`32/24/16/12/8/4/2/1`递减探测，首个可行点串行运行dense、independent interval200、shared interval200、shared interval800，各1个20 warmup + 800 measured cell。
- 结果：device batch`32/24/16` OOM、`12`通过，正式global/device batch48/12。dense/independent200/shared200/shared800分别为`515.79/449.70/427.06/363.27 ms`，相对dense快`12.81%/17.20%/29.57%`；peak为`14955/21138 MiB`。shared差分估计ordinary `342.007 ms`、单次refresh额外`17010.667 ms`，即interval200/800摊销`85.053/21.263 ms/update`。
- 状态：completed/diagnostic-positive；每配置单样本，不计算区间。720M FP32、小batch首次显示大幅端到端收益，但M002额外显存约`6.18 GiB`。
- 产物：`artifacts/compressed_muon/CM086-m002-score-gpt720m-fp32-max-batch-timing-ws4-s42/`。

## 2026-09-16：CM087/CM088 batch 与 bucket 归因

- CM087：当前代码、GPT-350M、FP32参数/gradient/bucket、BF16 autocast、seq256、global/device batch32/8、rank32、interval200；以bucket160和80分别运行dense/independent各一个20 warmup + 200 measured cell。bucket160对齐CM051以估计代码/环境变化，bucket160→80隔离bucket影响，再与CM085的batch64结果连接。
- CM088：GPT-720M保持CM086的FP32和bucket80口径，只将global/device batch从48/12降至32/8；运行dense、independent interval200、shared interval200/800各一个20 warmup + 800 measured cell，检验更小batch是否进一步扩大收益。
- CM087结果：4/4 cells成功。bucket160的dense/independent为`239.11/201.66 ms`，M002快`37.45 ms (15.66%)`，几乎复现CM051三重复的`15.97%`；bucket80为`252.70/215.93 ms`，M002快`36.77 ms (14.55%)`。这支持CM085的反转主要来自batch/工作负载几何而非代码退化；bucket差异只有单样本，不作因果claim。
- CM088结果：4/4 cells成功。batch8下dense/independent200/shared200/shared800为`458.74/428.41/416.94/352.90 ms`，相对dense快`6.61%/9.11%/23.07%`，反而小于CM086 batch12的`12.81%/17.20%/29.57%`。shared ordinary估计`331.553 ms`，refresh额外`17077.333 ms`，与batch12基本一致；固定M002成本使收益随batch变化并不单调。
- 状态：completed/diagnostic-positive；CM087/088均为探索性单样本，不计算重复区间。
- 产物：`artifacts/compressed_muon/CM087-m002-gpt350m-fp32-current-code-batch8-bucket-bridge-ws4-s42/`、`artifacts/compressed_muon/CM088-m002-gpt720m-fp32-device-batch8-timing-ws4-s42/`。

## 2026-09-16：sharded-SVD refresh 实现与 CM089–CM092 启动

- 动机：`local_svd` 在每个rank对相同全局平均corrected gradient重复完整basis分解；新增`basis_sync=sharded_svd`，按矩阵分解成本确定性LPT分配owner，各rank只计算自己的参数，再按稳定参数名顺序从对应owner广播完整basis。
- 生命周期：refresh hook仍完成corrected dense All-Reduce并正常返回bucket Future；全模型分片分解、basis广播、support更新和error清零集中放在backward之后、gradient clipping之前的`finish_step()`，避免跨bucket pending Future与现有tail形成依赖环。basis和error仍在所有rank复制，算法压缩位置不变；相对local-SVD新增完整basis广播payload。
- 验证：新增两rankGloo行为测试，确认分解延迟到`finish_step()`、两个rank各执行一个分解、广播顺序/字节和最终replicated basis一致；GreedyLore配置、单rank hook、两rankhook及训练入口共`69 passed`。另以2卡NCCL、BF16小模型运行4步，覆盖两次refresh，exit0、peak 252 MiB。未测试新增launcher脚本，遵循用户要求。
- 实验：2026-09-16在tmux `CM089-CM092-sharded-svd`启动单controller，使用GPU一次性选择且Agent不轮询。CM089复用CM078 full-isolation BF16几何；CM090复用CM085；CM091复用CM087的bucket160/80；CM092复用CM088的independent/shared及interval200/800。所有cell串行、`repeats=1`，不运行dense历史基线或profile。
- 结果：controller及7/7 cells均exit0。CM089为`198.67 ms`，较CM078历史full local-SVD低`2.50%`并与历史dense基本持平；CM090为`411.94 ms`，较CM085历史local-SVD低`4.67%`但仍比历史dense高`4.91%`；CM091 bucket160/80为`174.66/185.35 ms`，较历史local-SVD低`13.39%/14.16%`；CM092 independent200/shared200/shared800为`371.93/361.63/346.59 ms`，较历史对应local-SVD低`13.18%/13.27%/1.79%`。
- Refresh归因：由CM092同批shared interval200/800二点差分估计ordinary为`341.577 ms`、单次refresh额外`4010.667 ms`；CM088历史对应值为`331.553/17077.333 ms`。因此单次refresh缩短约`13.067 s (76.51%)`，接近四卡参数级分片的理想量级；ordinary的约`+10.02 ms`跨实验差异不归因为sharded-SVD，因为该模式只改变refresh步骤。
- 结论：completed/diagnostic-positive。分片计算在更大FP32模型上显著降低完整周期时间，验证了重复SVD是主要refresh瓶颈；CM085大physical batch下仍未超过dense，说明ordinary本地计算和通信比例仍决定最终盈亏。所有新cell均为单样本，与CM078/085/087/088的百分比属于历史跨实验比较，不给出置信区间或严格配对claim。
- 代码与产物：`dion/greedy_lore.py`、`dion/greedy_lore_ddp_hook.py`、`train_greedylore.py`、`benchmark/compressed_muon/run_greedy_lore_profiler.sh`、`benchmark/compressed_muon/run_cm089_cm092_sharded_svd_timing.sh`；controller产物为`artifacts/compressed_muon/CM089-CM092-sharded-svd-controller/`。
