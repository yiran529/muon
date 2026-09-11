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
