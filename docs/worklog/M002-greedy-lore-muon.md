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
