# M001 共享 ARC 同步层与 2×2 Benchmark 实施 Prompt

你是负责实现和执行实验的 agent。工作目录是 Dion 仓库根目录。请完整执行：

`docs/superpowers/plans/2026-09-04-arc-topk-adamw-muon-benchmark.md`

规格来源是：

- `AGENTS.md`
- `docs/compressed_muon/RESEARCH_GUIDE.md`
- `docs/worklog/M001-arc-topk-ef21m-muon.md` 中的 `### 推荐给后续 agent 的测试设置`

## 开始前

1. 完整阅读上述四个文件，以及适用的 `SKILL.md`；按仓库要求使用 TDD、执行计划和完成前验证流程。
2. 执行 `git status --short`、`git log --oneline -8`，记录初始 HEAD 和已有修改。保留所有非本任务改动，不得 reset、checkout、clean 或覆盖用户文件。
3. 不安装或升级依赖，不修改 PyTorch、CUDA、NCCL、Triton 或锁文件。
4. 先完成共享同步层、AdamW ARC、benchmark 和 CPU/Gloo 自动化测试；自动化测试通过前不得启动正式 GPU benchmark。

## 核心实现要求

- 不重写 `dion/arc_topk.py` 中已有的 projection、Top-K、scatter 或 EF21M 数学。
- 新增 `dion/arc_topk_sync.py`，统一负责：
  - 按首次出现顺序进行 `(shape, dtype)` 分组；
  - 所有 rank 都包含相同参数，局部 `grad=None` 使用零张量；
  - 初始化并传递 `arc_h_local`、`arc_g_local`、`arc_g_global`；
  - dense fallback；
  - task index/seed 语义；
  - 每 rank logical collective input bytes 统计。
- 修改 `ArcTopKMuon` 调用共享层；同步后的梯度仍严格进入现有 Muon momentum、Nesterov、Newton–Schulz、结果通信和参数更新路径。不要修改原始 `dion/muon.py` baseline。
- 新增纯 `ArcTopKAdamW`。它与 Muon ARC 压缩完全相同的 `model.transformer.h.parameters()`；embedding 和 lm_head 都走 dense All-Reduce。
- AdamW ARC 不得继承 `DistributedOrthoBase`，避免无意义地初始化正交化组件。
- 两个 ARC 消费端必须使用相同的分组函数、缺失梯度规则、ARC 调用和 collective 顺序，不能复制出两套近似实现。
- 保持 M001 checkpoint 兼容：旧状态缺少 `arc_start_compress_step` 时恢复为 `0`。

## TDD 与验证要求

严格按计划逐任务执行 RED → GREEN → 回归；每个 RED 都要确认是目标行为尚未实现导致，而不是测试拼写或环境错误。至少覆盖：

- 稳定 shape/dtype 分组；
- byte accounting 的手算值；
- 两 rank 首步 dense 初始化与 `ratio=1, eta=1` 等价性；
- rank-local missing gradient；
- 两个 optimizer 的 rank 参数一致性；
- AdamW moment/variance 数值；
- optimizer `state_dict()` 保存恢复；
- benchmark CLI、JSON schema、summary mean/std/CV 与三个 reduction ratio。

使用：

```bash
uv run --frozen --extra dev pytest ... -v
```

Gloo 测试若因沙箱禁止 localhost socket 而失败，申请执行该测试所需的最小权限；不要把环境限制误报为实现失败。

## GPU 执行要求

正式运行前执行：

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

只选择四张空闲 GPU。少于四张空闲时停止启动并记录阻塞，不得终止、暂停或干扰现有进程。长任务使用 tmux。

先执行 60M 的八个 smoke case：

```text
AdamW Dense / AdamW ARC / Muon Dense / Muon ARC
×
normal NCCL / NCCL_P2P_DISABLE=1,NCCL_SHM_DISABLE=0
```

smoke 必须验证退出码、finite loss/参数、四 rank checksum、collective signature、byte counters 和代表性 profiler 内容。全部通过后才登记并运行：

- CM002a–d：normal NCCL 2×2；
- CM003a–d：P2P disabled 2×2。

每个 cell 使用独立进程运行至少三次；每次至少 20 个不计时 warmup steps 和 100 个 measured steps。参数/计算 dtype 固定为 BF16，world size 4、local batch 1、sequence length 256、gradient accumulation 1、seed 42。ARC 固定 ratio 0.2、rank 4、eta 0.1、compression start 0；首步 dense 初始化和编译阶段不得进入正式计时。

测量区间不得执行 validation、checkpoint、W&B、tqdm 更新或同步文件 I/O。使用 CUDA events，并以四 rank 最大耗时作为关键路径；不能仅用未同步 CPU wall clock。Profiler 单独运行，不能混入正式 timing samples。

若任一 cell 的 step-time CV 超过 5%，先检查资源竞争、GPU 时钟/功耗和样本离群，再为受影响 cell 增加两个完整独立重复；不得静默删除离群值。

## 指标与结论

必须输出并归档：

- dense、sketch、selected values、uncompressed、Muon 其他通信字节；
- collective 类型、次数、消息大小和 NCCL GPU kernel 时间；
- forward/backward、optimizer、ARC projection/Top-K/EF21M、Muon Newton–Schulz 和完整 step 时间；
- tokens/s、峰值 allocated/reserved memory；
- 每组至少一个代表性 P2P-disabled profiler trace；
- repetitions 的均值、样本标准差和 CV；
- `R_bytes`、`R_grad_comm`、`R_step`。

严格应用 worklog 中预先约定的 5 个百分点判定方式。Profiler 若不能直接分离通信重叠，只报告直接测量值，并把 exposed communication 标记为推断。短 benchmark 不用于证明收敛、最终精度或 time-to-quality。

## 文档与汇报

- 原始产物放在 `artifacts/compressed_muon/<experiment-id>/`；可复用脚本放在 `benchmark/compressed_muon/`，不得放在仓库根目录。
- 启动正式实验时更新 `docs/compressed_muon/EXPERIMENTS.md`；无论成功、失败还是结论不明确，都在 `docs/worklog/M001-arc-topk-ef21m-muon.md` 追加中文记录。
- 只有结果稳定且具备正式比较价值时才更新 `RESULTS.md`。
- 60M 未通过正确性、指标完整性、CV≤5% 和 logical bytes 确实下降四个门槛时，不扩到 130M。130M、350M、1B 依次使用同样门槛和新的 CM 编号。

最终报告必须列出：完成内容、关键修改文件、提交、测试命令与 passed/failed/skipped、GPU 与 transport 证据、每个实验状态、主要指标、判定结论、原始产物路径、已知限制和下一步。不得只报告“跑通”或端到端 step time。
