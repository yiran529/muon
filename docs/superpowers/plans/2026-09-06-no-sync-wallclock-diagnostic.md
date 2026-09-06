# DDP no_sync Wall-Clock Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove and fix the backward-only `DDP.no_sync()` bug, then run a short serial GPT-350M dense-Muon versus optimizer-side ARC comparison under the corrected formal training entry.

**Architecture:** Add one reusable training-loop context selector whose context encloses both forward and backward. First characterize the historical and corrected DDP behaviors with a real two-rank Gloo test. Then launch two corrected, compiled, high-accumulation GPT-350M cells serially and preserve all commands, logs, exit codes, configuration, and timing output.

**Tech Stack:** Python 3.10, PyTorch 2.11 DDP/Gloo/NCCL, pytest, Bash, GNU timeout, existing FineWeb10B loader and GPT/Muon training entry points.

**Spec:** `docs/superpowers/specs/2026-09-06-arc-topk-ddp-bucket-hook-design.md`

## Global Constraints

- Follow strict red-green-refactor: no `train.py` behavior change before a real test fails for the expected reason.
- The corrected dense policy disables synchronization for the first `N-1` micro-batches and synchronizes the last.
- The corrected optimizer-side ARC policy disables DDP synchronization for all `N` micro-batches.
- Every `no_sync()` context must enclose both forward and backward.
- This plan does not implement the DDP ARC bucket hook.
- The GPU comparison is exploratory attribution evidence, not a formal performance claim: one serial run per cell and five timed optimizer steps are insufficient for a stable mean/CV.
- Do not mix historical CM019 results with corrected CM020 results.
- GPU cells dynamically select the lowest-numbered four idle GPUs from all visible devices, keep that same set for both cells, run serially, use normal NCCL, compile enabled, BF16, GPT-350M, sequence length 1024, device batch 1, global batch 1024, and seed 42.
- ARC uses ratio 0.2, projection rank 4, eta 0.1, and compression start 0 so all timed steps are compressed.
- The launcher must wait when fewer than four GPUs have less than 1024 MiB allocated. After selecting four GPUs, it must refuse to continue if any selected GPU reaches at least 1024 MiB before the ARC cell; missing data or a failed code/test gate also remains fail-closed.
- Primary wall-clock timing must not use `--time_optimizer`, because its explicit CUDA synchronizations perturb the measured critical path.

---

### Task 1: Characterize historical and corrected DDP synchronization

**Files:**
- Create: `benchmark/compressed_muon/no_sync_diagnostic.py`
- Create: `tests/test_no_sync_diagnostic.py`

**Interfaces:**
- Produces: `run_no_sync_case(pattern: str, world_size: int = 2) -> dict`
- Valid patterns: `backward_only`, `dense_correct`, `optimizer_correct`
- Produces JSON fields: `pattern`, `world_size`, `hook_calls_per_rank`, `pre_optimizer_gradients`, `gradient_ranges`, and `passed`.
- Later tasks use the diagnostic as the launcher’s CPU correctness gate.

- [ ] **Step 1: Write the failing real-distributed test**

Create a two-rank Gloo test using a one-layer bias-free `torch.nn.Linear(2, 1)` wrapped in real DDP. Register a communication hook that increments a counter, clones its local pre-reduction bucket input, performs SUM, divides by two, and returns `Future[Tensor]`.

Use two micro-batches with rank-distinct literal inputs. Exercise these literal policies:

```python
if pattern == "backward_only":
    loss = ddp(x).sum()
    with ddp.no_sync():
        loss.backward()
elif pattern == "dense_correct":
    context = ddp.no_sync() if micro_step == 0 else nullcontext()
    with context:
        ddp(x).sum().backward()
elif pattern == "optimizer_correct":
    with ddp.no_sync():
        ddp(x).sum().backward()
```

Assert literal outcomes:

```python
assert backward_only["hook_calls_per_rank"] == [2, 2]
assert dense_correct["hook_calls_per_rank"] == [1, 1]
assert optimizer_correct["hook_calls_per_rank"] == [0, 0]
assert dense_correct["gradient_ranges"]["max_abs"] == pytest.approx(0.0)
assert optimizer_correct["gradient_ranges"]["max_abs"] > 0.0
```

The production change that would make this test fail is moving forward outside the selected synchronization context or selecting the wrong final-micro-batch branch.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run --frozen --extra dev pytest tests/test_no_sync_diagnostic.py -v
```

Expected: FAIL because `benchmark.compressed_muon.no_sync_diagnostic` and `run_no_sync_case` do not exist.

- [ ] **Step 3: Implement the minimal diagnostic**

Implement `run_no_sync_case()` with `torch.multiprocessing.spawn`, a free localhost port, 30-second process-group timeout, and a queue or temporary JSON files for rank results. Keep model initialization identical across ranks and inputs deliberately different. Always destroy the process group in `finally`.

The CLI must run all three patterns serially:

```python
def main() -> int:
    results = [run_no_sync_case(name) for name in PATTERNS]
    payload = {"schema_version": 1, "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if all(item["passed"] for item in results) else 1
```

- [ ] **Step 4: Run the focused test and diagnostic**

Run:

```bash
uv run --frozen --extra dev pytest tests/test_no_sync_diagnostic.py -v
.venv/bin/python benchmark/compressed_muon/no_sync_diagnostic.py --output /tmp/no-sync-diagnostic.json
```

Expected: test PASS; command exits 0; JSON reports hook counts `2/1/0` for historical/dense-correct/optimizer-correct.

- [ ] **Step 5: Commit the characterization**

```bash
git add benchmark/compressed_muon/no_sync_diagnostic.py tests/test_no_sync_diagnostic.py
git commit -m "test: characterize DDP no_sync placement"
```

### Task 2: Fix the shared training loop

**Files:**
- Modify: `train.py:1-20`
- Modify: `train.py:1019-1047`
- Create: `tests/test_train_ddp_sync.py`
- Modify: `tests/test_train_arctopk.py`

**Interfaces:**
- Produces: `ddp_gradient_sync_context(model, *, micro_step: int, grad_accum_steps: int, optimizer_owns_gradient_sync: bool) -> ContextManager`
- Produces: `forward_backward_micro_step(model, x, y, *, autocast_ctx, micro_step: int, grad_accum_steps: int, optimizer_owns_gradient_sync: bool, before_backward: Optional[Callable[[], Any]] = None) -> tuple[Tensor, Any]`
- `micro_step` is one-based and must satisfy `1 <= micro_step <= grad_accum_steps`.
- `optimizer_owns_gradient_sync=True` corresponds to current `replicate_mesh_grad_sync=True` and disables reducer synchronization for every micro-batch.

- [ ] **Step 1: Write the failing context-policy test**

In a real two-rank Gloo test, import `forward_backward_micro_step` from `train`, run two micro-batches through that function, and assert:

```python
assert run_policy(optimizer_owns_gradient_sync=False) == [1, 1]
assert run_policy(optimizer_owns_gradient_sync=True) == [0, 0]
```

Also add literal argument validation tests:

```python
with pytest.raises(ValueError, match="micro_step"):
    ddp_gradient_sync_context(model, micro_step=0, grad_accum_steps=2,
                              optimizer_owns_gradient_sync=False)
```

The production mutations caught are using `<=` instead of `<` for dense accumulation, allowing the final ARC micro-step to sync, or moving forward outside the context inside `forward_backward_micro_step`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run --frozen --extra dev pytest tests/test_train_ddp_sync.py tests/test_train_arctopk.py -v
```

Expected: FAIL because `ddp_gradient_sync_context` and `forward_backward_micro_step` are absent.

- [ ] **Step 3: Add the minimal policy helper**

Import `nullcontext` and implement:

```python
def ddp_gradient_sync_context(
    model,
    *,
    micro_step: int,
    grad_accum_steps: int,
    optimizer_owns_gradient_sync: bool,
):
    if not 1 <= micro_step <= grad_accum_steps:
        raise ValueError(
            f"micro_step must be in [1, {grad_accum_steps}], got {micro_step}"
        )
    disable_sync = micro_step < grad_accum_steps or optimizer_owns_gradient_sync
    if isinstance(model, DDP) and disable_sync:
        return model.no_sync()
    return nullcontext()
```

- [ ] **Step 4: Add the tested complete micro-step function**

Implement the only forward/backward entry used by the shared loop:

```python
def forward_backward_micro_step(
    model,
    x,
    y,
    *,
    autocast_ctx,
    micro_step: int,
    grad_accum_steps: int,
    optimizer_owns_gradient_sync: bool,
    before_backward=None,
):
    with ddp_gradient_sync_context(
        model,
        micro_step=micro_step,
        grad_accum_steps=grad_accum_steps,
        optimizer_owns_gradient_sync=optimizer_owns_gradient_sync,
    ):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        callback_result = before_backward() if before_backward is not None else None
        loss.backward()
    return train_loss, callback_result
```

The optional callback preserves both the current next-batch fetch and the FSDP setters immediately before backward without duplicating the DDP context logic or changing data-loading overlap.

- [ ] **Step 5: Route the shared training loop through the function**

Replace the backward-only branch with this structure while retaining the existing FSDP flag logic:

```python
for i in range(1, grad_accum_steps + 1):
    def prepare_backward():
        next_batch = train_loader.next_batch()
        if isinstance(model, FSDPModule):
            model.set_is_last_backward(i == grad_accum_steps)
            if cli_args.fast_fsdp:
                model.set_reshard_after_backward(i == grad_accum_steps)
                model.set_requires_gradient_sync(i == grad_accum_steps)
            else:
                model.set_requires_gradient_sync(True)
        return next_batch

    train_loss, (x, y) = forward_backward_micro_step(
        model,
        x,
        y,
        autocast_ctx=autocast_ctx,
        micro_step=i,
        grad_accum_steps=grad_accum_steps,
        optimizer_owns_gradient_sync=hp.replicate_mesh_grad_sync,
        before_backward=prepare_backward,
    )
```

- [ ] **Step 6: Run focused and existing ARC tests**

Run:

```bash
uv run --frozen --extra dev pytest \
  tests/test_train_ddp_sync.py \
  tests/test_train_arctopk.py \
  tests/test_muon_arctopk.py \
  tests/test_muon_arctopk_distributed.py -v
```

Expected: PASS with no warnings or hanging subprocesses.

- [ ] **Step 7: Commit the training-loop fix**

```bash
git add train.py tests/test_train_ddp_sync.py tests/test_train_arctopk.py
git commit -m "fix: wrap DDP forward and backward in no_sync"
```

### Task 3: Add the corrected GPT-350M serial diagnostic launcher

**Files:**
- Create: `configs/compressed_muon/cm020a_dense_muon_gpt350m.yaml`
- Create: `configs/compressed_muon/cm020b_arc_muon_gpt350m.yaml`
- Create: `benchmark/compressed_muon/run_no_sync_wallclock_diagnostic.sh`
- Create: `tests/test_no_sync_wallclock_launcher.py`
- Modify: `docs/compressed_muon/EXPERIMENTS.md`
- Modify: `docs/worklog/M001-arc-topk-ef21m-muon.md`

**Interfaces:**
- Launcher writes controller state to `artifacts/compressed_muon/CM020-no-sync-wallclock-diagnostic/`.
- Cell artifacts are `CM020a-muon-dense-gpt350m-corrected-nosync` and `CM020b-m001-arc-muon-gpt350m-corrected-nosync`.
- Launcher runs CPU tests/diagnostic first, then CM020a and CM020b serially.
- Each cell records `command.txt`, `config.yaml`, `environment.txt`, `started_at.txt`, `finished_at.txt`, `exit_code.txt`, `stdout.log`, and `result.txt`.

- [ ] **Step 1: Write failing launcher contract tests**

Create `tests/test_no_sync_wallclock_launcher.py`. Run the launcher with a test-only `--print-plan` argument that performs no GPU work and emits JSON. Assert literal order and configuration:

```python
assert plan["cells"] == [
    "CM020a-muon-dense-gpt350m-corrected-nosync",
    "CM020b-m001-arc-muon-gpt350m-corrected-nosync",
]
assert plan["cuda_visible_devices"] == "dynamic"
assert plan["gpu_selection"] == {
    "scope": "all_visible",
    "count": 4,
    "max_memory_used_mib_exclusive": 1024,
    "poll_seconds": 60,
}
assert plan["model"] == {"dim": 1024, "layers": 20, "heads": 16}
assert plan["sequence_length"] == 1024
assert plan["batch_size"] == 1024
assert plan["device_batch_size"] == 1
assert plan["num_iterations"] == 15
assert plan["primary_uses_time_optimizer"] is False
```

The production changes caught are parallel cell launch, mismatched workloads, accidentally enabling perturbed timing, or running ARC before dense.

- [ ] **Step 2: Run launcher test and verify RED**

Run:

```bash
uv run --frozen --extra dev pytest tests/test_no_sync_wallclock_launcher.py -v
```

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Add two explicit diagnostic configs**

Both configs set:

```yaml
model_dim: 1024
n_layer: 20
n_head: 16
sequence_length: 1024
batch_size: 1024
device_batch_size: 1
num_iterations: 15
val_loss_every: 0
val_tokens: 4096
checkpoint_freq: 0
no_wandb: true
no_compile: false
no_triton: false
optimizer: muon
scalar_opt: adamw
adjust_lr: spectral_norm
mu: 0.95
weight_decay: 0.01
lr: 0.02
```

The ARC config changes `optimizer` to `arc_topk_muon`, sets `replicate_mesh_grad_sync: true`, `use_polar_express: true`, ratio `0.2`, projection rank `4`, eta `0.1`, seed `42`, and compression start `0`.

- [ ] **Step 4: Implement a fail-closed serial launcher**

The launcher must use `set -uo pipefail`, absolute repo/tool/data paths, and a one-hour timeout per cell. Its `--print-plan` path must return before checking GPUs or creating artifact directories.

Preflight order:

```text
focused pytest gate
CPU no_sync diagnostic and JSON gate
dataset directory check
GNU timeout check
wait for any four visible GPUs with memory usage <1024 MiB, then lock that set
```

Cell execution must remain serial:

```bash
run_cell dense || exit "$?"
preflight_selected_gpus || exit 78
run_cell arc || exit "$?"
```

Run each cell with normal NCCL and no W&B:

```bash
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 \
  "$ENTRY" --config "$CONFIG" --data_dir "$DATA_DIR" --no_wandb
```

Do not pass `--time_optimizer`. Extract the final `step_avg` line into `result.txt`, but retain raw stdout as the source of truth. A timeout, OOM, missing final timing, or nonzero exit must mark the cell failed and stop the queue.

- [ ] **Step 5: Pre-register exploratory experiments**

Append CM020a and CM020b rows to `docs/compressed_muon/EXPERIMENTS.md` with status `planned`, exact configuration, purpose, and artifact path. Append a worklog section stating that these are one-repeat, five-timed-step attribution probes and must not be reported as stable performance evidence.

- [ ] **Step 6: Run launcher tests and shell validation**

Run:

```bash
uv run --frozen --extra dev pytest tests/test_no_sync_wallclock_launcher.py -v
bash -n benchmark/compressed_muon/run_no_sync_wallclock_diagnostic.sh
benchmark/compressed_muon/run_no_sync_wallclock_diagnostic.sh --print-plan
```

Expected: PASS; JSON plan shows dense then ARC and no GPU work occurs.

- [ ] **Step 7: Commit the launcher and registrations**

```bash
git add \
  configs/compressed_muon/cm020a_dense_muon_gpt350m.yaml \
  configs/compressed_muon/cm020b_arc_muon_gpt350m.yaml \
  benchmark/compressed_muon/run_no_sync_wallclock_diagnostic.sh \
  tests/test_no_sync_wallclock_launcher.py \
  docs/compressed_muon/EXPERIMENTS.md \
  docs/worklog/M001-arc-topk-ef21m-muon.md
git commit -m "bench: add corrected no_sync wall-clock diagnostic"
```

### Task 4: Verify and launch the serial experiment

**Files:**
- Runtime output: `artifacts/compressed_muon/CM020-no-sync-wallclock-diagnostic/`
- Runtime output: `artifacts/compressed_muon/CM020a-muon-dense-gpt350m-corrected-nosync/`
- Runtime output: `artifacts/compressed_muon/CM020b-m001-arc-muon-gpt350m-corrected-nosync/`

**Interfaces:**
- Consumes the committed launcher and clean focused test suite.
- Produces a background PID plus durable controller/status logs.

- [ ] **Step 1: Run the complete CPU verification gate**

```bash
uv run --frozen --extra dev pytest \
  tests/test_no_sync_diagnostic.py \
  tests/test_train_ddp_sync.py \
  tests/test_train_arctopk.py \
  tests/test_muon_arctopk.py \
  tests/test_muon_arctopk_distributed.py \
  tests/test_no_sync_wallclock_launcher.py -v
```

Expected: all pass.

- [ ] **Step 2: Launch without foreground monitoring**

```bash
nohup benchmark/compressed_muon/run_no_sync_wallclock_diagnostic.sh \
  > artifacts/compressed_muon/CM020-no-sync-wallclock-diagnostic/controller.stdout \
  2>&1 &
```

Record `$!` in `controller.pid`. Do not poll continuously; the launcher itself writes status transitions and cell outputs.

- [ ] **Step 3: Perform one immediate startup check**

After launch, read `status.log` once. A valid handoff contains either `CPU_GATE START`, `GPU CELL START`, or a fail-closed `BLOCKED` reason. Report the PID and artifact paths to the user without waiting for experiment completion.

- [ ] **Step 4: Post-run follow-up boundary**

Do not update CM020 statuses or claim performance results until both `exit_code.txt` files and final timing lines exist. A later result-analysis turn will validate artifacts, calculate only exploratory ratios, update the registry/worklog, and decide whether the DDP bucket-hook implementation is justified.
