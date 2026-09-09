# ARC-TopK AdamW All-2D DDP Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a formally supported pure-AdamW training path that uses the existing ARC-TopK DDP hook for every two-dimensional parameter while preserving dense AdamW and existing Muon behavior.

**Architecture:** Make `ArcTopKDDPState` own only compressor lifecycle and remove its inspection of optimizer-private step state. Extract the current dense AdamW construction into one shared builder, then let `train_arctopk.py` select either ordinary Muon or ordinary AdamW before installing the same all-2D hook runtime. Keep optimizer-side `ArcTopKAdamW` unchanged as a benchmark control.

**Tech Stack:** Python 3.10, PyTorch DDP and distributed checkpointing, pytest, Gloo distributed tests, optional NCCL two-GPU smoke tests, YAML, Bash.

**Spec:** `docs/superpowers/specs/2026-09-09-arctopk-adamw-all2d-hook-design.md`

## Global Constraints

- Preserve existing `arc_topk_muon` optimizer-side and DDP-hook behavior.
- Preserve dense `optimizer=adamw` parameter groups and hyperparameters exactly, including zero weight decay on embedding and language-model-head groups.
- Compress every and only `parameter.ndim == 2` parameter in hook mode.
- Never allow DDP synchronization and optimizer-owned ARC synchronization at the same time.
- Do not modify ARC-TopK/EF21M mathematical primitives or optimizer-side `ArcTopKAdamW`.
- Keep DDP hook checkpoint schema and existing Muon-hook checkpoints compatible.
- Full paper-scale training is not part of implementation verification.

---

### Task 1: Remove optimizer-private step coupling from the hook state

**Files:**
- Modify: `dion/arc_topk_ddp_hook.py:92-228`
- Modify: `train_arctopk.py:188-204`
- Test: `tests/test_arc_topk_ddp_checkpoint.py:39-130`

**Interfaces:**
- Consumes: the existing `begin_step()`, `finish_step()`, `commit_step()`, `state_dict()`, and `load_state_dict()` lifecycle.
- Produces: `ArcTopKDDPState.__init__(self, *, process_group, fingerprint, parameter_specs, optimizer_parameters, config, find_unused_parameters=False)` with no `optimizer` argument and no inspection of optimizer state.

- [ ] **Step 1: Replace the Muon-step checkpoint test with compressor-boundary tests**

Remove `_StepOptimizer` and `test_snapshot_and_load_require_matching_committed_muon_step`. Add a test which completes one synthetic bucket, proves that snapshotting after `finish_step()` but before `commit_step()` fails, then commits and verifies `payload["shared"]["committed_step"] == 1`:

```python
def test_hook_state_constructor_has_no_optimizer_dependency():
    assert "optimizer" not in inspect.signature(ArcTopKDDPState).parameters


def test_snapshot_progress_is_owned_only_by_compressor_commit():
    state, matrix = _state()
    state.begin_step()
    context = state.note_bucket(
        _FakeBucket([matrix, state.parameter_specs[1].parameter])
    )
    context.completion_future.set_result(context.buffer)
    state.finish_step()
    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.state_dict()
    state.commit_step()
    assert state.state_dict()["shared"]["committed_step"] == 1
```

Update `_state()` so it no longer accepts or forwards `optimizer`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
.venv/bin/pytest -q tests/test_arc_topk_ddp_checkpoint.py
```

Expected: `test_hook_state_constructor_has_no_optimizer_dependency` fails because
`ArcTopKDDPState` still exposes the optional optimizer argument.

- [ ] **Step 3: Remove the optimizer dependency**

In `ArcTopKDDPState`:

```python
def __init__(
    self,
    *,
    process_group: Optional[ProcessGroup],
    fingerprint: str,
    parameter_specs: Sequence[ArcTopKDDPParameterSpec],
    optimizer_parameters: Sequence[Parameter],
    config: ArcTopKSyncConfig,
    find_unused_parameters: bool = False,
) -> None:
```

Delete `self.optimizer`, `_optimizer_step()`, and all optimizer-step comparisons
from `_require_committed_boundary()` and `load_state_dict()`. Keep the active
step and tail-Future checks. Remove `optimizer=optimizer` from
`train_arctopk.py`.

- [ ] **Step 4: Run hook checkpoint and state tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_arc_topk_ddp_checkpoint.py \
  tests/test_arc_topk_ddp_hook_state.py \
  tests/test_arc_topk_ddp_hook_future.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the independent lifecycle change**

```bash
git add dion/arc_topk_ddp_hook.py train_arctopk.py tests/test_arc_topk_ddp_checkpoint.py
git commit -m "refactor: decouple ARC hook state from optimizer steps"
```

---

### Task 2: Share dense AdamW construction between training entries

**Files:**
- Modify: `train.py:453-485`
- Modify: `train.py:698-709`
- Create: `tests/test_train_adamw_builder.py`

**Interfaces:**
- Consumes: existing `Hyperparameters`, already-created matrix/embedding/head parameter groups.
- Produces: `build_adamw_optimizer(param_groups: list[dict], hp: Hyperparameters) -> torch.optim.AdamW`.

- [ ] **Step 1: Add tests that freeze current dense AdamW semantics**

Create parameters and three representative groups, including explicit zero
weight decay on the latter two. Test the shared function directly:

```python
def test_build_adamw_optimizer_preserves_dense_training_semantics():
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    embedding = torch.nn.Parameter(torch.ones(3, 2))
    head = torch.nn.Parameter(torch.ones(2, 3))
    groups = [
        {"params": [matrix]},
        {"params": [embedding], "algorithm": "adamw", "weight_decay": 0.0},
        {"params": [head], "algorithm": "adamw", "weight_decay": 0.0},
    ]
    hp = train.Hyperparameters(lr=3e-4, weight_decay=0.1)
    optimizer = train.build_adamw_optimizer(groups, hp)
    assert type(optimizer) is torch.optim.AdamW
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4] * 3
    assert [group["betas"] for group in optimizer.param_groups] == [(0.9, 0.95)] * 3
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0, 0.0]
    assert [group["params"][0] for group in optimizer.param_groups] == [matrix, embedding, head]
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
.venv/bin/pytest -q tests/test_train_adamw_builder.py
```

Expected: failure because `train.build_adamw_optimizer` does not exist.

- [ ] **Step 3: Extract the behavior-preserving builder**

Add near `init_optimizer`:

```python
def build_adamw_optimizer(
    param_groups: list[dict], hp: Hyperparameters
) -> torch.optim.AdamW:
    print0("Using AdamW for all params, scalar optimizer will be ignored")
    print0("Setting all param groups to use unscaled base learning rate")
    for group in param_groups:
        group["lr"] = hp.lr
        group["betas"] = (0.9, 0.95)
    return torch.optim.AdamW(
        param_groups,
        lr=hp.lr,
        betas=(0.9, 0.95),
        weight_decay=hp.weight_decay,
    )
```

Replace the body of the existing `elif hp.optimizer == "adamw"` branch with
`opt = build_adamw_optimizer(param_groups, hp)`.

- [ ] **Step 4: Run focused and existing optimizer tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_train_adamw_builder.py \
  tests/test_dion3_alias.py \
  tests/test_adamw_foreach.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the shared builder**

```bash
git add train.py tests/test_train_adamw_builder.py
git commit -m "refactor: share standard AdamW optimizer construction"
```

---

### Task 3: Add the AdamW all-2D hook factory path

**Files:**
- Modify: `train_arctopk.py:29-212`
- Modify: `tests/test_train_arctopk.py`
- Modify: `tests/test_train_arctopk_ddp_hook.py:60-180`

**Interfaces:**
- Consumes: `train.build_adamw_optimizer`, `ArcTopKDDPState`, and the existing `GradientSyncRuntime` contract.
- Produces: `install_arc_topk_ddp_hook(model, ddp_model, optimizer, hp) -> train.GradientSyncRuntime` and the supported optimizer name `arc_topk_adamw`.

- [ ] **Step 1: Add parser/default and unsupported-combination tests**

Keep `arc_topk_muon` as the default. Add:

```python
def test_adamw_optimizer_side_mode_is_rejected():
    module = _module()
    with pytest.raises(ValueError, match="arc_topk_adamw.*ddp_hook"):
        module.init_arc_topk_optimizer(
            model=_StubModel(),
            device_mesh=None,
            ddp_model=_DDPStub(),
            hp=module.ArcTopKHyperparameters(
                optimizer="arc_topk_adamw",
                arc_sync_mode="optimizer",
            ),
            cli_args=_cli(),
        )
```

Also assert that any optimizer value outside `arc_topk_muon` and
`arc_topk_adamw` fails before construction.

- [ ] **Step 2: Add the AdamW hook construction test**

```python
def test_adamw_hook_mode_builds_standard_adamw_and_compresses_all_2d():
    module = _module()
    model = _StubModel()
    model.transformer.wte.scale = torch.nn.Parameter(torch.ones(8))
    ddp = _DDPStub()
    optimizer, runtime = module.init_arc_topk_optimizer(
        model=model,
        device_mesh=None,
        ddp_model=ddp,
        hp=module.ArcTopKHyperparameters(
            optimizer="arc_topk_adamw",
            arc_sync_mode="ddp_hook",
            lr=3e-4,
            weight_decay=0.1,
        ),
        cli_args=_cli(),
    )
    assert type(optimizer) is torch.optim.AdamW
    assert runtime.optimizer_owns_gradient_sync is False
    assert len(ddp.registrations) == 1
    state, hook = ddp.registrations[0]
    assert hook is module.arc_topk_ddp_hook
    roles = {spec.stable_name: spec.role for spec in state.parameter_specs}
    assert roles["transformer.h.weight"] == "arc_matrix"
    assert roles["transformer.wte.weight"] == "arc_matrix"
    assert roles["lm_head.weight"] == "arc_matrix"
    assert roles["transformer.wte.scale"] == "dense_aux"
```

- [ ] **Step 3: Run the new factory tests and verify they fail**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_train_arctopk.py \
  tests/test_train_arctopk_ddp_hook.py
```

Expected: failures because `arc_topk_adamw` is not recognized and hook setup is
still inlined after a hard-coded Muon construction.

- [ ] **Step 4: Extract one hook installer and add optimizer selection**

Introduce:

```python
def install_arc_topk_ddp_hook(
    model,
    ddp_model: DDP,
    optimizer: torch.optim.Optimizer,
    hp: ArcTopKHyperparameters,
) -> train.GradientSyncRuntime:
    config = ArcTopKSyncConfig(
        ratio=hp.arc_topk_ratio,
        projection_rank=hp.arc_projection_rank,
        eta=hp.arc_eta,
        seed=hp.arc_seed,
        start_compress_step=hp.arc_start_compress_step,
    )
    named_parameters = list(model.named_parameters())
    specs = tuple(
        ArcTopKDDPParameterSpec(
            parameter=parameter,
            stable_name=name,
            stable_id=stable_id,
            role="arc_matrix" if parameter.ndim == 2 else "dense_aux",
        )
        for stable_id, (name, parameter) in enumerate(named_parameters)
    )
    group_ranks = (
        tuple(dist.get_process_group_ranks(ddp_model.process_group))
        if ddp_model.process_group is not None
        else (0,)
    )
    fingerprint = canonical_arc_fingerprint(
        base_seed=config.seed,
        config=config,
        group_ranks=group_ranks,
        parameters=tuple(
            ArcParameterDescriptor(
                stable_name=spec.stable_name,
                stable_id=spec.stable_id,
                shape=tuple(spec.parameter.shape),
                dtype=str(spec.parameter.dtype).removeprefix("torch."),
                role=spec.role,
            )
            for spec in specs
        ),
    )
    if ddp_model.process_group is not None:
        validate_arc_fingerprint_across_ranks(
            fingerprint,
            ddp_model.process_group,
        )
    state = ArcTopKDDPState(
        process_group=ddp_model.process_group,
        fingerprint=fingerprint,
        parameter_specs=specs,
        optimizer_parameters=[
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ],
        config=config,
        find_unused_parameters=getattr(
            ddp_model, "find_unused_parameters", False
        ),
    )
    ddp_model.register_comm_hook(state, arc_topk_ddp_hook)
    return train.GradientSyncRuntime(
        optimizer_owns_gradient_sync=False,
        begin_step=state.begin_step,
        finish_step=state.finish_step,
        commit_step=state.commit_step,
        checkpoint_state=state,
    )
```

Use explicit early validation:

```python
if hp.optimizer not in ("arc_topk_muon", "arc_topk_adamw"):
    raise ValueError(f"Unsupported ARC optimizer: {hp.optimizer}")
if hp.optimizer == "arc_topk_adamw" and hp.arc_sync_mode != "ddp_hook":
    raise ValueError("arc_topk_adamw requires arc_sync_mode=ddp_hook")
```

For `arc_topk_adamw`, construct the existing three groups with
`train.build_adamw_optimizer(param_groups, hp)`, then call the shared installer.
For Muon hook mode, construct ordinary `Muon` and call the same installer.

- [ ] **Step 5: Run factory, ownership, and accumulation tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_train_arctopk.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_train_ddp_sync.py
```

Expected: all tests pass, including existing Muon cases.

- [ ] **Step 6: Commit the AdamW hook integration**

```bash
git add train_arctopk.py tests/test_train_arctopk.py tests/test_train_arctopk_ddp_hook.py
git commit -m "feat: add all-2d ARC DDP hook for AdamW"
```

---

### Task 4: Prove distributed AdamW correctness and checkpoint continuation

**Files:**
- Modify: `tests/test_train_arctopk_ddp_hook.py:180-360`
- Modify: `tests/test_arc_topk_ddp_checkpoint.py:160-330`

**Interfaces:**
- Consumes: `optimizer="arc_topk_adamw"`, `arc_sync_mode="ddp_hook"`, and the existing two-rank Gloo helpers.
- Produces: full-support dense equivalence, sparse rank agreement, and DCP resume coverage for standard AdamW plus hook-owned ARC state.

- [ ] **Step 1: Extend the real two-rank formal-loop test with AdamW modes**

Pass an explicit case name into the worker and construct either dense AdamW
through `train.init_optimizer` or hook AdamW through `train_arctopk`. Use
`arc_topk_ratio=1.0`, identical initial parameters, inputs, LR, betas, and weight
decay, then serialize final parameters. Assert exact or tight floating-point
agreement between dense and hook runs:

```python
for name in dense_parameters:
    torch.testing.assert_close(
        torch.tensor(hook_parameters[name]),
        torch.tensor(dense_parameters[name]),
        rtol=1e-6,
        atol=1e-7,
    )
```

- [ ] **Step 2: Add sparse AdamW rank-agreement coverage**

Run three `ratio=0.5`, `start_compress_step=0` hook steps with rank-distinct
inputs. Gather every model parameter after each optimizer step and assert all
ranks agree. Assert the compressor committed step is three and that embedding
and head tracker states exist.

- [ ] **Step 3: Run the Gloo tests and verify any missing coverage fails**

Run:

```bash
.venv/bin/pytest -q tests/test_train_arctopk_ddp_hook.py
```

Expected before completing worker support: failure for the new AdamW cases.

- [ ] **Step 4: Parameterize DCP round-trip for Muon and AdamW hook paths**

Change `_build_runtime` to accept `optimizer_name: str` and pass it into its
existing hyperparameter construction as
`ArcTopKHyperparameters(optimizer=optimizer_name, arc_sync_mode="ddp_hook", arc_topk_ratio=0.5, arc_projection_rank=2, arc_eta=0.25, arc_start_compress_step=0, scalar_opt="adamw", lr=0.01)`.
Parameterize the existing round-trip test:

```python
@pytest.mark.parametrize("optimizer_name", ["arc_topk_muon", "arc_topk_adamw"])
def test_real_dcp_two_rank_round_trip_preserves_local_state_and_continuation(
    optimizer_name,
):
    # Keep the existing temporary directories, two-rank spawn, JSON readback,
    # and unequal-rank-local-tracker assertion. Pass optimizer_name through
    # _dcp_round_trip_worker to both _build_runtime calls.
    run_dcp_round_trip_case(optimizer_name)
```

Extract the current test body into
`run_dcp_round_trip_case(optimizer_name: str) -> None`; it creates the existing
temporary checkpoint/output directories, spawns `_dcp_round_trip_worker` with
`optimizer_name` added to its arguments, reads both rank JSON files, and asserts
`results[0]["tracker_sum"] != results[1]["tracker_sum"]`. Update the worker to
pass `optimizer_name` to both its initial and rebuilt `_build_runtime` calls.

Keep comparison of model parameters and all `h_local`, `g_local`, and
`g_global` tensors before and after resume.

- [ ] **Step 5: Run all distributed CPU tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_arc_topk_ddp_checkpoint.py \
  tests/test_arc_topk_ddp_hook_distributed.py \
  tests/test_arc_topk_ddp_hook_future_distributed.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit distributed correctness coverage**

```bash
git add tests/test_train_arctopk_ddp_hook.py tests/test_arc_topk_ddp_checkpoint.py
git commit -m "test: cover distributed AdamW ARC hook training"
```

---

### Task 5: Extend the NCCL smoke gate to AdamW hook mode

**Files:**
- Modify: `tests/test_train_arctopk_ddp_hook_nccl.py`

**Interfaces:**
- Consumes: the existing `dense`, `optimizer`, and `ddp_hook` NCCL cases.
- Produces: an `adamw_ddp_hook` case with collective-signature and cross-rank parameter checks.

- [ ] **Step 1: Add an AdamW-hook case to the NCCL worker**

Extend `_build` so `adamw_ddp_hook` constructs:

```python
train_arctopk.init_arc_topk_optimizer(
    model=ddp.module,
    device_mesh=None,
    ddp_model=ddp,
    hp=train_arctopk.ArcTopKHyperparameters(
        optimizer="arc_topk_adamw",
        arc_sync_mode="ddp_hook",
        arc_topk_ratio=0.5,
        arc_projection_rank=2,
        arc_eta=0.25,
        arc_start_compress_step=0,
        model_dim=16,
        lr=0.01,
    ),
    cli_args=_cli(),
)
```

Iterate over `("dense", "optimizer", "ddp_hook", "adamw_ddp_hook")`. For both
hook modes assert `arc_hook/sketch` is present and `arc/sketch` is absent.

- [ ] **Step 2: Run the smoke gate when two GPUs are safely available**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/pytest -q \
  tests/test_train_arctopk_ddp_hook_nccl.py -m multi_gpu
```

Expected: pass when two GPUs are available; otherwise record the pytest skip
and rely on CPU distributed coverage without claiming NCCL verification.

- [ ] **Step 3: Commit the NCCL coverage**

```bash
git add tests/test_train_arctopk_ddp_hook_nccl.py
git commit -m "test: add AdamW ARC hook NCCL smoke coverage"
```

---

### Task 6: Add paired GPT-60M quality configurations and a dry-run launcher

**Files:**
- Create: `configs/compressed_muon/cm034a_dense_adamw_gpt60m_paperlike.yaml`
- Create: `configs/compressed_muon/cm034b_all2d_hook_adamw_gpt60m_paperlike.yaml`
- Create: `benchmark/compressed_muon/run_cm034_adamw_all2d_hook_quality.sh`
- Create: `tests/test_cm034_adamw_all2d_hook_quality_launcher.py`

**Interfaces:**
- Consumes: `train.py` dense AdamW, `train_arctopk.py` hook AdamW, FineWeb10B, and the CM033 probe/preflight conventions.
- Produces: two matched 8,393-step configurations and a launcher whose `--print-plan` is read-only and machine-checkable.

- [ ] **Step 1: Write failing config and launcher contract tests**

Assert both configs have `(model_dim, n_layer, n_head) == (512, 4, 8)`, sequence
length 256, global batch 512, 8,393 iterations, 10,485,760 validation tokens,
training seed supplied by the launcher, `lr=0.001`, and identical AdamW-relevant
training fields. Assert only the ARC config contains:

```yaml
optimizer: arc_topk_adamw
arc_sync_mode: ddp_hook
arc_topk_ratio: 0.2
arc_projection_rank: 4
arc_eta: 1.0
arc_seed: 42
arc_start_compress_step: 1000
bucket_cap_mb: 160
```

The dense config contains `optimizer: adamw`. Test `--print-plan` returns two
ordered cells, total tokens `1_100_087_296`, and
`hook_arc_scope="all_ndim_2_parameters"` for the ARC cell.

- [ ] **Step 2: Run the launcher test and verify it fails**

Run:

```bash
.venv/bin/pytest -q tests/test_cm034_adamw_all2d_hook_quality_launcher.py
```

Expected: failure because the CM034 files do not exist.

- [ ] **Step 3: Add the paired configurations**

Copy only the model/data-scale and validation schedule from CM033. Set both
configs to `lr: 0.001`, `weight_decay: 0.01`, `batch_size: 512`,
`device_batch_size: 128`, and `num_iterations: 8393`. Do not include
Muon-specific `mu`, `adjust_lr`, or `use_polar_express` fields in the new files.

- [ ] **Step 4: Add the serial launcher**

Follow CM033's safe preflight and OOM fallback. The launcher must:

- refuse to overwrite an existing controller artifact root;
- validate the configured four-GPU list and required free memory;
- probe physical device batches 128, 64, 32, and 16 while preserving global
  batch 512;
- run dense first, then ARC only after dense succeeds;
- store config, command, git commit/status, GPU snapshots, stdout/stderr, exit
  code, and final validation/memory lines per cell;
- calculate PPL from the parsed final loss with Python `math.exp`, without
  changing the training evaluator;
- support `--print-plan` without accessing GPUs, W&B, artifacts, or data.

- [ ] **Step 5: Run static and dry-run verification**

Run:

```bash
bash -n benchmark/compressed_muon/run_cm034_adamw_all2d_hook_quality.sh
.venv/bin/pytest -q tests/test_cm034_adamw_all2d_hook_quality_launcher.py
benchmark/compressed_muon/run_cm034_adamw_all2d_hook_quality.sh --print-plan
```

Expected: syntax check and tests pass; the last command emits valid JSON and
does not create an artifact directory.

- [ ] **Step 6: Commit experiment scaffolding**

```bash
git add \
  configs/compressed_muon/cm034a_dense_adamw_gpt60m_paperlike.yaml \
  configs/compressed_muon/cm034b_all2d_hook_adamw_gpt60m_paperlike.yaml \
  benchmark/compressed_muon/run_cm034_adamw_all2d_hook_quality.sh \
  tests/test_cm034_adamw_all2d_hook_quality_launcher.py
git commit -m "exp: add GPT-60M AdamW ARC hook quality pair"
```

---

### Task 7: Run the full regression gate and document implementation status

**Files:**
- Modify: `docs/worklog/M001-arc-topk-ef21m-muon.md`

**Interfaces:**
- Consumes: all implementation and test changes from Tasks 1-6.
- Produces: one verified implementation record; it does not claim formal quality results.

- [ ] **Step 1: Run formatting and diff checks**

```bash
git diff --check
```

Expected: no output and exit code zero.

- [ ] **Step 2: Run the complete relevant CPU suite**

```bash
.venv/bin/pytest -q \
  tests/test_arc_topk.py \
  tests/test_arc_topk_sync.py \
  tests/test_adamw_arctopk.py \
  tests/test_adamw_arctopk_distributed.py \
  tests/test_muon_arctopk.py \
  tests/test_muon_arctopk_distributed.py \
  tests/test_arc_topk_ddp_hook.py \
  tests/test_arc_topk_ddp_hook_state.py \
  tests/test_arc_topk_ddp_hook_future.py \
  tests/test_arc_topk_ddp_hook_distributed.py \
  tests/test_arc_topk_ddp_hook_future_distributed.py \
  tests/test_arc_topk_ddp_checkpoint.py \
  tests/test_train_arctopk.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_train_adamw_builder.py \
  tests/test_cm033_all2d_hook_quality_launcher.py \
  tests/test_cm034_adamw_all2d_hook_quality_launcher.py
```

Expected: all tests pass.

- [ ] **Step 3: Run a one-process debug smoke for both training entries**

Use a temporary checkpoint/artifact directory and existing local FineWeb data:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --standalone --nproc_per_node=1 \
  train.py --config configs/compressed_muon/cm034a_dense_adamw_gpt60m_paperlike.yaml \
  --data_dir data/fineweb10B --debug --no_wandb --no_compile

CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --standalone --nproc_per_node=1 \
  train_arctopk.py \
  --config configs/compressed_muon/cm034b_all2d_hook_adamw_gpt60m_paperlike.yaml \
  --data_dir data/fineweb10B --debug --no_wandb --no_compile \
  --arc_start_compress_step 0
```

Expected: both complete their debug iterations and final validation without
exception. If GPU 0 is occupied, select one verified idle GPU rather than
interrupting another process; if none is safe, record the smoke as blocked and
do not claim it passed.

- [ ] **Step 4: Record implementation evidence**

Append a dated worklog section stating:

- standard AdamW now supports the shared all-2D ARC DDP hook;
- optimizer-side `ArcTopKAdamW` remains a benchmark control;
- dense and ARC AdamW share the same update builder;
- the exact CPU/NCCL/debug commands run and their outcomes;
- CM034 configs and launcher exist but the paper-scale run has not yet been
  launched;
- no quality or performance conclusion is claimed.

- [ ] **Step 5: Commit the verified implementation record**

```bash
git add docs/worklog/M001-arc-topk-ef21m-muon.md
git commit -m "docs: record AdamW all-2d ARC hook implementation"
```

- [ ] **Step 6: Confirm repository state**

```bash
git status --short
git log --oneline -8
```

Expected: clean worktree and the task commits visible in order.
