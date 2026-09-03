# ARC-TopK-EF21M-Muon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DDP-only ARC-TopK-EF21M-Muon optimizer that replaces DDP's dense matrix-gradient All-Reduce with the complete ARC-TopK Algorithm 1 and feeds the EF21M global estimate into Dion's existing Muon update path.

**Architecture:** Refactor `train.py` only enough to inject an extended argument parser, hyperparameter factory, and optimizer factory while preserving its direct-entry defaults. Put the compressor/EF21M recursion in `dion/arc_topk.py`, integrate it through a separate `ArcTopKMuon` class in `dion/muon_arctopk.py`, and expose a thin DDP-only `train_arctopk.py` entry point. Matrix gradients use ARC-TopK plus EF21M; AdamW/Lion groups use dense optimizer-side averaging because DDP backward runs under `no_sync()`.

**Tech Stack:** Python 3.10+, PyTorch distributed (`torch.distributed`, Gloo for CPU tests, NCCL at runtime), pytest, YAML.

**Spec:** `docs/compressed_muon/methods/M001_arc_topk_ef21m_muon.md`

## Global Constraints

- Preserve the original `Muon` implementation and its default execution path.
- Direct execution of `python train.py` must retain its existing defaults and behavior.
- The first version supports DDP only; reject FSDP, HSDP, and TP configurations.
- Implement rank-0 seed synchronization, Gaussian sketching, shared row support, index-free selected-value All-Reduce, and EF21M equations 11a–11c.
- Feed the EF21M global estimate into the existing Muon momentum/Nesterov/orthogonalization/update path.
- Do not modify or upgrade PyTorch, CUDA, NCCL, Triton, or the virtual environment.
- Do not run formal training, benchmark, profiler, or long GPU jobs; automated tests are the completion gate.
- Preserve unrelated worktree changes, including the existing `.gitignore` modification and `artifacts/` content.

## File Map

- Modify `train.py`: add generic parser and factory injection only.
- Create `train_arctopk.py`: ARC-specific DDP entry, CLI fields, param grouping, and optimizer factory.
- Create `dion/arc_topk.py`: pure tensor helpers, distributed ARC-TopK, and EF21M state recursion.
- Create `dion/muon_arctopk.py`: optimizer integration and dense scalar-gradient synchronization.
- Modify `dion/__init__.py`: export `ArcTopKMuon`.
- Create `configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`: first-version defaults.
- Create `tests/test_train_factories.py`: regression coverage for `train.py` injection.
- Create `tests/test_arc_topk.py`: local compressor and EF21M tests.
- Create `tests/test_arc_topk_distributed.py`: two-rank Algorithm 1 collective tests.
- Create `tests/test_muon_arctopk.py`: optimizer construction, state, recurrence, and serialization tests.
- Create `tests/test_muon_arctopk_distributed.py`: two-rank matrix/scalar synchronization tests.
- Create `tests/test_train_arctopk.py`: thin-entry configuration and DDP-only checks.
- Create `docs/compressed_muon/VALIDATE_M001_PROMPT.md`: read-only verification prompt for another agent.
- Modify `docs/compressed_muon/METHOD_INDEX.md`: advance M001 status after tests pass.

---

### Task 1: Make `train.py` Injectible Without Changing Defaults

**Files:**

- Modify: `train.py:91-241`
- Modify: `train.py:695-700`
- Modify: `train.py:844-851`
- Create: `tests/test_train_factories.py`

**Interfaces:**

- Consumes: existing `Hyperparameters`, `parse_cli_args()`, `init_optimizer()`, and `main()`.
- Produces: `parse_cli_args(configure_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None)` and `main(hyperparameters_factory=Hyperparameters, optimizer_factory=init_optimizer, configure_parser=None)`.

- [ ] **Step 1: Write failing parser-extension and signature tests**

```python
def test_parse_cli_args_accepts_extension():
    train = _import_train()

    def configure(parser):
        parser.add_argument("--arc_eta", type=float, default=None)

    with patch.object(sys, "argv", ["train.py", "--arc_eta", "0.25"]):
        args = train.parse_cli_args(configure_parser=configure)
    assert args.arc_eta == 0.25


def test_main_exposes_defaulted_factories():
    train = _import_train()
    signature = inspect.signature(train.main)
    assert signature.parameters["hyperparameters_factory"].default is train.Hyperparameters
    assert signature.parameters["optimizer_factory"].default is train.init_optimizer
    assert signature.parameters["configure_parser"].default is None
```

- [ ] **Step 2: Run the focused tests and confirm the expected failure**

Run: `python -m pytest tests/test_train_factories.py -v`

Expected: FAIL because `parse_cli_args` and `main` do not accept the injected arguments.

- [ ] **Step 3: Add parser and factory injection**

Implement the following structure without adding ARC-specific names to `train.py`:

```python
def parse_cli_args(configure_parser=None):
    parser = argparse.ArgumentParser()
    # existing arguments remain unchanged
    if configure_parser is not None:
        configure_parser(parser)
    cli_args = parser.parse_args()
    # existing YAML merge remains unchanged
    return cli_args


def main(
    hyperparameters_factory=Hyperparameters,
    optimizer_factory=init_optimizer,
    configure_parser=None,
):
    cli_args = parse_cli_args(configure_parser=configure_parser)
    hp = hyperparameters_factory()
    hp = override_args_from_cli(hp, cli_args)
    # existing setup remains unchanged
    optimizer = optimizer_factory(
        model=raw_model,
        device_mesh=device_mesh,
        ddp_model=model if isinstance(model, DDP) else None,
        hp=hp,
        cli_args=cli_args,
    )
```

Keep the bottom-level `main()` call unchanged so defaults are exercised by normal execution.

- [ ] **Step 4: Run focused and existing train/config tests**

Run: `python -m pytest tests/test_train_factories.py tests/test_configs.py tests/test_dion3_alias.py -v`

Expected: PASS, with optional training dependencies producing only the repository's existing skips.

- [ ] **Step 5: Commit the injection seam**

```bash
git add train.py tests/test_train_factories.py
git commit -m "refactor: make training factories injectable"
```

---

### Task 2: Implement Local ARC-TopK Tensor Operations and EF21M Recurrence

**Files:**

- Create: `dion/arc_topk.py`
- Create: `tests/test_arc_topk.py`

**Interfaces:**

- Produces: `validate_arc_topk_config(ratio: float, projection_rank: int, eta: float) -> None`.
- Produces: `make_gaussian_projection(batch: int, columns: int, rank: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> Tensor` returning `[batch, columns, rank]`.
- Produces: `arc_topk_local_sketch(delta: Tensor, projection: Tensor) -> Tensor` returning `[batch, rows, rank]`.
- Produces: `arc_topk_support(global_sketch: Tensor, k: int) -> Tensor` returning `[batch, k]`.
- Produces: `gather_rows(values: Tensor, indices: Tensor) -> Tensor` and `scatter_rows(selected: Tensor, indices: Tensor, rows: int) -> Tensor`.
- Produces: `ef21m_update_tracker_(tracker: Tensor, gradient: Tensor, eta: float) -> Tensor` and `ef21m_apply_delta_(local_estimate: Tensor, global_estimate: Tensor, local_delta: Tensor, averaged_delta: Tensor) -> None`.

- [ ] **Step 1: Write failing validation and projection tests**

```python
@pytest.mark.parametrize("ratio,rank,eta", [(0.0, 2, 0.5), (1.1, 2, 0.5), (0.5, 0, 0.5), (0.5, 2, 0.0), (0.5, 2, 1.1)])
def test_invalid_arc_config(ratio, rank, eta):
    with pytest.raises(ValueError):
        validate_arc_topk_config(ratio, rank, eta)


def test_projection_is_seeded_gaussian_and_has_expected_shape():
    a = make_gaussian_projection(2, 5, 3, seed=17, device=torch.device("cpu"), dtype=torch.float32)
    b = make_gaussian_projection(2, 5, 3, seed=17, device=torch.device("cpu"), dtype=torch.float32)
    assert a.shape == (2, 5, 3)
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all()
```

- [ ] **Step 2: Run the tests and confirm imports fail**

Run: `python -m pytest tests/test_arc_topk.py -v`

Expected: FAIL because `dion.arc_topk` does not exist.

- [ ] **Step 3: Implement validation and Gaussian projection**

Use a private `torch.Generator(device=device)` and `manual_seed(seed)`. Generate float32 Gaussian values, then cast to the requested computation dtype only when required by the caller.

```python
def validate_arc_topk_config(ratio, projection_rank, eta):
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if isinstance(projection_rank, bool) or projection_rank < 1:
        raise ValueError(f"projection_rank must be a positive integer, got {projection_rank!r}")
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
```

- [ ] **Step 4: Write failing sketch, support, gather, and scatter tests**

```python
def test_support_selects_largest_global_sketch_rows():
    sketch = torch.tensor([[[3.0], [1.0], [4.0], [2.0]]])
    indices = arc_topk_support(sketch, k=2)
    assert set(indices[0].tolist()) == {0, 2}


def test_gather_scatter_preserves_selected_rows():
    x = torch.arange(24.0).reshape(2, 4, 3)
    idx = torch.tensor([[3, 1], [0, 2]])
    selected = gather_rows(x, idx)
    rebuilt = scatter_rows(selected, idx, rows=4)
    torch.testing.assert_close(gather_rows(rebuilt, idx), selected)
    assert torch.count_nonzero(rebuilt).item() == torch.count_nonzero(selected).item()
```

- [ ] **Step 5: Implement the local tensor operations**

Compute the paper's normalized sketch in float32:

```python
def arc_topk_local_sketch(delta, projection):
    return torch.bmm(delta.float(), projection.float()) / math.sqrt(projection.shape[-1])


def arc_topk_support(global_sketch, k):
    scores = global_sketch.square().sum(dim=-1)
    return scores.topk(k=k, dim=-1, sorted=True).indices
```

Use `torch.gather` and `scatter_` with expanded `[batch, k, columns]` indices; do not use Python loops over rows.

- [ ] **Step 6: Write failing multi-step EF21M recurrence tests**

```python
def test_ef21m_recurrence_matches_equations_11a_and_11b():
    h = torch.zeros(1, 2, 2)
    g_local = torch.zeros_like(h)
    g_global = torch.zeros_like(h)
    grad = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    ef21m_update_tracker_(h, grad, eta=0.25)
    torch.testing.assert_close(h, 0.25 * grad)
    local_delta = torch.tensor([[[0.5, 1.0], [0.0, 0.0]]])
    avg_delta = local_delta.clone()
    ef21m_apply_delta_(g_local, g_global, local_delta, avg_delta)
    torch.testing.assert_close(g_local, local_delta)
    torch.testing.assert_close(g_global, avg_delta)
```

- [ ] **Step 7: Implement the in-place EF21M helpers and run tests**

Run: `python -m pytest tests/test_arc_topk.py -v`

Expected: PASS.

- [ ] **Step 8: Commit the local algorithm**

```bash
git add dion/arc_topk.py tests/test_arc_topk.py
git commit -m "feat: add ARC-TopK and EF21M tensor operations"
```

---

### Task 3: Implement Complete Distributed ARC-TopK Algorithm 1

**Files:**

- Modify: `dion/arc_topk.py`
- Create: `tests/test_arc_topk_distributed.py`

**Interfaces:**

- Consumes: all Task 2 helpers.
- Produces: `arc_topk_ef21m_async(gradients: List[Tensor], trackers: List[Tensor], local_estimates: List[Tensor], global_estimates: List[Tensor], *, process_group: Optional[ProcessGroup], ratio: float, projection_rank: int, eta: float, base_seed: int, step: int, task_index: int) -> Generator[None, None, List[Tensor]]`.
- The returned tensors are views/copies of the updated global estimates in original list order.

- [ ] **Step 1: Write a two-rank Gloo test for shared support and averaged selected values**

Create a module-level spawn worker. Each rank initializes Gloo, supplies deliberately different `[1, 4, 3]` gradients, exhausts the generator through `AsyncTask`/`AsyncRuntime`, and saves the output through a multiprocessing queue or rank-specific temporary file.

The parent computes the expected result directly:

```python
global_sketch = (sketch_rank0 + sketch_rank1) / 2
indices = arc_topk_support(global_sketch, k=2)
expected = scatter_rows(
    (gather_rows(delta0, indices) + gather_rows(delta1, indices)) / 2,
    indices,
    rows=4,
)
```

Assert both ranks return the same `g_global` and that it equals `expected`.

- [ ] **Step 2: Run the distributed test and confirm the missing-interface failure**

Run: `python -m pytest tests/test_arc_topk_distributed.py -v`

Expected: FAIL because `arc_topk_ef21m_async` is undefined.

- [ ] **Step 3: Implement seed broadcast and the two All-Reduces**

Use one device `torch.int64` seed tensor per shape-group task. Rank 0 fills it with `base_seed + step * 1_000_003 + task_index`, then calls `dist.broadcast(seed_tensor, src=group_source_rank, group=process_group, async_op=True)` and yields before `wait()`.

Stack gradients and state buffers, update trackers, and compute `delta = tracker - local_estimate`. Generate `[batch, columns, projection_rank]` Gaussian projections from the synchronized seed.

For sketch and selected values, use `dist.all_reduce(global_sketch, op=dist.ReduceOp.SUM, group=process_group, async_op=True)` and `dist.all_reduce(averaged_selected, op=dist.ReduceOp.SUM, group=process_group, async_op=True)`. Yield and wait after each call, then divide each buffer by `world_size`. Do not rely on `ReduceOp.AVG`, so the same code works under Gloo and NCCL.

Update state as follows:

```python
local_compressed = scatter_rows(local_selected, indices, rows)
averaged_compressed = scatter_rows(averaged_selected, indices, rows)
local_estimate.add_(local_compressed)
global_estimate.add_(averaged_compressed)
```

Copy stacked state back into the original per-parameter state tensors before returning.

- [ ] **Step 4: Add a ratio-one dense-average test**

With `eta=1`, zero initial estimates, and `ratio=1`, assert the first returned `g_global` equals the manually averaged dense gradients on both ranks.

- [ ] **Step 5: Add a rank-local missing-gradient test**

Represent a missing local gradient as zeros in the worker while the other rank supplies a real gradient. Assert both ranks complete, use identical support, and return the same global estimate.

- [ ] **Step 6: Run local and distributed ARC tests**

Run: `python -m pytest tests/test_arc_topk.py tests/test_arc_topk_distributed.py -v`

Expected: PASS with no hangs.

- [ ] **Step 7: Commit the distributed compressor**

```bash
git add dion/arc_topk.py tests/test_arc_topk_distributed.py
git commit -m "feat: add distributed ARC-TopK EF21M"
```

---

### Task 4: Integrate ARC-TopK-EF21M With Muon

**Files:**

- Create: `dion/muon_arctopk.py`
- Modify: `dion/__init__.py`
- Create: `tests/test_muon_arctopk.py`

**Interfaces:**

- Consumes: `arc_topk_ef21m_async` from Task 3 and the existing `Muon`, `muon_update_pre_orthogonalize`, `megabatch_orthogonalize_async`, LR adjustment helpers, and post-orthogonalize update.
- Produces: `ArcTopKMuon(params, distributed_mesh=None, *, arc_topk_ratio=0.2, arc_projection_rank=4, arc_eta=0.1, arc_seed=42, **muon_kwargs)`.
- Produces optimizer state keys `momentum`, `arc_h_local`, `arc_g_local`, and `arc_g_global` for every Muon parameter.

- [ ] **Step 1: Write failing constructor, validation, and state-prepopulation tests**

```python
def test_arc_topk_muon_prepopulates_full_state():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = ArcTopKMuon([p], arc_topk_ratio=0.5, arc_projection_rank=2, arc_eta=0.25)
    assert opt.state[p].keys() >= {
        "momentum", "arc_h_local", "arc_g_local", "arc_g_global"
    }
    for name in ("arc_h_local", "arc_g_local", "arc_g_global"):
        assert opt.state[p][name].shape == p.shape
        assert torch.count_nonzero(opt.state[p][name]) == 0
```

Also assert invalid ratio/rank/eta raise during construction and that `flatten=True`, `num_heads>1`, and `split_sizes` are rejected with explicit messages.

- [ ] **Step 2: Run the focused test and confirm import failure**

Run: `python -m pytest tests/test_muon_arctopk.py -v`

Expected: FAIL because `ArcTopKMuon` is not defined/exported.

- [ ] **Step 3: Implement the class and prepopulated state**

Set ARC attributes before calling `super().__init__()` because `DistributedOrthoBase.__init__()` dynamically calls `_get_or_initialize_state()`.

```python
class ArcTopKMuon(Muon):
    def __init__(
        self,
        params,
        distributed_mesh=None,
        *,
        arc_topk_ratio=0.2,
        arc_projection_rank=4,
        arc_eta=0.1,
        arc_seed=42,
        **kwargs,
    ):
        validate_arc_topk_config(
            arc_topk_ratio, arc_projection_rank, arc_eta
        )
        self._arc_topk_ratio = arc_topk_ratio
        self._arc_projection_rank = arc_projection_rank
        self._arc_eta = arc_eta
        self._arc_seed = arc_seed
        super().__init__(params, distributed_mesh=distributed_mesh, **kwargs)

    def _get_or_initialize_state(self, param, algo):
        state = super()._get_or_initialize_state(param, algo)
        if algo == "muon":
            state.setdefault("arc_h_local", torch.zeros_like(param))
            state.setdefault("arc_g_local", torch.zeros_like(param))
            state.setdefault("arc_g_global", torch.zeros_like(param))
        return state
```

- [ ] **Step 4: Write a failing single-process recurrence-to-Muon test**

Use a deterministic identity orthogonalizer injected through `newton_schulz_func=lambda x, epsilon: x`. Set `arc_topk_ratio=1`, `arc_eta=1`, `mu=0`, `nesterov=False`, `weight_decay=0`, and `adjust_lr=None`. After one step, assert `arc_g_global == grad` and the parameter update equals `-lr * grad`.

- [ ] **Step 5: Implement rank-symmetric task creation and update generator**

Copy only the shape-group orchestration needed from `Muon._create_ortho_tasks()`. Iterate all Muon parameters in stable param-group order; replace local `None` gradients with `torch.zeros_like(to_local(param))`.

For each shape group:

```python
G_global = yield from arc_topk_ef21m_async(
    gradients=gradients,
    trackers=trackers,
    local_estimates=local_estimates,
    global_estimates=global_estimates,
    process_group=self._process_group,
    ratio=self._arc_topk_ratio,
    projection_rank=self._arc_projection_rank,
    eta=self._arc_eta,
    base_seed=self._arc_seed,
    step=group["step"],
    task_index=task_index,
)
U = muon_update_pre_orthogonalize(
    G=G_global,
    M=momentums,
    momentum=momentum,
    nesterov=nesterov,
)
U = yield from megabatch_orthogonalize_async(
    U,
    comm_dim=None,
    device_rank=self._device_rank,
    world_size=self._world_size,
    process_group=self._process_group,
    newton_schulz_func=self._newton_schulz_func,
    flatten=False,
    epsilon=epsilon,
)
muon_update_post_orthogonalize(
    X=params,
    U=U,
    base_lr=lr,
    adjusted_lr=adjusted_lr,
    weight_decay=weight_decay,
    cautious_wd=cautious_wd,
)
```

Reject non-2D Muon params and unsupported `flatten`, `num_heads`, and `split_sizes` options in the ARC class before collective creation.

- [ ] **Step 6: Add and pass state-dict round-trip tests**

Run two local steps, serialize `state_dict()`, load it into a fresh optimizer with a cloned parameter, and assert all four state tensors and ARC param-group fields match.

Run: `python -m pytest tests/test_muon_arctopk.py tests/test_state_prepopulation.py -v`

Expected: PASS and no regression in existing state-prepopulation tests.

- [ ] **Step 7: Commit Muon integration**

```bash
git add dion/muon_arctopk.py dion/__init__.py tests/test_muon_arctopk.py
git commit -m "feat: integrate ARC-TopK EF21M with Muon"
```

---

### Task 5: Synchronize Scalar Optimizer Groups and Verify Two-Rank Consistency

**Files:**

- Modify: `dion/muon_arctopk.py`
- Create: `tests/test_muon_arctopk_distributed.py`

**Interfaces:**

- Consumes: Task 4 `ArcTopKMuon`.
- Produces: overridden `_create_lion_tasks()` and `_create_adamw_tasks()` that average local gradients over the same DDP process group before invoking the existing scalar update kernels.

- [ ] **Step 1: Write a two-rank matrix synchronization test**

Spawn two Gloo ranks with identical parameters but different local matrix gradients. Inject the identity orthogonalizer, use `ratio=1`, `eta=1`, `mu=0`, no Nesterov, no decay, and one optimizer step. Assert both ranks have identical:

```text
parameter
momentum
arc_g_global
```

Also assert the update equals the dense average-gradient reference.

- [ ] **Step 2: Write failing AdamW and Lion synchronization tests**

Construct mixed parameter groups with one matrix group and one `algorithm="adamw"` or `algorithm="lion"` group. Give the scalar-group parameter different gradients on each rank, run one step, and assert the scalar parameters match across ranks.

- [ ] **Step 3: Run tests and confirm scalar divergence**

Run: `python -m pytest tests/test_muon_arctopk_distributed.py -v`

Expected: matrix test passes after Task 4; scalar tests FAIL because base Muon assumes gradients were already synchronized.

- [ ] **Step 4: Implement asynchronous dense scalar-gradient averaging**

Add a private generator that stacks or coalesces same-shaped scalar gradients, performs `SUM` All-Reduce, yields, waits, divides by world size, then invokes the current `lion_update_foreach_async` or `adamw_update_foreach_async` path with the averaged tensors.

Do not import the legacy Dion scalar helpers because the current base path carries capturable AdamW `state_steps`; preserve those arguments even though ARC itself is eager-only.

- [ ] **Step 5: Add rank-asymmetric matrix-gradient coverage**

On rank 0 set the matrix gradient tensor; on rank 1 leave `.grad=None`. Ensure both ranks still create the matrix task, complete all collectives, and finish with identical parameter and global state.

- [ ] **Step 6: Run distributed and optimizer regression tests**

Run: `python -m pytest tests/test_muon_arctopk_distributed.py tests/test_muon_arctopk.py tests/test_optimizers.py -v`

Expected: PASS, allowing only existing hardware-dependent skips.

- [ ] **Step 7: Commit distributed optimizer behavior**

```bash
git add dion/muon_arctopk.py tests/test_muon_arctopk_distributed.py
git commit -m "feat: synchronize ARC Muon scalar groups"
```

---

### Task 6: Add the Thin `train_arctopk.py` Entry Point and Configuration

**Files:**

- Create: `train_arctopk.py`
- Create: `configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`
- Create: `tests/test_train_arctopk.py`

**Interfaces:**

- Consumes: Task 1 injectable `train.main` and Task 4 `ArcTopKMuon`.
- Produces: `ArcTopKHyperparameters`, `configure_arc_topk_parser(parser)`, `init_arc_topk_optimizer(model, device_mesh, ddp_model, hp, cli_args)`, and the script entry point.

- [ ] **Step 1: Write failing import, parser, and DDP-only tests**

```python
def test_arc_defaults_enable_optimizer_side_sync():
    module = _import_train_arctopk()
    hp = module.ArcTopKHyperparameters()
    assert hp.optimizer == "arc_topk_muon"
    assert hp.replicate_mesh_grad_sync is True


def test_arc_parser_adds_method_parameters():
    module = _import_train_arctopk()
    parser = argparse.ArgumentParser()
    module.configure_arc_topk_parser(parser)
    args = parser.parse_args(["--arc_topk_ratio", "0.3", "--arc_projection_rank", "2", "--arc_eta", "0.4"])
    assert (args.arc_topk_ratio, args.arc_projection_rank, args.arc_eta) == (0.3, 2, 0.4)


def test_arc_optimizer_factory_rejects_device_mesh():
    with pytest.raises(ValueError, match="DDP only"):
        init_arc_topk_optimizer(model=stub, device_mesh=object(), ddp_model=None, hp=hp, cli_args=args)
```

- [ ] **Step 2: Run focused tests and confirm module import failure**

Run: `python -m pytest tests/test_train_arctopk.py -v`

Expected: FAIL because `train_arctopk.py` does not exist.

- [ ] **Step 3: Implement ARC hyperparameters and parser extension**

```python
@dataclass
class ArcTopKHyperparameters(train.Hyperparameters):
    optimizer: str = "arc_topk_muon"
    replicate_mesh_grad_sync: bool = True
    arc_topk_ratio: float = 0.2
    arc_projection_rank: int = 4
    arc_eta: float = 0.1
    arc_seed: int = 42
```

Add numeric CLI arguments with `default=None` so YAML/default precedence matches `train.py`.

- [ ] **Step 4: Implement the DDP-only optimizer factory**

Reject `device_mesh is not None`, require `ddp_model`, build the same matrix/embedding/lm-head param groups as base `init_optimizer`, and construct:

```python
ArcTopKMuon(
    param_groups,
    distributed_mesh=ddp_model.process_group,
    lr=hp.lr,
    mu=hp.mu,
    weight_decay=hp.weight_decay,
    nesterov=True,
    adjust_lr=hp.adjust_lr,
    arc_topk_ratio=hp.arc_topk_ratio,
    arc_projection_rank=hp.arc_projection_rank,
    arc_eta=hp.arc_eta,
    arc_seed=hp.arc_seed,
    use_gram_newton_schulz=cli_args.use_gram_newton_schulz,
    use_triton=not cli_args.no_triton,
    use_polar_express=cli_args.use_polar_express,
)
```

- [ ] **Step 5: Add the thin main call and sample config**

```python
if __name__ == "__main__":
    train.main(
        hyperparameters_factory=ArcTopKHyperparameters,
        optimizer_factory=init_arc_topk_optimizer,
        configure_parser=configure_arc_topk_parser,
    )
```

The YAML must set `optimizer: arc_topk_muon`, `replicate_mesh_grad_sync: true`, ARC defaults, and `checkpoint_freq: 0`, while omitting `dp_size`, `fs_size`, and `tp_size` so `train.init_distributed()` selects DDP.

- [ ] **Step 6: Test entry-point configuration and base regressions**

Run: `python -m pytest tests/test_train_arctopk.py tests/test_train_factories.py tests/test_configs.py -v`

Expected: PASS. The nested ARC config is tested explicitly by `test_train_arctopk.py`; existing `tests/test_configs.py` only scans top-level `configs/*.yaml`.

- [ ] **Step 7: Commit the entry point**

```bash
git add train_arctopk.py configs/compressed_muon/m001_arc_topk_muon_ddp.yaml tests/test_train_arctopk.py
git commit -m "feat: add ARC-TopK Muon DDP training entry"
```

---

### Task 7: Add the External Verification Prompt and Run the Automated Gate

**Files:**

- Create: `docs/compressed_muon/VALIDATE_M001_PROMPT.md`
- Modify: `docs/compressed_muon/METHOD_INDEX.md`

**Interfaces:**

- Consumes: all implementation and test files from Tasks 1–6.
- Produces: a self-contained prompt for a separate verification-only agent and a truthful M001 status.

- [ ] **Step 1: Write the verification-agent prompt**

The prompt must instruct the other agent to:

```text
Read AGENTS.md, the M001 spec, and the implementation plan.
Do not edit code, install dependencies, start long training, or interfere with existing processes.
Inspect git diff and enumerate the exact M001 files.
Run the focused local tests.
Run the two-rank Gloo distributed tests with a bounded timeout.
Run the relevant existing regression tests.
If at least two suitable GPUs are idle, optionally run the CUDA/NCCL distributed tests; otherwise report SKIPPED without treating it as failure.
Check that train.py direct defaults remain unchanged.
Check that ARC performs seed broadcast, sketch SUM/divide, common Top-K support, selected-value SUM/divide, and no index communication.
Check EF21M state equations and that g_global, Muon momentum, parameters, and scalar parameters agree across ranks.
Report every command, exit code, passed/failed/skipped counts, and any semantic discrepancy. Do not fix failures.
```

Include exact commands from Steps 2–4 below and a final report template with `PASS`, `FAIL`, and `NOT RUN` sections.

- [ ] **Step 2: Run focused M001 tests**

Run:

```bash
python -m pytest \
  tests/test_train_factories.py \
  tests/test_arc_topk.py \
  tests/test_arc_topk_distributed.py \
  tests/test_muon_arctopk.py \
  tests/test_muon_arctopk_distributed.py \
  tests/test_train_arctopk.py \
  -v
```

Expected: all collected M001 tests PASS; hardware- or optional-dependency skips must be listed explicitly.

- [ ] **Step 3: Run relevant existing regression tests**

Run:

```bash
python -m pytest \
  tests/test_configs.py \
  tests/test_state_prepopulation.py \
  tests/test_optimizers.py \
  tests/test_dion3_alias.py \
  -v
```

Expected: PASS except for existing hardware-dependent skips.

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests -v`

Expected: PASS except for environment-dependent skips. If the suite exposes an unrelated pre-existing failure, record the exact failing test and prove the focused M001 tests still pass; do not silently classify the full gate as passing.

- [ ] **Step 5: Check formatting and import integrity**

Run:

```bash
python -m compileall -q dion train.py train_arctopk.py tests
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 6: Update method status from evidence**

If focused and relevant regression tests pass, change M001 from `idea` to `testing`. Do not mark it `active` because no training or performance experiment has run.

- [ ] **Step 7: Commit the validation handoff**

```bash
git add docs/compressed_muon/VALIDATE_M001_PROMPT.md docs/compressed_muon/METHOD_INDEX.md
git commit -m "docs: add M001 verification handoff"
```

## Final Review Checklist

- [ ] Compare every implementation file against `docs/compressed_muon/methods/M001_arc_topk_ef21m_muon.md`.
- [ ] Confirm `train.py` contains no ARC-specific option or optimizer branch.
- [ ] Confirm `dion/muon.py` has no behavior changes.
- [ ] Confirm every distributed collective has the same order and tensor shape on every rank.
- [ ] Confirm the final response distinguishes automated-test success from unmeasured training/communication performance.
