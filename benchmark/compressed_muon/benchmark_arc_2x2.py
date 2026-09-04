"""A small, reproducible DDP benchmark for dense and ARC-TopK optimizers.

The benchmark deliberately keeps configuration and result construction usable on
CPU.  The actual runner requires CUDA because its measurements use CUDA events.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
from torch.profiler import record_function

from dion.adamw_arctopk import ArcTopKAdamW
from dion.arc_topk_sync import ArcTopKLogicalBytes, ArcTopKSyncConfig, estimate_arc_logical_bytes, group_parameters_by_shape_dtype
from dion.muon import Muon
from dion.muon_arctopk import ArcTopKMuon
from dion.collective_observer import (CollectiveObserver, aggregate_observed,
                                      set_active_observer, signatures_agree)
from models.gpt_model import GPT, GPTConfig

MODEL_PRESETS = {
    "gpt60m": dict(model_dim=512, n_layer=4, n_head=8),
    "gpt130m": dict(model_dim=768, n_layer=8, n_head=12),
    "gpt350m": dict(model_dim=1024, n_layer=20, n_head=16),
    "gpt1b": dict(model_dim=1536, n_layer=30, n_head=24),
}


@dataclass
class BenchmarkConfig:
    experiment_id: str
    optimizer: str
    sync: str
    model: str
    warmup_steps: int
    measure_steps: int
    seed: int
    output: Optional[str]
    profile_output: Optional[str]
    compile_model: bool = False
    world_size: int = 4
    local_batch: int = 1
    sequence_length: int = 256
    gradient_accumulation: int = 1
    formal: bool = True
    transport: str = "normal"
    profile_only: bool = False
    profiles: Optional[list[str]] = None
    ratio: float = 0.2
    projection_rank: int = 4
    eta: float = 0.1
    start_compress_step: int = 0


def validate_config(config: BenchmarkConfig) -> BenchmarkConfig:
    if config.optimizer not in {"adamw", "muon"}:
        raise ValueError("optimizer must be one of: adamw, muon")
    if config.sync not in {"dense", "arc"}:
        raise ValueError("sync must be one of: dense, arc")
    if config.model not in MODEL_PRESETS:
        raise ValueError(f"unknown model preset: {config.model}")
    if config.sync == "arc" and config.optimizer not in {"adamw", "muon"}:
        raise ValueError(f"ARC sync is unsupported for optimizer {config.optimizer!r}")
    if config.formal and config.warmup_steps < 20:
        raise ValueError("formal benchmark requires warmup_steps >= 20")
    if config.formal and config.measure_steps < 100:
        raise ValueError("formal benchmark requires measure_steps >= 100")
    if config.formal and config.world_size != 4:
        raise ValueError("formal benchmark requires world_size == 4")
    if config.local_batch < 1 or config.sequence_length < 1:
        raise ValueError("local_batch and sequence_length must be positive")
    if config.gradient_accumulation != 1:
        raise ValueError("gradient_accumulation must be exactly 1")
    if config.profile_output and not config.profile_only:
        raise ValueError("profile_output requires explicit --profile mode")
    if config.profile_only and not config.profile_output:
        raise ValueError("--profile requires --profile-output")
    if config.transport == "p2p_disabled" and (
        os.environ.get("NCCL_P2P_DISABLE") != "1" or os.environ.get("NCCL_SHM_DISABLE") != "0"
    ):
        raise ValueError("p2p_disabled requires NCCL_P2P_DISABLE=1 and NCCL_SHM_DISABLE=0")
    parts = config.experiment_id.split("-")
    expected_parts = [config.optimizer, config.sync, config.model, "ddp",
                      f"ws{config.world_size}", f"s{config.seed}"]
    valid_id = parts and parts[0].startswith("CM") and any(
        parts[i:i + len(expected_parts)] == expected_parts for i in range(1, len(parts) - len(expected_parts) + 1)
    )
    if not valid_id:
        raise ValueError(
            "experiment_id must encode optimizer, sync, model, world size and seed "
            f"(expected *-{config.optimizer}-{config.sync}-{config.model}-ddp-"
            f"ws{config.world_size}-s{config.seed})"
        )
    return config


def parse_args(argv: Optional[list[str]] = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), required=True)
    parser.add_argument("--sync", choices=("dense", "arc"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_PRESETS), required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--measure-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-output")
    parser.add_argument("--profiles", nargs="+", help="profiler summaries associated with this timing result")
    parser.add_argument("--profile", dest="profile_only", action="store_true",
                        help="run only the independent 3+3+5 profiler schedule")
    parser.add_argument("--compile-model", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile-model", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=False)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--local-batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="relax formal minimums")
    parser.add_argument("--transport", choices=("normal", "p2p_disabled"), default="normal")
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        experiment_id=args.experiment_id, optimizer=args.optimizer, sync=args.sync,
        model=args.model, warmup_steps=args.warmup_steps, measure_steps=args.measure_steps,
        seed=args.seed, output=args.output, profile_output=args.profile_output,
        compile_model=args.compile_model, world_size=args.world_size,
        local_batch=args.local_batch, sequence_length=args.sequence_length,
        gradient_accumulation=args.gradient_accumulation,
        formal=not args.smoke, transport=args.transport,
        profile_only=args.profile_only,
        profiles=args.profiles,
    )
    return validate_config(config)


def _parameter_count(model_name: str) -> int:
    p = MODEL_PRESETS[model_name]
    d, l = p["model_dim"], p["n_layer"]
    # GPT ties neither embedding nor lm_head; this avoids constructing 1B models
    return 2 * 50304 * d + 12 * l * d * d


def build_result_skeleton(config: BenchmarkConfig) -> dict[str, Any]:
    validate_config(config)
    preset = MODEL_PRESETS[config.model]
    optimizer_config = (
        {"lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01}
        if config.optimizer == "adamw"
        else {"lr": 0.02, "mu": 0.95, "weight_decay": 0.01}
    )
    if config.optimizer == "muon":
        from dion.newton_schulz_triton import TRITON_AVAILABLE
        optimizer_config["use_triton"] = bool(TRITON_AVAILABLE)
        optimizer_config["use_polar_express"] = True
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "optimizer": config.optimizer,
        "sync_mode": config.sync,
        "transport": config.transport,
        "model": {"label": config.model, "dim": preset["model_dim"],
                  "layers": preset["n_layer"], "heads": preset["n_head"],
                  "parameters": _parameter_count(config.model)},
        "workload": {"world_size": config.world_size, "local_batch": config.local_batch,
                      "sequence_length": config.sequence_length,
                      "gradient_accumulation": config.gradient_accumulation, "dtype": "bfloat16"},
        "arc": {"ratio": config.ratio, "projection_rank": config.projection_rank,
                "eta": config.eta, "seed": config.seed,
                "start_compress_step": config.start_compress_step},
        "optimizer_config": optimizer_config,
        "timing_ms": {"step_samples": [], "fwd_bwd_samples": [],
                      "optimizer_samples": [], "step_mean": 0.0},
        "throughput": {"tokens_per_second": 0.0},
        "memory": {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0},
        "communication": {"dense_gradient_bytes": 0, "arc_seed_bytes": 0,
                          "arc_sketch_bytes": 0, "arc_selected_values_bytes": 0,
                          "uncompressed_bytes": 0},
        "profiler": {"trace_path": None, "nccl_kernel_time_ms": None, "collectives": []},
        "correctness": {"finite_loss": None, "finite_parameters": None,
                        "parameter_checksum": None, "parameter_checksum_squared": None,
                        "parameter_checksum_agreement": None,
                        "collective_signature": {"all_ranks_match": None, "per_rank": []}},
        "environment": {"git_commit": "", "torch": torch.__version__,
                         "cuda": torch.version.cuda or "", "nccl": "", "gpu_names": []},
    }


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_model(config: BenchmarkConfig, device: torch.device) -> GPT:
    torch.manual_seed(config.seed)
    model = GPT(GPTConfig(sequence_len=config.sequence_length, **{
        "n_embd": MODEL_PRESETS[config.model]["model_dim"],
        "n_layer": MODEL_PRESETS[config.model]["n_layer"],
        "n_head": MODEL_PRESETS[config.model]["n_head"],
    }))
    model.init_weights()
    model.to(device=device, dtype=torch.bfloat16)
    if config.compile_model:
        model.compile()
    return model


def _optimizer_parameters(model: GPT) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    compressed = list(model.transformer.h.parameters())
    uncompressed = [*model.transformer.wte.parameters(), *model.lm_head.parameters()]
    return compressed, uncompressed


def communication_result(
    config: BenchmarkConfig,
    compressed_batches: list[list[torch.Tensor]],
    uncompressed: list[torch.Tensor],
    *,
    step: int | None = None,
) -> dict[str, int]:
    """Return schema-named logical communication bytes for one measured step."""
    step = max(2, config.start_compress_step + 1) if step is None else step
    if config.sync == "dense":
        estimate = ArcTopKLogicalBytes(
            dense_gradient=sum(p.numel() * p.element_size() for batch in compressed_batches for p in batch)
            + sum(p.numel() * p.element_size() for p in uncompressed),
            arc_seed=0, arc_sketch=0, arc_selected_values=0, uncompressed=0,
        )
    else:
        estimate = estimate_arc_logical_bytes(
            compressed_batches=compressed_batches, uncompressed_params=uncompressed,
            config=ArcTopKSyncConfig(config.ratio, config.projection_rank, config.eta,
                                     config.seed, config.start_compress_step), step=step,
        )
    return {
        "dense_gradient_bytes": estimate.dense_gradient,
        "arc_seed_bytes": estimate.arc_seed,
        "arc_sketch_bytes": estimate.arc_sketch,
        "arc_selected_values_bytes": estimate.arc_selected_values,
        "uncompressed_bytes": estimate.uncompressed,
    }


def build_optimizer(config: BenchmarkConfig, model: GPT, process_group=None):
    compressed, uncompressed = _optimizer_parameters(model)
    if config.optimizer == "adamw":
        if config.sync == "arc":
            return ArcTopKAdamW(
                [{"params": compressed, "arc_compress": True}, {"params": uncompressed}],
                process_group=process_group, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
                arc_topk_ratio=config.ratio, arc_projection_rank=config.projection_rank,
                arc_eta=config.eta, arc_seed=config.seed,
                arc_start_compress_step=config.start_compress_step,
            )
        return torch.optim.AdamW([*compressed, *uncompressed], lr=1e-3,
                                 betas=(0.9, 0.999), weight_decay=0.01)
    groups = [{"params": compressed, "algorithm": "muon"},
              {"params": uncompressed, "algorithm": "adamw"}]
    cls = ArcTopKMuon if config.sync == "arc" else Muon
    from dion.newton_schulz_triton import TRITON_AVAILABLE
    kwargs = dict(lr=0.02, mu=0.95, weight_decay=0.01, adjust_lr="spectral_norm",
                  use_triton=bool(TRITON_AVAILABLE), use_polar_express=True)
    if cls is ArcTopKMuon:
        kwargs.update(arc_topk_ratio=config.ratio, arc_projection_rank=config.projection_rank,
                      arc_eta=config.eta, arc_seed=config.seed,
                      arc_start_compress_step=config.start_compress_step)
    # Dense Muon is a standard DDP baseline: DDP owns gradient synchronization
    # and Muon's internal result-sharding collectives must stay disabled.
    distributed_mesh = process_group if config.sync == "arc" else None
    return cls(groups, distributed_mesh=distributed_mesh, **kwargs)


def _environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = ""
    names = []
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"git_commit": commit, "torch": torch.__version__, "cuda": torch.version.cuda or "",
            "nccl": getattr(torch.cuda, "nccl", None) and torch.cuda.nccl.version(),
            "gpu_names": names, "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", ""),
            "nccl_shm_disable": os.environ.get("NCCL_SHM_DISABLE", ""),
            "muon_use_triton": bool(__import__("dion.newton_schulz_triton", fromlist=["TRITON_AVAILABLE"]).TRITON_AVAILABLE)}


def _collective_signature(config: BenchmarkConfig) -> list[str]:
    if config.sync == "dense":
        return ["ddp_gradient"]
    return ["arc_seed", "arc_sketch", "arc_selected_values", "arc_dense_uncompressed"]


def reduce_correctness_flags(finite_loss: bool, finite_parameters: bool) -> tuple[bool, bool]:
    """Reduce correctness booleans with logical AND (MIN) across ranks."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return bool(finite_loss), bool(finite_parameters)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    flags = torch.tensor([int(finite_loss), int(finite_parameters)], device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    return bool(flags[0]), bool(flags[1])


def validate_actual_world_size(configured_world_size: int) -> int:
    """Reject launch metadata that disagrees with the initialized job."""
    actual = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if actual != configured_world_size:
        raise RuntimeError(
            f"actual distributed world size {actual} does not match "
            f"configured {configured_world_size}"
        )
    return actual


def sync_context(config: BenchmarkConfig, ddp_model):
    """Disable DDP reduction for ARC, while supporting a single-rank smoke."""
    if config.sync == "arc" and hasattr(ddp_model, "no_sync"):
        return ddp_model.no_sync()
    return _nullcontext()


def checksums_agree(checksums: list[tuple[float, float]]) -> bool:
    """Compare two independent parameter moments across ranks."""
    if not checksums:
        return False
    reference = checksums[0]
    return all(value == reference for value in checksums[1:])


@contextmanager
def observer_scope(observer):
    """Install an observer only for the work that can emit collectives."""
    set_active_observer(observer)
    try:
        yield observer
    finally:
        set_active_observer(None)


def _correctness(model: torch.nn.Module, config: BenchmarkConfig, last_loss=None,
                 observer: CollectiveObserver | None = None) -> dict[str, Any]:
    params = list(_raw_model(model).parameters())
    finite_parameters = all(bool(torch.isfinite(p).all()) for p in params)
    checksum = float(sum(p.detach().double().sum() for p in params))
    checksum_squared = float(sum(p.detach().double().square().sum() for p in params))
    checksum_pair = (checksum, checksum_squared)
    per_rank = [observer.signature() if observer is not None else []]
    checksums = [checksum_pair]
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, per_rank[0])
        per_rank = gathered
        checksum_values = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(checksum_values, checksum_pair)
        checksums = checksum_values
    finite_loss = bool(last_loss is not None and torch.isfinite(last_loss).all())
    finite_loss, finite_parameters = reduce_correctness_flags(finite_loss, finite_parameters)
    return {"finite_loss": finite_loss, "finite_parameters": finite_parameters,
            "parameter_checksum": checksum,
            "parameter_checksum_squared": checksum_squared,
            "parameter_checksum_agreement": checksums_agree(checksums),
            "collective_signature": {"all_ranks_match": signatures_agree(per_rank),
                                      "per_rank": per_rank}}


def register_dense_ddp_hook(ddp_model, observer: CollectiveObserver):
    """Register SUM/world-size averaging while observing real DDP buckets."""
    process_group = getattr(ddp_model, "process_group", None)
    world_size = dist.get_world_size(process_group) if process_group is not None else 1

    def hook(state, bucket):
        buffer = bucket.buffer()
        with record_function("DDP bucket All-Reduce"):
            observer.record("ddp_gradient", "all_reduce", buffer.numel(), buffer.dtype,
                            buffer.numel() * buffer.element_size())
            work = dist.all_reduce(buffer, op=dist.ReduceOp.SUM, group=process_group,
                                   async_op=True)
        def average(future):
            value = future.value()
            reduced = value[0] if isinstance(value, (list, tuple)) else value
            return reduced.div(world_size)
        return work.get_future().then(average)

    ddp_model.register_comm_hook(None, hook)
    return ddp_model


def _run_steps(model, optimizer, batches, config, ddp_model=None, measure=False):
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark execution requires CUDA; configuration helpers are CPU-safe")
    measured_events = []
    fwd_samples, opt_samples, step_samples = [], [], []
    last_loss = None
    for tokens, targets in batches:
        step_start = torch.cuda.Event(enable_timing=True)
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        opt_start = torch.cuda.Event(enable_timing=True)
        opt_end = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record(); fwd_start.record()
        context = sync_context(config, ddp_model)
        with context:
            with record_function("benchmark/forward_backward"):
                loss = model(tokens, targets=targets)
                loss.backward()
                last_loss = loss.detach()
        fwd_end.record(); opt_start.record()
        with record_function("benchmark/optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        opt_end.record(); step_end.record()
        if measure: measured_events.append((fwd_start, fwd_end, opt_start, opt_end, step_start, step_end))
    torch.cuda.synchronize()
    for fwd_start, fwd_end, opt_start, opt_end, step_start, step_end in measured_events:
        fwd_samples.append(fwd_start.elapsed_time(fwd_end))
        opt_samples.append(opt_start.elapsed_time(opt_end))
        step_samples.append(step_start.elapsed_time(step_end))
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 and measure:
        # Materialize all local event samples first, then report the slowest rank.
        for samples in (fwd_samples, opt_samples, step_samples):
            sample_tensor = torch.tensor(samples, device=torch.cuda.current_device(), dtype=torch.float64)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.MAX)
            samples[:] = sample_tensor.cpu().tolist()
    return fwd_samples, opt_samples, step_samples, last_loss


def run_profiler(model, optimizer, batches, config, *, ddp_model=None, output=None):
    """Capture exactly 3 wait, 3 warmup, and 5 active profiler steps."""
    if not torch.cuda.is_available():
        raise RuntimeError("profiler execution requires CUDA")
    output = output or config.profile_output
    if not output:
        raise ValueError("a profile output path is required")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    trace_path = str(output) if rank == 0 else None
    def on_trace_ready(prof):
        if trace_path:
            prof.export_chrome_trace(trace_path)
    schedule = torch.profiler.schedule(wait=3, warmup=3, active=5, repeat=1)
    if len(batches) < 11:
        batches = (list(batches) * ((11 + len(batches) - 1) // len(batches)))[:11]
    last_loss = None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule, on_trace_ready=on_trace_ready, record_shapes=False,
    ) as prof:
        for tokens, targets in batches[:11]:
            context = sync_context(config, ddp_model)
            with context:
                with record_function("benchmark/forward_backward"):
                    loss = model(tokens, targets=targets)
                    loss.backward()
                    last_loss = loss.detach()
            with record_function("benchmark/optimizer"):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            prof.step()
    run_profiler.last_loss = last_loss
    return trace_path


def run_profile_mode(config: BenchmarkConfig) -> dict[str, Any]:
    """Run an independent profiler process setup and write a profiler summary."""
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiler execution")
    rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    if config.world_size > 1 and not dist.is_initialized(): dist.init_process_group("nccl")
    validate_actual_world_size(config.world_size)
    group = dist.group.WORLD if dist.is_initialized() else None
    device = torch.device("cuda", local_rank)
    model = build_model(config, device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if group else model
    optimizer = build_optimizer(config, model, group)
    observer = CollectiveObserver()
    if config.sync == "dense" and group:
        register_dense_ddp_hook(ddp_model, observer)
    random.seed(config.seed + rank); torch.manual_seed(config.seed + rank)
    batches = [(torch.randint(0, 50304, (config.local_batch, config.sequence_length), device=device),
                torch.randint(0, 50304, (config.local_batch, config.sequence_length), device=device)) for _ in range(11)]
    with observer_scope(observer):
        trace_path = run_profiler(ddp_model, optimizer, batches, config, ddp_model=ddp_model)
    result = build_result_skeleton(config)
    raw = _raw_model(ddp_model); compressed, uncompressed = _optimizer_parameters(raw)
    result["communication"] = communication_result(config, group_parameters_by_shape_dtype(compressed), uncompressed)
    result["profiler"]["trace_path"] = trace_path
    result["profiler"]["observed_collectives"] = aggregate_observed(observer)
    result["correctness"] = _correctness(ddp_model, config, getattr(run_profiler, "last_loss", None), observer)
    if rank == 0:
        from benchmark.compressed_muon.profiler_trace import attribute_trace
        trace_summary = attribute_trace(trace_path)
        result["profiler"].update(trace_summary)
        if config.output:
            Path(config.output).parent.mkdir(parents=True, exist_ok=True)
            Path(config.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run a configured CUDA benchmark and optionally write its JSON result."""
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark execution")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if config.world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    validate_actual_world_size(config.world_size)
    group = dist.group.WORLD if dist.is_initialized() else None
    random.seed(config.seed + rank); torch.manual_seed(config.seed + rank)
    model = build_model(config, device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if group else model
    optimizer = build_optimizer(config, model, group)
    # Rank-local data are deterministic and independent of model initialization.
    random.seed(config.seed + rank); torch.manual_seed(config.seed + rank)
    batches = [(torch.randint(0, 50304, (config.local_batch, config.sequence_length), device=device),
                torch.randint(0, 50304, (config.local_batch, config.sequence_length), device=device))
               for _ in range(config.warmup_steps + config.measure_steps)]
    # Compile-trigger calls are intentionally isolated from measured optimizer state.
    _run_steps(ddp_model, optimizer, batches[:config.warmup_steps], config, ddp_model)
    model = build_model(config, device); ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if group else model
    optimizer = build_optimizer(config, model, group)
    observer = CollectiveObserver()
    if config.sync == "dense" and group:
        register_dense_ddp_hook(ddp_model, observer)
    with observer_scope(observer):
        _run_steps(ddp_model, optimizer, batches[:config.warmup_steps], config, ddp_model)
    observer.events.clear()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    with observer_scope(observer):
        fwd, opt, step, last_loss = _run_steps(ddp_model, optimizer, batches[config.warmup_steps:], config, ddp_model, True)
    result = build_result_skeleton(config)
    result["timing_ms"].update(step_samples=step, fwd_bwd_samples=fwd, optimizer_samples=opt,
                                step_mean=statistics.mean(step) if step else 0.0)
    result["throughput"]["tokens_per_second"] = (config.local_batch * config.sequence_length * config.world_size / (statistics.mean(step) / 1000)) if step else 0.0
    result["memory"] = {"peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
    raw = _raw_model(ddp_model); compressed, uncompressed = _optimizer_parameters(raw)
    result["communication"] = communication_result(
        config, group_parameters_by_shape_dtype(compressed), uncompressed,
    )
    result["environment"] = _environment()
    result["correctness"] = _correctness(ddp_model, config, last_loss, observer)
    result["profiler"]["observed_collectives"] = aggregate_observed(observer)
    if rank == 0 and config.output:
        Path(config.output).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    config = parse_args(argv)
    try:
        if config.profile_only:
            run_profile_mode(config)
        else:
            run_benchmark(config)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
