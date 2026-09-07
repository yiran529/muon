"""Opt-in profiler for one final accumulation micro-step and optimizer step."""

from __future__ import annotations

import os

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import record_function


@dataclass(frozen=True)
class TrainingProfileConfig:
    output_dir: Path
    profile_step: int


def validate_training_profile_request(profile_step: int, num_iterations: int) -> None:
    if profile_step < 1:
        raise ValueError(f"profile_step must be at least 1, got {profile_step}")
    if profile_step >= num_iterations:
        raise ValueError(
            f"profile_step must be before num_iterations ({num_iterations}), got {profile_step}"
        )


def should_start_profile(
    config: TrainingProfileConfig,
    *,
    step: int,
    micro_step: int,
    grad_accum_steps: int,
) -> bool:
    return step == config.profile_step and micro_step == grad_accum_steps


def rank_trace_path(config: TrainingProfileConfig, rank: int) -> Path:
    return config.output_dir / f"rank-{rank}.json"


def profiled_optimizer_step(optimizer, profile_capture) -> None:
    """Run the optimizer and close an active capture at its exact boundary."""
    with profile_capture.range("train/optimizer"):
        optimizer.step()
    profile_capture.finish()


class TrainingProfileCapture:
    """Manage a single targeted Kineto capture without affecting normal runs."""

    def __init__(self, config: TrainingProfileConfig | None):
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.profiler = None
        self.window = None
        self.active = False
        self.completed = False

    def maybe_start(self, *, step: int, micro_step: int, grad_accum_steps: int) -> bool:
        if (
            self.config is None
            or self.completed
            or not should_start_profile(
                self.config,
                step=step,
                micro_step=micro_step,
                grad_accum_steps=grad_accum_steps,
            )
        ):
            return False
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize()
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.profiler.start()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        self.window = record_function("train/profile_window")
        self.window.__enter__()
        self.active = True
        return True

    def range(self, name: str):
        return record_function(name) if self.active else nullcontext()

    def finish(self) -> None:
        if not self.active:
            return
        torch.cuda.synchronize()
        self.window.__exit__(None, None, None)
        self.profiler.stop()
        self.profiler.export_chrome_trace(str(rank_trace_path(self.config, self.rank)))
        self.active = False
        self.completed = True


def make_training_profile_capture(
    *, output_dir: str | None, profile_step: int | None, num_iterations: int
) -> TrainingProfileCapture:
    if output_dir is None and profile_step is None:
        return TrainingProfileCapture(None)
    if output_dir is None or profile_step is None:
        raise ValueError("--profile-output-dir and --profile-step must be provided together")
    validate_training_profile_request(profile_step, num_iterations)
    return TrainingProfileCapture(
        TrainingProfileConfig(Path(output_dir).resolve(), profile_step)
    )
