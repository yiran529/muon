from pathlib import Path

import pytest

import train
from benchmark.compressed_muon.training_profiler import (
    TrainingProfileConfig,
    profiled_optimizer_step,
    rank_trace_path,
    should_start_profile,
    validate_training_profile_request,
)


def test_profile_request_requires_a_nonfinal_training_step(tmp_path: Path):
    validate_training_profile_request(12, 13)

    with pytest.raises(ValueError, match="profile_step"):
        validate_training_profile_request(0, 13)
    with pytest.raises(ValueError, match="before num_iterations"):
        validate_training_profile_request(13, 13)


def test_capture_starts_only_on_selected_final_micro_step(tmp_path: Path):
    config = TrainingProfileConfig(output_dir=tmp_path, profile_step=12)

    assert not should_start_profile(config, step=11, micro_step=256, grad_accum_steps=256)
    assert not should_start_profile(config, step=12, micro_step=255, grad_accum_steps=256)
    assert should_start_profile(config, step=12, micro_step=256, grad_accum_steps=256)


def test_each_rank_gets_a_distinct_trace_path(tmp_path: Path):
    config = TrainingProfileConfig(output_dir=tmp_path, profile_step=12)

    assert rank_trace_path(config, rank=0) == tmp_path / "rank-0.json"
    assert rank_trace_path(config, rank=2) == tmp_path / "rank-2.json"


def test_shared_training_cli_accepts_targeted_profiler_arguments(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--profile-output-dir",
            str(tmp_path),
            "--profile-step",
            "12",
            "--training-seed",
            "42",
            "--val_tokens",
            "3072",
            "--timing-warmup-steps",
            "7",
            "--bucket-cap-mb",
            "5",
        ],
    )

    args = train.parse_cli_args()

    assert args.profile_output_dir == str(tmp_path)
    assert args.profile_step == 12
    assert args.training_seed == 42
    assert args.val_tokens == 3072
    assert args.timing_warmup_steps == 7
    assert args.bucket_cap_mb == 5


def test_nondefault_timing_warmup_excludes_exactly_completed_warmup_steps():
    assert train.completed_timed_steps(step=7, timing_warmup_steps=7) == 0
    assert train.completed_timed_steps(step=8, timing_warmup_steps=7) == 1
    assert train.completed_timed_steps(step=107, timing_warmup_steps=7) == 100


def test_profile_capture_finishes_immediately_after_real_optimizer_step():
    parameter = train.torch.nn.Parameter(train.torch.tensor([1.0]))
    parameter.grad = train.torch.tensor([2.0])
    optimizer = train.torch.optim.SGD([parameter], lr=0.25)

    class Capture:
        def __init__(self):
            self.value_at_finish = None

        def range(self, _name):
            return train.nullcontext()

        def finish(self):
            self.value_at_finish = parameter.detach().item()

    capture = Capture()
    profiled_optimizer_step(optimizer, capture)

    assert parameter.item() == pytest.approx(0.5)
    assert capture.value_at_finish == pytest.approx(0.5)
