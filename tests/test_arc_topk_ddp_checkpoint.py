"""DCP and compatibility contracts for ARC DDP compressor checkpoints."""

import copy
import argparse
import json
import os
import socket
import tempfile

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.arc_topk_ddp_hook import ArcTopKDDPParameterSpec, ArcTopKDDPState
from dion.arc_topk_sync import ArcTopKSyncConfig


class _FakeBucket:
    def __init__(self, parameters):
        self._parameters = tuple(parameters)
        self._gradients = tuple(torch.zeros_like(parameter) for parameter in parameters)
        self._buffer = torch.cat([gradient.flatten() for gradient in self._gradients])

    def parameters(self):
        return list(self._parameters)

    def gradients(self):
        return list(self._gradients)

    def buffer(self):
        return self._buffer


def _state(*, optimizer=None):
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    auxiliary = torch.nn.Parameter(torch.zeros(4))
    config = ArcTopKSyncConfig(
        ratio=0.5,
        projection_rank=2,
        eta=0.25,
        seed=17,
        start_compress_step=0,
    )
    state = ArcTopKDDPState(
        process_group=None,
        fingerprint="a" * 64,
        parameter_specs=(
            ArcTopKDDPParameterSpec(matrix, "matrix", 0, "arc_matrix"),
            ArcTopKDDPParameterSpec(auxiliary, "auxiliary", 1, "dense_aux"),
        ),
        optimizer_parameters=(matrix, auxiliary),
        config=config,
        optimizer=optimizer,
    )
    return state, matrix


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda payload: payload["shared"].__setitem__("schema_version", 999), "schema"),
        (lambda payload: payload["shared"].__setitem__("dp_world_size", 2), "world size"),
        (lambda payload: payload["shared"].__setitem__("group_ranks", [1]), "rank membership"),
        (lambda payload: payload["shared"].__setitem__("config_fingerprint", "b" * 64), "fingerprint"),
        (lambda payload: payload["shared"].__setitem__("seed_scheme_version", 999), "seed scheme"),
        (lambda payload: payload["shared"]["config"].__setitem__("eta", 0.5), "config"),
        (lambda payload: payload["shared"]["ordered_parameter_table"][0].__setitem__("role", "dense_aux"), "parameter table"),
        (lambda payload: payload["shared"]["ordered_parameter_table"].append(payload["shared"]["ordered_parameter_table"][0]), "parameter table"),
        (lambda payload: payload["shared"]["ordered_parameter_table"][0].__setitem__("shape", [12]), "parameter table"),
        (lambda payload: payload["shared"]["ordered_parameter_table"][0].__setitem__("dtype", "float64"), "parameter table"),
        (lambda payload: payload["rank_0"].pop("matrix"), "missing.*matrix"),
        (lambda payload: payload["shared"]["g_global"].pop("matrix"), "missing.*matrix"),
        (lambda payload: payload["rank_0"]["matrix"].__setitem__("h_local", torch.zeros(12)), "tensor schema"),
        (lambda payload: payload["rank_0"]["matrix"].__setitem__("g_local", torch.zeros(3, 4, dtype=torch.float64)), "tensor schema"),
    ],
)
def test_load_rejects_incompatible_or_incomplete_compressor_state(mutation, match):
    source, _ = _state()
    payload = copy.deepcopy(source.state_dict())
    mutation(payload)
    destination, _ = _state()

    with pytest.raises(ValueError, match=match):
        destination.load_state_dict(payload)


def test_checkpoint_metadata_validation_does_not_overwrite_runtime_identity():
    state, _ = _state()
    metadata = copy.deepcopy(state.checkpoint_metadata())
    metadata["config_fingerprint"] = "b" * 64

    with pytest.raises(ValueError, match="fingerprint"):
        state.validate_checkpoint_metadata(metadata)

    assert state.fingerprint == "a" * 64


class _StepOptimizer:
    def __init__(self, step):
        self.param_groups = [{"step": step}]


def test_snapshot_and_load_require_matching_committed_muon_step():
    state, _ = _state(optimizer=_StepOptimizer(step=1))

    with pytest.raises(RuntimeError, match="Muon.*compressor"):
        state.state_dict()

    source, _ = _state(optimizer=_StepOptimizer(step=0))
    payload = source.state_dict()
    payload["shared"]["committed_step"] = 2
    destination, _ = _state(optimizer=_StepOptimizer(step=1))
    with pytest.raises(ValueError, match="Muon.*compressor"):
        destination.load_state_dict(payload)


def test_snapshot_rejects_an_active_step_or_in_flight_future():
    state, matrix = _state()
    state.begin_step()

    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.state_dict()


def test_snapshot_rejects_after_finish_but_before_optimizer_commit():
    state, matrix = _state()
    state.begin_step()
    context = state.note_bucket(
        _FakeBucket([matrix, state.parameter_specs[1].parameter])
    )
    context.completion_future.set_result(context.buffer)
    state.finish_step()

    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.state_dict()


class _Loader:
    def __init__(self):
        self.position = 0

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state_dict):
        self.position = state_dict["position"]


class _CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.Linear(8, 8, bias=False)
        self.transformer.wte = torch.nn.Embedding(4, 8)
        self.lm_head = torch.nn.Linear(8, 4, bias=False)

    def forward(self, x, _target):
        matrix_loss = (x @ self.transformer.h.weight.t()).sum()
        auxiliary_coverage = self.transformer.wte.weight.sum() + self.lm_head.weight.sum()
        return matrix_loss + auxiliary_coverage * 0.0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _cli():
    return argparse.Namespace(
        use_gram_newton_schulz=False,
        no_triton=True,
        use_polar_express=False,
        _explicit_replicate_mesh_grad_sync=False,
    )


def _build_runtime(bucket_cap_mb):
    import train_arctopk

    torch.manual_seed(123)
    ddp = DDP(_CheckpointModel(), bucket_cap_mb=bucket_cap_mb)
    optimizer, runtime = train_arctopk.init_arc_topk_optimizer(
        model=ddp.module,
        device_mesh=None,
        ddp_model=ddp,
        hp=train_arctopk.ArcTopKHyperparameters(
            arc_sync_mode="ddp_hook",
            arc_topk_ratio=0.5,
            arc_projection_rank=2,
            arc_eta=0.25,
            arc_start_compress_step=0,
            scalar_opt="adamw",
            lr=0.01,
        ),
        cli_args=_cli(),
    )
    return ddp, optimizer, runtime


def _run_step(ddp, optimizer, runtime, *, rank, step):
    import train

    runtime.begin_step()
    values = torch.arange(1, 9, dtype=torch.float32).unsqueeze(0)
    x = values * (rank + 1) * step
    train.forward_backward_micro_step(
        ddp,
        x,
        None,
        autocast_ctx=nullcontext(),
        micro_step=1,
        grad_accum_steps=1,
        optimizer_owns_gradient_sync=False,
    )
    runtime.finish_step()
    optimizer.step()
    runtime.commit_step()
    ddp.zero_grad(set_to_none=True)


def _snapshot(ddp, state):
    compressor = {}
    for spec in state.parameter_specs:
        if spec.role == "arc_matrix":
            parameter_state = state.parameter_state(spec.parameter)
            compressor[spec.stable_name] = {
                "h_local": parameter_state.h_local.detach().clone(),
                "g_local": parameter_state.g_local.detach().clone(),
                "g_global": parameter_state.g_global.detach().clone(),
            }
    return {
        "parameters": {
            name: parameter.detach().clone()
            for name, parameter in ddp.module.named_parameters()
        },
        "compressor": compressor,
        "step": state.committed_step,
    }


def _assert_snapshot_equal(actual, expected):
    assert actual["step"] == expected["step"]
    assert actual["parameters"].keys() == expected["parameters"].keys()
    for name in actual["parameters"]:
        torch.testing.assert_close(actual["parameters"][name], expected["parameters"][name])
    assert actual["compressor"].keys() == expected["compressor"].keys()
    for name in actual["compressor"]:
        for field in ("h_local", "g_local", "g_global"):
            torch.testing.assert_close(
                actual["compressor"][name][field],
                expected["compressor"][name][field],
            )


def _dcp_round_trip_worker(rank, world_size, port, checkpoint_dir, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        import train

        ddp, optimizer, runtime = _build_runtime(bucket_cap_mb=25)
        state = runtime.checkpoint_state
        train_loader = _Loader()
        val_loader = _Loader()
        manager = train.CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=ddp,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            extra_stateful={"arc_compressor": state},
        )
        for step in range(1, 4):
            _run_step(ddp, optimizer, runtime, rank=rank, step=step)
        saved = _snapshot(ddp, state)
        manager.save(step=3)
        for step in range(4, 6):
            _run_step(ddp, optimizer, runtime, rank=rank, step=step)
        uninterrupted = _snapshot(ddp, state)

        rebuilt_ddp, rebuilt_optimizer, rebuilt_runtime = _build_runtime(
            bucket_cap_mb=0.0001
        )
        rebuilt_state = rebuilt_runtime.checkpoint_state
        rebuilt_manager = train.CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=rebuilt_ddp,
            optimizer=rebuilt_optimizer,
            train_loader=_Loader(),
            val_loader=_Loader(),
            extra_stateful={"arc_compressor": rebuilt_state},
        )
        rebuilt_manager.load()
        _assert_snapshot_equal(_snapshot(rebuilt_ddp, rebuilt_state), saved)
        for step in range(4, 6):
            _run_step(
                rebuilt_ddp,
                rebuilt_optimizer,
                rebuilt_runtime,
                rank=rank,
                step=step,
            )
        _assert_snapshot_equal(
            _snapshot(rebuilt_ddp, rebuilt_state),
            uninterrupted,
        )
        local_tracker = rebuilt_state.parameter_state(
            rebuilt_ddp.module.transformer.h.weight
        ).h_local
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps({"tracker_sum": float(local_tracker.sum())})
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_dcp_two_rank_round_trip_preserves_local_state_and_continuation():
    with tempfile.TemporaryDirectory(prefix="arc-dcp-") as root:
        checkpoint_dir = str(Path(root, "checkpoint-root"))
        output_dir = str(Path(root, "output"))
        Path(output_dir).mkdir()
        mp.spawn(
            _dcp_round_trip_worker,
            args=(2, _free_port(), checkpoint_dir, output_dir),
            nprocs=2,
            join=True,
        )
        results = [
            json.loads(Path(output_dir, f"rank-{rank}.json").read_text())
            for rank in range(2)
        ]

    assert results[0]["tracker_sum"] != results[1]["tracker_sum"]
