"""DCP and compatibility contracts for GreedyLore DDP checkpoints."""

import argparse
import copy
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
from torch.distributed.checkpoint.api import CheckpointException
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.greedy_lore import GreedyLoreConfig
from dion.greedy_lore_ddp_hook import GreedyLoreDDPParameterSpec, GreedyLoreDDPState


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


def _state(*, config=None):
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    auxiliary = torch.nn.Parameter(torch.zeros(4))
    config = config or GreedyLoreConfig(
        rank=2,
        update_interval=3,
        seed=17,
        start_compress_step=0,
        basis_sync="local_svd",
    )
    state = GreedyLoreDDPState(
        process_group=None,
        fingerprint="a" * 64,
        parameter_specs=(
            GreedyLoreDDPParameterSpec(matrix, "matrix", 0, "matrix"),
            GreedyLoreDDPParameterSpec(auxiliary, "auxiliary", 1, "dense_aux"),
        ),
        optimizer_parameters=(matrix, auxiliary),
        config=config,
    )
    return state, matrix


def test_metadata_enumerates_exact_stable_name_tensor_schema():
    state, _ = _state()

    assert state.checkpoint_metadata()["tensor_schema"] == {
        "shared": {
            "schema_version": {"shape": [], "dtype": "int64"},
            "committed_step": {"shape": [], "dtype": "int64"},
        },
        "rank_0": {
            "matrix": {
                "error": {"shape": [3, 4], "dtype": "float32"},
                "basis": {"shape": [3, 3], "dtype": "float32"},
                "last_support": {"shape": [2], "dtype": "int64"},
            }
        },
    }


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda metadata: metadata.__setitem__("schema_version", 999), "schema"),
        (lambda metadata: metadata.__setitem__("dp_world_size", 2), "world size"),
        (lambda metadata: metadata.__setitem__("group_ranks", [1]), "rank membership"),
        (
            lambda metadata: metadata.__setitem__("config_fingerprint", "b" * 64),
            "fingerprint",
        ),
        (
            lambda metadata: metadata.__setitem__("seed_scheme_version", 999),
            "seed scheme",
        ),
        (lambda metadata: metadata["config"].__setitem__("rank", 1), "config"),
        (
            lambda metadata: metadata["config"].__setitem__("update_interval", 2),
            "config",
        ),
        (
            lambda metadata: metadata["config"].__setitem__("start_compress_step", 2),
            "config",
        ),
        (
            lambda metadata: metadata["config"].__setitem__("basis_sync", "broadcast"),
            "config",
        ),
        (
            lambda metadata: metadata["ordered_parameter_table"][0].__setitem__(
                "role", "dense_aux"
            ),
            "parameter table",
        ),
        (
            lambda metadata: metadata["ordered_parameter_table"][0].__setitem__(
                "stable_name", "renamed"
            ),
            "parameter table",
        ),
        (
            lambda metadata: metadata["ordered_parameter_table"].reverse(),
            "parameter table",
        ),
        (
            lambda metadata: metadata["ordered_parameter_table"].append(
                metadata["ordered_parameter_table"][0]
            ),
            "parameter table",
        ),
        (
            lambda metadata: metadata["ordered_parameter_table"][0].__setitem__(
                "shape", [12]
            ),
            "parameter table",
        ),
        (
            lambda metadata: metadata["ordered_parameter_table"][0].__setitem__(
                "dtype", "float64"
            ),
            "parameter table",
        ),
        (
            lambda metadata: metadata["tensor_schema"]["rank_0"].pop("matrix"),
            "missing.*matrix",
        ),
        (
            lambda metadata: metadata["tensor_schema"]["rank_0"]["matrix"][
                "error"
            ].__setitem__("shape", [12]),
            "tensor schema",
        ),
        (
            lambda metadata: metadata["tensor_schema"]["rank_0"]["matrix"][
                "basis"
            ].__setitem__("dtype", "float64"),
            "tensor schema",
        ),
        (lambda metadata: metadata.__setitem__("committed_step", -1), "committed step"),
    ],
)
def test_metadata_validation_rejects_incompatible_greedylore_checkpoint(
    mutation, match
):
    state, _ = _state()
    metadata = copy.deepcopy(state.checkpoint_metadata())
    mutation(metadata)

    with pytest.raises(ValueError, match=match):
        state.validate_checkpoint_metadata(metadata)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (
            lambda payload: payload["shared"].__setitem__(
                "schema_version", torch.tensor(999)
            ),
            "schema",
        ),
        (
            lambda payload: payload["shared"].__setitem__(
                "committed_step", torch.tensor(-1)
            ),
            "committed step",
        ),
        (lambda payload: payload.pop("rank_0"), "rank_0"),
        (lambda payload: payload["rank_0"].pop("matrix"), "missing.*matrix"),
        (
            lambda payload: payload["rank_0"]["matrix"].__setitem__(
                "error", torch.zeros(12)
            ),
            "tensor schema",
        ),
        (
            lambda payload: payload["rank_0"]["matrix"].__setitem__(
                "basis", torch.zeros(3, 3, dtype=torch.float64)
            ),
            "tensor schema",
        ),
        (
            lambda payload: payload["rank_0"]["matrix"].__setitem__(
                "last_support", torch.zeros(2, dtype=torch.int32)
            ),
            "tensor schema",
        ),
    ],
)
def test_load_state_dict_rejects_incompatible_or_incomplete_tensor_payload(
    mutation, match
):
    source, _ = _state()
    payload = copy.deepcopy(source.state_dict())
    mutation(payload)
    destination, _ = _state()

    with pytest.raises(ValueError, match=match):
        destination.load_state_dict(payload)


def test_load_requires_payload_committed_step_to_match_validated_metadata():
    source, _ = _state()
    metadata = source.checkpoint_metadata()
    payload = copy.deepcopy(source.state_dict())
    payload["shared"]["committed_step"] = torch.tensor(1, dtype=torch.int64)
    destination, _ = _state()
    destination.validate_checkpoint_metadata(metadata)

    with pytest.raises(ValueError, match="committed step"):
        destination.load_state_dict(payload)


def test_snapshot_and_load_reject_active_step_or_unfinished_tail():
    state, matrix = _state()
    state.begin_step()
    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.state_dict()
    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.load_state_dict(_state()[0].state_dict())

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
        self.transformer.h = torch.nn.Sequential(
            torch.nn.Linear(8, 8, bias=False),
            torch.nn.Linear(8, 8, bias=False),
        )
        self.transformer.wte = torch.nn.Embedding(4, 8)
        self.lm_head = torch.nn.Linear(8, 4, bias=False)

    def forward(self, x, _target):
        first = self.transformer.h[0](x).sum()
        second = self.transformer.h[1](x).sum()
        auxiliary = self.transformer.wte.weight.sum() + self.lm_head.weight.sum()
        return first + second + auxiliary * 0.0


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
    import train_greedylore

    torch.manual_seed(123)
    ddp = DDP(_CheckpointModel(), bucket_cap_mb=bucket_cap_mb)
    optimizer, runtime = train_greedylore.init_greedy_lore_optimizer(
        model=ddp.module,
        device_mesh=None,
        ddp_model=ddp,
        hp=train_greedylore.GreedyLoreHyperparameters(
            greedy_lore_rank=1,
            greedy_lore_update_interval=3,
            greedy_lore_seed=17,
            greedy_lore_start_compress_step=0,
            greedy_lore_basis_sync="local_svd",
            scalar_opt="adamw",
            lr=0.01,
            weight_decay=0.1,
            adjust_lr=None,
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
    reconstructed_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in ddp.module.named_parameters()
        if parameter.grad is not None
    }
    runtime.finish_step()
    optimizer.step()
    runtime.commit_step()
    ddp.zero_grad(set_to_none=True)
    return reconstructed_gradients


def _snapshot(ddp, optimizer, state, reconstructed_gradients=None):
    compressor = {}
    for spec in state.parameter_specs:
        if spec.role != "matrix":
            continue
        parameter_state = state.parameter_state(spec.parameter)
        compressor[spec.stable_name] = {
            "error": parameter_state.error.detach().clone(),
            "basis": parameter_state.basis.detach().clone(),
            "last_support": parameter_state.last_support.detach().clone(),
        }
    return {
        "parameters": {
            name: parameter.detach().clone()
            for name, parameter in ddp.module.named_parameters()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "compressor": compressor,
        "gradients": reconstructed_gradients or {},
        "step": state.committed_step,
    }


def _assert_optimizer_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    assert len(actual["param_groups"]) == len(expected["param_groups"])
    for actual_group, expected_group in zip(
        actual["param_groups"], expected["param_groups"]
    ):
        assert actual_group.keys() == expected_group.keys()
        for key, expected_value in expected_group.items():
            actual_value = actual_group[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in expected["state"]:
        assert (
            actual["state"][parameter_id].keys()
            == expected["state"][parameter_id].keys()
        )
        for key, expected_value in expected["state"][parameter_id].items():
            actual_value = actual["state"][parameter_id][key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _assert_snapshot_equal(actual, expected):
    assert actual["step"] == expected["step"]
    assert actual["parameters"].keys() == expected["parameters"].keys()
    for name in actual["parameters"]:
        torch.testing.assert_close(
            actual["parameters"][name], expected["parameters"][name]
        )
    assert actual["compressor"].keys() == expected["compressor"].keys()
    for name in actual["compressor"]:
        for field in actual["compressor"][name]:
            torch.testing.assert_close(
                actual["compressor"][name][field],
                expected["compressor"][name][field],
            )
    assert actual["gradients"].keys() == expected["gradients"].keys()
    for name in actual["gradients"]:
        torch.testing.assert_close(
            actual["gradients"][name], expected["gradients"][name]
        )
    _assert_optimizer_state_equal(actual["optimizer"], expected["optimizer"])


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
        manager = train.CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=ddp,
            optimizer=optimizer,
            train_loader=_Loader(),
            val_loader=_Loader(),
            extra_stateful={"greedy_lore_compressor": state},
        )
        for step in range(1, 4):
            _run_step(ddp, optimizer, runtime, rank=rank, step=step)
        state.validate_replicated_basis_across_ranks()
        saved = _snapshot(ddp, optimizer, state)
        manager.save(step=3)
        local_error_sum = float(
            state.parameter_state(ddp.module.transformer.h[0].weight).error.sum()
        )

        for step in range(4, 7):
            gradients = _run_step(ddp, optimizer, runtime, rank=rank, step=step)
        state.validate_replicated_basis_across_ranks()
        uninterrupted = _snapshot(ddp, optimizer, state, gradients)

        del manager, runtime, state, optimizer, ddp

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
            extra_stateful={"greedy_lore_compressor": rebuilt_state},
        )
        rebuilt_manager.load()
        rebuilt_state.validate_replicated_basis_across_ranks()
        assert rebuilt_state.committed_step == saved["step"]
        _assert_snapshot_equal(
            _snapshot(rebuilt_ddp, rebuilt_optimizer, rebuilt_state), saved
        )
        for step in range(4, 7):
            rebuilt_gradients = _run_step(
                rebuilt_ddp,
                rebuilt_optimizer,
                rebuilt_runtime,
                rank=rank,
                step=step,
            )
        rebuilt_state.validate_replicated_basis_across_ranks()
        _assert_snapshot_equal(
            _snapshot(
                rebuilt_ddp,
                rebuilt_optimizer,
                rebuilt_state,
                rebuilt_gradients,
            ),
            uninterrupted,
        )
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps({"local_error_sum": local_error_sum})
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _corrupt_dcp_payload_worker(rank, world_size, port, checkpoint_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        import train

        ddp, optimizer, runtime = _build_runtime(bucket_cap_mb=25)
        state = runtime.checkpoint_state
        manager = train.CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=ddp,
            optimizer=optimizer,
            train_loader=_Loader(),
            val_loader=_Loader(),
            extra_stateful={"greedy_lore_compressor": state},
        )
        _run_step(ddp, optimizer, runtime, rank=rank, step=1)
        manager.save(step=1)
        if rank == 0:
            payload = next(Path(checkpoint_dir, "checkpoint").glob("*.distcp"))
            payload.write_bytes(payload.read_bytes()[:1])
        dist.barrier()

        rebuilt_ddp, rebuilt_optimizer, rebuilt_runtime = _build_runtime(
            bucket_cap_mb=25
        )
        rebuilt_manager = train.CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=rebuilt_ddp,
            optimizer=rebuilt_optimizer,
            train_loader=_Loader(),
            val_loader=_Loader(),
            extra_stateful={"greedy_lore_compressor": rebuilt_runtime.checkpoint_state},
        )
        with pytest.raises(CheckpointException):
            rebuilt_manager.load()
    finally:
        dist.destroy_process_group()


def test_real_two_rank_dcp_round_trip_preserves_local_state_and_continuation():
    with tempfile.TemporaryDirectory(prefix="greedylore-dcp-") as root:
        checkpoint_dir = str(Path(root, "checkpoint-root"))
        output_dir = Path(root, "output")
        output_dir.mkdir()
        mp.spawn(
            _dcp_round_trip_worker,
            args=(2, _free_port(), checkpoint_dir, str(output_dir)),
            nprocs=2,
            join=True,
        )
        results = [
            json.loads(Path(output_dir, f"rank-{rank}.json").read_text())
            for rank in range(2)
        ]

    assert results[0]["local_error_sum"] != results[1]["local_error_sum"]


def test_corrupted_real_dcp_payload_reports_failure():
    with tempfile.TemporaryDirectory(prefix="greedylore-dcp-corrupt-") as root:
        checkpoint_dir = str(Path(root, "checkpoint-root"))
        mp.spawn(
            _corrupt_dcp_payload_worker,
            args=(2, _free_port(), checkpoint_dir),
            nprocs=2,
            join=True,
        )
