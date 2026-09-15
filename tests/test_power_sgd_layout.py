from dataclasses import replace
import os
import socket
from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.power_sgd import PowerSGDConfig
import dion.power_sgd_layout as layout


def _parameters():
    return (
        layout.PowerSGDParameterDescriptor(
            "transformer.h.0.weight", 0, (5, 3), "float32", "matrix"
        ),
        layout.PowerSGDParameterDescriptor(
            "lm_head.weight", 1, (8, 3), "float32", "dense_aux"
        ),
    )


def _fingerprint(*, config=None, group_ranks=(0, 1), parameters=None):
    return layout.canonical_power_sgd_fingerprint(
        config=config or PowerSGDConfig(rank=2),
        group_ranks=group_ranks,
        parameters=parameters or _parameters(),
    )


def test_equal_value_layouts_have_equal_sha256_fingerprints():
    first = _fingerprint()
    second = _fingerprint(parameters=tuple(replace(item) for item in _parameters()))
    assert first == second
    assert len(first) == 64
    int(first, 16)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: replace(c, rank=c.rank + 1),
        lambda c: replace(c, start_compress_step=c.start_compress_step + 1),
        lambda c: replace(c, min_compression_rate=c.min_compression_rate + 1),
        lambda c: replace(c, error_feedback="none"),
        lambda c: replace(c, warm_start=not c.warm_start),
        lambda c: replace(c, seed=c.seed + 1),
        lambda c: replace(c, orthogonalization_epsilon=c.orthogonalization_epsilon * 2),
    ],
)
def test_fingerprint_is_sensitive_to_each_runtime_configuration_value(mutation):
    config = PowerSGDConfig(rank=2)
    assert _fingerprint(config=mutation(config)) != _fingerprint(config=config)


def test_fingerprint_is_sensitive_to_ordered_group_membership():
    assert _fingerprint(group_ranks=(0, 2)) != _fingerprint(group_ranks=(0, 1))
    assert _fingerprint(group_ranks=(1, 0)) != _fingerprint(group_ranks=(0, 1))


@pytest.mark.parametrize(
    "parameters",
    [
        lambda p: tuple(reversed(p)),
        lambda p: (replace(p[0], stable_name="renamed"), p[1]),
        lambda p: (replace(p[0], stable_id=7), p[1]),
        lambda p: (replace(p[0], shape=(3, 5)), p[1]),
        lambda p: (replace(p[0], dtype="float64"), p[1]),
        lambda p: (replace(p[0], role="dense_aux"), p[1]),
    ],
)
def test_fingerprint_is_sensitive_to_ordered_parameter_values(parameters):
    original = _parameters()
    assert _fingerprint(parameters=parameters(original)) != _fingerprint(parameters=original)


@pytest.mark.parametrize(
    "parameters,match",
    [
        (
            lambda p: (p[0], replace(p[1], stable_name=p[0].stable_name)),
            "stable names",
        ),
        (
            lambda p: (p[0], replace(p[1], stable_id=p[0].stable_id)),
            "stable IDs",
        ),
    ],
)
def test_fingerprint_rejects_duplicate_stable_identity(parameters, match):
    with pytest.raises(ValueError, match=match):
        _fingerprint(parameters=parameters(_parameters()))


def _validation_worker(rank, world_size, port, mismatch):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))
    try:
        fingerprint = _fingerprint(group_ranks=(0, 1))
        if rank == 1 and mismatch:
            fingerprint = _fingerprint(group_ranks=(1, 0))
        message = None
        try:
            layout.validate_power_sgd_fingerprint_across_ranks(fingerprint, dist.group.WORLD)
        except layout.PowerSGDLayoutMismatch as exc:
            message = str(exc)
        messages = [None] * world_size
        dist.all_gather_object(messages, message)
        if mismatch:
            assert messages[0] == messages[1]
            assert "PowerSGD layout mismatch" in messages[0]
        else:
            assert messages == [None, None]
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("mismatch", [False, True])
def test_distributed_validation_is_symmetric(mismatch):
    mp.spawn(_validation_worker, args=(2, _free_port(), mismatch), nprocs=2, join=True)
