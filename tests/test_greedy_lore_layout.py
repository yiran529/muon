"""Canonical GreedyLore layout fingerprint contracts."""

from dataclasses import replace

import pytest

from dion.greedy_lore import GreedyLoreConfig
import dion.greedy_lore_layout as layout


def _parameters():
    return (
        layout.GreedyLoreParameterDescriptor(
            "transformer.h.0.weight", 0, (5, 3), "float32", "matrix"
        ),
        layout.GreedyLoreParameterDescriptor(
            "lm_head.weight", 1, (8, 3), "float32", "dense_aux"
        ),
    )


def _fingerprint(*, config=None, group_ranks=(0, 1), parameters=None):
    return layout.canonical_greedy_lore_fingerprint(
        config=config or GreedyLoreConfig(rank=2),
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
        lambda c: replace(c, seed=c.seed + 1),
        lambda c: replace(c, rank=c.rank + 1),
        lambda c: replace(c, update_interval=c.update_interval + 1),
        lambda c: replace(c, start_compress_step=c.start_compress_step + 1),
        lambda c: replace(c, basis_sync="broadcast"),
    ],
)
def test_fingerprint_is_sensitive_to_each_runtime_configuration_value(mutation):
    config = GreedyLoreConfig(rank=2)
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
    assert _fingerprint(parameters=parameters(original)) != _fingerprint(
        parameters=original
    )


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
