"""Tests for stable Sparse-K parameter layouts."""

from dataclasses import replace

import pytest

from dion.sparse_k import SparseKConfig
from dion.sparse_k_layout import (
    SparseKParameterDescriptor,
    canonical_sparse_k_fingerprint,
)


def _parameters():
    return (
        SparseKParameterDescriptor(
            "transformer.h.0.weight", 0, (4, 3), "float32", "sparse_matrix"
        ),
        SparseKParameterDescriptor("scale", 1, (3,), "float32", "dense_aux"),
    )


def _fingerprint(parameters=None, config=None, group_ranks=(0, 1)):
    return canonical_sparse_k_fingerprint(
        config=config or SparseKConfig(method="randk", ratio=0.5, seed=17),
        group_ranks=group_ranks,
        parameters=_parameters() if parameters is None else parameters,
    )


def test_equal_sparse_k_layouts_have_the_same_sha256_fingerprint():
    assert _fingerprint() == _fingerprint(
        tuple(replace(item) for item in _parameters())
    )
    assert len(_fingerprint()) == 64


@pytest.mark.parametrize(
    "parameters",
    [
        (replace(_parameters()[0], stable_name="renamed"), _parameters()[1]),
        (replace(_parameters()[0], stable_id=9), _parameters()[1]),
        (replace(_parameters()[0], shape=(3, 4)), _parameters()[1]),
        (replace(_parameters()[0], dtype="bfloat16"), _parameters()[1]),
        (replace(_parameters()[0], role="dense_aux"), _parameters()[1]),
        tuple(reversed(_parameters())),
    ],
)
def test_every_parameter_layout_field_affects_fingerprint(parameters):
    assert _fingerprint(parameters) != _fingerprint()


def test_method_config_and_group_membership_affect_fingerprint():
    baseline = _fingerprint()

    assert (
        _fingerprint(config=SparseKConfig(method="topk", ratio=0.5, seed=17))
        != baseline
    )
    assert _fingerprint(group_ranks=(1, 0)) != baseline


@pytest.mark.parametrize(
    "field,match", [("stable_name", "stable name"), ("stable_id", "stable ID")]
)
def test_duplicate_stable_identity_is_rejected(field, match):
    replacement = {field: getattr(_parameters()[0], field)}
    parameters = (_parameters()[0], replace(_parameters()[1], **replacement))

    with pytest.raises(ValueError, match=match):
        _fingerprint(parameters)
