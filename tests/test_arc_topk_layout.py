"""Tests for canonical ARC parameter and optimizer-task layouts."""

from dataclasses import replace

import pytest

import dion.arc_topk_layout as layout
from dion.arc_topk_sync import ArcTopKSyncConfig


_DEFAULT = object()


def _parameters():
    return (
        layout.ArcParameterDescriptor(
            stable_name="transformer.h.0.weight",
            stable_id=0,
            shape=(4, 3),
            dtype="float32",
            role="arc_matrix",
        ),
        layout.ArcParameterDescriptor(
            stable_name="lm_head.weight",
            stable_id=1,
            shape=(8, 3),
            dtype="float32",
            role="dense_aux",
        ),
    )


def _tasks(config=None):
    config = config or ArcTopKSyncConfig(
        ratio=0.5,
        projection_rank=2,
        eta=0.25,
        seed=17,
        start_compress_step=0,
    )
    return (
        layout.ArcOptimizerTaskDescriptor(
            group_id=0,
            task_id=0,
            ordered_parameter_names=("transformer.h.0.weight",),
            shape=(4, 3),
            dtype="float32",
            config=config,
        ),
    )


def _fingerprint(*, parameters=None, tasks=_DEFAULT, base_seed=17, config=None):
    config = config or ArcTopKSyncConfig(
        ratio=0.5,
        projection_rank=2,
        eta=0.25,
        seed=17,
        start_compress_step=0,
    )
    return layout.canonical_arc_fingerprint(
        base_seed=base_seed,
        config=config,
        group_ranks=(0, 1),
        parameters=_parameters() if parameters is None else parameters,
        optimizer_tasks=_tasks(config) if tasks is _DEFAULT else tasks,
    )


def test_equal_value_layouts_have_the_same_sha256_fingerprint():
    first = _fingerprint()
    second = _fingerprint(
        parameters=tuple(replace(parameter) for parameter in _parameters()),
        tasks=tuple(replace(task) for task in _tasks()),
    )

    assert first == second
    assert len(first) == 64
    int(first, 16)


@pytest.mark.parametrize(
    "parameters",
    [
        (replace(_parameters()[0], stable_name="renamed.weight"), _parameters()[1]),
        (replace(_parameters()[0], stable_id=9), _parameters()[1]),
        (replace(_parameters()[0], shape=(5, 3)), _parameters()[1]),
        (replace(_parameters()[0], dtype="bfloat16"), _parameters()[1]),
        (replace(_parameters()[0], role="dense_aux"), _parameters()[1]),
        tuple(reversed(_parameters())),
    ],
)
def test_parameter_name_id_order_shape_dtype_and_role_affect_fingerprint(parameters):
    assert _fingerprint(parameters=parameters) != _fingerprint()


def test_seed_group_membership_and_arc_config_affect_fingerprint():
    baseline = _fingerprint()
    changed_config = ArcTopKSyncConfig(
        ratio=0.25,
        projection_rank=2,
        eta=0.25,
        seed=17,
        start_compress_step=0,
    )

    assert _fingerprint(base_seed=18) != baseline
    assert layout.canonical_arc_fingerprint(
        base_seed=17,
        config=ArcTopKSyncConfig(0.5, 2, 0.25, 17, 0),
        group_ranks=(1, 0),
        parameters=_parameters(),
        optimizer_tasks=_tasks(),
    ) != baseline
    assert _fingerprint(config=changed_config) != baseline


def test_optimizer_task_membership_order_and_config_affect_fingerprint():
    baseline = _fingerprint()
    task = _tasks()[0]
    second_name = "transformer.h.1.weight"
    two_member_task = replace(
        task,
        ordered_parameter_names=(task.ordered_parameter_names[0], second_name),
    )
    reversed_task = replace(
        two_member_task,
        ordered_parameter_names=tuple(reversed(two_member_task.ordered_parameter_names)),
    )
    changed_config_task = replace(
        task,
        config=replace(task.config, projection_rank=3),
    )

    assert _fingerprint(tasks=(two_member_task,)) != baseline
    assert _fingerprint(tasks=(reversed_task,)) != _fingerprint(tasks=(two_member_task,))
    assert _fingerprint(tasks=(replace(task, group_id=1),)) != baseline
    assert _fingerprint(tasks=(replace(task, task_id=1),)) != baseline
    assert _fingerprint(tasks=(changed_config_task,)) != baseline


def test_hook_and_optimizer_layouts_are_distinct_canonical_payloads():
    hook_fingerprint = _fingerprint(tasks=None)
    optimizer_fingerprint = _fingerprint()

    assert hook_fingerprint != optimizer_fingerprint


@pytest.mark.parametrize(
    "parameters,match",
    [
        (
            (_parameters()[0], replace(_parameters()[1], stable_name=_parameters()[0].stable_name)),
            "stable name",
        ),
        (
            (_parameters()[0], replace(_parameters()[1], stable_id=_parameters()[0].stable_id)),
            "stable ID",
        ),
    ],
)
def test_duplicate_stable_names_and_ids_are_rejected(parameters, match):
    with pytest.raises(ValueError, match=match):
        _fingerprint(parameters=parameters)
