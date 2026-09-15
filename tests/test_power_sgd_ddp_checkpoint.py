"""State lifecycle and checkpoint contracts for the PowerSGD DDP hook."""

import copy

import pytest
import torch

from dion.power_sgd import PowerSGDConfig
from dion.power_sgd_ddp_hook import PowerSGDDDPParameterSpec, PowerSGDDDPState


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


def _state(*, dtype=torch.float32, config=None):
    matrix = torch.nn.Parameter(torch.zeros(8, 16, dtype=dtype))
    auxiliary = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
    state = PowerSGDDDPState(
        process_group=None,
        fingerprint="a" * 64,
        parameter_specs=(
            PowerSGDDDPParameterSpec(matrix, "matrix", 0, "matrix"),
            PowerSGDDDPParameterSpec(auxiliary, "auxiliary", 1, "dense_aux"),
        ),
        optimizer_parameters=(matrix, auxiliary),
        config=config
        or PowerSGDConfig(rank=2, start_compress_step=1, min_compression_rate=2.0),
    )
    return state, matrix, auxiliary


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_state_preallocates_native_matrix_state_only_when_profitable(dtype):
    state, matrix, auxiliary = _state(dtype=dtype)
    item = state.parameter_state(matrix)

    assert item.error.shape == (8, 16)
    assert item.error.dtype == dtype
    assert item.q_memory.shape == (16, 2)
    assert item.q_memory.dtype == dtype
    assert item.q_initialized.shape == torch.Size([])
    assert item.q_initialized.dtype == torch.bool
    assert not bool(item.q_initialized)
    with pytest.raises(ValueError, match="dense auxiliary"):
        state.parameter_state(auxiliary)

    unprofitable = torch.nn.Parameter(torch.zeros(4, 4, dtype=dtype))
    dense_state = PowerSGDDDPState(
        process_group=None,
        fingerprint="b" * 64,
        parameter_specs=(
            PowerSGDDDPParameterSpec(unprofitable, "unprofitable", 0, "matrix"),
        ),
        optimizer_parameters=(unprofitable,),
        config=PowerSGDConfig(rank=2, min_compression_rate=2.0),
    )
    with pytest.raises(ValueError, match="dense fallback"):
        dense_state.parameter_state(unprofitable)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_name", "stable names"),
        ("duplicate_id", "stable IDs"),
        ("duplicate_parameter", "unique parameters"),
        ("duplicate_optimizer", "optimizer parameters"),
        ("unsupported_role", "role"),
    ],
)
def test_state_rejects_ambiguous_frozen_identity_tables(mutation, match):
    first = torch.nn.Parameter(torch.zeros(8, 16))
    second = torch.nn.Parameter(torch.zeros(8, 16))
    parameters = [(first, "first", "matrix"), (second, "second", "matrix")]
    specs = [
        PowerSGDDDPParameterSpec(first, "first", 0, "matrix"),
        PowerSGDDDPParameterSpec(second, "second", 1, "matrix"),
    ]
    if mutation == "duplicate_name":
        specs[1] = PowerSGDDDPParameterSpec(second, "first", 1, "matrix")
    elif mutation == "duplicate_id":
        specs[1] = PowerSGDDDPParameterSpec(second, "second", 0, "matrix")
    elif mutation == "duplicate_parameter":
        specs[1] = PowerSGDDDPParameterSpec(first, "second", 1, "matrix")
    elif mutation == "unsupported_role":
        specs[1] = PowerSGDDDPParameterSpec(second, "second", 1, "other")
    if mutation == "duplicate_optimizer":
        parameters[1] = (first, "second", "matrix")

    with pytest.raises(ValueError, match=match):
        PowerSGDDDPState(
            process_group=None,
            fingerprint="a" * 64,
            parameter_specs=specs,
            optimizer_parameters=tuple(item[0] for item in parameters),
            config=PowerSGDConfig(rank=2),
        )


def test_specs_and_optimizer_parameters_require_exact_identity_coverage():
    model_only = torch.nn.Parameter(torch.zeros(8, 16))
    optimizer_only = torch.nn.Parameter(torch.zeros(8, 16))
    spec = PowerSGDDDPParameterSpec(model_only, "model", 0, "matrix")

    with pytest.raises(ValueError, match="not owned by the optimizer"):
        PowerSGDDDPState(
            process_group=None,
            fingerprint="a" * 64,
            parameter_specs=(spec,),
            optimizer_parameters=(optimizer_only,),
            config=PowerSGDConfig(rank=2),
        )
    with pytest.raises(ValueError, match="absent from the model"):
        PowerSGDDDPState(
            process_group=None,
            fingerprint="a" * 64,
            parameter_specs=(),
            optimizer_parameters=(model_only,),
            config=PowerSGDConfig(rank=2),
        )


@pytest.mark.parametrize(
    "parameter,role,match",
    [
        (torch.nn.Parameter(torch.zeros(8)), "matrix", "two-dimensional"),
        (
            torch.nn.Parameter(torch.zeros(8, 16, dtype=torch.float64)),
            "matrix",
            "FP32 or BF16",
        ),
        (
            torch.nn.Parameter(torch.zeros(8, dtype=torch.float64)),
            "dense_aux",
            "FP32 or BF16",
        ),
    ],
)
def test_state_rejects_unsupported_shape_or_dtype(parameter, role, match):
    with pytest.raises(ValueError, match=match):
        PowerSGDDDPState(
            process_group=None,
            fingerprint="a" * 64,
            parameter_specs=(PowerSGDDDPParameterSpec(parameter, "item", 0, role),),
            optimizer_parameters=(parameter,),
            config=PowerSGDConfig(rank=2),
        )


def test_state_rejects_find_unused_parameters_mode():
    state, matrix, auxiliary = _state()
    with pytest.raises(ValueError, match="find_unused_parameters=False"):
        PowerSGDDDPState(
            process_group=None,
            fingerprint=state.fingerprint,
            parameter_specs=state.parameter_specs,
            optimizer_parameters=(matrix, auxiliary),
            config=state.config,
            find_unused_parameters=True,
        )


def test_lifecycle_requires_one_step_exact_coverage_and_completed_futures():
    state, matrix, auxiliary = _state()

    with pytest.raises(RuntimeError, match="active PowerSGD step"):
        state.note_bucket(_FakeBucket((matrix, auxiliary)))
    assert state.begin_step() == 1
    with pytest.raises(RuntimeError, match="already active"):
        state.begin_step()
    context = state.note_bucket(_FakeBucket((matrix,)))
    assert context.step == 1
    assert context.phase is None
    assert context.parameter_states == (state.parameter_state(matrix),)
    with pytest.raises(RuntimeError, match="more than once"):
        state.note_bucket(_FakeBucket((matrix,)))
    with pytest.raises(RuntimeError, match="missing.*auxiliary"):
        state.finish_step()
    auxiliary_context = state.note_bucket(_FakeBucket((auxiliary,)))
    auxiliary_context.completion_future.set_result(auxiliary_context.buffer)
    with pytest.raises(RuntimeError, match="in flight"):
        state.finish_step()
    with pytest.raises(RuntimeError, match="before finish"):
        state.commit_step()

    context.completion_future.set_result(context.buffer)
    state.finish_step()
    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.state_dict()
    state.commit_step()
    assert state.committed_step == 1
    with pytest.raises(RuntimeError, match="no active"):
        state.commit_step()


def test_checkpoint_operations_reject_an_active_step():
    state, _, _ = _state()
    state.begin_step()

    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.checkpoint_metadata()
    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.validate_checkpoint_metadata({})
    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.state_dict()


def test_finish_waits_for_every_bucket_not_only_the_latest():
    state, matrix, auxiliary = _state()
    state.begin_step()
    first = state.note_bucket(_FakeBucket((matrix,)))
    second = state.note_bucket(_FakeBucket((auxiliary,)))
    second.completion_future.set_result(second.buffer)

    with pytest.raises(RuntimeError, match="in flight"):
        state.finish_step()

    first.completion_future.set_result(first.buffer)
    state.finish_step()


def test_metadata_enumerates_exact_stable_name_tensor_schema():
    state, _, _ = _state()

    assert state.checkpoint_metadata()["tensor_schema"] == {
        "shared": {
            "schema_version": {"shape": [], "dtype": "int64"},
            "committed_step": {"shape": [], "dtype": "int64"},
        },
        "rank_0": {
            "matrix": {
                "error": {"shape": [8, 16], "dtype": "float32"},
                "q_memory": {"shape": [16, 2], "dtype": "float32"},
                "q_initialized": {"shape": [], "dtype": "bool"},
            }
        },
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_round_trip_preserves_rank_local_state_and_step(dtype):
    source, source_matrix, _ = _state(dtype=dtype)
    source_state = source.parameter_state(source_matrix)
    source_state.error.fill_(0.25)
    source_state.q_memory.copy_(torch.arange(32, dtype=dtype).reshape(16, 2))
    source_state.q_initialized.fill_(True)
    source.committed_step = 4
    metadata = copy.deepcopy(source.checkpoint_metadata())
    payload = copy.deepcopy(source.state_dict())

    destination, destination_matrix, _ = _state(dtype=dtype)
    destination.validate_checkpoint_metadata(metadata)
    destination.load_state_dict(payload)

    restored = destination.parameter_state(destination_matrix)
    assert destination.committed_step == 4
    torch.testing.assert_close(restored.error, source_state.error)
    torch.testing.assert_close(restored.q_memory, source_state.q_memory)
    assert bool(restored.q_initialized)


def test_checkpoint_contains_no_transient_bucket_state():
    state, _, _ = _state()

    payload = state.state_dict()

    assert set(payload) == {"shared", "rank_0"}
    assert set(payload["shared"]) == {"schema_version", "committed_step"}
    assert set(payload["rank_0"]) == {"matrix"}
    assert set(payload["rank_0"]["matrix"]) == {
        "error",
        "q_memory",
        "q_initialized",
    }
    assert all(
        torch.is_tensor(value)
        for root in payload.values()
        for entry in root.values()
        for value in (entry.values() if isinstance(entry, dict) else (entry,))
    )


def test_cross_dtype_checkpoint_restore_fails_before_payload_copy():
    source, _, _ = _state(dtype=torch.bfloat16)
    destination, _, _ = _state(dtype=torch.float32)

    with pytest.raises(ValueError, match="parameter table"):
        destination.validate_checkpoint_metadata(source.checkpoint_metadata())


@pytest.mark.parametrize(
    "mutation,match",
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
            lambda metadata: metadata["ordered_parameter_table"][0].__setitem__(
                "shape", [128]
            ),
            "parameter table",
        ),
        (
            lambda metadata: metadata["ordered_parameter_table"][0].__setitem__(
                "dtype", "float64"
            ),
            "parameter table",
        ),
        (lambda metadata: metadata["tensor_schema"].pop("rank_0"), "missing.*rank_0"),
        (
            lambda metadata: metadata["tensor_schema"]["rank_0"].pop("matrix"),
            "missing.*matrix",
        ),
        (
            lambda metadata: metadata["tensor_schema"]["rank_0"]["matrix"].__setitem__(
                "initialized",
                metadata["tensor_schema"]["rank_0"]["matrix"].pop("q_initialized"),
            ),
            "tensor schema",
        ),
        (
            lambda metadata: metadata["tensor_schema"]["rank_0"]["matrix"][
                "q_memory"
            ].__setitem__("shape", [32]),
            "tensor schema",
        ),
        (
            lambda metadata: metadata["tensor_schema"]["rank_0"]["matrix"][
                "q_initialized"
            ].__setitem__("dtype", "int64"),
            "tensor schema",
        ),
        (lambda metadata: metadata.__setitem__("committed_step", -1), "committed step"),
    ],
)
def test_metadata_validation_rejects_incompatible_checkpoint(mutation, match):
    state, _, _ = _state()
    metadata = copy.deepcopy(state.checkpoint_metadata())
    mutation(metadata)

    with pytest.raises(ValueError, match=match):
        state.validate_checkpoint_metadata(metadata)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload.pop("shared"), "shared"),
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
                "initialized", payload["rank_0"]["matrix"].pop("q_initialized")
            ),
            "fields",
        ),
        (
            lambda payload: payload["rank_0"]["matrix"].__setitem__(
                "error", torch.zeros(128)
            ),
            "tensor schema",
        ),
        (
            lambda payload: payload["rank_0"]["matrix"].__setitem__(
                "q_memory", torch.zeros(16, 2, dtype=torch.float64)
            ),
            "tensor schema",
        ),
        (
            lambda payload: payload["rank_0"]["matrix"].__setitem__(
                "q_initialized", torch.zeros((), dtype=torch.int64)
            ),
            "tensor schema",
        ),
    ],
)
def test_load_rejects_incompatible_payload_before_copying_any_tensor(mutation, match):
    source, source_matrix, _ = _state()
    source_state = source.parameter_state(source_matrix)
    source_state.error.fill_(3)
    source_state.q_memory.fill_(4)
    source_state.q_initialized.fill_(True)
    payload = copy.deepcopy(source.state_dict())
    mutation(payload)
    destination, destination_matrix, _ = _state()
    original = destination.parameter_state(destination_matrix)

    with pytest.raises(ValueError, match=match):
        destination.load_state_dict(payload)

    assert torch.count_nonzero(original.error) == 0
    assert torch.count_nonzero(original.q_memory) == 0
    assert not bool(original.q_initialized)
    assert destination.committed_step == 0


def test_load_requires_payload_step_to_match_validated_metadata():
    source, _, _ = _state()
    metadata = source.checkpoint_metadata()
    payload = copy.deepcopy(source.state_dict())
    payload["shared"]["committed_step"] = torch.tensor(1, dtype=torch.int64)
    destination, _, _ = _state()
    destination.validate_checkpoint_metadata(metadata)

    with pytest.raises(ValueError, match="committed step"):
        destination.load_state_dict(payload)
