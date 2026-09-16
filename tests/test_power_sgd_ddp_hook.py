"""Packing, averaging, and asynchronous lifecycle for the PowerSGD hook."""

import threading

import pytest
import torch

from dion.collective_observer import CollectiveObserver, set_active_observer
from dion.power_sgd import PowerSGDConfig
import dion.power_sgd_ddp_hook as hook_module
from dion.power_sgd_ddp_hook import PowerSGDDDPParameterSpec, PowerSGDDDPState


class FakeGradBucket:
    def __init__(self, parameters, values, index=0):
        self._parameters = tuple(parameters)
        self._buffer = torch.cat([value.flatten() for value in values])
        self._gradients = []
        self._index = index
        offset = 0
        for parameter in parameters:
            self._gradients.append(
                self._buffer[offset : offset + parameter.numel()].view_as(parameter)
            )
            offset += parameter.numel()

    def parameters(self):
        return list(self._parameters)

    def gradients(self):
        return list(self._gradients)

    def buffer(self):
        return self._buffer

    def index(self):
        return self._index


def make_state(parameters, *, config=None, process_group=None):
    return PowerSGDDDPState(
        process_group=process_group,
        fingerprint="a" * 64,
        parameter_specs=[
            PowerSGDDDPParameterSpec(p, f"parameter_{i}", 10 + i, role)
            for i, (p, role) in enumerate(parameters)
        ],
        optimizer_parameters=[p for p, _ in parameters],
        config=config or PowerSGDConfig(rank=1, start_compress_step=0),
    )


class PendingWork:
    def __init__(self):
        self.future = torch.futures.Future()

    def get_future(self):
        return self.future

    def wait(self):
        raise AssertionError("hook must not wait on Work")


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def all_reduce(tensor, *, group, async_op):
        assert async_op is True
        work = PendingWork()
        calls.append((tensor, work))
        return work

    monkeypatch.setattr(hook_module.dist, "all_reduce", all_reduce)
    observer = CollectiveObserver()
    set_active_observer(observer)
    yield calls, observer
    set_active_observer(None)


def complete(calls, index):
    tensor, work = calls[index]
    tensor.mul_(2)
    work.future.set_result([tensor])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_warmup_uses_one_dense_collective_and_averages_once(transport, dtype):
    calls, observer = transport
    matrix = torch.nn.Parameter(torch.zeros(8, 12, dtype=dtype))
    auxiliary = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
    state = make_state(
        [(matrix, "matrix"), (auxiliary, "dense_aux")],
        config=PowerSGDConfig(start_compress_step=1),
    )
    state.world_size = 2
    values = [torch.ones_like(matrix) * 3, torch.tensor([2, 4, 6], dtype=dtype)]
    bucket = FakeGradBucket([matrix, auxiliary], values)
    state.begin_step()
    result = hook_module.power_sgd_ddp_hook(state, bucket)
    assert not result.done()
    assert len(calls) == 1
    assert calls[0][0].numel() == 99
    complete(calls, 0)
    assert result.wait() is bucket.buffer()
    torch.testing.assert_close(result.value(), torch.cat([v.flatten() for v in values]))
    assert observer.signature() == [
        (
            "powersgd_hook/dense",
            "all_reduce",
            99,
            str(dtype).removeprefix("torch."),
            99 * matrix.element_size(),
        )
    ]
    assert state.collective_tail.done()
    assert not state._active_contexts
    assert not bool(state.parameter_state(matrix).q_initialized)
    state.finish_step()
    state.commit_step()


def test_mixed_bucket_packs_dense_fallback_and_p_then_q_only(transport):
    calls, observer = transport
    wide = torch.nn.Parameter(torch.zeros(8, 12))
    auxiliary = torch.nn.Parameter(torch.zeros(3))
    small = torch.nn.Parameter(torch.zeros(2, 2))
    tall = torch.nn.Parameter(torch.zeros(12, 8))
    parameters = [wide, auxiliary, small, tall]
    state = make_state(
        [(p, "dense_aux" if p is auxiliary else "matrix") for p in parameters]
    )
    state.world_size = 2
    values = [torch.full_like(p, i + 1) for i, p in enumerate(parameters)]
    bucket = FakeGradBucket(parameters, values)
    state.begin_step()
    result = hook_module.power_sgd_ddp_hook(state, bucket)
    assert not result.done()
    assert [t.numel() for t, _ in calls] == [27]  # dense 3+4; P 8+12
    complete(calls, 0)
    assert [t.numel() for t, _ in calls] == [27, 20]  # Q 12+8
    assert not result.done()
    complete(calls, 1)
    assert result.wait() is bucket.buffer()
    for actual, expected in zip(bucket.gradients(), values):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert observer.signature() == [
        ("powersgd_hook/p_plus_aux", "all_reduce", 27, "float32", 108),
        ("powersgd_hook/q", "all_reduce", 20, "float32", 80),
    ]
    for parameter in (wide, tall):
        item = state.parameter_state(parameter)
        torch.testing.assert_close(
            item.error, torch.zeros_like(parameter), atol=2e-5, rtol=0
        )
        assert bool(item.q_initialized)
    assert state.collective_tail.done()
    assert not state._active_contexts
    state.finish_step()


@pytest.mark.parametrize("failure_stage", [0, 1])
def test_collective_failure_completes_both_state_placeholders(transport, failure_stage):
    calls, _ = transport
    parameter = torch.nn.Parameter(torch.zeros(8, 12))
    state = make_state([(parameter, "matrix")])
    state.world_size = 2
    state.begin_step()
    result = hook_module.power_sgd_ddp_hook(
        state, FakeGradBucket([parameter], [torch.ones_like(parameter)])
    )
    if failure_stage:
        complete(calls, 0)
    calls[failure_stage][1].future.set_exception(RuntimeError("transport failed"))
    with pytest.raises(RuntimeError, match="transport failed"):
        result.wait()
    assert state.tail_future.done()
    assert state.collective_tail.done()
    assert not state._active_contexts
    with pytest.raises(RuntimeError, match="transport failed"):
        state.collective_tail.value()
    with pytest.raises(RuntimeError, match="transport failed"):
        state.finish_step()


@pytest.mark.parametrize(
    "failure_point",
    ["preparation", "launch", "previous_collective_tail", "reconstruction"],
)
def test_non_transport_failures_release_both_placeholders(
    monkeypatch, transport, failure_point
):
    calls, _ = transport
    parameter = torch.nn.Parameter(torch.zeros(8, 12))
    state = make_state([(parameter, "matrix")])
    state.world_size = 2
    state.begin_step()

    def broken(*args, **kwargs):
        raise RuntimeError("injected failure")

    if failure_point == "previous_collective_tail":
        state.collective_tail = torch.futures.Future()
        state.collective_tail.set_exception(RuntimeError("injected failure"))
    else:
        monkeypatch.setattr(
            hook_module,
            {
                "preparation": "corrected_gradient",
                "launch": "_all_reduce_future",
                "reconstruction": "reconstruct",
            }[failure_point],
            broken,
        )
    result = hook_module.power_sgd_ddp_hook(
        state, FakeGradBucket([parameter], [torch.ones_like(parameter)])
    )
    if failure_point == "reconstruction":
        complete(calls, 0)
        complete(calls, 1)
    assert result.done()
    with pytest.raises(RuntimeError, match="injected failure"):
        result.wait()
    assert state.tail_future.done()
    assert state.collective_tail.done()
    assert not state._active_contexts
    if failure_point == "reconstruction":
        state.collective_tail.value()
    else:
        with pytest.raises(RuntimeError, match="injected failure"):
            state.collective_tail.value()


def test_second_bucket_collectives_wait_for_previous_q(transport):
    calls, observer = transport
    first = torch.nn.Parameter(torch.zeros(8, 12))
    second = torch.nn.Parameter(torch.zeros(8, 12))
    state = make_state([(first, "matrix"), (second, "matrix")])
    state.world_size = 2
    state.begin_step()
    first_result = hook_module.power_sgd_ddp_hook(
        state, FakeGradBucket([first], [torch.ones_like(first)])
    )
    second_result = hook_module.power_sgd_ddp_hook(
        state, FakeGradBucket([second], [torch.ones_like(second)], index=1)
    )
    assert len(calls) == 1
    complete(calls, 0)
    assert len(calls) == 2
    assert not first_result.done()
    assert not second_result.done()
    complete(calls, 1)
    assert first_result.done()
    assert len(calls) == 3
    assert not second_result.done()
    complete(calls, 2)
    complete(calls, 3)
    assert second_result.done()
    state.finish_step()
    assert [event.category for event in observer.events] == [
        "powersgd_hook/p_plus_aux",
        "powersgd_hook/q",
        "powersgd_hook/p_plus_aux",
        "powersgd_hook/q",
    ]


def test_second_bucket_prepares_while_first_collective_is_pending(
    monkeypatch, transport
):
    calls, _ = transport
    parameters = [torch.nn.Parameter(torch.zeros(8, 12)) for _ in range(2)]
    state = make_state([(p, "matrix") for p in parameters])
    state.world_size = 2
    state.begin_step()
    prepared = []
    original = hook_module._prepare_compressed_bucket

    def prepare(current_state, context):
        result = original(current_state, context)
        prepared.append(context.bucket_index)
        return result

    monkeypatch.setattr(hook_module, "_prepare_compressed_bucket", prepare)
    futures = [
        hook_module.power_sgd_ddp_hook(
            state, FakeGradBucket([p], [torch.ones_like(p)], index=i)
        )
        for i, p in enumerate(parameters)
    ]
    assert prepared == [0, 1]
    assert len(calls) == 1
    assert not any(f.done() for f in futures)
    for i in range(4):
        complete(calls, i)
    state.finish_step()


def test_collectives_advance_while_reconstruction_and_step_finish_are_pending(
    monkeypatch, transport
):
    calls, observer = transport
    parameters = [torch.nn.Parameter(torch.zeros(8, 12)) for _ in range(2)]
    state = make_state([(p, "matrix") for p in parameters])
    state.world_size = 2
    state.begin_step()
    futures = [
        hook_module.power_sgd_ddp_hook(
            state, FakeGradBucket([p], [torch.ones_like(p)], index=i)
        )
        for i, p in enumerate(parameters)
    ]
    entered = threading.Event()
    release = threading.Event()
    original = hook_module.reconstruct

    def delayed_reconstruct(p, q):
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=10), "test did not release reconstruction"
        return original(p, q)

    monkeypatch.setattr(hook_module, "reconstruct", delayed_reconstruct)
    complete(calls, 0)
    worker = threading.Thread(target=complete, args=(calls, 1))
    worker.start()
    try:
        assert entered.wait(timeout=10)
        assert len(calls) == 3  # P(B) is submitted before reconstruction(A).
        assert not futures[0].done()
        complete(calls, 2)
        complete(calls, 3)
        assert futures[1].done()
        assert state.collective_tail.done()
        assert not state.tail_future.done()
        with pytest.raises(hook_module.PowerSGDStateError, match="still in flight"):
            state.finish_step()
    finally:
        release.set()
        worker.join(timeout=10)
    assert not worker.is_alive()
    futures[0].value()
    state.finish_step()
    assert [event.category for event in observer.events] == [
        "powersgd_hook/p_plus_aux",
        "powersgd_hook/q",
        "powersgd_hook/p_plus_aux",
        "powersgd_hook/q",
    ]


@pytest.mark.parametrize("stage", ["p", "q", "reconstruction"])
def test_pipeline_failure_preserves_collective_submission_contract(
    monkeypatch, transport, stage
):
    calls, _ = transport
    parameters = [torch.nn.Parameter(torch.zeros(8, 12)) for _ in range(2)]
    state = make_state([(p, "matrix") for p in parameters])
    state.world_size = 2
    state.begin_step()
    futures = [
        hook_module.power_sgd_ddp_hook(
            state, FakeGradBucket([p], [torch.ones_like(p)], index=i)
        )
        for i, p in enumerate(parameters)
    ]
    if stage == "reconstruction":
        original = hook_module.reconstruct
        failed = False

        def fail_once(p, q):
            nonlocal failed
            if not failed:
                failed = True
                raise KeyboardInterrupt("reconstruction cancelled")
            return original(p, q)

        monkeypatch.setattr(hook_module, "reconstruct", fail_once)
        complete(calls, 0)
        complete(calls, 1)
        assert len(calls) == 3
        complete(calls, 2)
        complete(calls, 3)
        futures[1].value()
        state.collective_tail.value()
    else:
        if stage == "q":
            complete(calls, 0)
        calls[-1][1].future.set_exception(RuntimeError("transport cancelled"))
        assert len(calls) == (1 if stage == "p" else 2)
        assert futures[1].done()
        with pytest.raises(RuntimeError, match="cancelled"):
            futures[1].value()
    assert futures[0].done()
    with pytest.raises(RuntimeError, match="cancelled"):
        futures[0].value()
    with pytest.raises(RuntimeError, match="cancelled"):
        state.finish_step()
    assert not state._active_contexts


@pytest.mark.parametrize(
    "exception_type", [KeyboardInterrupt, SystemExit, BaseException]
)
def test_base_exception_settles_all_futures_and_releases_context(
    monkeypatch, exception_type
):
    parameter = torch.nn.Parameter(torch.zeros(8, 12))
    state = make_state([(parameter, "matrix")])
    state.begin_step()

    def cancelled(*args, **kwargs):
        raise exception_type("preparation cancelled")

    monkeypatch.setattr(hook_module, "corrected_gradient", cancelled)
    result = hook_module.power_sgd_ddp_hook(
        state, FakeGradBucket([parameter], [torch.ones_like(parameter)])
    )
    # Check completion before reading errors so a regression cannot hang pytest.
    assert result.done()
    assert state.tail_future.done()
    assert state.collective_tail.done()
    assert not state._active_contexts
    for future in (result, state.tail_future, state.collective_tail):
        with pytest.raises(
            RuntimeError, match=f"{exception_type.__name__}.*preparation cancelled"
        ):
            future.value()
    with pytest.raises(RuntimeError, match="preparation cancelled"):
        state.finish_step()


@pytest.mark.parametrize("warm_start", [True, False])
def test_seeded_factors_use_stable_identity_phase_and_preserve_global_rng(warm_start):
    first = torch.nn.Parameter(torch.zeros(4, 4))
    second = torch.nn.Parameter(torch.zeros(4, 4))
    state = make_state(
        [(first, "matrix"), (second, "matrix")],
        config=PowerSGDConfig(
            start_compress_step=0,
            min_compression_rate=1,
            warm_start=warm_start,
            error_feedback="none",
            orthogonalization_epsilon=0,
        ),
    )
    diagonal = torch.tensor([1.0, 2.0, 3.0, 4.0])
    gradient = torch.diag(diagonal)
    previous_q = {}
    for phase in (0, 1):
        state.begin_step()
        # Reverse and split bucket membership across steps; seeds follow parameters.
        order = [second, first] if phase == 0 else [first, second]
        random_state = torch.random.get_rng_state()
        for parameter in order:
            item = state.parameter_state(parameter)
            # A stale error must not affect the no-EF path.
            item.error.fill_(99)
            if warm_start and phase:
                initial = previous_q[id(parameter)]
            else:
                seed = (52 if parameter is first else 53) + phase * 1_000_003
                initial = torch.randn(4, generator=torch.Generator().manual_seed(seed))
            direction = diagonal * (initial / torch.linalg.vector_norm(initial))
            p = direction / torch.linalg.vector_norm(direction)
            expected_q = diagonal * p
            expected = torch.outer(p, expected_q)
            bucket = FakeGradBucket([parameter], [gradient], index=17 - phase)
            result = hook_module.power_sgd_ddp_hook(state, bucket).wait()
            torch.testing.assert_close(result.view(4, 4), expected)
            torch.testing.assert_close(item.q_memory[:, 0], expected_q)
            torch.testing.assert_close(item.error, torch.full((4, 4), 99.0))
            previous_q[id(parameter)] = expected_q
        assert torch.equal(torch.random.get_rng_state(), random_state)
        state.finish_step()
        state.commit_step()


def test_bf16_compression_preserves_bucket_and_state_dtype():
    parameter = torch.nn.Parameter(torch.zeros(8, 12, dtype=torch.bfloat16))
    state = make_state([(parameter, "matrix")])
    state.begin_step()
    bucket = FakeGradBucket([parameter], [torch.ones_like(parameter)])
    result = hook_module.power_sgd_ddp_hook(state, bucket).wait()
    item = state.parameter_state(parameter)
    assert result is bucket.buffer()
    assert result.dtype == item.error.dtype == item.q_memory.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.ones_like(result), atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        item.error, torch.ones_like(parameter) - result.view_as(parameter)
    )
    state.finish_step()
