"""Optional two-device correctness checks for PowerSGD stream visibility."""

import os
import socket
import subprocess
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.collective_observer import CollectiveObserver, set_active_observer
from dion.power_sgd import PowerSGDConfig
import dion.power_sgd_ddp_hook as hook_module
from test_power_sgd_ddp_hook import FakeGradBucket, make_state


def _exclusive_devices():
    """Inspect availability without creating contexts or touching other jobs."""
    try:
        gpu_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        process_rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    busy = set(process_rows.splitlines())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = None if visible is None else set(visible.split(","))
    devices = []
    for line in gpu_rows.splitlines():
        index, uuid, memory = [part.strip() for part in line.split(",")]
        if allowed is not None and not any(
            token == index or (token.startswith("GPU-") and uuid.startswith(token))
            for token in allowed
        ):
            continue
        if uuid not in busy and int(memory) <= 32:
            devices.append(uuid)
    return devices[:2]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join_workers(context):
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=max(0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                pytest.fail("PowerSGD CUDA correctness smoke exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def _finish_visibility_worker(rank):
    """Do not let bucket.wait() hide a missing aggregate stream dependency."""
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    parameters = [
        torch.nn.Parameter(torch.zeros(64, 96, device=device)) for _ in range(2)
    ]
    state = make_state([(p, "matrix") for p in parameters])
    consumer = torch.cuda.Stream(device=device)
    reconstruction_streams = [torch.cuda.Stream(device=device) for _ in parameters]
    # Warm up the real hook so CUDA module loading/allocation cannot consume
    # the deliberate reconstruction delay before the consumer is submitted.
    state.begin_step()
    for i, parameter in enumerate(parameters):
        state._reconstruction_streams[device] = reconstruction_streams[i]
        hook_module.power_sgd_ddp_hook(
            state, FakeGradBucket([parameter], [torch.ones_like(parameter)], index=i)
        ).wait()
    state.finish_step()
    state.commit_step()
    torch.cuda.current_stream(device).synchronize()

    for parameter in parameters:
        item = state.parameter_state(parameter)
        item.error.fill_(3)
        item.q_memory.zero_()
        item.q_initialized.fill_(False)
    consumer.wait_stream(torch.cuda.current_stream(device))
    original = hook_module.reconstruct
    reconstruction_index = 0

    def delayed_reconstruct(p, q):
        nonlocal reconstruction_index
        torch.cuda._sleep(500_000_000 if reconstruction_index == 0 else 50_000_000)
        reconstruction_index += 1
        return original(p, q)

    hook_module.reconstruct = delayed_reconstruct
    try:
        state.begin_step()
        buckets = [
            FakeGradBucket([p], [torch.ones_like(p)], index=i)
            for i, p in enumerate(parameters)
        ]
        results = []
        for i, bucket in enumerate(buckets):
            # Independent streams make B finish before A. This exercises the
            # aggregate dependency on every bucket, not just the final tensor.
            state._reconstruction_streams[device] = reconstruction_streams[i]
            results.append(hook_module.power_sgd_ddp_hook(state, bucket))
        assert state.tail_future.done()
        with torch.cuda.stream(consumer):
            state.finish_step()
            snapshots = [
                (
                    bucket.buffer().clone(),
                    state.parameter_state(p).error.clone(),
                    state.parameter_state(p).q_memory.clone(),
                    state.parameter_state(p).q_initialized.clone(),
                )
                for p, bucket in zip(parameters, buckets)
            ]
        consumer.synchronize()
        for output, error, q, initialized in snapshots:
            assert (
                initialized.item()
            ), "finish_step did not expose reconstruction writes"
            torch.testing.assert_close(
                output, torch.full_like(output, 4), atol=5e-5, rtol=5e-5
            )
            torch.testing.assert_close(
                error, torch.zeros_like(error), atol=5e-5, rtol=0
            )
            assert torch.linalg.vector_norm(q).item() > 0
        # Only after observing the finish boundary may the test consume bucket
        # Futures. Their waits must not supply the dependency being tested.
        for result in results:
            result.wait()
        state.commit_step()
    finally:
        hook_module.reconstruct = original
        for stream in reconstruction_streams:
            stream.synchronize()


@pytest.mark.multi_gpu
def test_finish_step_exports_all_reconstruction_writes_without_bucket_wait(monkeypatch):
    devices = _exclusive_devices()
    if len(devices) < 2:
        pytest.skip("requires two exclusive CUDA devices")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(devices))
    _join_workers(mp.spawn(_finish_visibility_worker, nprocs=2, join=False))


def _worker(rank, port, dtype_name):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = getattr(torch, dtype_name)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    original_prepare = hook_module._prepare_compressed_bucket
    original_reconstruct = hook_module.reconstruct
    observer = CollectiveObserver()
    set_active_observer(observer)
    try:
        parameters = [
            torch.nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
            for shape in ((64, 96), (96, 64), (7,))
        ]
        state = make_state(
            [(p, "matrix" if p.ndim == 2 else "dense_aux") for p in parameters],
            process_group=dist.group.WORLD,
            config=PowerSGDConfig(rank=2, start_compress_step=0),
        )
        producer = torch.cuda.Stream(device=device)
        consumer = torch.cuda.Stream(device=device)
        churn = torch.cuda.Stream(device=device)

        def delayed_prepare(current_state, context):
            if (rank + context.bucket_index) % 2:
                torch.cuda._sleep(2_000_000)
            return original_prepare(current_state, context)

        def delayed_reconstruct(p, q):
            torch.cuda._sleep(4_000_000 if rank == 0 else 1_000_000)
            return original_reconstruct(p, q)

        hook_module._prepare_compressed_bucket = delayed_prepare
        hook_module.reconstruct = delayed_reconstruct
        for step in range(3):
            state.begin_step()
            producer.wait_stream(torch.cuda.current_stream(device))
            futures = []
            buckets = []
            corrected = []
            with torch.cuda.stream(producer):
                for index, members in enumerate(
                    ([parameters[0], parameters[2]], [parameters[1]])
                ):
                    torch.cuda._sleep(1_000_000 * (rank + 1))
                    values = []
                    for p in members:
                        base = torch.ones_like(p)
                        if p.ndim == 2:
                            # Two independent blocks give rank two. A constant
                            # matrix is rank one and would test degenerate
                            # Gram-Schmidt rounding instead of stream ordering.
                            base.zero_()
                            m, n = p.shape
                            base[: m // 2, : n // 2] = 1
                            base[m // 2 :, n // 2 :] = 2
                        values.append(base * (rank + step + 1.0))
                    bucket = FakeGradBucket(members, values, index=index)
                    item = state.parameter_state(members[0])
                    corrected.append((values[0] + item.error).clone())
                    buckets.append(bucket)
                    futures.append(hook_module.power_sgd_ddp_hook(state, bucket))
            # Reuse similarly sized allocations while preparation, NCCL, and
            # reconstruction are still in flight on their separate streams.
            with torch.cuda.stream(churn):
                for _ in range(40):
                    for size in (128, 192, 327, 6144):
                        torch.empty(size, device=device, dtype=dtype).fill_(-999)
            snapshots = []
            with torch.cuda.stream(consumer):
                for bucket, result in zip(buckets, futures):
                    output = result.wait().clone()
                    item = state.parameter_state(bucket.parameters()[0])
                    snapshots.append(
                        (
                            output,
                            item.error.clone(),
                            item.q_memory.clone(),
                            item.q_initialized.clone(),
                        )
                    )
            consumer.synchronize()
            tolerance = 0.08 if dtype == torch.bfloat16 else 5e-5
            for bucket, expected_h, (output, error, q, initialized) in zip(
                buckets, corrected, snapshots
            ):
                matrix_output = output[:6144].view_as(expected_h)
                torch.testing.assert_close(
                    error, expected_h - matrix_output, atol=tolerance, rtol=tolerance
                )
                assert initialized.item()
                assert torch.isfinite(q).all()
                if step == 0:
                    expected = torch.zeros_like(matrix_output)
                    m, n = expected.shape
                    expected[: m // 2, : n // 2] = 1.5
                    expected[m // 2 :, n // 2 :] = 3.0
                    torch.testing.assert_close(
                        matrix_output, expected, atol=tolerance, rtol=tolerance
                    )
                gathered = [torch.empty_like(output) for _ in range(2)]
                dist.all_gather(gathered, output)
                torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
            torch.testing.assert_close(
                snapshots[0][0][-7:], torch.full_like(snapshots[0][0][-7:], step + 1.5)
            )
            state.finish_step()
            state.commit_step()
            assert not state._active_contexts
            # The next preparation must observe state exported by this step.
            torch.cuda.current_stream(device).wait_stream(consumer)
        signatures = [None, None]
        dist.all_gather_object(signatures, observer.signature())
        assert signatures[0] == signatures[1]
        assert [entry[0] for entry in signatures[0]] == [
            "powersgd_hook/p_plus_aux",
            "powersgd_hook/q",
            "powersgd_hook/p_plus_aux",
            "powersgd_hook/q",
        ] * 3

        # Exercise the actual reducer's Future consumption and rebuilt buckets.
        hook_module._prepare_compressed_bucket = original_prepare
        hook_module.reconstruct = original_reconstruct
        model = torch.nn.Sequential(
            torch.nn.Linear(96, 64, bias=False),
            torch.nn.Linear(64, 96, bias=False),
        ).to(device=device, dtype=dtype)
        ddp = DDP(model, device_ids=[rank], bucket_cap_mb=0.001)
        ddp_state = make_state(
            [(p, "matrix") for p in model.parameters()],
            process_group=dist.group.WORLD,
        )
        ddp.register_comm_hook(ddp_state, hook_module.power_sgd_ddp_hook)
        for _ in range(3):
            ddp.zero_grad(set_to_none=True)
            ddp_state.begin_step()
            ddp(
                torch.full((4, 96), rank + 1.0, device=device, dtype=dtype)
            ).float().sum().backward()
            ddp_state.finish_step()
            ddp_state.commit_step()
            for parameter in model.parameters():
                gradients = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gradients, parameter.grad)
                torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
        churn.synchronize()
    finally:
        hook_module._prepare_compressed_bucket = original_prepare
        hook_module.reconstruct = original_reconstruct
        set_active_observer(None)
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_pipeline_nccl_visibility_allocator_churn_and_rank_order(
    monkeypatch, dtype_name
):
    devices = _exclusive_devices()
    if len(devices) < 2 or not dist.is_nccl_available():
        pytest.skip("requires two exclusive CUDA devices and NCCL")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(devices))
    context = mp.spawn(_worker, args=(_free_port(), dtype_name), nprocs=2, join=False)
    _join_workers(context)
