"""Two-GPU formal-loop smoke gate for all ARC synchronization modes."""

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

from dion.collective_observer import CollectiveObserver, set_active_observer


class _TinyFormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.Sequential(
            torch.nn.Linear(16, 16, bias=False),
            torch.nn.Linear(16, 16, bias=False),
        )
        self.transformer.wte = torch.nn.Embedding(8, 16)
        self.lm_head = torch.nn.Linear(16, 8, bias=False)

    def forward(self, x, _target):
        hidden = self.transformer.h(x)
        auxiliary = self.transformer.wte.weight.sum() + self.lm_head.weight.sum()
        return hidden.sum() + auxiliary * 0.0


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


def _build(mode, device):
    import train
    import train_arctopk

    torch.manual_seed(123)
    model = _TinyFormalModel().to(device)
    ddp = DDP(model, device_ids=[device.index], bucket_cap_mb=0.0005)
    if mode == "dense":
        hp = train.Hyperparameters(
            optimizer="muon",
            scalar_opt="adamw",
            model_dim=16,
            lr=0.01,
        )
        factory_result = train.init_optimizer(
            model=ddp.module,
            device_mesh=None,
            ddp_model=ddp,
            hp=hp,
            cli_args=_cli(),
        )
        return ddp, *train.normalize_gradient_sync_runtime(
            factory_result,
            optimizer_owns_gradient_sync=False,
        )
    optimizer, runtime = train_arctopk.init_arc_topk_optimizer(
        model=ddp.module,
        device_mesh=None,
        ddp_model=ddp,
        hp=train_arctopk.ArcTopKHyperparameters(
            arc_sync_mode=mode,
            arc_topk_ratio=0.5,
            arc_projection_rank=2,
            arc_eta=0.25,
            arc_start_compress_step=0,
            scalar_opt="adamw",
            model_dim=16,
            lr=0.01,
        ),
        cli_args=_cli(),
    )
    return ddp, optimizer, runtime


def _worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        import train

        results = {}
        device = torch.device("cuda", rank)
        for mode in ("dense", "optimizer", "ddp_hook"):
            ddp, optimizer, runtime = _build(mode, device)
            observer = CollectiveObserver()
            set_active_observer(observer)
            for step in range(1, 4):
                if runtime.begin_step is not None:
                    runtime.begin_step()
                x = torch.arange(1, 17, device=device, dtype=torch.float32).view(1, 16)
                x.mul_((rank + 1) * step)
                train.forward_backward_micro_step(
                    ddp,
                    x,
                    None,
                    autocast_ctx=nullcontext(),
                    micro_step=1,
                    grad_accum_steps=1,
                    optimizer_owns_gradient_sync=runtime.optimizer_owns_gradient_sync,
                )
                if runtime.finish_step is not None:
                    runtime.finish_step()
                optimizer.step()
                if runtime.commit_step is not None:
                    runtime.commit_step()
                ddp.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            for parameter in ddp.module.parameters():
                gathered = [torch.empty_like(parameter) for _ in range(world_size)]
                dist.all_gather(gathered, parameter)
                for other in gathered[1:]:
                    torch.testing.assert_close(parameter, other, rtol=1e-5, atol=1e-6)
            signatures = [None] * world_size
            dist.all_gather_object(signatures, observer.signature())
            assert all(signature == signatures[0] for signature in signatures[1:])
            categories = [event.category for event in observer.events]
            assert "arc/seed" not in categories
            if mode == "optimizer":
                assert "arc/sketch" in categories
                assert not any(category.startswith("arc_hook/") for category in categories)
            elif mode == "ddp_hook":
                assert "arc_hook/sketch" in categories
                assert "arc/sketch" not in categories
            results[mode] = {"categories": categories}
            set_active_observer(None)
            del optimizer, ddp
            torch.cuda.empty_cache()
            dist.barrier()
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(results))
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_three_mode_formal_loop_nccl_smoke():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    with tempfile.TemporaryDirectory(prefix="arc-formal-nccl-") as output_dir:
        mp.spawn(
            _worker,
            args=(2, _free_port(), output_dir),
            nprocs=2,
            join=True,
        )
        rank_results = [
            json.loads(Path(output_dir, f"rank-{rank}.json").read_text())
            for rank in range(2)
        ]
    assert rank_results[0].keys() == rank_results[1].keys()
