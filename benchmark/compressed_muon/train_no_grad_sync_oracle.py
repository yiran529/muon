"""Timing-only dense Muon entry that skips DDP gradient collectives."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

import train


def local_gradient_identity_hook(
    _state: None, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """Return each rank's local bucket without launching a collective."""

    future: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    future.set_result(bucket.buffer())
    return future


def init_no_grad_sync_oracle(
    model,
    device_mesh: DeviceMesh | None,
    ddp_model: DDP | None,
    hp: train.Hyperparameters,
    cli_args,
):
    if device_mesh is not None or ddp_model is None:
        raise ValueError("no-gradient-sync timing oracle requires DDP")
    if hp.replicate_mesh_grad_sync:
        raise ValueError("optimizer-owned gradient sync is incompatible with this oracle")

    optimizer = train.init_optimizer(model, device_mesh, ddp_model, hp, cli_args)
    ddp_model.register_comm_hook(None, local_gradient_identity_hook)
    train.print0(
        "TIMING ORACLE: DDP gradient All-Reduce disabled; rank parameters may diverge"
    )
    return optimizer


if __name__ == "__main__":
    train.main(optimizer_factory=init_no_grad_sync_oracle)
