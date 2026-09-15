import torch
import torch.distributed as dist

from benchmark.compressed_muon.train_no_grad_sync_oracle import (
    local_gradient_identity_hook,
)


class _Bucket:
    def __init__(self, buffer: torch.Tensor) -> None:
        self._buffer = buffer

    def buffer(self) -> torch.Tensor:
        return self._buffer


def test_local_gradient_identity_hook_returns_bucket_without_collective():
    buffer = torch.arange(4, dtype=torch.float32)

    result = local_gradient_identity_hook(None, _Bucket(buffer)).wait()

    assert result is buffer


def test_local_gradient_identity_hook_uses_ddp_required_bucket_annotation():
    assert local_gradient_identity_hook.__annotations__["bucket"] is dist.GradBucket
