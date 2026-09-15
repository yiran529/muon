"""Small in-stack NCCL All-Reduce transport and bandwidth diagnostic."""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", default="1,32,80")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes_mib = tuple(int(value) for value in args.sizes_mib.split(","))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    bytes_per_element = torch.empty((), dtype=torch.bfloat16).element_size()
    maximum_elements = max(sizes_mib) * 1024 * 1024 // bytes_per_element
    storage = torch.ones(maximum_elements, dtype=torch.bfloat16, device=device)
    results = []

    for size_mib in sizes_mib:
        elements = size_mib * 1024 * 1024 // bytes_per_element
        payload = storage[:elements]
        dist.barrier()
        for _ in range(args.warmups):
            dist.all_reduce(payload)
        torch.cuda.synchronize()

        samples_ms = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(payload)
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))

        gathered: list[list[float] | None] = [None] * world_size
        dist.all_gather_object(gathered, samples_ms)
        if rank == 0:
            rank_medians = [statistics.median(samples) for samples in gathered if samples]
            conservative_ms = max(rank_medians)
            payload_bytes = elements * bytes_per_element
            algorithm_bandwidth = payload_bytes / (conservative_ms / 1000) / 1e9
            results.append(
                {
                    "size_mib": size_mib,
                    "dtype": "bfloat16",
                    "iterations": args.iterations,
                    "rank_median_ms": rank_medians,
                    "max_rank_median_ms": conservative_ms,
                    "algorithm_bandwidth_gbps": algorithm_bandwidth,
                    "ring_equivalent_bus_bandwidth_gbps": (
                        algorithm_bandwidth * 2 * (world_size - 1) / world_size
                    ),
                }
            )

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "world_size": world_size,
                    "backend": dist.get_backend(),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": torch.cuda.nccl.version(),
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
