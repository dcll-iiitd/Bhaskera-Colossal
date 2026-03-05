"""
bhaskera.launcher.torchrun_entry
=================================
Entry point for torchrun (single-node multi-GPU or multi-node without SLURM).

Usage:
    # Single node, all GPUs:
    torchrun --nproc_per_node=auto -m bhaskera.launcher.torchrun_entry \\
             --config config_fsdp.yaml

    # Multi-node (run on EACH node):
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \\
             --master_addr=<head_ip> --master_port=29500 \\
             -m bhaskera.launcher.torchrun_entry --config config_fsdp.yaml
"""
from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _run_worker(args: argparse.Namespace) -> None:
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][torchrun][rank {global_rank}] %(levelname)s %(message)s",
        force=True,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from bhaskera.config_loader import load_config
    cfg = load_config(args.config)

    from bhaskera.launcher.worker_core import WorkerContext, run_worker
    ctx = WorkerContext(
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        launcher="torchrun",
    )
    run_worker(ctx, cfg)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description="Bhaskera torchrun entrypoint")
    p.add_argument("--config", required=True, help="YAML config path")
    _run_worker(p.parse_args())


if __name__ == "__main__":
    main()
