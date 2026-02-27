"""
bhaskera.launcher.slurm_entry
==============================
torch.distributed entry point — launched by srun inside a SLURM job.

Responsibilities of this module (only):
  1. Read SLURM env vars to resolve rank/world_size
  2. Initialise torch.distributed via env://
  3. Set CUDA device
  4. Call worker_core.run_worker()

All actual training logic lives in worker_core.py.

Usage (via submit_multinode.sh):
    srun python -m bhaskera.launcher.slurm_entry --config config.yaml

Usage (manual debug, no SLURM):
    MASTER_ADDR=localhost MASTER_PORT=29500 \\
    SLURM_PROCID=0 SLURM_LOCALID=0 SLURM_NTASKS=1 \\
    python -m bhaskera.launcher.slurm_entry --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import socket

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _resolve_slurm_env() -> tuple[int, int, int]:
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank  = int(os.environ.get("SLURM_LOCALID", 0))
    world_size  = int(os.environ.get("SLURM_NTASKS",  1))
    return global_rank, local_rank, world_size


def _init_distributed(global_rank: int, world_size: int, local_rank: int) -> None:
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"]        = str(global_rank)
    os.environ["WORLD_SIZE"]  = str(world_size)
    os.environ["LOCAL_RANK"]  = str(local_rank)

    # Suppress deprecated env var warning — use the PyTorch name instead
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=global_rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    dist.barrier()
    logger.info(
        f"[Rank {global_rank}/{world_size}] dist.init OK on "
        f"{socket.gethostname()} (master={master_addr}:{master_port})"
    )


def _run_worker(args: argparse.Namespace) -> None:
    global_rank, local_rank, world_size = _resolve_slurm_env()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][slurm][rank {global_rank}] %(levelname)s %(message)s",
    )

    _init_distributed(global_rank, world_size, local_rank)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Change to submit dir so relative config paths work on all nodes
    submit_dir = os.environ.get("SLURM_SUBMIT_DIR", ".")
    os.chdir(submit_dir)

    from bhaskera.config_loader import load_config
    cfg = load_config(args.config)

    if global_rank == 0:
        logger.info(f"model={cfg.MODEL_NAME} strategy={cfg.distributed.strategy}")

    from bhaskera.launcher.worker_core import WorkerContext, run_worker
    ctx = WorkerContext(
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        launcher="slurm",
    )
    run_worker(ctx, cfg)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description="Bhaskera SLURM entrypoint")
    p.add_argument("--config", required=True, help="Path to YAML config")
    _run_worker(p.parse_args())


if __name__ == "__main__":
    main()