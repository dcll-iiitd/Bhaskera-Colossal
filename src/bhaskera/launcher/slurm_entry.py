"""
bhaskera.launcher.slurm_entry
==============================
Direct SLURM entrypoint — no Ray, no torchrun.

Each `srun` task runs this module.  We read SLURM environment variables to
set up torch.distributed ourselves, then hand off to the same train_func that
the Ray path uses.  This means you get identical training code on both paths.

Environment variables consumed (all set by SLURM automatically):
    SLURM_PROCID            — global rank of this task (0 … WORLD_SIZE-1)
    SLURM_LOCALID           — local rank within this node (0 … GPUs_per_node-1)
    SLURM_NTASKS            — total number of tasks == WORLD_SIZE
    MASTER_ADDR             — hostname of rank-0 node (set in submit script)
    MASTER_PORT             — port rank-0 listens on   (set in submit script)

Usage (via srun in submit_multinode.sh):
    srun python -m bhaskera.launcher.slurm_entry --config config_fsdp.yaml

Usage (manual, for debugging single-node without SLURM):
    MASTER_ADDR=localhost MASTER_PORT=29500 \\
    SLURM_PROCID=0 SLURM_LOCALID=0 SLURM_NTASKS=1 \\
    python -m bhaskera.launcher.slurm_entry --config config_fsdp.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import socket

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed bootstrap
# ---------------------------------------------------------------------------

def _resolve_slurm_env() -> tuple[int, int, int]:
    """
    Read SLURM variables and return (global_rank, local_rank, world_size).

    Falls back to rank=0 / world_size=1 when not running under SLURM so that
    the same entry-point works for quick local debugging.
    """
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank  = int(os.environ.get("SLURM_LOCALID", 0))
    world_size  = int(os.environ.get("SLURM_NTASKS",  1))
    return global_rank, local_rank, world_size


def _init_distributed(global_rank: int, world_size: int) -> None:
    """
    Initialise torch.distributed using the env:// init method.
    MASTER_ADDR and MASTER_PORT must already be in the environment
    (set by the SLURM submission script).
    """
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # torch.distributed env:// reads MASTER_ADDR / MASTER_PORT from env
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"]        = str(global_rank)
    os.environ["WORLD_SIZE"]  = str(world_size)
    os.environ["LOCAL_RANK"]  = str(int(os.environ.get("SLURM_LOCALID", 0)))

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=global_rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),  # add this
    )

    # Verify the group is healthy
    dist.barrier()
    logger.info(
        f"[Rank {global_rank}/{world_size}] dist.init_process_group OK "
        f"on {socket.gethostname()} "
        f"(master={master_addr}:{master_port})"
    )


# ---------------------------------------------------------------------------
# Main worker logic (mirrors cli.py / newtrain.py train_func)
# ---------------------------------------------------------------------------

def _run_worker(args: argparse.Namespace) -> None:
    global_rank, local_rank, world_size = _resolve_slurm_env()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][rank {global_rank}] %(levelname)s %(message)s",
    )

    logger.info(
        f"Starting worker | node={socket.gethostname()} "
        f"global_rank={global_rank} local_rank={local_rank} world_size={world_size}"
    )

    # ---- torch.distributed ----
    _init_distributed(global_rank, world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ---- config ----
    from bhaskera.config_loader import load_config
    cfg = load_config(args.config)

    if global_rank == 0:
        logger.info(f"Config loaded: model={cfg.MODEL_NAME} strategy={cfg.distributed.strategy}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token      = tokenizer.eos_token
    tokenizer.padding_side   = "right"

    # ---- dataset ----
    from bhaskera.data.registry import build_dataset
    dataset = build_dataset(cfg, tokenizer, global_rank, world_size)
    loader  = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, pin_memory=True)

    # ---- model ----
    from bhaskera.models.registry import build_model
    model_device = (
        torch.device("cpu")
        if cfg.distributed.strategy.lower() == "fsdp"
        else device
    )
    model = build_model(cfg, model_device)

    # ---- distributed wrap ----
    from bhaskera.distributed.wrapper import wrap_model_distributed
    model = wrap_model_distributed(
        model=model,
        cfg=cfg,
        local_rank=local_rank,
        device=device,
    )

    # ---- optimizer ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    if global_rank == 0:
        logger.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    # ---- experiment logger (rank 0 only) ----
    logger_obj = None
    if cfg.TRACKER and global_rank == 0:
        from bhaskera.utils.logger_factory import build_logger
        logger_obj = build_logger(cfg, log_gpu=True, gpu_log_every_n_steps=1)

    # ---- train ----
    from bhaskera.trainer.train_loop import train
    train(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=device,
        grad_accum_steps=cfg.GRAD_ACCUM,
        max_steps=cfg.MAX_STEPS,
        local_rank=local_rank,
        global_rank=global_rank,
        cfg=cfg,
        logger_obj=logger_obj,
        num_epochs=getattr(cfg, "NUM_EPOCHS", 1),
        checkpoint_dir=cfg.CHECKPOINT_DIR if cfg.CHECKPOINT_ENABLED else None,
    )

    # ---- teardown ----
    if global_rank == 0:
        logger.info("Training finished — cleaning up.")
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bhaskera SLURM multi-node entrypoint")
    p.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. config_fsdp.yaml)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _run_worker(args)


if __name__ == "__main__":
    main()