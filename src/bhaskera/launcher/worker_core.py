"""
bhaskera.launcher.worker_core
==============================
Shared worker body — called identically by both entry points:

    launcher/slurm_entry.py   (torch.distributed / srun)
    launcher/ray_entry.py     (Ray Train / TorchTrainer)

Both entry points are responsible for:
  - initialising torch.distributed
  - setting the CUDA device
  - passing a fully-resolved WorkerContext to run_worker()

This module owns everything AFTER that point:
  tokenizer → dataset → model → distributed wrap → optimizer → train

Having a single source of truth here means:
  - no duplicated code between backends
  - bug fixes automatically apply to both paths
  - adding a third backend (e.g. torchrun) only requires a new entry point
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker context — the contract between entry points and worker_core
# ---------------------------------------------------------------------------

@dataclass
class WorkerContext:
    """
    Everything an entry point knows about its rank/device after distributed
    init.  Both slurm_entry and ray_entry fill this and pass it to run_worker.
    """
    global_rank: int
    local_rank: int
    world_size: int
    device: torch.device
    launcher: str          # "slurm" | "ray"  — for logging / checkpointing


# ---------------------------------------------------------------------------
# Main shared worker
# ---------------------------------------------------------------------------

def run_worker(ctx: WorkerContext, cfg) -> None:
    """
    Core training worker.  Called by every rank in both launcher backends.

    Args:
        ctx:  WorkerContext filled by the entry point.
        cfg:  Bhaskera Config dataclass (from config_loader.load_config).
    """
    _setup_logging(ctx)

    logger.info(
        f"[{ctx.launcher}] rank={ctx.global_rank}/{ctx.world_size} "
        f"local={ctx.local_rank} device={ctx.device}"
    )

    # ---- tokenizer --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- dataset ----------------------------------------------------------
    from bhaskera.data.registry import build_dataset
    dataset = build_dataset(cfg, tokenizer, ctx.global_rank, ctx.world_size)
    loader  = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, pin_memory=True)

    # ---- model ------------------------------------------------------------
    from bhaskera.models.registry import build_model
    model_device = (
        torch.device("cpu")
        if cfg.distributed.strategy.lower() == "fsdp"
        else ctx.device
    )
    model = build_model(cfg, model_device)

    # ---- distributed wrap (DDP or FSDP) -----------------------------------
    from bhaskera.distributed.wrapper import wrap_model_distributed
    model = wrap_model_distributed(
        model=model,
        cfg=cfg,
        local_rank=ctx.local_rank,
        device=ctx.device,
    )

    # ---- optimizer --------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    if ctx.global_rank == 0:
        logger.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    # ---- experiment logger (rank 0 only) ----------------------------------
    logger_obj = None
    if cfg.TRACKER and ctx.global_rank == 0:
        from bhaskera.utils.logger_factory import build_logger
        logger_obj = build_logger(cfg, log_gpu=True, gpu_log_every_n_steps=1)

    # ---- checkpoint dir ---------------------------------------------------
    checkpoint_dir = cfg.CHECKPOINT_DIR if cfg.CHECKPOINT_ENABLED else None

    # ---- train ------------------------------------------------------------
    from bhaskera.trainer.train_loop import train
    train(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=ctx.device,
        grad_accum_steps=cfg.GRAD_ACCUM,
        max_steps=cfg.MAX_STEPS,
        local_rank=ctx.local_rank,
        global_rank=ctx.global_rank,
        cfg=cfg,
        logger_obj=logger_obj,
        num_epochs=getattr(cfg, "NUM_EPOCHS", 1),
        checkpoint_dir=checkpoint_dir,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(ctx: WorkerContext) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"[%(asctime)s][{ctx.launcher}]"
            f"[rank {ctx.global_rank}] %(levelname)s %(message)s"
        ),
        force=True,
    )