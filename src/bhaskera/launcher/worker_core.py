"""
bhaskera.launcher.worker_core
==============================
The single training body — called identically by ALL entry points:

    slurm_entry.py    (torch.distributed / srun)
    ray_entry.py      (Ray Train / TorchTrainer)
    torchrun_entry.py (torchrun, local dev)

Entry points are responsible for:
  - initialising torch.distributed
  - setting the CUDA device
  - filling WorkerContext and calling run_worker()

This module owns everything AFTER that point:
  tokenizer → dataset → model → distributed wrap → optimizer → train
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class WorkerContext:
    """Passed by every entry point to run_worker."""
    global_rank: int
    local_rank:  int
    world_size:  int
    device:      torch.device
    launcher:    str   # "slurm" | "ray" | "torchrun"


def run_worker(ctx: WorkerContext, cfg) -> None:
    _setup_logging(ctx)

    logger.info(
        f"[{ctx.launcher}] rank={ctx.global_rank}/{ctx.world_size} "
        f"local={ctx.local_rank} device={ctx.device}"
    )

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── dataset with per-rank sharding ────────────────────────────────────────
    from bhaskera.data.registry import build_dataset
    dataset = build_dataset(cfg, tokenizer, ctx.global_rank, ctx.world_size)
    # IterableDataset with rank-level sharding — no DistributedSampler needed.
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    # FSDP: build on CPU — FSDP's device_id moves each shard to GPU during init.
    # DDP:  build on GPU.
    is_fsdp      = cfg.distributed.strategy.lower() == "fsdp"
    model_device = torch.device("cpu") if is_fsdp else ctx.device

    from bhaskera.models.registry import build_model
    model = build_model(cfg, model_device)

    # ── distributed wrap ──────────────────────────────────────────────────────
    from bhaskera.distributed.wrapper import wrap_model_distributed
    model = wrap_model_distributed(
        model=model,
        cfg=cfg,
        local_rank=ctx.local_rank,
        device=ctx.device,
    )

    # ── optimizer ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    if ctx.global_rank == 0:
        total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in trainable)
        logger.info(
            f"Trainable params: {n_trainable/1e6:.2f}M / {total/1e6:.2f}M "
            f"({n_trainable/total*100:.4f}%)"
        )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    # ── experiment logger (rank 0 only) ───────────────────────────────────────
    logger_obj = None
    if cfg.TRACKER and ctx.global_rank == 0:
        from bhaskera.utils.logger_factory import build_logger
        logger_obj = build_logger(cfg, log_gpu=True, gpu_log_every_n_steps=10)

    # ── checkpoint dir ────────────────────────────────────────────────────────
    checkpoint_dir = cfg.CHECKPOINT_DIR if cfg.CHECKPOINT_ENABLED else None

    # ── train ─────────────────────────────────────────────────────────────────
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
        num_epochs=cfg.NUM_EPOCHS,
        checkpoint_dir=checkpoint_dir,
    )


def _setup_logging(ctx: WorkerContext) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"[%(asctime)s][{ctx.launcher}]"
            f"[rank {ctx.global_rank}] %(levelname)s %(message)s"
        ),
        force=True,
    )
