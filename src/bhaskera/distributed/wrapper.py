"""
bhaskera.distributed.wrapper
=============================
DDP/FSDP wrapping and checkpoint save/load.

Checkpoint contract
-------------------
FSDP:  ALL ranks must call save_checkpoint / load_checkpoint simultaneously.
       FULL_STATE_DICT is a collective all-gather. If rank 0 calls it alone
       it hangs forever waiting for the other ranks to join.
DDP:   Only rank 0 needs to call — no collective involved.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


# ── model wrapping ─────────────────────────────────────────────────────────────

def wrap_model_distributed(model, cfg, local_rank: int, device: torch.device):
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before wrapping the model.")

    strategy = cfg.distributed.strategy.lower()
    if strategy == "ddp":
        logger.info(f"[Rank {dist.get_rank()}] Wrapping with DDP")
        return _wrap_ddp(model, cfg, local_rank, device)
    elif strategy == "fsdp":
        logger.info(f"[Rank {dist.get_rank()}] Wrapping with FSDP")
        from .fsdp_utils import wrap_model_fsdp
        return wrap_model_fsdp(model, cfg, local_rank)
    else:
        raise ValueError(f"Unknown distributed strategy: '{strategy}'. Use 'ddp' or 'fsdp'.")


def _wrap_ddp(model, cfg, local_rank: int, device: torch.device) -> DDP:
    model = model.to(device)
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=cfg.distributed.ddp_broadcast_buffers,
        find_unused_parameters=cfg.distributed.ddp_find_unused_parameters,
        gradient_as_bucket_view=cfg.distributed.ddp_gradient_as_bucket_view,
    )


# ── model type helpers ─────────────────────────────────────────────────────────

def is_fsdp_model(model) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except Exception:
        return False


def is_ddp_model(model) -> bool:
    return isinstance(model, DDP)


# ── checkpoint: save ──────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step: int, cfg,
                    global_rank: int, checkpoint_path: str) -> None:
    """
    Save model + optimizer + scheduler state.

    FSDP: ALL ranks must call this. Rank 0 writes to disk.
    DDP:  Only rank 0 calls this (caller's responsibility to gate on rank 0).
    """
    if is_fsdp_model(model):
        _save_fsdp(model, optimizer, scheduler, step, global_rank, checkpoint_path)
    else:
        _save_ddp(model, optimizer, scheduler, step, global_rank, checkpoint_path)


def _save_ddp(model, optimizer, scheduler, step: int,
              global_rank: int, path: str) -> None:
    if global_rank != 0:
        return
    torch.save({
        "model_state_dict":     model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
    }, path)
    logger.info(f"[DDP] Saved checkpoint → {path}")


def _save_fsdp(model, optimizer, scheduler, step: int,
               global_rank: int, path: str) -> None:
    try:
        # Modern API: PyTorch >= 2.1
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)
        if global_rank == 0:
            torch.save({
                "model_state_dict":     model_sd,
                "optimizer_state_dict": optim_sd,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "step": step,
            }, path)
            logger.info(f"[FSDP] Saved checkpoint → {path}")
    except ImportError:
        _save_fsdp_legacy(model, optimizer, scheduler, step, global_rank, path)


def _save_fsdp_legacy(model, optimizer, scheduler, step: int,
                       global_rank: int, path: str) -> None:
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_sd = model.state_dict()
        optim_sd = FSDP.optim_state_dict(model, optimizer)
    if global_rank == 0:
        torch.save({
            "model_state_dict":     model_sd,
            "optimizer_state_dict": optim_sd,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
        }, path)
        logger.info(f"[FSDP legacy] Saved checkpoint → {path}")


# ── checkpoint: load ──────────────────────────────────────────────────────────

def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str,
                    cfg, device: torch.device) -> int:
    """Load checkpoint. Returns the step number to resume from."""
    if is_fsdp_model(model):
        return _load_fsdp(model, optimizer, scheduler, checkpoint_path, device)
    else:
        return _load_ddp(model, optimizer, scheduler, checkpoint_path, device)


def _load_ddp(model, optimizer, scheduler, path: str, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    model.module.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt["step"]
    logger.info(f"[DDP] Loaded checkpoint from {path} (step {step})")
    return step


def _load_fsdp(model, optimizer, scheduler, path: str, device: torch.device) -> int:
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict
        ckpt = torch.load(path, map_location=device)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_state_dict(
            model, optimizer,
            model_state_dict=ckpt["model_state_dict"],
            optim_state_dict=ckpt["optimizer_state_dict"],
            options=opts,
        )
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        step = ckpt["step"]
        logger.info(f"[FSDP] Loaded checkpoint from {path} (step {step})")
        return step
    except ImportError:
        return _load_fsdp_legacy(model, optimizer, scheduler, path, device)


def _load_fsdp_legacy(model, optimizer, scheduler, path: str, device: torch.device) -> int:
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
    ckpt = torch.load(path, map_location=device)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model.load_state_dict(ckpt["model_state_dict"])
        optim_state = FSDP.optim_state_dict_to_load(model, optimizer, ckpt["optimizer_state_dict"])
        optimizer.load_state_dict(optim_state)
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt["step"]
    logger.info(f"[FSDP legacy] Loaded checkpoint from {path} (step {step})")
    return step
