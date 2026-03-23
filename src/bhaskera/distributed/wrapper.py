"""
bhaskera.distributed.wrapper
=============================
DDP / FSDP2 wrapping and checkpoint save/load.

Checkpoint contract
-------------------
FSDP2:  ALL ranks must call save_checkpoint / load_checkpoint simultaneously.
        get_state_dict() is a collective operation. If rank 0 calls it alone
        it will hang forever waiting for other ranks.
DDP:    Only rank 0 needs to call — no collective involved.

FSDP2 changes vs FSDP1
-----------------------
- is_fsdp_model() checks for FSDPModule (not isinstance(model, FSDP))
- _save_fsdp / _load_fsdp use torch.distributed.checkpoint.state_dict
  (get_state_dict / set_state_dict) — the legacy FSDP.state_dict_type
  context manager does NOT exist in FSDP2
- clip_grad_norm_ is called normally (FSDP2 handles the allreduce internally)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


# ── model type helpers ─────────────────────────────────────────────────────────

def is_fsdp_model(model) -> bool:
    """Check if model has been wrapped with FSDP2 (fully_shard)."""
    try:
        from torch.distributed._composable.fsdp import FSDPModule
        return isinstance(model, FSDPModule)
    except ImportError:
        # Fallback: also accept old FSDP1 wrapper (transitional safety)
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return isinstance(model, FSDP)
        except Exception:
            return False


def is_ddp_model(model) -> bool:
    return isinstance(model, DDP)


# ── model wrapping ─────────────────────────────────────────────────────────────

def wrap_model_distributed(model, cfg, local_rank: int, device: torch.device):
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before wrapping the model."
        )

    strategy = cfg.distributed.strategy.lower()
    if strategy == "ddp":
        logger.info(f"[Rank {dist.get_rank()}] Wrapping with DDP")
        return _wrap_ddp(model, cfg, local_rank, device)
    elif strategy == "fsdp":
        logger.info(f"[Rank {dist.get_rank()}] Wrapping with FSDP2")
        from .fsdp_utils import wrap_model_fsdp
        return wrap_model_fsdp(model, cfg, local_rank)
    else:
        raise ValueError(
            f"Unknown distributed strategy: '{strategy}'. Use 'ddp' or 'fsdp'."
        )


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


# ── checkpoint: save ──────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step: int, cfg,
                    global_rank: int, checkpoint_path: str) -> None:
    """
    Save model + optimizer + scheduler state.

    FSDP2: ALL ranks must call this. Only rank 0 writes to disk.
    DDP:   Only rank 0 should call this (caller's responsibility).
    """
    if is_fsdp_model(model):
        _save_fsdp2(model, optimizer, scheduler, step, global_rank, checkpoint_path)
    else:
        _save_ddp(model, optimizer, scheduler, step, global_rank, checkpoint_path)


def _save_ddp(model, optimizer, scheduler, step: int,
              global_rank: int, path: str) -> None:
    if global_rank != 0:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "model_state_dict":     model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
    }, path)
    logger.info(f"[DDP] Saved checkpoint → {path}")


def _save_fsdp2(model, optimizer, scheduler, step: int,
                global_rank: int, path: str) -> None:
    """
    Save FSDP2 checkpoint using the modern distributed state_dict API.

    get_state_dict() is a COLLECTIVE call — every rank must call this together.
    Only rank 0 writes the file to disk.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
    )

    # full_state_dict=True → gather all shards to a single full state dict
    # cpu_offload=True     → move gathered tensors to CPU to save GPU memory
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)

    # COLLECTIVE: all ranks participate
    model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)

    if global_rank == 0:
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True,
        )
        torch.save({
            "model_state_dict":     model_sd,
            "optimizer_state_dict": optim_sd,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
        }, path)
        logger.info(f"[FSDP2] Saved checkpoint → {path}")


# ── checkpoint: load ──────────────────────────────────────────────────────────

def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str,
                    cfg, device: torch.device) -> int:
    """Load checkpoint. Returns the step number to resume from."""
    if is_fsdp_model(model):
        return _load_fsdp2(model, optimizer, scheduler, checkpoint_path)
    else:
        return _load_ddp(model, optimizer, scheduler, checkpoint_path, device)


def _load_ddp(model, optimizer, scheduler, path: str,
              device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.module.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt.get("step", 0)
    logger.info(f"[DDP] Loaded checkpoint from {path} (step {step})")
    return step


def _load_fsdp2(model, optimizer, scheduler, path: str) -> int:
    """
    Load FSDP2 checkpoint using the modern distributed state_dict API.

    set_state_dict() is a COLLECTIVE call — every rank must call this together.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_state_dict,
    )

    # Load the full checkpoint on CPU from rank 0 storage
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)

    # COLLECTIVE: distributes the loaded state to each rank's shards
    set_state_dict(
        model,
        optimizer,
        model_state_dict=ckpt["model_state_dict"],
        optim_state_dict=ckpt["optimizer_state_dict"],
        options=opts,
    )

    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    step = ckpt.get("step", 0)
    logger.info(f"[FSDP2] Loaded checkpoint from {path} (step {step})")
    return step
