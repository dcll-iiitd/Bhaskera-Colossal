"""
Unified distributed training wrapper supporting both DDP and FSDP.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def wrap_model_distributed(
    model: torch.nn.Module,
    cfg,
    local_rank: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Wrap a model for distributed training using either DDP or FSDP.

    Args:
        model:      The model to wrap (on device for DDP, CPU or device for FSDP)
        cfg:        Configuration object with distributed settings
        local_rank: Local rank (GPU index on this node)
        device:     Device to use

    Returns:
        Wrapped model (DDP or FSDP)
    """
    # Ensure distributed is initialized
    if not dist.is_initialized():
        logger.error("Distributed process group not initialized!")
        raise RuntimeError(
            "torch.distributed must be initialized before wrapping model. "
            "This should be done automatically by Ray Train."
        )

    logger.info(f"Distributed initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")

    strategy = cfg.distributed.strategy.lower()

    if strategy == "ddp":
        logger.info("Using DDP (DistributedDataParallel)")
        return wrap_model_ddp(model, cfg, local_rank)

    elif strategy == "fsdp":
        logger.info("Using FSDP (FullyShardedDataParallel)")
        from .fsdp_utils import wrap_model_fsdp
        return wrap_model_fsdp(model, cfg, local_rank)

    else:
        raise ValueError(
            f"Unknown distributed strategy: {strategy}. "
            "Must be 'ddp' or 'fsdp'"
        )


def wrap_model_ddp(
    model: torch.nn.Module,
    cfg,
    local_rank: int,
) -> DDP:
    """
    Wrap a model with DDP.

    Args:
        model:      The model to wrap (should already be on device)
        cfg:        Configuration object with DDP settings
        local_rank: Local rank (GPU index)

    Returns:
        DDP-wrapped model
    """
    logger.info(f"Wrapping model with DDP:")
    logger.info(f"  - Broadcast buffers: {cfg.distributed.ddp_broadcast_buffers}")
    logger.info(f"  - Find unused parameters: {cfg.distributed.ddp_find_unused_parameters}")
    logger.info(f"  - Gradient as bucket view: {cfg.distributed.ddp_gradient_as_bucket_view}")

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=cfg.distributed.ddp_broadcast_buffers,
        find_unused_parameters=cfg.distributed.ddp_find_unused_parameters,
        gradient_as_bucket_view=cfg.distributed.ddp_gradient_as_bucket_view,
    )

    return ddp_model


def get_model_params_for_optimizer(model: torch.nn.Module):
    """
    Get model parameters for optimizer.
    Works with both DDP and FSDP models.
    """
    return model.parameters()


def is_fsdp_model(model: torch.nn.Module) -> bool:
    """Check if model is wrapped with FSDP."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return isinstance(model, FSDP)


def is_ddp_model(model: torch.nn.Module) -> bool:
    """Check if model is wrapped with DDP."""
    return isinstance(model, DDP)


# -----------------------------------------------------------
# Checkpoint saving
# -----------------------------------------------------------
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg,
    global_rank: int,
    checkpoint_path: str,
):
    """
    Save checkpoint compatible with both DDP and FSDP.

    Args:
        model:           Wrapped model (DDP or FSDP)
        optimizer:       Optimizer
        step:            Current training step
        cfg:             Configuration object
        global_rank:     Global rank
        checkpoint_path: Path to save checkpoint
    """
    if is_fsdp_model(model):
        save_fsdp_checkpoint(model, optimizer, step, cfg, global_rank, checkpoint_path)
    else:
        save_ddp_checkpoint(model, optimizer, step, global_rank, checkpoint_path)


def save_ddp_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    global_rank: int,
    checkpoint_path: str,
):
    """Save DDP checkpoint (only on rank 0)."""
    if global_rank == 0:
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved DDP checkpoint to {checkpoint_path}")


def save_fsdp_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg,
    global_rank: int,
    checkpoint_path: str,
):
    """Save FSDP checkpoint using specified state dict type."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from .fsdp_utils import get_state_dict_type

    state_dict_type = get_state_dict_type(cfg.distributed.fsdp_state_dict_type)

    if state_dict_type == StateDictType.FULL_STATE_DICT:
        # Gather full state dict on rank 0
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state_dict = model.state_dict()
            optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)

        if global_rank == 0:
            checkpoint = {
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'step': step,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved FSDP full checkpoint to {checkpoint_path}")

    else:
        # FIX: was using ._replace() which doesn't exist on dataclasses.
        # Temporarily override the field, recurse, then restore.
        logger.warning("Sharded checkpoint saving not fully implemented. Falling back to FULL_STATE_DICT.")
        orig_type = cfg.distributed.fsdp_state_dict_type
        cfg.distributed.fsdp_state_dict_type = "FULL_STATE_DICT"
        try:
            save_fsdp_checkpoint(model, optimizer, step, cfg, global_rank, checkpoint_path)
        finally:
            cfg.distributed.fsdp_state_dict_type = orig_type


# -----------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------
def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    cfg,
    device: torch.device,
) -> int:
    """
    Load checkpoint compatible with both DDP and FSDP.

    Returns:
        Step number from checkpoint
    """
    if is_fsdp_model(model):
        return load_fsdp_checkpoint(model, optimizer, checkpoint_path, cfg, device)
    else:
        return load_ddp_checkpoint(model, optimizer, checkpoint_path, device)


def load_ddp_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    """Load DDP checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    logger.info(f"Loaded DDP checkpoint from {checkpoint_path} at step {step}")
    return step


def load_fsdp_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    cfg,
    device: torch.device,
) -> int:
    """Load FSDP checkpoint."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from .fsdp_utils import get_state_dict_type

    state_dict_type = get_state_dict_type(cfg.distributed.fsdp_state_dict_type)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if state_dict_type == StateDictType.FULL_STATE_DICT:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model.load_state_dict(checkpoint['model_state_dict'])
            optim_state = FSDP.optim_state_dict_to_load(
                model,
                optimizer,
                checkpoint['optimizer_state_dict']
            )
            optimizer.load_state_dict(optim_state)

    step = checkpoint['step']
    logger.info(f"Loaded FSDP checkpoint from {checkpoint_path} at step {step}")
    return step
