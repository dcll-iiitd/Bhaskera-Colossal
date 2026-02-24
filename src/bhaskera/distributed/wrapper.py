import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def wrap_model_distributed(model, cfg, local_rank, device):
    if not dist.is_initialized():
        raise RuntimeError('torch.distributed must be initialized before wrapping model.')
    strategy = cfg.distributed.strategy.lower()
    if strategy == 'ddp':
        logger.info('Using DDP')
        return wrap_model_ddp(model, cfg, local_rank)
    elif strategy == 'fsdp':
        logger.info('Using FSDP')
        from .fsdp_utils import wrap_model_fsdp
        return wrap_model_fsdp(model, cfg, local_rank)
    else:
        raise ValueError(f'Unknown distributed strategy: {strategy}')


def wrap_model_ddp(model, cfg, local_rank):
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=cfg.distributed.ddp_broadcast_buffers,
        find_unused_parameters=cfg.distributed.ddp_find_unused_parameters,
        gradient_as_bucket_view=cfg.distributed.ddp_gradient_as_bucket_view,
    )


def is_fsdp_model(model):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except Exception:
        return False


def is_ddp_model(model):
    return isinstance(model, DDP)


def _resolve_state_dict_type(cfg) -> str:
    # Try config_loader.py dataclass path first
    try:
        return cfg.distributed.fsdp_state_dict_type
    except AttributeError:
        pass
    # Try cli.py nested Config path
    try:
        return cfg.training.distributed.fsdp.state_dict_type
    except AttributeError:
        pass
    logger.warning('Cannot resolve fsdp_state_dict_type, defaulting to FULL_STATE_DICT.')
    return 'FULL_STATE_DICT'


def save_checkpoint(model, optimizer, step, cfg, global_rank, checkpoint_path):
    """
    Save checkpoint.

    FSDP: ALL ranks must call this simultaneously.
    FSDP FULL_STATE_DICT is a collective -- every rank participates in the
    all-gather so FSDP can assemble shards into one state dict.
    Only rank 0 writes to disk. If only rank 0 calls this, it hangs forever
    waiting for the other ranks to join -- the deadlock you hit.

    DDP: Only rank 0 needs to call this (no collective involved).
    """
    if is_fsdp_model(model):
        save_fsdp_checkpoint(model, optimizer, step, cfg, global_rank, checkpoint_path)
    else:
        save_ddp_checkpoint(model, optimizer, step, global_rank, checkpoint_path)


def save_ddp_checkpoint(model, optimizer, step, global_rank, checkpoint_path):
    if global_rank == 0:
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }, checkpoint_path)
        logger.info(f'Saved DDP checkpoint to {checkpoint_path}')


def save_fsdp_checkpoint(model, optimizer, step, cfg, global_rank, checkpoint_path):
    """
    Save FSDP checkpoint. ALL ranks must call this together.

    Uses the modern torch.distributed.checkpoint.state_dict API
    (get_state_dict) instead of the deprecated FSDP.state_dict_type()
    context manager that was causing the FutureWarning.
    Falls back to the legacy API if the modern one is unavailable
    (PyTorch < 2.1).
    """
    try:
        # Modern API: PyTorch >= 2.1, no deprecation warning
        from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        if global_rank == 0:
            torch.save({
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'step': step,
            }, checkpoint_path)
            logger.info(f'Saved FSDP checkpoint to {checkpoint_path}')
    except ImportError:
        # Legacy API fallback (PyTorch < 2.1)
        _save_fsdp_checkpoint_legacy(model, optimizer, step, cfg, global_rank, checkpoint_path)


def _save_fsdp_checkpoint_legacy(model, optimizer, step, cfg, global_rank, checkpoint_path):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from .fsdp_utils import get_state_dict_type
    state_dict_type_str = _resolve_state_dict_type(cfg)
    state_dict_type = get_state_dict_type(state_dict_type_str)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    if global_rank == 0:
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'step': step,
        }, checkpoint_path)
        logger.info(f'Saved FSDP checkpoint (legacy API) to {checkpoint_path}')


def load_checkpoint(model, optimizer, checkpoint_path, cfg, device):
    if is_fsdp_model(model):
        return load_fsdp_checkpoint(model, optimizer, checkpoint_path, cfg, device)
    else:
        return load_ddp_checkpoint(model, optimizer, checkpoint_path, device)


def load_ddp_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    logger.info(f'Loaded DDP checkpoint from {checkpoint_path} at step {step}')
    return step


def load_fsdp_checkpoint(model, optimizer, checkpoint_path, cfg, device):
    try:
        from torch.distributed.checkpoint.state_dict import set_state_dict, StateDictOptions
        checkpoint = torch.load(checkpoint_path, map_location=device)
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_state_dict(
            model, optimizer,
            model_state_dict=checkpoint['model_state_dict'],
            optim_state_dict=checkpoint['optimizer_state_dict'],
            options=options,
        )
        step = checkpoint['step']
        logger.info(f'Loaded FSDP checkpoint from {checkpoint_path} at step {step}')
        return step
    except ImportError:
        return _load_fsdp_checkpoint_legacy(model, optimizer, checkpoint_path, cfg, device)


def _load_fsdp_checkpoint_legacy(model, optimizer, checkpoint_path, cfg, device):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from .fsdp_utils import get_state_dict_type
    state_dict_type_str = _resolve_state_dict_type(cfg)
    state_dict_type = get_state_dict_type(state_dict_type_str)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model.load_state_dict(checkpoint['model_state_dict'])
        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, checkpoint['optimizer_state_dict']
        )
        optimizer.load_state_dict(optim_state)
    step = checkpoint['step']
    logger.info(f'Loaded FSDP checkpoint (legacy) from {checkpoint_path} at step {step}')
    return step