"""
bhaskera.distributed.fsdp_utils
================================
FSDP wrapping and activation checkpointing helpers.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import List, Optional, Set

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

logger = logging.getLogger(__name__)

# ── dtype helpers ──────────────────────────────────────────────────────────────

def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}.get(dtype_str, torch.float32)


def get_sharding_strategy(s: str) -> ShardingStrategy:
    return {
        "FULL_SHARD":           ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP":        ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD":             ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD":         ShardingStrategy.HYBRID_SHARD,
        "_HYBRID_SHARD_ZERO2":  ShardingStrategy._HYBRID_SHARD_ZERO2,
    }.get(s, ShardingStrategy.FULL_SHARD)


def get_backward_prefetch(s: Optional[str]) -> Optional[BackwardPrefetch]:
    if not s or s.lower() in ("null", "none"):
        return None
    return {"BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST}.get(s, BackwardPrefetch.BACKWARD_PRE)


def get_state_dict_type(s: str) -> StateDictType:
    return {
        "FULL_STATE_DICT":    StateDictType.FULL_STATE_DICT,
        "SHARDED_STATE_DICT": StateDictType.SHARDED_STATE_DICT,
        "LOCAL_STATE_DICT":   StateDictType.LOCAL_STATE_DICT,
    }.get(s, StateDictType.FULL_STATE_DICT)

# ── layer discovery ────────────────────────────────────────────────────────────

def find_transformer_layers(model: torch.nn.Module,
                             names: List[str]) -> Set[type]:
    found: Set[type] = set()
    seen:  Set[str]  = set()
    for _, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in names and cls_name not in seen:
            found.add(module.__class__)
            seen.add(cls_name)
            logger.info(f"  FSDP wrap layer: {cls_name}")
    return found

# ── main wrap function ─────────────────────────────────────────────────────────

def wrap_model_fsdp(model: torch.nn.Module, cfg, device_id: int) -> FSDP:
    dc = cfg.distributed   # DistributedConfig

    sharding_strategy = get_sharding_strategy(dc.fsdp_sharding_strategy)
    backward_prefetch = get_backward_prefetch(dc.fsdp_backward_prefetch)
    cpu_offload       = CPUOffload(offload_params=True) if dc.fsdp_cpu_offload else None

    mixed_precision = MixedPrecision(
        param_dtype=get_dtype(dc.fsdp_mixed_precision_param),
        reduce_dtype=get_dtype(dc.fsdp_mixed_precision_reduce),
        buffer_dtype=get_dtype(dc.fsdp_mixed_precision_buffer),
    )

    # Build auto-wrap policy
    if dc.fsdp_auto_wrap_policy == "transformer_auto_wrap":
        layer_classes = find_transformer_layers(model, dc.fsdp_transformer_layer_cls)
        if layer_classes:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=layer_classes,
            )
        else:
            logger.warning(
                "No transformer layers matched config names. "
                "Falling back to size-based auto wrap (min 100M params)."
            )
            auto_wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=int(dc.fsdp_min_num_params),
            )
    else:
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(dc.fsdp_min_num_params),
        )

    logger.info(
        f"FSDP | shard={dc.fsdp_sharding_strategy} "
        f"mixed_prec={dc.fsdp_mixed_precision_param} "
        f"cpu_offload={dc.fsdp_cpu_offload} "
        f"act_ckpt={dc.fsdp_activation_checkpointing}"
    )

    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        forward_prefetch=dc.fsdp_forward_prefetch,
        device_id=device_id,        # moves each shard CPU→GPU during init
        limit_all_gathers=True,
        use_orig_params=True,       # required for LoRA params to be seen by optimizer
    )

    if dc.fsdp_activation_checkpointing:
        _apply_activation_checkpointing(fsdp_model, dc.fsdp_transformer_layer_cls)

    return fsdp_model


def _apply_activation_checkpointing(fsdp_model: FSDP, layer_cls_names: List[str]):
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        layer_classes = find_transformer_layers(fsdp_model, layer_cls_names)
        if not layer_classes:
            logger.warning("Activation checkpointing: no matching layers found, skipping.")
            return

        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda m: isinstance(m, tuple(layer_classes)),
        )
        logger.info(f"Activation checkpointing applied to: {[c.__name__ for c in layer_classes]}")
    except Exception as e:
        logger.warning(f"Activation checkpointing failed (non-fatal): {e}")
