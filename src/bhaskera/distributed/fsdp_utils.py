"""
bhaskera.distributed.fsdp_utils
================================
FSDP2 wrapping and activation checkpointing helpers.

Requires: torch >= 2.4  (FSDP2 / fully_shard is stable from 2.4+)
torch 2.10 (your version) ships FSDP2 fully stable.

Key differences from FSDP1:
  - Use fully_shard() instead of FSDP(model, ...)
  - fully_shard() modifies the model IN-PLACE — no wrapper returned
  - Apply activation checkpointing BEFORE calling fully_shard()
  - clip_grad_norm_ works directly (no custom all-reduce needed)
  - MixedPrecisionPolicy replaces MixedPrecision
"""
from __future__ import annotations

import logging
from functools import partial
from typing import List, Optional, Set

import torch
import torch.distributed as dist

# FSDP2 imports
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

logger = logging.getLogger(__name__)


# ── dtype helpers ──────────────────────────────────────────────────────────────

def get_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_str, torch.float32)


# ── layer discovery ────────────────────────────────────────────────────────────

def find_transformer_layers(model: torch.nn.Module,
                             names: List[str]) -> Set[type]:
    """Walk the model and collect the actual classes matching config names."""
    found: Set[type] = set()
    seen:  Set[str]  = set()
    for _, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in names and cls_name not in seen:
            found.add(module.__class__)
            seen.add(cls_name)
            logger.info(f"  FSDP2 wrap layer found: {cls_name}")
    return found


# ── activation checkpointing ───────────────────────────────────────────────────

def _apply_activation_checkpointing(model: torch.nn.Module,
                                     layer_cls_names: List[str]) -> None:
    """
    Apply activation (gradient) checkpointing BEFORE fully_shard().
    This is required by FSDP2 — checkpointing after sharding is unsupported.
    """
    layer_classes = find_transformer_layers(model, layer_cls_names)
    if not layer_classes:
        logger.warning(
            "Activation checkpointing: no matching layers found — skipping. "
            "Check fsdp_transformer_layer_cls in your config."
        )
        return

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=lambda m: isinstance(m, tuple(layer_classes)),
    )
    logger.info(
        f"Activation checkpointing applied to: {[c.__name__ for c in layer_classes]}"
    )


# ── main FSDP2 wrap function ───────────────────────────────────────────────────

def wrap_model_fsdp(model: torch.nn.Module, cfg, device_id: int) -> torch.nn.Module:
    """
    Wrap a model with FSDP2 (fully_shard).

    FSDP2 workflow:
      1. Apply activation checkpointing to each transformer layer (if enabled)
      2. Call fully_shard() on each transformer layer
      3. Call fully_shard() on the root model

    The model is modified IN-PLACE. The same object is returned for
    compatibility with the rest of the codebase, but there is no FSDP
    wrapper class around it — it's an FSDPModule via __torch_dispatch__.
    """
    dc = cfg.distributed   # DistributedConfig

    # Build MixedPrecisionPolicy (FSDP2 style)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=get_dtype(dc.fsdp_mixed_precision_param),
        reduce_dtype=get_dtype(dc.fsdp_mixed_precision_reduce),
        # output_dtype is optional; buffer_dtype is not a separate field in FSDP2
    )

    # CPU offload in FSDP2
    cpu_offload = None
    if dc.fsdp_cpu_offload:
        from torch.distributed._composable.fsdp import CPUOffloadPolicy
        cpu_offload = CPUOffloadPolicy()
        logger.info("FSDP2 | CPU offload enabled")

    # Discover transformer layer classes
    layer_classes = find_transformer_layers(model, dc.fsdp_transformer_layer_cls)

    if not layer_classes:
        logger.warning(
            "No transformer layers matched config names. "
            "The root model will be sharded as a single unit. "
            "This works but is less memory-efficient than per-layer sharding."
        )

    # Step 1 — Activation checkpointing (must happen BEFORE fully_shard)
    if dc.fsdp_activation_checkpointing:
        _apply_activation_checkpointing(model, dc.fsdp_transformer_layer_cls)

    # Step 2 — Shard each transformer layer individually
    fsdp_kwargs: dict = {"mp_policy": mp_policy}
    if cpu_offload is not None:
        fsdp_kwargs["offload_policy"] = cpu_offload

    for module in model.modules():
        if layer_classes and isinstance(module, tuple(layer_classes)):
            fully_shard(module, **fsdp_kwargs)

    # Step 3 — Shard the root model
    fully_shard(model, **fsdp_kwargs)

    logger.info(
        f"FSDP2 | shard=FULL_SHARD "
        f"mixed_prec={dc.fsdp_mixed_precision_param} "
        f"cpu_offload={dc.fsdp_cpu_offload} "
        f"act_ckpt={dc.fsdp_activation_checkpointing} "
        f"layers_sharded={[c.__name__ for c in layer_classes]}"
    )

    return model
