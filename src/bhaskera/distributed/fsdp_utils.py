"""
FSDP wrapper utilities for distributed training.
Works with any transformer architecture by dynamically detecting layer types.
"""
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from functools import partial
from typing import Optional, List, Set
import logging

logger = logging.getLogger(__name__)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def get_sharding_strategy(strategy_str: str) -> ShardingStrategy:
    """Convert string to ShardingStrategy enum."""
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        "_HYBRID_SHARD_ZERO2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    return strategy_map.get(strategy_str, ShardingStrategy.FULL_SHARD)


def get_backward_prefetch(prefetch_str: Optional[str]) -> Optional[BackwardPrefetch]:
    """Convert string to BackwardPrefetch enum."""
    if prefetch_str is None or prefetch_str.lower() == "null":
        return None
    
    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    return prefetch_map.get(prefetch_str, BackwardPrefetch.BACKWARD_PRE)


def get_state_dict_type(state_dict_str: str) -> StateDictType:
    """Convert string to StateDictType enum."""
    state_dict_map = {
        "FULL_STATE_DICT": StateDictType.FULL_STATE_DICT,
        "SHARDED_STATE_DICT": StateDictType.SHARDED_STATE_DICT,
        "LOCAL_STATE_DICT": StateDictType.LOCAL_STATE_DICT,
    }
    return state_dict_map.get(state_dict_str, StateDictType.FULL_STATE_DICT)


def find_transformer_layers(model: torch.nn.Module, layer_class_names: List[str]) -> Set[type]:
    """
    Dynamically find transformer layer classes in the model.
    
    Args:
        model: The model to search
        layer_class_names: List of potential transformer layer class names
    
    Returns:
        Set of layer classes found in the model
    """
    found_layers = set()
    
    for name, module in model.named_modules():
        module_class_name = module.__class__.__name__
        if module_class_name in layer_class_names:
            found_layers.add(module.__class__)
            logger.info(f"Found transformer layer: {module_class_name} at {name}")
    
    return found_layers


def get_auto_wrap_policy(model: torch.nn.Module, cfg):
    """
    Get the auto wrap policy for FSDP based on configuration.
    
    Args:
        model: The model to wrap
        cfg: Configuration object with FSDP settings
    
    Returns:
        Auto wrap policy function or None
    """
    policy_type = cfg.distributed.fsdp_auto_wrap_policy
    
    if policy_type is None or policy_type.lower() == "null":
        return None
    
    if policy_type == "transformer_auto_wrap":
        # Dynamically find transformer layers
        layer_classes = find_transformer_layers(
            model,
            cfg.distributed.fsdp_transformer_layer_cls
        )
        
        if not layer_classes:
            logger.warning(
                "No transformer layers found with names: "
                f"{cfg.distributed.fsdp_transformer_layer_cls}. "
                "Falling back to size-based wrapping."
            )
            return partial(
                size_based_auto_wrap_policy,
                min_num_params=int(cfg.distributed.fsdp_min_num_params)
            )
        
        logger.info(f"Using transformer auto wrap policy with layers: {layer_classes}")
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=layer_classes,
        )
    
    elif policy_type == "size_based_auto_wrap":
        return partial(
            size_based_auto_wrap_policy,
            min_num_params=int(cfg.distributed.fsdp_min_num_params)
        )
    
    else:
        logger.warning(f"Unknown auto wrap policy: {policy_type}. Using None.")
        return None


def setup_mixed_precision(cfg) -> MixedPrecision:
    """
    Setup mixed precision policy for FSDP.
    
    Args:
        cfg: Configuration object with FSDP settings
    
    Returns:
        MixedPrecision policy
    """
    param_dtype = get_dtype(cfg.distributed.fsdp_mixed_precision_param)
    reduce_dtype = get_dtype(cfg.distributed.fsdp_mixed_precision_reduce)
    buffer_dtype = get_dtype(cfg.distributed.fsdp_mixed_precision_buffer)
    
    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )


def setup_cpu_offload(cfg) -> Optional[CPUOffload]:
    """
    Setup CPU offload for FSDP.
    
    Args:
        cfg: Configuration object with FSDP settings
    
    Returns:
        CPUOffload object or None
    """
    if cfg.distributed.fsdp_cpu_offload:
        return CPUOffload(offload_params=True)
    return None


def wrap_model_fsdp(
    model: torch.nn.Module,
    cfg,
    device_id: int,
) -> FSDP:
    """
    Wrap a model with FSDP.
    
    Args:
        model: The model to wrap (should already be on the correct device)
        cfg: Configuration object with FSDP settings
        device_id: Local device ID
    
    Returns:
        FSDP-wrapped model
    """
    # Get sharding strategy
    sharding_strategy = get_sharding_strategy(cfg.distributed.fsdp_sharding_strategy)
    
    # Get auto wrap policy
    auto_wrap_policy = get_auto_wrap_policy(model, cfg)
    
    # Setup mixed precision
    mixed_precision = setup_mixed_precision(cfg)
    
    # Setup CPU offload
    cpu_offload = setup_cpu_offload(cfg)
    
    # Get backward prefetch
    backward_prefetch = get_backward_prefetch(cfg.distributed.fsdp_backward_prefetch)
    
    # Get forward prefetch
    forward_prefetch = cfg.distributed.fsdp_forward_prefetch
    
    logger.info(f"Wrapping model with FSDP:")
    logger.info(f"  - Sharding strategy: {sharding_strategy}")
    logger.info(f"  - Mixed precision: param={mixed_precision.param_dtype}, "
                f"reduce={mixed_precision.reduce_dtype}, buffer={mixed_precision.buffer_dtype}")
    logger.info(f"  - CPU offload: {cfg.distributed.fsdp_cpu_offload}")
    logger.info(f"  - Backward prefetch: {backward_prefetch}")
    logger.info(f"  - Forward prefetch: {forward_prefetch}")
    
    # Wrap the model with FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        forward_prefetch=forward_prefetch,
        device_id=device_id,
        limit_all_gathers=True,
        use_orig_params=True,  # Required for optimizer state dict
    )
    
    # Apply activation checkpointing if enabled
    if cfg.distributed.fsdp_activation_checkpointing:
        apply_activation_checkpointing(fsdp_model, cfg)
    
    return fsdp_model


def apply_activation_checkpointing(model: FSDP, cfg):
    """
    Apply activation checkpointing to FSDP-wrapped model.
    
    Args:
        model: FSDP-wrapped model
        cfg: Configuration object
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing as apply_ac,
    )
    
    # Find transformer layers again for checkpointing
    layer_classes = find_transformer_layers(
        model,
        cfg.distributed.fsdp_transformer_layer_cls
    )
    
    if not layer_classes:
        logger.warning("No transformer layers found for activation checkpointing")
        return
    
    def check_fn(submodule):
        """Check if this submodule should be checkpointed."""
        return isinstance(submodule, tuple(layer_classes))
    
    logger.info(f"Applying activation checkpointing to: {layer_classes}")
    
    apply_ac(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )
