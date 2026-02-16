"""
Bhaskera CLI - Main entry point for training with FSDP support
"""
import argparse
import yaml
import torch
import ray
import logging

from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from bhaskera.data.registry import build_dataset
from bhaskera.models.registry import build_model
from bhaskera.trainer.train_loop import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
# Configuration Loading - Handles Nested YAML
# ==========================================================
class Config:
    """Config class that handles nested dictionaries and provides flat access"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __getattr__(self, name):
        """Provide backward compatibility for flat attribute access"""
        # Try to get from self first
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        # Map old flat names to new nested structure
        mappings = {
            'MODEL_NAME': ('model', 'name'),
            'ATTN_IMPL': ('model', 'attn_impl'),
            'DATASET_NAME': ('dataset', 'name'),
            'SEQ_LEN': ('dataset', 'seq_len'),
            'BATCH_SIZE': ('training', 'batch_size'),
            'GRAD_ACCUM': ('training', 'grad_accum'),
            'LR': ('training', 'lr'),
            'MAX_STEPS': ('training', 'max_steps'),
            'PEFT': ('peft', 'method'),
            'LORA': ('peft', 'lora'),
            'TRACKER': ('training', 'tracker'),
            'CHECKPOINT_ENABLED': ('training', 'checkpoint', 'enabled'),
            'CHECKPOINT_INTERVAL': ('training', 'checkpoint', 'interval'),
            'CHECKPOINT_DIR': ('training', 'checkpoint', 'dir'),
            'CHECKPOINT_KEEP_LAST_N': ('training', 'checkpoint', 'keep_last_n'),
        }

        if name in mappings:
            path = mappings[name]
            obj = self
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    raise AttributeError(f"Config has no attribute '{name}' (mapped to {path})")
            return obj

        raise AttributeError(f"Config has no attribute '{name}'")


def load_config(path: str) -> Config:
    """Load YAML config file"""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


# ==========================================================
# Distributed Wrapper - Supports DDP and FSDP
# ==========================================================
def wrap_model_distributed(model, config, local_rank, device):
    """
    Wrap model with DDP or FSDP based on config.

    Args:
        model: Model to wrap
        config: Config object
        local_rank: Local GPU rank
        device: Device

    Returns:
        Wrapped model (DDP or FSDP)
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Check if FSDP is requested
    try:
        strategy = config.training.distributed.strategy.lower()
        use_fsdp = (strategy == "fsdp")
    except AttributeError:
        use_fsdp = False

    if use_fsdp:
        logger.info(f"[Rank {torch.distributed.get_rank()}] Using FSDP")
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
                MixedPrecision,
                BackwardPrefetch,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from functools import partial

            # Get FSDP config
            fsdp_cfg = config.training.distributed.fsdp

            # Sharding strategy
            sharding_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
                "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
            }
            sharding_strategy = sharding_map.get(
                fsdp_cfg.sharding_strategy,
                ShardingStrategy.FULL_SHARD
            )

            # Mixed precision
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            mixed_precision = MixedPrecision(
                param_dtype=dtype_map.get(fsdp_cfg.mixed_precision.param_dtype, torch.float32),
                reduce_dtype=dtype_map.get(fsdp_cfg.mixed_precision.reduce_dtype, torch.float32),
                buffer_dtype=dtype_map.get(fsdp_cfg.mixed_precision.buffer_dtype, torch.float32),
            )

            # Auto wrap policy - find transformer layers
            layer_classes = []
            for layer_name in fsdp_cfg.auto_wrap_policy.transformer_layer_cls:
                for name, module in model.named_modules():
                    if module.__class__.__name__ == layer_name:
                        layer_classes.append(module.__class__)
                        logger.info(f"Found transformer layer: {layer_name}")
                        break

            auto_wrap_policy = None
            if layer_classes:
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=set(layer_classes),
                )

            # Backward prefetch
            backward_prefetch = None
            if fsdp_cfg.backward_prefetch == "BACKWARD_PRE":
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE
            elif fsdp_cfg.backward_prefetch == "BACKWARD_POST":
                backward_prefetch = BackwardPrefetch.BACKWARD_POST

            # Wrap with FSDP
            model = FSDP(
                model,
                sharding_strategy=sharding_strategy,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                backward_prefetch=backward_prefetch,
                device_id=local_rank,
                limit_all_gathers=True,
                use_orig_params=True,
            )

            logger.info(f"[Rank {torch.distributed.get_rank()}] Model wrapped with FSDP")
            return model

        except Exception as e:
            logger.error(f"FSDP wrapping failed: {e}")
            logger.warning("Falling back to DDP")
            use_fsdp = False

    # Default to DDP
    logger.info(f"[Rank {torch.distributed.get_rank()}] Using DDP")
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    return model


# ==========================================================
# Ray Worker Function
# ==========================================================
def train_func(config):
    """
    Training function that runs on each Ray worker.
    Supports both DDP and FSDP.
    """
    import torch.distributed as dist

    # Get Ray Train context
    ctx = ray.train.get_context()

    local_rank = ctx.get_local_rank()
    global_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    # Check if distributed is initialized
    if not dist.is_initialized():
        logger.warning(f"[Rank {global_rank}] Distributed not initialized, initializing...")
        dist.init_process_group(backend="nccl")

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"[Rank {global_rank}/{world_size}] Initialized on GPU {local_rank}")

    # -----------------------
    # Tokenizer
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -----------------------
    # Dataset
    # -----------------------
    logger.info(f"[Rank {global_rank}] Building dataset...")
    dataset = build_dataset(
        config,
        tokenizer,
        global_rank,
        world_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=True,
    )

    # -----------------------
    # Model
    # -----------------------
    logger.info(f"[Rank {global_rank}] Building model...")

    # For FSDP, start on CPU so FSDP can shard across GPUs
    # For DDP, start directly on GPU
    try:
        use_fsdp = config.training.distributed.strategy.lower() == "fsdp"
        model_device = torch.device("cpu") if use_fsdp else device
    except AttributeError:
        model_device = device

    model = build_model(config, model_device)

    # Wrap with DDP or FSDP
    logger.info(f"[Rank {global_rank}] Wrapping model for distributed training...")
    model = wrap_model_distributed(model, config, local_rank, device)

    # -----------------------
    # Optimizer
    # -----------------------
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )


    # -----------------------
    # Logger (rank-0 only)
    # -----------------------
    logger_obj = None
    try:
        if config.TRACKER and global_rank == 0:
            from bhaskera.utils.logger_factory import build_logger
            logger_obj = build_logger(config)
    except AttributeError:
        pass  # TRACKER not configured, skip

    logger.info(f"[Rank {global_rank}] Starting training...")

    # -----------------------
    # Train
    # -----------------------
    train(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=device,
        grad_accum_steps=config.GRAD_ACCUM,
        max_steps=config.MAX_STEPS,
        local_rank=local_rank,
        global_rank=global_rank,
        cfg=config,
        logger_obj=logger_obj,
    )

    logger.info(f"[Rank {global_rank}] Training complete!")


# ==========================================================
# CLI Entry Point
# ==========================================================
def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Bhaskera Training Framework with DDP/FSDP support"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Ray workers (GPUs to use)",
    )

    args = parser.parse_args()

    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)

    # Log strategy
    try:
        strategy = cfg.training.distributed.strategy
        logger.info(f"Distributed strategy: {strategy}")
    except AttributeError:
        logger.info("Distributed strategy: DDP (default)")

    logger.info(f"Initializing Ray with {args.num_workers} workers...")

    # Shutdown any existing Ray instance
    if ray.is_initialized():
        logger.info("Shutting down existing Ray instance...")
        ray.shutdown()

    # Initialize Ray with explicit GPU configuration
    ray.init(num_gpus=args.num_workers)

    # Log Ray resources
    resources = ray.available_resources()
    logger.info(f"Ray resources available: {resources}")

    if resources.get('GPU', 0) < args.num_workers:
        logger.warning(
            f"⚠️  Ray only sees {resources.get('GPU', 0)} GPUs, "
            f"but {args.num_workers} workers requested!"
        )
        logger.warning("Make sure CUDA_VISIBLE_DEVICES is set correctly")

    # Create and run trainer
    logger.info("Creating TorchTrainer...")
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
            resources_per_worker={
                "GPU": 1,
                "CPU": 2,
            },
        ),
        torch_config=TorchConfig(
            backend="nccl",
            timeout_s=1800,
        ),
    )

    logger.info(f"Starting training with {args.num_workers} workers...")
    result = trainer.fit()

    logger.info("Training complete!")
    return result


if __name__ == "__main__":
    main()
