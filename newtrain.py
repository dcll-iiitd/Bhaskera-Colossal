"""
Bhaskera training launcher with unified DDP/FSDP support.
Usage:
    python newtrain.py --config config.yaml --num-workers 4
    python newtrain.py --num-workers 2  # Uses default DDP config
"""
import argparse
import torch
import ray
import logging

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import bhaskera.config as legacy_config
from bhaskera.data.registry import build_dataset
from bhaskera.models.registry import build_model

# Import new modules
from bhaskera.config_loader import load_config
from bhaskera.distributed_wrapper import wrap_model_distributed
from bhaskera.train_loop_unified import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# Worker function with unified DDP/FSDP support
# =====================================================
def train_func(train_loop_config):
    """
    Training function that runs on each worker.
    Supports both DDP and FSDP based on configuration.
    """
    import torch.distributed as dist

    # Ray Train automatically sets up the distributed backend
    # But we need to ensure it's initialized for FSDP
    if not dist.is_initialized():
        # This shouldn't happen with Ray Train, but just in case
        logger.warning("Distributed not initialized by Ray, initializing manually")
        dist.init_process_group(backend="nccl")

    ctx = ray.train.get_context()

    local_rank = ctx.get_local_rank()
    global_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"[Rank {global_rank}/{world_size}] Initialized on GPU {local_rank}")

    # Load config (either from YAML or use legacy config)
    config_path = train_loop_config.get("config_path")
    if config_path:
        cfg = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
        logger.info(f"Using distributed strategy: {cfg.distributed.strategy}")
    else:
        # Backward compatibility: use legacy config module
        cfg = legacy_config
        # Add default distributed config for backward compatibility
        from bhaskera.config_loader import DistributedConfig
        if not hasattr(cfg, 'distributed'):
            cfg.distributed = DistributedConfig(strategy="ddp")
        logger.info("Using legacy config (DDP mode)")

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Dataset
    dataset = build_dataset(
        cfg,
        tokenizer,
        global_rank,
        world_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        pin_memory=True,
    )

    # Build model (on CPU or GPU depending on strategy)
    # For FSDP, model can be on CPU initially
    # For DDP, model should be on GPU
    if cfg.distributed.strategy.lower() == "fsdp":
        # For FSDP, we can build on CPU to save memory
        # The FSDP wrapper will handle device placement
        model_device = torch.device("cpu")
    else:
        # For DDP, build directly on GPU
        model_device = device

    model = build_model(cfg, model_device)

    # Wrap model with distributed strategy (DDP or FSDP)
    model = wrap_model_distributed(
        model=model,
        cfg=cfg,
        local_rank=local_rank,
        device=device,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )


    # Gradient scaler
    scaler = torch.amp.GradScaler("cuda")

    # Setup logger if configured
    logger_obj = None
    if cfg.TRACKER and global_rank == 0:
        from bhaskera.utils.logger_factory import build_logger
        logger_obj = build_logger(cfg)

    # Run training
    train(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        grad_accum_steps=cfg.GRAD_ACCUM,
        max_steps=cfg.MAX_STEPS,
        local_rank=local_rank,
        global_rank=global_rank,
        cfg=cfg,
        logger_obj=logger_obj,
    )


# =====================================================
# Ray launcher
# =====================================================
def launch_ray(num_workers: int, config_path: str = None):
    """
    Launch distributed training with Ray.

    Args:
        num_workers: Number of GPUs/workers
        config_path: Path to YAML config file (optional)
    """
    # Shutdown any existing Ray instance
    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray with explicit GPU configuration
    logger.info(f"Initializing Ray with {num_workers} GPUs")
    ray.init(
        num_gpus=num_workers,
        # Set log verbosity if needed for debugging
        # logging_level=logging.INFO,
    )

    # Ray Train will automatically set up the torch distributed backend
    # We just need to configure it properly
    from ray.train.torch import TorchConfig

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "config_path": config_path,
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            # Explicitly allocate 1 GPU per worker
            resources_per_worker={
                "GPU": 1,  # Each worker gets 1 GPU
                "CPU": 2,  # Each worker gets 2 CPUs (adjust as needed)
            },
        ),
        # Explicitly configure torch distributed backend for FSDP
        torch_config=TorchConfig(
            backend="nccl",  # Use NCCL for GPU communication
            timeout_s=1800,  # 30 minute timeout for operations
        ),
    )

    logger.info(f"Starting training with {num_workers} workers")
    result = trainer.fit()
    return result


# =====================================================
# CLI entrypoint
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bhaskera training launcher with DDP/FSDP support"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Ray workers (GPUs)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional, defaults to legacy config.py)",
    )

    args = parser.parse_args()

    if args.config:
        logger.info(f"Using config file: {args.config}")
    else:
        logger.info("No config file provided, using legacy config.py (DDP mode)")

    launch_ray(args.num_workers, args.config)


if __name__ == "__main__":
    main()
