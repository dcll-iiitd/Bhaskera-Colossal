"""
Bhaskera CLI - Main entry point for training with FSDP/DDP support.

Usage:
    bhaskera --config config_fsdp.yaml --num-workers 4
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

# Sentinel — distinguishes "attribute value is None" from "attribute does not exist"
_MISSING = object()


# ==========================================================
# Configuration Loading - Handles Nested YAML
# ==========================================================
class Config:
    """Config class that handles nested dictionaries and provides flat access."""

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __getattr__(self, name):
        """
        Flat-name -> nested-path lookup for backward compatibility.
        Uses _MISSING sentinel so legitimate None values are returned correctly.
        """
        if name.startswith('_'):
            raise AttributeError(name)

        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        # (path, coerce_fn) — coerce_fn=None means return as-is
        mappings = {
            'MODEL_NAME':             (('model', 'name'),                         None),
            'ATTN_IMPL':              (('model', 'attn_impl'),                    None),
            'DTYPE':                  (('model', 'dtype'),                        None),
            'DATASET_NAME':           (('dataset', 'name'),                       None),
            'SEQ_LEN':                (('dataset', 'seq_len'),                    int),
            'BATCH_SIZE':             (('training', 'batch_size'),                int),
            'GRAD_ACCUM':             (('training', 'grad_accum'),                int),
            'LR':                     (('training', 'lr'),                        float),
            'MAX_STEPS':              (('training', 'max_steps'),                 int),
            'NUM_EPOCHS':             (('training', 'num_epochs'),                int),
            'PEFT':                   (('peft', 'method'),                        None),
            'LORA':                   (('peft', 'lora'),                          None),
            'TRACKER':                (('logging', 'tracker'),                    None),
            'CHECKPOINT_ENABLED':     (('checkpointing', 'enabled'),              bool),
            'CHECKPOINT_INTERVAL':    (('checkpointing', 'save_interval'),        int),
            'CHECKPOINT_DIR':         (('checkpointing', 'save_dir'),             None),
            'CHECKPOINT_KEEP_LAST_N': (('checkpointing', 'keep_last_n'),          int),
        }

        if name in mappings:
            path, coerce = mappings[name]
            obj = self
            for attr in path:
                obj = getattr(obj, attr, _MISSING)
                if obj is _MISSING:
                    raise AttributeError(
                        f"Config has no attribute '{name}' "
                        f"(missing key '{attr}' in path {path})"
                    )
            if obj is not None and coerce is not None:
                obj = coerce(obj)
            return obj

        raise AttributeError(f"Config has no attribute '{name}'")


def load_config(path: str) -> Config:
    """Load YAML config file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


# ==========================================================
# Distributed Wrapper - Supports DDP and FSDP
# ==========================================================
def wrap_model_distributed(model, config, local_rank, device):
    """
    Wrap model with DDP or FSDP based on config.

    IMPORTANT: For FSDP, the model must be on CPU when passed in.
    FSDP's device_id parameter handles moving each shard to GPU.

    For DDP, the model must already be fully on GPU before wrapping.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        strategy = config.training.distributed.strategy.lower()
    except AttributeError:
        strategy = "ddp"

    # ------------------------------------------------------------------
    # FSDP path
    # ------------------------------------------------------------------
    if strategy == "fsdp":
        logger.info(f"[Rank {torch.distributed.get_rank()}] Wrapping with FSDP")

        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            BackwardPrefetch,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial

        fsdp_cfg = config.training.distributed.fsdp

        sharding_map = {
            "FULL_SHARD":    ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD":      ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD":  ShardingStrategy.HYBRID_SHARD,
        }
        sharding_strategy = sharding_map.get(
            fsdp_cfg.sharding_strategy,
            ShardingStrategy.FULL_SHARD
        )

        dtype_map = {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }
        mixed_precision = MixedPrecision(
            param_dtype=dtype_map.get(fsdp_cfg.mixed_precision.param_dtype, torch.bfloat16),
            reduce_dtype=dtype_map.get(fsdp_cfg.mixed_precision.reduce_dtype, torch.bfloat16),
            buffer_dtype=dtype_map.get(fsdp_cfg.mixed_precision.buffer_dtype, torch.bfloat16),
        )

        # Find the actual transformer layer classes present in this model
        layer_classes = []
        for layer_name in fsdp_cfg.auto_wrap_policy.transformer_layer_cls:
            for name, module in model.named_modules():
                if module.__class__.__name__ == layer_name:
                    layer_classes.append(module.__class__)
                    logger.info(f"  Found transformer layer for FSDP wrap: {layer_name}")
                    break

        if not layer_classes:
            # Safety net — fall back to size-based wrapping rather than crashing
            logger.warning(
                "No transformer layers matched the names in config. "
                "Falling back to size-based auto wrap (min 100M params)."
            )
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(1e8))
        else:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=set(layer_classes),
            )

        backward_prefetch = None
        if fsdp_cfg.backward_prefetch == "BACKWARD_PRE":
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE
        elif fsdp_cfg.backward_prefetch == "BACKWARD_POST":
            backward_prefetch = BackwardPrefetch.BACKWARD_POST

        # NOTE: No try/except here — if FSDP fails, we want the real error,
        # not a silent fallback to DDP that produces a confusing secondary error.
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            backward_prefetch=backward_prefetch,
            device_id=local_rank,   # FSDP moves each shard from CPU to this GPU
            limit_all_gathers=True,
            use_orig_params=True,   # Required for optimizer to see LoRA params
        )

        # Activation checkpointing (applied after FSDP wrapping)
        if getattr(fsdp_cfg, "activation_checkpointing", False):
            _apply_activation_checkpointing(model, layer_classes)

        logger.info(f"[Rank {torch.distributed.get_rank()}] FSDP wrap complete")
        return model

    # ------------------------------------------------------------------
    # DDP path — model must already be on GPU
    # ------------------------------------------------------------------
    logger.info(f"[Rank {torch.distributed.get_rank()}] Wrapping with DDP")

    # If the model somehow ended up on CPU (e.g. misconfigured), move it now
    model = model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    return model


def _apply_activation_checkpointing(fsdp_model, layer_classes):
    """Apply activation checkpointing to FSDP-wrapped model."""
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
        from functools import partial

        if not layer_classes:
            return

        def check_fn(submodule):
            return isinstance(submodule, tuple(layer_classes))

        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=check_fn,
        )
        logger.info(f"Activation checkpointing applied to: {[c.__name__ for c in layer_classes]}")
    except Exception as e:
        logger.warning(f"Activation checkpointing failed (non-fatal): {e}")


# ==========================================================
# Ray Worker Function
# ==========================================================
def train_func(config):
    """Training function that runs on each Ray worker. Supports DDP and FSDP."""
    import torch.distributed as dist

    ctx         = ray.train.get_context()
    local_rank  = ctx.get_local_rank()
    global_rank = ctx.get_world_rank()
    world_size  = ctx.get_world_size()

    if not dist.is_initialized():
        logger.warning(f"[Rank {global_rank}] Distributed not initialized, initializing...")
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"[Rank {global_rank}/{world_size}] Initialized on GPU {local_rank}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info(f"[Rank {global_rank}] Building dataset...")
    dataset = build_dataset(config, tokenizer, global_rank, world_size)
    loader  = DataLoader(dataset, batch_size=config.BATCH_SIZE, pin_memory=True)

    # ------------------------------------------------------------------
    # Model device selection
    # FSDP: build on CPU — FSDP's device_id moves each shard to GPU during init.
    # DDP:  build on GPU — DDP requires the full model already on device.
    # ------------------------------------------------------------------
    try:
        use_fsdp = config.training.distributed.strategy.lower() == "fsdp"
    except AttributeError:
        use_fsdp = False

    model_device = torch.device("cpu") if use_fsdp else device
    logger.info(f"[Rank {global_rank}] Building model on {model_device} (FSDP={use_fsdp})")
    model = build_model(config, model_device)

    # ------------------------------------------------------------------
    # Distributed wrap
    # ------------------------------------------------------------------
    logger.info(f"[Rank {global_rank}] Wrapping model for distributed training...")
    model = wrap_model_distributed(model, config, local_rank, device)

    # ------------------------------------------------------------------
    # Optimizer — only trainable params (LoRA adapters)
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if global_rank == 0:
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    # ------------------------------------------------------------------
    # Experiment logger (rank-0 only)
    # ------------------------------------------------------------------
    logger_obj = None
    if config.TRACKER and global_rank == 0:
        from bhaskera.utils.logger_factory import build_logger
        log_gpu               = getattr(config, "log_gpu",               True)
        gpu_log_every_n_steps = getattr(config, "gpu_log_every_n_steps", 1)
        logger_obj = build_logger(
            config,
            log_gpu=log_gpu,
            gpu_log_every_n_steps=gpu_log_every_n_steps,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    try:
        ckpt_dir = config.CHECKPOINT_DIR if config.CHECKPOINT_ENABLED else None
    except AttributeError:
        ckpt_dir = None

    try:
        num_epochs = config.NUM_EPOCHS
    except AttributeError:
        num_epochs = 1

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info(f"[Rank {global_rank}] Starting training...")
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
        num_epochs=num_epochs,
        checkpoint_dir=ckpt_dir,
    )
    logger.info(f"[Rank {global_rank}] Training complete!")


# ==========================================================
# CLI Entry Point
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bhaskera Training Framework with DDP/FSDP support"
    )
    parser.add_argument("--config",      type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of Ray workers (GPUs to use)")
    args = parser.parse_args()

    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)

    try:
        strategy = cfg.training.distributed.strategy
        logger.info(f"Distributed strategy: {strategy}")
    except AttributeError:
        logger.info("Distributed strategy: DDP (default)")

    if ray.is_initialized():
        logger.info("Shutting down existing Ray instance...")
        ray.shutdown()

    logger.info(f"Initializing Ray with {args.num_workers} workers...")
    ray.init(num_gpus=args.num_workers)

    resources = ray.available_resources()
    logger.info(f"Ray resources available: {resources}")
    if resources.get('GPU', 0) < args.num_workers:
        logger.warning(
            f"Ray only sees {resources.get('GPU', 0)} GPUs "
            f"but {args.num_workers} workers requested!"
        )

    logger.info("Creating TorchTrainer...")
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
            resources_per_worker={"GPU": 1},
        ),
        torch_config=TorchConfig(backend="nccl", timeout_s=1800),
    )

    logger.info(f"Starting training with {args.num_workers} workers...")
    result = trainer.fit()
    logger.info("Training complete!")
    return result


if __name__ == "__main__":
    main()