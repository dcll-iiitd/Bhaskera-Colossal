"""
bhaskera.launcher.torchrun_entry
=================================
Entry point for torchrun (single-node multi-GPU or torch elastic multi-node).

This is the recommended path for:
  - Local development / debugging  (torchrun --nproc_per_node=2 ...)
  - Multi-node without SLURM       (torchrun --nnodes=2 --node_rank=... ...)

torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
automatically before calling this module, so no manual env wrangling is needed.

Usage:
    # Single node, 2 GPUs:
    torchrun --nproc_per_node=2 -m bhaskera.launcher.torchrun_entry \\
             --config config_fsdp.yaml

    # Multi-node (run on EACH node):
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \\
             --master_addr=<head_ip> --master_port=29500 \\
             -m bhaskera.launcher.torchrun_entry --config config_fsdp.yaml
"""
from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _run_worker(args: argparse.Namespace) -> None:
    # torchrun sets these:
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][rank {global_rank}] %(levelname)s %(message)s",
    )

    # torchrun has already called init_process_group via its own mechanism,
    # but only if using the default elastic launch. With the standard
    # torchrun (non-elastic) we init manually.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from bhaskera.config_loader import load_config
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from bhaskera.data.registry import build_dataset
    dataset = build_dataset(cfg, tokenizer, global_rank, world_size)
    loader  = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, pin_memory=True)

    from bhaskera.models.registry import build_model
    model_device = (
        torch.device("cpu")
        if cfg.distributed.strategy.lower() == "fsdp"
        else device
    )
    model = build_model(cfg, model_device)

    from bhaskera.distributed.wrapper import wrap_model_distributed
    model = wrap_model_distributed(model=model, cfg=cfg, local_rank=local_rank, device=device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    logger_obj = None
    if cfg.TRACKER and global_rank == 0:
        from bhaskera.utils.logger_factory import build_logger
        logger_obj = build_logger(cfg)

    from bhaskera.trainer.train_loop import train
    train(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=device,
        grad_accum_steps=cfg.GRAD_ACCUM,
        max_steps=cfg.MAX_STEPS,
        local_rank=local_rank,
        global_rank=global_rank,
        cfg=cfg,
        logger_obj=logger_obj,
        num_epochs=getattr(cfg, "NUM_EPOCHS", 1),
        checkpoint_dir=cfg.CHECKPOINT_DIR if cfg.CHECKPOINT_ENABLED else None,
    )

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description="Bhaskera torchrun entrypoint")
    p.add_argument("--config", required=True, help="YAML config path")
    _run_worker(p.parse_args())


if __name__ == "__main__":
    main()