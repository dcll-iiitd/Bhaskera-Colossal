"""
bhaskera.launcher.worker_core
==============================
Single training body called by all entry points.
Only rank 0 prints. All other ranks are completely silent.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ── logging: rank 0 prints to stdout, all others → /dev/null ─────────────────

def _setup_logging(global_rank: int) -> None:
    if global_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",      # clean — no timestamps, no module name
            stream=sys.stdout,
            force=True,
        )
    else:
        # Silence every logger on non-zero ranks completely
        logging.basicConfig(
            level=logging.CRITICAL,
            stream=open(os.devnull, "w"),
            force=True,
        )
        # Also silence the root logger and all third-party noisy loggers
        logging.getLogger().setLevel(logging.CRITICAL)


def _log(msg: str, global_rank: int) -> None:
    """Print only on rank 0, always flushed."""
    if global_rank == 0:
        print(msg, flush=True)


@dataclass
class WorkerContext:
    global_rank: int
    local_rank:  int
    world_size:  int
    device:      torch.device
    launcher:    str


def run_worker(ctx: WorkerContext, cfg) -> None:
    _setup_logging(ctx.global_rank)

    # ── Audio jobs: route entirely to the Whisper pipeline ───────────────────
    # Must be checked BEFORE any LLM-specific code (tokenizer, dataset, model).
    if cfg.DATASET_NAME.lower() == "audio":
        from bhaskera.audio.worker import run_audio_worker
        run_audio_worker(ctx, cfg)
        return

    is_rank0 = ctx.global_rank == 0

    # ── config summary (rank 0 only) ──────────────────────────────────────────
    if is_rank0:
        strategy = cfg.distributed.strategy.upper()
        print("=" * 60, flush=True)
        print(f"  Bhaskera — {strategy} Training", flush=True)
        print("=" * 60, flush=True)
        print(f"  Launcher   : {ctx.launcher}", flush=True)
        print(f"  Backend    : {strategy}", flush=True)
        print(f"  World size : {ctx.world_size}", flush=True)
        print(f"  Model      : {cfg.MODEL_NAME}", flush=True)
        print(f"  Dataset    : {cfg.DATASET_NAME}  seq_len={cfg.SEQ_LEN}", flush=True)
        print(f"  PEFT       : {cfg.PEFT}", flush=True)
        print(f"  Batch/GPU  : {cfg.BATCH_SIZE}  grad_accum={cfg.GRAD_ACCUM}", flush=True)
        print(f"  Eff. batch : {cfg.BATCH_SIZE * cfg.GRAD_ACCUM * ctx.world_size}", flush=True)
        print(f"  LR         : {cfg.LR:.2e}  warmup={cfg.WARMUP_STEPS}", flush=True)
        print(f"  Max steps  : {cfg.MAX_STEPS}  epochs={cfg.NUM_EPOCHS}", flush=True)
        print(f"  Checkpoint : {'enabled → ' + cfg.CHECKPOINT_DIR if cfg.CHECKPOINT_ENABLED else 'disabled'}", flush=True)
        print("=" * 60, flush=True)

    # ── tokenizer ─────────────────────────────────────────────────────────────
    _log("\n[1/4] Loading tokenizer...", ctx.global_rank)
    # Suppress HF download noise on all ranks
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _log(f"     tokenizer: {cfg.MODEL_NAME}  vocab={tokenizer.vocab_size:,}", ctx.global_rank)

    # ── dataset ───────────────────────────────────────────────────────────────
    _log("\n[2/4] Building dataset...", ctx.global_rank)
    from bhaskera.data.registry import build_dataset
    dataset = build_dataset(cfg, tokenizer, ctx.global_rank, ctx.world_size)
    loader  = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
    )
    _log(f"     dataset : {cfg.DATASET_NAME}  (rank {ctx.global_rank} shard, streaming)", ctx.global_rank)

    # ── model ─────────────────────────────────────────────────────────────────
    _log("\n[3/4] Loading model...", ctx.global_rank)
    is_fsdp      = cfg.distributed.strategy.lower() == "fsdp"
    model_device = torch.device("cpu") if is_fsdp else ctx.device

    from bhaskera.models.registry import build_model
    model = build_model(cfg, model_device)

    # ── distributed wrap ──────────────────────────────────────────────────────
    from bhaskera.distributed.wrapper import wrap_model_distributed
    model = wrap_model_distributed(
        model=model, cfg=cfg,
        local_rank=ctx.local_rank, device=ctx.device,
    )

    # Model stats — only rank 0
    if is_rank0:
        total       = sum(p.numel() for p in model.parameters())
        trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"     model   : {cfg.MODEL_NAME}", flush=True)
        print(f"     params  : {total/1e9:.2f}B total  {trainable/1e6:.2f}M trainable ({trainable/total*100:.4f}%)", flush=True)
        print(f"     dtype   : {cfg.DTYPE}  peft={cfg.PEFT}", flush=True)
        print(f"     strategy: {cfg.distributed.strategy.upper()}  act_ckpt={cfg.distributed.fsdp_activation_checkpointing}", flush=True)

    # ── optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
    )

    # ── experiment logger ─────────────────────────────────────────────────────
    logger_obj = None
    if cfg.TRACKER and is_rank0:
        from bhaskera.utils.logger_factory import build_logger
        logger_obj = build_logger(cfg, log_gpu=True, gpu_log_every_n_steps=10)

    checkpoint_dir = cfg.CHECKPOINT_DIR if cfg.CHECKPOINT_ENABLED else None

    # ── train ─────────────────────────────────────────────────────────────────
    _log("\n[4/4] Training...\n", ctx.global_rank)

    from bhaskera.trainer.train_loop import train
    train(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=ctx.device,
        grad_accum_steps=cfg.GRAD_ACCUM,
        max_steps=cfg.MAX_STEPS,
        local_rank=ctx.local_rank,
        global_rank=ctx.global_rank,
        cfg=cfg,
        logger_obj=logger_obj,
        num_epochs=cfg.NUM_EPOCHS,
        checkpoint_dir=checkpoint_dir,
    )
