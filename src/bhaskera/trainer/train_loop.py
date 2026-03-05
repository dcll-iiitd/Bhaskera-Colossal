"""
bhaskera.trainer.train_loop
============================
Stable training loop for FSDP / DDP.

Key guarantees:
  - FSDP checkpoint: ALL ranks call save_checkpoint (collective all-gather).
  - Scheduler state is saved and restored on resume (prevents warmup restart).
  - NaN/Inf loss and grad-norm are silently skipped — training never crashes.
  - Best-N checkpoint management: keeps the N lowest-loss checkpoints.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def is_fsdp_model(model) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except Exception:
        return False


def build_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then constant LR. Prevents NaN from LR spike in bfloat16+FSDP."""
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def clip_grads_fsdp(model, max_norm: float = 1.0) -> float:
    """
    Correct global grad-norm clip for FULL_SHARD FSDP.
    Each rank only holds sharded params — we must all-reduce the squared norms.
    """
    local_sq = torch.tensor(0.0, device="cuda")
    for p in model.parameters():
        if p.grad is not None:
            local_sq += p.grad.detach().float().norm(2) ** 2
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    global_norm = local_sq.sqrt().item()
    if global_norm > max_norm:
        coef = max_norm / (global_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(coef)
    return global_norm


def _manage_checkpoints(current_path: str, current_loss: float,
                         best: list, keep_n: int) -> list:
    best.append((current_loss, current_path))
    best.sort(key=lambda x: x[0])
    to_keep = {path for _, path in best[:keep_n]}
    updated = []
    for loss, path in best:
        if path in to_keep:
            updated.append((loss, path))
        else:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"[Checkpoint] Pruned: {path}")
            except OSError as e:
                logger.warning(f"[Checkpoint] Could not prune {path}: {e}")
    return updated


# ── main train function ────────────────────────────────────────────────────────

def train(
    *,
    model,
    dataloader,
    optimizer,
    device: torch.device,
    grad_accum_steps: int,
    max_steps: int,
    local_rank: int,
    global_rank: int,
    cfg,
    scaler=None,
    logger_obj=None,
    num_epochs: int = 1,
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> None:
    from bhaskera.distributed.wrapper import is_fsdp_model, load_checkpoint, save_checkpoint

    is_fsdp     = is_fsdp_model(model)
    strategy    = "FSDP" if is_fsdp else "DDP"
    warmup_steps = getattr(cfg, "WARMUP_STEPS", 20)

    scheduler = build_warmup_scheduler(optimizer, warmup_steps, max_steps)

    # ── resume from checkpoint ─────────────────────────────────────────────
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(model, optimizer, scheduler, resume_from, cfg, device)
        if global_rank == 0:
            logger.info(f"Resumed from {resume_from} at step {start_step}")

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if global_rank == 0:
        logger.info(f"[{strategy}] Training start | epochs={num_epochs} max_steps={max_steps}")
        logger.info(
            f"  batch/GPU={cfg.BATCH_SIZE} grad_accum={grad_accum_steps} "
            f"LR={cfg.LR:.2e} warmup={warmup_steps}"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step            = start_step
    best_checkpoints: list = []
    keep_n          = getattr(cfg, "CHECKPOINT_KEEP_LAST_N", 3)
    save_interval   = getattr(cfg, "CHECKPOINT_INTERVAL", 100)

    for epoch in range(num_epochs):
        if step >= max_steps:
            break

        micro = 0

        for batch in dataloader:
            if step >= max_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)
            loss    = outputs.loss / grad_accum_steps

            if not torch.isfinite(loss):
                if global_rank == 0:
                    logger.warning(
                        f"[{strategy}][e{epoch}][s{step}] "
                        "NaN/Inf loss — skipping batch."
                    )
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            loss.backward()
            micro += 1

            if micro % grad_accum_steps != 0:
                continue

            # ── optimizer step ─────────────────────────────────────────────
            if is_fsdp:
                grad_norm = clip_grads_fsdp(model, 1.0)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

            if not math.isfinite(grad_norm):
                if global_rank == 0:
                    logger.warning(
                        f"[{strategy}][e{epoch}][s{step}] "
                        f"Non-finite grad norm ({grad_norm:.4f}) — skipping step."
                    )
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro = 0

            actual_loss = loss.item() * grad_accum_steps
            current_lr  = scheduler.get_last_lr()[0]

            if global_rank == 0:
                logger.info(
                    f"[{strategy}][e{epoch}][s{step}] "
                    f"loss={actual_loss:.4f} lr={current_lr:.2e} "
                    f"gnorm={grad_norm:.4f}"
                )
                if logger_obj:
                    logger_obj.log(
                        {"loss": actual_loss, "lr": current_lr,
                         "grad_norm": grad_norm, "epoch": epoch},
                        step=step,
                    )

            step += 1

            # ── per-step checkpoint ─────────────────────────────────────────
            if checkpoint_dir and save_interval > 0 and step % save_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
                # FSDP: ALL ranks must call this
                save_checkpoint(model, optimizer, scheduler, step, cfg,
                                global_rank, ckpt_path)
                if global_rank == 0:
                    best_checkpoints = _manage_checkpoints(
                        ckpt_path, actual_loss, best_checkpoints, keep_n
                    )

        # ── end-of-epoch checkpoint ─────────────────────────────────────────
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
            # FSDP: ALL ranks must call this
            save_checkpoint(model, optimizer, scheduler, step, cfg,
                            global_rank, ckpt_path)
            if global_rank == 0:
                best_checkpoints = _manage_checkpoints(
                    ckpt_path, actual_loss if micro == 0 else 999.0,
                    best_checkpoints, keep_n
                )

    if global_rank == 0 and logger_obj:
        logger_obj.finish()

    if global_rank == 0:
        logger.info(f"[{strategy}] Training complete at step {step}.")
