"""
bhaskera.trainer.train_loop
============================
Only rank 0 prints. Clean loss-per-step output, nothing else.

FSDP2 changes:
  - is_fsdp_model() now delegates to wrapper.py (checks FSDPModule)
  - clip_grads_fsdp() uses torch.nn.utils.clip_grad_norm_ directly.
    FSDP2 handles the all-reduce of grad norms internally — the old
    manual all_reduce + local norm loop was only needed for FSDP1.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.distributed as dist


def _p(msg: str, global_rank: int) -> None:
    """Print only from rank 0, always flushed."""
    if global_rank == 0:
        print(msg, flush=True)


def build_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def clip_grads_fsdp(model, max_norm: float = 1.0) -> float:
    """
    Gradient clipping for FSDP2.

    With FSDP2 (fully_shard), torch.nn.utils.clip_grad_norm_() works
    correctly — FSDP2 performs the allreduce of the global grad norm
    internally. The old manual all_reduce loop was only required for FSDP1.
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


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
            except OSError:
                pass
    return updated


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

    is_fsdp      = is_fsdp_model(model)
    warmup_steps = getattr(cfg, "WARMUP_STEPS", 20)
    scheduler    = build_warmup_scheduler(optimizer, warmup_steps, max_steps)

    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(model, optimizer, scheduler, resume_from, cfg, device)
        _p(f"  Resumed from {resume_from} at step {start_step}", global_rank)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Print header once
    _p(f"  {'epoch':>5}  {'step':>6}  {'loss':>9}  {'lr':>10}  {'gnorm':>8}", global_rank)
    _p(f"  {'-'*5}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*8}", global_rank)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step             = start_step
    best_checkpoints : list = []
    keep_n           = getattr(cfg, "CHECKPOINT_KEEP_LAST_N", 3)
    save_interval    = getattr(cfg, "CHECKPOINT_INTERVAL", 100)
    actual_loss      = float("nan")

    for epoch in range(num_epochs):
        if step >= max_steps:
            break

        micro = 0

        for batch in dataloader:
            if step >= max_steps:
                break

            batch   = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss / grad_accum_steps

            if not torch.isfinite(loss):
                _p(f"  {'':>5}  {step:>6}  {'NaN/Inf — skipping batch':>30}", global_rank)
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            loss.backward()
            micro += 1

            if micro % grad_accum_steps != 0:
                continue

            # ── optimizer step ──────────────────────────────────────────────
            # clip_grads_fsdp works for both FSDP2 and DDP:
            # - FSDP2: clip_grad_norm_ performs an allreduce internally
            # - DDP:   clip_grad_norm_ works on local replica (grads already synced)
            if is_fsdp:
                grad_norm = clip_grads_fsdp(model, 1.0)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                ).item()

            if not math.isfinite(grad_norm):
                _p(
                    f"  {'':>5}  {step:>6}  {'non-finite grad norm — skipping':>30}",
                    global_rank,
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

            _p(
                f"  {epoch:>5}  {step:>6}  {actual_loss:>9.4f}"
                f"  {current_lr:>10.2e}  {grad_norm:>8.4f}",
                global_rank,
            )

            if logger_obj and global_rank == 0:
                logger_obj.log(
                    {
                        "loss":      actual_loss,
                        "lr":        current_lr,
                        "grad_norm": grad_norm,
                        "epoch":     epoch,
                    },
                    step=step,
                )

            step += 1

            if checkpoint_dir and save_interval > 0 and step % save_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
                save_checkpoint(
                    model, optimizer, scheduler, step, cfg, global_rank, ckpt_path
                )
                if global_rank == 0:
                    best_checkpoints = _manage_checkpoints(
                        ckpt_path, actual_loss, best_checkpoints, keep_n
                    )
                _p(f"  → checkpoint saved: {ckpt_path}", global_rank)

        # end-of-epoch checkpoint
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                model, optimizer, scheduler, step, cfg, global_rank, ckpt_path
            )
            if global_rank == 0:
                best_checkpoints = _manage_checkpoints(
                    ckpt_path, actual_loss, best_checkpoints, keep_n
                )
            _p(f"  → epoch {epoch} checkpoint saved: {ckpt_path}", global_rank)

    if logger_obj and global_rank == 0:
        logger_obj.finish()

    _p(f"\n  Training complete. Final step: {step}", global_rank)
