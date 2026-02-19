"""
Stable training loop for FSDP / DDP.
Fixes NaN issues with bf16 + FSDP.
Supports epoch-level best-checkpoint keeping (top-2 best + current).
"""

import os
import torch
import torch.distributed as dist
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# helpers
# -----------------------------------------------------------
def is_fsdp_model(model):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except Exception:
        return False


def clip_grads_fsdp(model, max_norm: float):
    """
    Proper global grad norm for FULL_SHARD FSDP.
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


# -----------------------------------------------------------
# Checkpoint management
# -----------------------------------------------------------
def manage_best_checkpoints(
    current_path: str,
    current_loss: float,
    best_checkpoints: list,
    keep_n_best: int = 2,
) -> list:
    """
    Keeps only the top-N best checkpoints + the most recent (current) one.
    Deletes checkpoint files that are no longer needed.

    Args:
        current_path:      Path of the checkpoint just saved.
        current_loss:      Epoch loss for this checkpoint (lower = better).
        best_checkpoints:  Mutable list of (loss, path) maintained across epochs.
        keep_n_best:       How many best checkpoints to keep (default 2).

    Returns:
        Updated best_checkpoints list.
    """
    # Add current to the pool
    best_checkpoints.append((current_loss, current_path))

    # Sort ascending by loss (lower = better)
    best_checkpoints.sort(key=lambda x: x[0])

    # The set of paths we want to keep: top-N best + the latest (current)
    to_keep = set(path for _, path in best_checkpoints[:keep_n_best])
    to_keep.add(current_path)  # always keep the most recent regardless of rank

    # Delete files not in the keep set and rebuild the list
    updated = []
    for loss, path in best_checkpoints:
        if path in to_keep:
            updated.append((loss, path))
        else:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"[Checkpoint] Removed old checkpoint: {path}")
                except OSError as e:
                    logger.warning(f"[Checkpoint] Could not remove {path}: {e}")

    return updated


# -----------------------------------------------------------
# TRAIN
# -----------------------------------------------------------
def train(
    *,
    model,
    dataloader,
    optimizer,
    scaler=None,
    device,
    grad_accum_steps,
    max_steps,
    local_rank,
    global_rank,
    cfg,
    logger_obj=None,
    num_epochs: int = 1,
    checkpoint_dir: Optional[str] = None,
):
    from bhaskera.distributed.wrapper import save_checkpoint

    is_fsdp = is_fsdp_model(model)
    strategy = "FSDP" if is_fsdp else "DDP"

    if global_rank == 0:
        logger.info(f"Starting training with {strategy}")
        logger.info(f"Epochs: {num_epochs} | Max steps: {max_steps}")
        logger.info(f"Grad accum: {grad_accum_steps}")
        logger.info(f"Batch/GPU: {cfg.BATCH_SIZE}")
        if checkpoint_dir:
            logger.info(f"Best-epoch checkpoints → {checkpoint_dir}  (keep top-2 + current)")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0
    best_checkpoints = []   # list of (loss, path), maintained across epochs

    # -------------------------------------------------------
    for epoch in range(num_epochs):

        micro = 0
        epoch_loss_sum = 0.0
        epoch_steps = 0

        # ---------------------------------------------------
        for batch in dataloader:

            if step >= max_steps:
                break

            # move to GPU
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # ---------------------------------------------------
            # 🚨 NO AUTOCAST HERE (FSDP already handles bf16)
            # ---------------------------------------------------
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

            # ---------------------------------------------------
            # NaN guard
            # ---------------------------------------------------
            if not torch.isfinite(loss):
                if global_rank == 0:
                    print("🚨 NaN loss detected — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            # backward
            loss.backward()
            micro += 1

            # ---------------------------------------------------
            # optimizer step after grad accum
            # ---------------------------------------------------
            if micro % grad_accum_steps == 0:

                # grad clip
                if is_fsdp:
                    grad_norm = clip_grads_fsdp(model, 1.0)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    ).item()

                # grad explosion guard
                if not math.isfinite(grad_norm):
                    if global_rank == 0:
                        print(f"🚨 Non-finite grad norm {grad_norm}, skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    micro = 0
                    continue

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                actual_loss = loss.item() * grad_accum_steps
                epoch_loss_sum += actual_loss
                epoch_steps += 1

                # logging
                if global_rank == 0:
                    print(
                        f"[{strategy}][epoch {epoch}][step {step}] "
                        f"loss={actual_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"grad_norm={grad_norm:.4f}"
                    )

                    if logger_obj:
                        logger_obj.log(
                            {
                                "loss": actual_loss,
                                "lr": optimizer.param_groups[0]["lr"],
                                "grad_norm": grad_norm,
                                "step": step,
                                "epoch": epoch,
                            },
                            step=step,
                        )

                step += 1

        # -------------------------------------------------------
        # End of epoch — checkpoint if enabled (rank 0 only)
        # -------------------------------------------------------
        if epoch_steps == 0:
            logger.warning(f"[Epoch {epoch}] No steps completed, skipping checkpoint.")
            if step >= max_steps:
                break
            continue

        avg_epoch_loss = epoch_loss_sum / epoch_steps

        if global_rank == 0:
            print(f"[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f}")

            if logger_obj:
                logger_obj.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch}, step=step)

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"epoch_{epoch:04d}_loss_{avg_epoch_loss:.4f}.pt"
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    cfg=cfg,
                    global_rank=global_rank,
                    checkpoint_path=ckpt_path,
                )
                logger.info(f"[Epoch {epoch}] Saved checkpoint → {ckpt_path}")

                best_checkpoints = manage_best_checkpoints(
                    current_path=ckpt_path,
                    current_loss=avg_epoch_loss,
                    best_checkpoints=best_checkpoints,
                    keep_n_best=2,
                )

                kept = [p for _, p in best_checkpoints]
                logger.info(f"[Epoch {epoch}] Checkpoints kept: {kept}")

        if step >= max_steps:
            break

    # -------------------------------------------------------
    if logger_obj and global_rank == 0:
        logger_obj.finish()

    if global_rank == 0:
        print("Training finished.")
        if best_checkpoints:
            print("Final best checkpoints:")
            for loss, path in best_checkpoints:
                print(f"  loss={loss:.4f}  →  {path}")
