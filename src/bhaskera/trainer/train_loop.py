"""
Stable training loop for FSDP / DDP.

Key fix: FSDP checkpoint saving requires ALL ranks to participate in the
all-gather collective. The previous code called save_checkpoint only inside
`if global_rank == 0`, causing rank 0 to hang waiting for rank 1 to join
the FSDP barrier - a classic FSDP deadlock.

Correct pattern:
  - ALL ranks call save_checkpoint (so FSDP collective can complete)
  - Only rank 0 does the actual torch.save() to disk (inside save_fsdp_checkpoint)
"""

import os
import torch
import torch.distributed as dist
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def is_fsdp_model(model):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except Exception:
        return False


def clip_grads_fsdp(model, max_norm: float):
    """Proper global grad-norm clip for FULL_SHARD FSDP."""
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


def build_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then constant LR. Prevents NaN from LR spike in bfloat16+FSDP."""
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def manage_best_checkpoints(current_path, current_loss, best_checkpoints, keep_n_best=2):
    best_checkpoints.append((current_loss, current_path))
    best_checkpoints.sort(key=lambda x: x[0])
    to_keep = set(path for _, path in best_checkpoints[:keep_n_best])
    to_keep.add(current_path)
    updated = []
    for loss, path in best_checkpoints:
        if path in to_keep:
            updated.append((loss, path))
        else:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"[Checkpoint] Removed: {path}")
                except OSError as e:
                    logger.warning(f"[Checkpoint] Could not remove {path}: {e}")
    return updated


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

    warmup_steps = getattr(cfg, "WARMUP_STEPS", None)
    if warmup_steps is None:
        try:
            warmup_steps = int(cfg.training.warmup_steps)
        except (AttributeError, TypeError):
            warmup_steps = 0

    scheduler = build_warmup_scheduler(optimizer, warmup_steps, max_steps)

    if global_rank == 0:
        logger.info(f"Starting training with {strategy}")
        logger.info(f"Epochs: {num_epochs} | Max steps: {max_steps}")
        logger.info(f"Grad accum: {grad_accum_steps} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Batch/GPU: {cfg.BATCH_SIZE}")
        if checkpoint_dir:
            logger.info(f"Checkpoints -> {checkpoint_dir}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0
    best_checkpoints = []

    for epoch in range(num_epochs):

        micro = 0
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in dataloader:

            if step >= max_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

            if not torch.isfinite(loss):
                if global_rank == 0:
                    logger.warning(
                        f"[{strategy}][epoch {epoch}][step {step}] "
                        "NaN/Inf loss -- discarding batch and resetting grad window."
                    )
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            loss.backward()
            micro += 1

            if micro % grad_accum_steps == 0:

                if is_fsdp:
                    grad_norm = clip_grads_fsdp(model, 1.0)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    ).item()

                if not math.isfinite(grad_norm):
                    if global_rank == 0:
                        logger.warning(
                            f"[{strategy}][epoch {epoch}][step {step}] "
                            f"Non-finite grad norm ({grad_norm:.4f}) -- skipping step."
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
                epoch_loss_sum += actual_loss
                epoch_steps    += 1

                if global_rank == 0:
                    logger.info(
                        f"[{strategy}][epoch {epoch}][step {step}] "
                        f"loss={actual_loss:.4f} "
                        f"lr={current_lr:.2e} "
                        f"grad_norm={grad_norm:.4f}"
                    )
                    if logger_obj:
                        logger_obj.log(
                            {"loss": actual_loss, "lr": current_lr,
                             "grad_norm": grad_norm, "step": step, "epoch": epoch},
                            step=step,
                        )

                step += 1

        # ------------------------------------------------------------------
        # End-of-epoch checkpoint
        #
        # CRITICAL FOR FSDP: save_checkpoint MUST be called by ALL ranks.
        #
        # Why: FSDP FULL_STATE_DICT is a collective op. Every GPU must call
        # state_dict_type() together so FSDP can all-gather shards from all
        # ranks into one full parameter tensor. If rank 0 calls it alone,
        # it blocks waiting for rank 1 to join -- causing an infinite hang.
        #
        # How we handle it:
        #   - ALL ranks enter the checkpoint block and call save_checkpoint()
        #   - Inside save_fsdp_checkpoint(), all ranks join the collective
        #   - Only rank 0 calls torch.save() once the state dict is gathered
        #   - Bookkeeping (manage_best_checkpoints) stays rank-0 only since
        #     it only touches the local filesystem
        # ------------------------------------------------------------------
        if epoch_steps == 0:
            logger.warning(f"[Epoch {epoch}] No steps completed.")
            if step >= max_steps:
                break
            continue

        avg_epoch_loss = epoch_loss_sum / epoch_steps

        if global_rank == 0:
            logger.info(f"[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f}")
            if logger_obj:
                logger_obj.log(
                    {"epoch_avg_loss": avg_epoch_loss, "epoch": epoch},
                    step=step,
                )

        if checkpoint_dir:
            # All ranks must execute this block together.
            # makedirs is safe to call on all ranks (exist_ok=True).
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"epoch_{epoch:04d}_loss_{avg_epoch_loss:.4f}.pt"
            )

            # ALL ranks call save_checkpoint -- FSDP collective happens here.
            # torch.save() inside only runs on rank 0.
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                cfg=cfg,
                global_rank=global_rank,
                checkpoint_path=ckpt_path,
            )

            # Bookkeeping: rank 0 only (filesystem operations)
            if global_rank == 0:
                logger.info(f"[Epoch {epoch}] Saved -> {ckpt_path}")
                best_checkpoints = manage_best_checkpoints(
                    current_path=ckpt_path,
                    current_loss=avg_epoch_loss,
                    best_checkpoints=best_checkpoints,
                    keep_n_best=2,
                )
                kept = [p for _, p in best_checkpoints]
                logger.info(f"[Epoch {epoch}] Kept: {kept}")

        if step >= max_steps:
            break

    if logger_obj and global_rank == 0:
        logger_obj.finish()

    if global_rank == 0:
        logger.info("Training finished.")
        if best_checkpoints:
            logger.info("Final best checkpoints:")
            for loss, path in best_checkpoints:
                logger.info(f"  loss={loss:.4f}  ->  {path}")