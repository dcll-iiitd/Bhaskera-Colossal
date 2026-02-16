"""
Stable training loop for FSDP / DDP.
Fixes NaN issues with bf16 + FSDP.
"""

import torch
import torch.distributed as dist
import logging
import math

logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# helpers
# -----------------------------------------------------------
def is_fsdp_model(model):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except:
        return False


def clip_grads_fsdp(model, max_norm: float):
    """
    Proper global grad norm for FULL_SHARD FSDP
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
):

    is_fsdp = is_fsdp_model(model)
    strategy = "FSDP" if is_fsdp else "DDP"

    if global_rank == 0:
        logger.info(f"Starting training with {strategy}")
        logger.info(f"Max steps: {max_steps}")
        logger.info(f"Grad accum: {grad_accum_steps}")
        logger.info(f"Batch/GPU: {cfg.BATCH_SIZE}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0
    micro = 0

    # -------------------------------------------------------
    for batch in dataloader:

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
        # step after grad accum
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

            # logging
            if global_rank == 0:
                actual_loss = loss.item() * grad_accum_steps
                print(
                    f"[{strategy}][step {step}] "
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
                        },
                        step=step,
                    )

            step += 1

            if step >= max_steps:
                break

    if logger_obj and global_rank == 0:
        logger_obj.finish()

    if global_rank == 0:
        print("Training finished.")
