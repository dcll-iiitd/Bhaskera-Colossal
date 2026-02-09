import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def train(
    *,
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    grad_accum_steps: int,
    max_steps: int,
    local_rank: int,
    global_rank: int,
    logger=None,
):
    """
    Pure PyTorch training loop.

    Args:
        model: torch.nn.Module (NOT wrapped)
        dataloader: PyTorch DataLoader
        optimizer: torch.optim.Optimizer
        scaler: torch.cuda.amp.GradScaler
        device: torch.device
        grad_accum_steps: gradient accumulation steps
        max_steps: optimizer steps
        local_rank: GPU index on this node
        global_rank: global rank across all workers
        logger: optional experiment logger (rank-0 only)
    """

    # ------------------------
    # DDP WRAP
    # ------------------------
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0
    micro_step = 0

    # ------------------------
    # TRAINING LOOP
    # ------------------------
    for batch in dataloader:
        batch = {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
        }

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

        scaler.scale(loss).backward()
        micro_step += 1

        # ------------------------
        # OPTIMIZER STEP
        # ------------------------
        if micro_step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # ------------------------
            # LOGGING (rank-0 only)
            # ------------------------
            if logger is not None and global_rank == 0:
                logger.log(
                    {
                        "loss": loss.item() * grad_accum_steps,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )

            if global_rank == 0:
                print(
                    f"[step {step}] "
                    f"loss={loss.item() * grad_accum_steps:.4f}"
                )

            step += 1

            if step >= max_steps:
                break

    # ------------------------
    # FINALIZE LOGGER
    # ------------------------
    if logger is not None and global_rank == 0:
        logger.finish()
