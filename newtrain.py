import torch
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import config
from data.registry import build_dataset
from models.registry import build_model
from trainer.train_loop import train


def train_func(_):
    ctx = ray.train.get_context()

    local_rank = ctx.get_local_rank()
    global_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # -----------------------
    # Tokenizer
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -----------------------
    # Dataset
    # -----------------------
    dataset = build_dataset(
        config,
        tokenizer,
        global_rank,
        world_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=True,
    )

    # -----------------------
    # Model (PEFT handled inside registry)
    # -----------------------
    model = build_model(config, device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )

    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
    )
    scaler = torch.amp.GradScaler("cuda")

    # -----------------------
    # Train
    # -----------------------
    train(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        grad_accum_steps=config.GRAD_ACCUM,
        max_steps=config.MAX_STEPS,
        local_rank=local_rank,
        global_rank=global_rank,
    )


def main():
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
        ),
    )

    trainer.fit()


if __name__ == "__main__":
    main()
