import argparse
import torch
import ray

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import bhaskera.config as config
from bhaskera.data.registry import build_dataset
from bhaskera.models.registry import build_model
from bhaskera.trainer.train_loop import train


# =====================================================
# Worker function (UNCHANGED LOGIC)
# =====================================================
def train_func(_):
    ctx = ray.train.get_context()

    local_rank = ctx.get_local_rank()
    global_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    model = build_model(config, device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
    )
    scaler = torch.amp.GradScaler("cuda")

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


# =====================================================
# Ray launcher (NEW, clean separation)
# =====================================================
def launch_ray(num_workers: int):
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
        ),
    )

    trainer.fit()


# =====================================================
# CLI entrypoint
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bhaskera training launcher"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Ray workers (GPUs)",
    )

    args = parser.parse_args()
    launch_ray(args.num_workers)


if __name__ == "__main__":
    main()
