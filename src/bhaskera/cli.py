import argparse
import yaml
import torch
import ray

from types import SimpleNamespace
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from bhaskera.data.registry import build_dataset
from bhaskera.models.registry import build_model
from bhaskera.trainer.train_loop import train

# ==========================================================
# Utility: Load YAML config
# ==========================================================
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)

            # ---- Type normalization ----
            if key in {"LR"}:
                value = float(value)
            if key in {"BATCH_SIZE", "GRAD_ACCUM", "MAX_STEPS", "SEQ_LEN"}:
                value = int(value)

            setattr(self, key, value)


def load_config(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)

# def load_config(path: str):
#     with open(path, "r") as f:
#         data = yaml.safe_load(f)
#     return SimpleNamespace(**data)


# ==========================================================
# Ray Worker Function
# ==========================================================
def train_func(config):
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
    # Model
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


# ==========================================================
# CLI Entry Point
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bhaskera Training Framework"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,  # 🔥 THIS ENFORCES CONFIG
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Ray workers (GPUs)",
    )

    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Initialize Ray
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=cfg,   # 🔥 Pass config to workers
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
        ),
    )

    trainer.fit()


if __name__ == "__main__":
    main()
