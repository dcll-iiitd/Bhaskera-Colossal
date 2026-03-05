from .ultrachat import build_ultrachat
from .openassistant import build_openassistant


def build_dataset(cfg, tokenizer, rank: int, world_size: int):
    """Dispatch to the correct dataset builder based on cfg.DATASET_NAME."""
    name = cfg.DATASET_NAME.lower()
    if name == "ultrachat":
        return build_ultrachat(cfg, tokenizer, rank, world_size)
    elif name in ("openassistant", "oasst1"):
        return build_openassistant(cfg, tokenizer, rank, world_size)
    else:
        raise ValueError(
            f"Unknown dataset: '{cfg.DATASET_NAME}'. "
            "Supported: ultrachat, openassistant"
        )
