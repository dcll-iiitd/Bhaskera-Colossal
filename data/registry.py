from .ultrachat import build_ultrachat
from .redpajama import build_redpajama


def build_dataset(cfg, tokenizer, rank, world_size):
    if cfg.DATASET_NAME == "ultrachat":
        return build_ultrachat(cfg, tokenizer, rank, world_size)
    elif cfg.DATASET_NAME == "redpajama":
        return build_redpajama(cfg, tokenizer, rank, world_size)
    else:
        raise ValueError(f"Unknown dataset: {cfg.DATASET_NAME}")

