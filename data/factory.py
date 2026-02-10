from .ultrachat import build_ultrachat
from .openassistant import build_openassistant

def build_dataset(cfg, tokenizer, rank, world_size):
    if cfg.name == "ultrachat":
        return build_ultrachat(cfg, tokenizer, rank, world_size)
    elif cfg.name == "openassistant":
        return build_openassistant(cfg, tokenizer, rank, world_size)
    else:
        raise ValueError(cfg.name)
