from .ultrachat import build_ultrachat
from .openassistant import build_openassistant


def build_dataset(cfg, tokenizer, rank: int, world_size: int):
    """Dispatch to the correct dataset builder based on cfg.DATASET_NAME."""
    name = cfg.DATASET_NAME.lower()
    if name == "ultrachat":
        return build_ultrachat(cfg, tokenizer, rank, world_size)
    elif name in ("openassistant", "oasst1"):
        return build_openassistant(cfg, tokenizer, rank, world_size)
    elif name == "audio":
        # Audio jobs never reach here — they are handled entirely by
        # bhaskera.audio.worker before the text DataLoader is constructed.
        raise RuntimeError(
            "build_dataset() should not be called for audio jobs. "
            "Ensure worker_core.py routes cfg.DATASET_NAME='audio' to "
            "bhaskera.audio.worker.run_audio_worker()."
        )
    else:
        raise ValueError(
            f"Unknown dataset: '{cfg.DATASET_NAME}'. "
            "Supported: ultrachat, openassistant, audio"
        )
