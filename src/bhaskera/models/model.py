import torch
from transformers import AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def build_hf_model(cfg, device: torch.device):
    """Load a HuggingFace CausalLM, with optional flash-attention."""
    raw = cfg.ATTN_IMPL
    attn_impl = None if (raw is None or str(raw).strip().lower() in ("none", "null", "")) else str(raw)

    dtype = _DTYPE_MAP.get(cfg.DTYPE, torch.bfloat16)
    kwargs: dict = {"torch_dtype": dtype}
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl

    logger.info(f"Loading {cfg.MODEL_NAME} dtype={cfg.DTYPE} attn={attn_impl} device={device}")

    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, **kwargs)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if device.type != "cpu":
        model = model.to(device)

    return model
