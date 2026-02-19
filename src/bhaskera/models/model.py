import torch
from transformers import AutoModelForCausalLM


def build_hf_model(cfg, device):
    # Sanitize attn_impl — must never pass the string "None" to from_pretrained.
    # Only pass attn_implementation kwarg when it is a real, non-empty string.
    raw = getattr(cfg, 'ATTN_IMPL', None)
    if raw is None or str(raw).strip().lower() in ("none", "null", ""):
        attn_impl = None
    else:
        attn_impl = str(raw).strip()

    kwargs = dict(torch_dtype=torch.bfloat16)
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        **kwargs,
    ).to(device)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model
