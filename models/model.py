import torch
from transformers import AutoModelForCausalLM


def build_hf_model(cfg, device):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        dtype=torch.float16,
        attn_implementation=cfg.ATTN_IMPL,
    ).to(device)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model
