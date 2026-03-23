# factory.py — thin wrapper, does NOT import from registry.py
from transformers import AutoModelForCausalLM


def build_model(cfg, device):
    if (cfg.PEFT or "").lower() == "qlora":
        from .qlora import build_qlora
        return build_qlora(cfg, device)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        torch_dtype=cfg.DTYPE,
        attn_implementation=cfg.ATTN_IMPL,
    ).to(device)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if (cfg.PEFT or "").lower() == "lora":
        from .lora import apply_lora
        model = apply_lora(model, cfg.LORA)

    return model
