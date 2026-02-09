
from transformers import AutoModelForCausalLM


def build_model(cfg, device):
    if cfg.peft == "qlora":
        from .qlora import build_qlora
        return build_qlora(cfg.model_name, cfg.lora, device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=cfg.dtype,
            attn_implementation=cfg.attn_impl,
        ).to(device)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        if cfg.peft == "lora":
            from .lora import apply_lora
            model = apply_lora(model, cfg.lora)

        return model
