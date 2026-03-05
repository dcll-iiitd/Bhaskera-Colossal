from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)

# Map from architecture detection to target module names.
# Extend this dict when adding new architectures.
_ARCH_TARGETS = {
    "q_proj":          ["q_proj", "k_proj", "v_proj", "o_proj"],     # LLaMA/Mistral/Gemma/Phi
    "query_key_value": ["query_key_value", "dense"],                   # Falcon
    "c_attn":          ["c_attn", "c_proj"],                           # GPT-2/J
    "qkv_proj":        ["qkv_proj", "o_proj"],                         # Phi-3
}


def infer_lora_targets(model) -> list:
    names = {n for n, _ in model.named_modules()}
    for probe, targets in _ARCH_TARGETS.items():
        if any(probe in n for n in names):
            logger.info(f"Inferred LoRA targets via '{probe}': {targets}")
            return targets
    # Last resort: target all nn.Linear layers (safe but less efficient)
    logger.warning(
        "Could not infer LoRA targets from architecture — "
        "targeting all nn.Linear modules. Add explicit support for this model."
    )
    return None   # caller handles None -> task_type default


def apply_lora(model, lora_cfg: dict):
    """
    Apply LoRA adapters to model.

    Args:
        lora_cfg: dict with keys r, alpha, dropout
    """
    r       = lora_cfg.get("r",       16)
    alpha   = lora_cfg.get("alpha",   32)
    dropout = lora_cfg.get("dropout", 0.05)

    target_modules = infer_lora_targets(model)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,   # None → PEFT picks defaults
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
