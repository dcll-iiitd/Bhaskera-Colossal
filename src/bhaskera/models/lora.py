from peft import LoraConfig, get_peft_model


def infer_lora_targets(model):
    names = {n for n, _ in model.named_modules()}

    # LLaMA / Mistral
    if any("q_proj" in n for n in names):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Falcon
    if any("query_key_value" in n for n in names):
        return ["query_key_value", "dense"]

    raise ValueError(
        "Could not infer LoRA target modules for this model. "
        "Please add support explicitly."
    )


def apply_lora(model, lora_cfg):
    target_modules = infer_lora_targets(model)

    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model
