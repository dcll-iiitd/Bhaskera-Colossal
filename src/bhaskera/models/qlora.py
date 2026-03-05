"""
QLoRA builder.

NOTE: QLoRA is ONLY compatible with DDP / single-GPU.
FSDP cannot shard BitsAndBytes integer-dtype tensors.
"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .lora import infer_lora_targets


def build_qlora(cfg, device: torch.device):
    """Build a 4-bit QLoRA model. cfg is a Config dataclass."""
    lora_raw = cfg.LORA
    if isinstance(lora_raw, dict):
        r, alpha, dropout = (
            lora_raw.get("r", 16),
            lora_raw.get("alpha", 32),
            lora_raw.get("dropout", 0.05),
        )
    else:
        r       = getattr(lora_raw, "r",       16)
        alpha   = getattr(lora_raw, "alpha",   32)
        dropout = getattr(lora_raw, "dropout", 0.05)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device},
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=infer_lora_targets(model),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
