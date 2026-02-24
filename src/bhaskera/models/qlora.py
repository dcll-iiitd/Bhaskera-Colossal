"""
QLoRA builder.

NOTE: QLoRA (4-bit quantization via BitsAndBytes) is ONLY compatible with
DDP or single-GPU training. It cannot be used with FSDP because BitsAndBytes
stores weights as integer dtypes (uint8/int8), which FSDP's parameter
flattening engine cannot shard.
"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .lora import infer_lora_targets


def build_qlora(cfg, device):
    """
    Build a QLoRA model from config.

    Args:
        cfg:    Config object. cfg.LORA may be a dict OR an object with
                .r / .alpha / .dropout attributes — both are handled.
        device: Device to load the model on.

    Returns:
        PEFT-wrapped model with 4-bit quantized base weights.
    """

    # ------------------------------------------------------------------
    # Resolve LoRA hyperparams — handle both dict and attribute access
    # ------------------------------------------------------------------
    lora_raw = cfg.LORA
    if isinstance(lora_raw, dict):
        lora_r       = lora_raw.get("r",       16)
        lora_alpha   = lora_raw.get("alpha",   32)
        lora_dropout = lora_raw.get("dropout", 0.05)
    else:
        # Legacy config.py uses dict(r=8, alpha=32) which is still a dict,
        # but handle object-with-attributes just in case.
        lora_r       = getattr(lora_raw, "r",       16)
        lora_alpha   = getattr(lora_raw, "alpha",   32)
        lora_dropout = getattr(lora_raw, "dropout", 0.05)

    # ------------------------------------------------------------------
    # BitsAndBytes 4-bit config
    # ------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,   # bfloat16 is more stable than float16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ------------------------------------------------------------------
    # Load quantized base model
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device},
    )

    # ------------------------------------------------------------------
    # QLoRA prep
    # ------------------------------------------------------------------
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # Infer LoRA target modules from architecture
    # ------------------------------------------------------------------
    target_modules = infer_lora_targets(model)

    # ------------------------------------------------------------------
    # Build and apply LoRA config
    # ------------------------------------------------------------------
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    return model