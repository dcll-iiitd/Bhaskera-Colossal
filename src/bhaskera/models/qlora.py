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
        r            = lora_raw.get("r",            16)
        alpha        = lora_raw.get("alpha",        32)
        dropout      = lora_raw.get("dropout",      0.05)
        use_dora     = lora_raw.get("use_dora",     False)
        init_weights = lora_raw.get("init_weights", "lora")
    else:
        r            = getattr(lora_raw, "r",            16)
        alpha        = getattr(lora_raw, "alpha",        32)
        dropout      = getattr(lora_raw, "dropout",      0.05)
        use_dora     = getattr(lora_raw, "use_dora",     False)
        init_weights = getattr(lora_raw, "init_weights", "lora")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # device_map must use the integer index, not a torch.device object.
    # {"": 0} tells BitsAndBytes to place everything on GPU 0 (or whichever
    # local rank is passed in). For multi-GPU DDP each process sets its own
    # CUDA_VISIBLE_DEVICES so GPU 0 is always the right device.
    device_index = device.index if device.type == "cuda" and device.index is not None else 0

    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device_index},
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
        use_dora=use_dora,
        init_lora_weights=init_weights,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
