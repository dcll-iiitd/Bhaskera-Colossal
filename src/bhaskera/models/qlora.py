import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .lora import infer_lora_targets


def build_qlora(cfg, device):
    """
    Build a QLoRA model from config.
    Contract: (cfg, device) -> nn.Module
    """

    # 1. BitsAndBytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 2. Load quantized base model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device},
    )

    # 3. QLoRA-required prep
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # 4. 🔑 Infer target modules from architecture
    target_modules = infer_lora_targets(model)

    # 5. LoRA config (NO target_modules in config.py)
    peft_cfg = LoraConfig(
        r=cfg.LORA.r,
        lora_alpha=cfg.LORA.alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # 6. Apply LoRA
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    return model
