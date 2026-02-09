from .model import build_hf_model
from .lora import apply_lora
from .qlora import build_qlora


def build_model(cfg, device):
    """
    Central model factory.
    """

    if cfg.PEFT == "qlora":
        model = build_qlora(cfg, device)
    else:
        model = build_hf_model(cfg, device)

        if cfg.PEFT == "lora":
            model = apply_lora(model, cfg.LORA)

    return model
