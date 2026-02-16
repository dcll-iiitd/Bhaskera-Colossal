from .model import build_hf_model
from .lora import apply_lora
from .qlora import build_qlora
import torch


def build_model(cfg, device):

    if cfg.PEFT == "qlora":
        model = build_qlora(cfg, device)
        return model

    model = build_hf_model(cfg, device)

    if cfg.PEFT == "lora":
        model = apply_lora(model, cfg.LORA)

        # freeze base
        for name, p in model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

        # 🔥 CRITICAL: unify dtype for FSDP
        target_dtype = torch.bfloat16
        for p in model.parameters():
            p.data = p.data.to(target_dtype)

        # sanity print
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(
            f"🔥 LoRA active | trainable: {trainable/1e6:.2f}M "
            f"/ total: {total/1e6:.2f}M "
            f"({trainable/total*100:.4f}%)"
        )

    return model
