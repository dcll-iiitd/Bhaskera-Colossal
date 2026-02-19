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

        # Freeze base weights
        for name, p in model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

        # Re-enable gradient checkpointing AFTER LoRA wrapping.
        # PEFT replaces the model's forward method, which silently breaks
        # gradient checkpointing that was enabled before apply_lora().
        # use_reentrant=False is required for PEFT compatibility.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

        # Unify dtype for FSDP
        target_dtype = torch.bfloat16
        for p in model.parameters():
            p.data = p.data.to(target_dtype)

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"🔥 LoRA active | trainable: {trainable/1e6:.2f}M "
            f"/ total: {total/1e6:.2f}M "
            f"({trainable/total*100:.4f}%)"
        )

    return model
