"""
Model registry.

FSDP compatibility notes:
  - qlora is INCOMPATIBLE with FSDP (BitsAndBytes stores uint8/int8 tensors).
  - lora IS compatible with FSDP (weights stay float/bfloat16).
  - For FSDP: call build_model(cfg, device=cpu). FSDP moves shards to GPU.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def build_model(cfg, device: torch.device):
    """
    Build model according to cfg.PEFT.

    Args:
        cfg:    Config dataclass.
        device: torch.device — pass cpu for FSDP, cuda for DDP/single-GPU.
    """
    from .model import build_hf_model
    from .lora import apply_lora
    from .qlora import build_qlora

    peft = (cfg.PEFT or "").lower()

    # ── QLoRA path ────────────────────────────────────────────────────────────
    if peft == "qlora":
        if cfg.distributed.strategy.lower() == "fsdp":
            raise ValueError(
                "QLoRA is incompatible with FSDP — BitsAndBytes 4-bit weights "
                "cannot be sharded. Use peft.method='lora' or strategy='ddp'."
            )
        return build_qlora(cfg, device)

    # ── Base model ────────────────────────────────────────────────────────────
    model = build_hf_model(cfg, device)

    # ── LoRA path ─────────────────────────────────────────────────────────────
    if peft == "lora":
        model = apply_lora(model, cfg.LORA)

        # Freeze base weights — only LoRA adapters train
        for name, p in model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

        # Re-enable gradient checkpointing AFTER LoRA wrapping.
        # PEFT replaces forward(), which silently disables grad checkpointing.
        # use_reentrant=False is required for PEFT + FSDP.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

        # Cast ALL parameters to a single dtype for FSDP homogeneity.
        # FSDP cannot shard a module that has mixed dtypes.
        target_dtype = _DTYPE_MAP.get(cfg.DTYPE, torch.bfloat16)
        for p in model.parameters():
            p.data = p.data.to(target_dtype)

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"LoRA active | trainable: {trainable/1e6:.2f}M / "
            f"total: {total/1e6:.2f}M ({trainable/total*100:.4f}%)"
        )

    return model
