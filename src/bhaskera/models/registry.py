"""
Model registry — dispatches to the correct builder based on cfg.PEFT.

FSDP compatibility notes:
  - "qlora" is INCOMPATIBLE with FSDP. BitsAndBytes stores weights as uint8/int8,
    and FSDP's flattening engine cannot shard integer-dtype tensors. Use qlora
    only with DDP or single-GPU.
  - "lora" IS compatible with FSDP. Weights stay in bfloat16/float32 (floats),
    which FSDP can shard freely.
  - When using FSDP, always call build_model() with model_device=cpu.
    FSDP will move each shard to GPU during its own init.
"""
from .model import build_hf_model
from .lora import apply_lora
from .qlora import build_qlora
import torch


def _get_lora_cfg(cfg):
    """
    Safely extract LoRA hyperparameters from config regardless of whether
    cfg.LORA is a dict (from the new YAML dataclass loader) or an object
    with .r / .alpha attributes (legacy config.py).
    
    Returns a simple namespace with .r, .alpha, .dropout attributes.
    """
    lora_raw = cfg.LORA

    if isinstance(lora_raw, dict):
        # New YAML path: cfg.LORA = {"r": 16, "alpha": 32, "dropout": 0.05}
        class _LoraArgs:
            pass
        args = _LoraArgs()
        args.r       = lora_raw.get("r",       16)
        args.alpha   = lora_raw.get("alpha",   32)
        args.dropout = lora_raw.get("dropout", 0.05)
        return args
    else:
        # Legacy path: cfg.LORA is already an object with attributes
        return lora_raw


def build_model(cfg, device):
    """
    Build a model according to cfg.PEFT.

    Args:
        cfg:    Config object (new dataclass or legacy module).
        device: torch.device — use cpu for FSDP, cuda for DDP/single-GPU.

    Returns:
        nn.Module ready to be wrapped with DDP or FSDP.
    """

    # ------------------------------------------------------------------
    # QLoRA path — NOT compatible with FSDP
    # ------------------------------------------------------------------
    if cfg.PEFT == "qlora":
        # Warn loudly if someone accidentally uses qlora with fsdp config
        dist_strategy = getattr(
            getattr(cfg, "distributed", None), "strategy", "ddp"
        )
        if dist_strategy.lower() == "fsdp":
            raise ValueError(
                "QLoRA is incompatible with FSDP. "
                "BitsAndBytes stores weights as 4-bit integers (uint8/int8), "
                "and FSDP cannot flatten/shard integer-dtype tensors. "
                "Fix: set peft.method = 'lora' in your YAML config, "
                "or switch to strategy = 'ddp'."
            )
        model = build_qlora(cfg, device)
        return model

    # ------------------------------------------------------------------
    # Base model (no PEFT, or LoRA applied below)
    # ------------------------------------------------------------------
    model = build_hf_model(cfg, device)

    # ------------------------------------------------------------------
    # LoRA path — compatible with FSDP
    # ------------------------------------------------------------------
    if cfg.PEFT == "lora":
        lora_cfg = _get_lora_cfg(cfg)
        model = apply_lora(model, lora_cfg)

        # Freeze base weights — only LoRA adapters train
        for name, p in model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

        # Re-enable gradient checkpointing AFTER LoRA wrapping.
        # PEFT replaces the model's forward() method, which silently disables
        # gradient checkpointing that was set before apply_lora().
        # use_reentrant=False is required for PEFT + FSDP compatibility.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

        # Cast ALL parameters to a single dtype for FSDP.
        # FSDP requires a homogeneous dtype across the module it wraps.
        # Mixed dtypes (e.g. some layers float32, some bfloat16) will cause
        # FSDP to raise or silently produce wrong results.
        # We match the dtype from config; fall back to bfloat16 if not set.
        dtype_str = getattr(cfg, "DTYPE", "bfloat16")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        target_dtype = dtype_map.get(dtype_str, torch.bfloat16)
        for p in model.parameters():
            p.data = p.data.to(target_dtype)

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"LoRA active | trainable: {trainable/1e6:.2f}M "
            f"/ total: {total/1e6:.2f}M "
            f"({trainable/total*100:.4f}%)"
        )

    return model