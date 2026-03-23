from peft import LoraConfig, get_peft_model
import inspect
import logging

logger = logging.getLogger(__name__)

_ARCH_TARGETS = {
    "q_proj":          ["q_proj", "k_proj", "v_proj", "o_proj"],
    "query_key_value": ["query_key_value", "dense"],
    "c_attn":          ["c_attn", "c_proj"],
    "qkv_proj":        ["qkv_proj", "o_proj"],
}

# Valid string values for init_lora_weights in this PEFT version.
# "lora" is NOT valid — default standard init is bool True.
_VALID_STRING_INITS = {
    "gaussian", "eva", "olora", "pissa", "corda", "loftq", "orthogonal"
}


def infer_lora_targets(model) -> list:
    names = {n for n, _ in model.named_modules()}
    for probe, targets in _ARCH_TARGETS.items():
        if any(probe in n for n in names):
            logger.info(f"Inferred LoRA targets via '{probe}': {targets}")
            return targets
    logger.warning("Could not infer LoRA targets — targeting all nn.Linear modules.")
    return None


def _resolve_init_weights(init_weights):
    """
    Convert the YAML init_weights value to what this PEFT version accepts.

    Valid values:
        True / None / "" / "lora" / "default"  →  True  (standard Kaiming init)
        "gaussian"                              →  "gaussian"
        "pissa"                                 →  "pissa"
        "pissa_niter_16"                        →  "pissa_niter_16"  (fast SVD)
        "olora" / "eva" / "loftq" / etc        →  passed through
    """
    if init_weights in (None, "", "lora", "default", True):
        return True
    # pissa_niter_N — check prefix
    if isinstance(init_weights, str) and init_weights.startswith("pissa_niter_"):
        return init_weights
    if isinstance(init_weights, str) and init_weights in _VALID_STRING_INITS:
        return init_weights
    logger.warning(
        f"Unknown init_weights='{init_weights}'. "
        f"Valid options: True, {sorted(_VALID_STRING_INITS)}, pissa_niter_N. "
        "Falling back to True."
    )
    return True


def _peft_supports(param_name: str) -> bool:
    try:
        return param_name in inspect.signature(LoraConfig.__init__).parameters
    except Exception:
        return False


def apply_lora(model, lora_cfg: dict):
    """
    Apply LoRA / DoRA / PiSSA adapters to model.

    lora_cfg keys:
        r            – rank (default 16)
        alpha        – alpha (default 32)
        dropout      – dropout (default 0.05)
        use_dora     – DoRA (default False)
        init_weights – True | "gaussian" | "pissa" | "pissa_niter_16" | "olora" etc.
                       NOTE: "lora" is NOT a valid string — use True for standard init.
    """
    r            = lora_cfg.get("r",            16)
    alpha        = lora_cfg.get("alpha",        32)
    dropout      = lora_cfg.get("dropout",      0.05)
    use_dora     = lora_cfg.get("use_dora",     False)
    init_weights = lora_cfg.get("init_weights", True)

    kwargs = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=infer_lora_targets(model),
        task_type="CAUSAL_LM",
        bias="none",
        init_lora_weights=_resolve_init_weights(init_weights),
    )

    if use_dora:
        if _peft_supports("use_dora"):
            kwargs["use_dora"] = True
        else:
            logger.warning("use_dora=True not supported by this PEFT version. Ignoring.")

    peft_config = LoraConfig(**kwargs)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
