"""
bhaskera.audio.asr_registry
-----------------------------
Central registry that maps ASR model *families* to their pipeline builders.

Supported families
------------------
  whisper      openai/whisper-*           Seq2Seq encoder-decoder
  wav2vec2     facebook/wav2vec2-*        CTC encoder-only
  mms          facebook/mms-*             CTC / Seq2Seq (MMS-1B-FL102 etc.)
  seamless     facebook/seamless-*        Multitask Seq2Seq (SeamlessM4T)
  nemo         nvidia/parakeet-*          CTC/RNNT via NeMo (optional dep)

The registry is intentionally open: third parties can call
  register_asr_model("myfamily", MyASRPipeline)
before calling run_audio_training().

Pipeline contract
-----------------
Every builder must be a class or factory that, when instantiated with (cfg),
exposes:
  .processor          — object that satisfies feature-extraction + tokenisation
  .model              — nn.Module ready for training
  .training_args      — Seq2SeqTrainingArguments or TrainingArguments
  .data_collator      — callable(List[dict]) -> dict of tensors
  .compute_metrics    — callable(EvalPrediction) -> dict[str, float]
  .post_init(trainer) — optional hook called after Trainer is built
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Type

logger = logging.getLogger(__name__)

# Family → builder callable (instantiated with cfg)
_REGISTRY: Dict[str, Callable] = {}


def register_asr_model(family: str, builder: Callable) -> None:
    """Register an ASR pipeline builder under *family* (lower-case key)."""
    key = family.lower()
    if key in _REGISTRY:
        logger.warning("ASR registry: overwriting existing entry '%s'", key)
    _REGISTRY[key] = builder
    logger.debug("ASR registry: registered '%s' → %s", key, builder)


def get_asr_pipeline(cfg) -> object:
    """
    Resolve cfg.AUDIO_MODEL_FAMILY → instantiate and return the pipeline.

    Resolution order
    ----------------
    1. cfg.AUDIO_MODEL_FAMILY  (explicit override, e.g. "wav2vec2")
    2. Auto-detect from cfg.MODEL_NAME prefix
    """
    # Lazy-import so the registry module itself has no heavy deps at import time
    _ensure_defaults_registered()

    family = _resolve_family(cfg)
    logger.info("ASR registry: resolved family='%s' for model='%s'", family, cfg.MODEL_NAME)

    if family not in _REGISTRY:
        raise ValueError(
            f"Unknown ASR model family '{family}'. "
            f"Registered families: {sorted(_REGISTRY.keys())}. "
            "Use cfg.AUDIO_MODEL_FAMILY to set the family explicitly, "
            "or register a custom pipeline with register_asr_model()."
        )

    builder = _REGISTRY[family]
    return builder(cfg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_family(cfg) -> str:
    """Return the lower-case model family string."""
    # 1. Explicit override
    explicit = getattr(cfg, "AUDIO_MODEL_FAMILY", None)
    if explicit:
        return explicit.lower()

    # 2. Auto-detect from model name
    name = cfg.MODEL_NAME.lower()
    if "whisper" in name:
        return "whisper"
    if "wav2vec2" in name or "wav2vec-2" in name:
        return "wav2vec2"
    if "mms" in name:
        return "mms"
    if "seamless" in name:
        return "seamless"
    if "parakeet" in name or "stt_en" in name:
        return "nemo"

    raise ValueError(
        f"Cannot auto-detect ASR family from model name '{cfg.MODEL_NAME}'. "
        "Set cfg.AUDIO_MODEL_FAMILY explicitly in your config."
    )


_defaults_registered = False


def _ensure_defaults_registered() -> None:
    global _defaults_registered
    if _defaults_registered:
        return
    # Import lazily so heavy HF deps are not loaded unless actually needed
    from bhaskera.audio.pipelines.whisper_pipeline import WhisperASRPipeline
    from bhaskera.audio.pipelines.wav2vec2_pipeline import Wav2Vec2ASRPipeline
    from bhaskera.audio.pipelines.mms_pipeline import MMSASRPipeline

    register_asr_model("whisper",  WhisperASRPipeline)
    register_asr_model("wav2vec2", Wav2Vec2ASRPipeline)
    register_asr_model("mms",      MMSASRPipeline)

    _defaults_registered = True
