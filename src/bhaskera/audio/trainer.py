"""
bhaskera.audio.trainer
-----------------------
Generic ASR fine-tuning trainer.

Delegates all model-specific logic to a pipeline object resolved from
bhaskera.audio.asr_registry — supports Whisper, Wav2Vec2, MMS, and any
custom pipeline registered via register_asr_model().

Called by bhaskera.audio.worker — never invoked directly by the framework.

Pipeline resolution order
-------------------------
1.  cfg.AUDIO_MODEL_FAMILY  (explicit, e.g. "wav2vec2")
2.  Auto-detect from cfg.MODEL_NAME prefix
    "whisper"  → WhisperASRPipeline
    "wav2vec2" → Wav2Vec2ASRPipeline
    "mms"      → MMSASRPipeline

Each pipeline provides:
  • load_processor()   — processor / tokenizer / feature extractor
  • load_model()       — nn.Module
  • preprocess_fn      — example-level mapping function (injected into dataset.map)
  • get_collator()     — data collator
  • get_metrics()      — compute_metrics
  • build_trainer()    — returns a ready-to-train HF Trainer
"""
from __future__ import annotations

import logging

from bhaskera.audio.asr_registry import get_asr_pipeline
from bhaskera.audio.dataset import load_and_merge, preprocess

logger = logging.getLogger(__name__)


def run_audio_training(cfg, global_rank: int = 0) -> None:
    """
    Full ASR fine-tuning pipeline driven by a bhaskera Config.
    Model family is resolved automatically from cfg.MODEL_NAME or
    cfg.AUDIO_MODEL_FAMILY.
    """
    is_rank0 = global_rank == 0

    # ── 1. Resolve pipeline ───────────────────────────────────────────────────
    if is_rank0:
        logger.info("[1/5] Resolving ASR pipeline for '%s'…", cfg.MODEL_NAME)

    pipeline = get_asr_pipeline(cfg)

    # ── 2. Load processor / tokenizer / feature extractor ────────────────────
    if is_rank0:
        logger.info("[2/5] Loading processor…")

    pipeline.load_processor()

    # ── 3. Load and preprocess datasets ──────────────────────────────────────
    if is_rank0:
        logger.info("[3/5] Loading and preprocessing audio datasets…")

    raw      = load_and_merge(cfg)
    encoded  = preprocess(
        raw,
        preprocess_fn=pipeline.make_preprocess_fn(),
        num_proc=cfg.AUDIO_NUM_PROC,
    )

    # ── 4. Load model ─────────────────────────────────────────────────────────
    if is_rank0:
        logger.info("[4/5] Loading model '%s'…", cfg.MODEL_NAME)

    pipeline.load_model()

    # ── 5. Build trainer and train ────────────────────────────────────────────
    if is_rank0:
        logger.info("[5/5] Starting trainer…")

    trainer = pipeline.build_trainer(encoded)
    pipeline.post_init(trainer)

    trainer.train()
    trainer.save_model(cfg.AUDIO_OUTPUT_DIR)
    pipeline.processor.save_pretrained(cfg.AUDIO_OUTPUT_DIR)

    if is_rank0:
        logger.info("ASR training complete. Model saved to %s", cfg.AUDIO_OUTPUT_DIR)

