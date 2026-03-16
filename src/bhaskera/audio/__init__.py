"""
bhaskera.audio
--------------
ASR fine-tuning sub-package.

Public API
----------
  run_audio_training(cfg, global_rank)  — main entry point
  register_asr_model(family, builder)   — extend the model registry
  get_asr_pipeline(cfg)                 — resolve + instantiate a pipeline
"""
from bhaskera.audio.trainer import run_audio_training
from bhaskera.audio.asr_registry import register_asr_model, get_asr_pipeline

__all__ = ["run_audio_training", "register_asr_model", "get_asr_pipeline"]
