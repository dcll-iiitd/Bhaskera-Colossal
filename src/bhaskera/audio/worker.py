"""
bhaskera.audio.worker
----------------------
Drop-in replacement for bhaskera.launcher.worker_core.run_worker
when cfg.DATASET_NAME == "audio".

The framework calls run_worker(ctx, cfg) identically for all job types.
This module intercepts audio jobs and routes to the generic ASR trainer
which resolves the correct pipeline (Whisper, Wav2Vec2, MMS, …) from
the registry based on cfg.MODEL_NAME / cfg.AUDIO_MODEL_FAMILY.
"""
from __future__ import annotations

import logging
import sys
import os

from bhaskera.launcher.worker_core import WorkerContext

logger = logging.getLogger(__name__)


def _setup_logging(global_rank: int) -> None:
    if global_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stdout,
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.CRITICAL,
            stream=open(os.devnull, "w"),
            force=True,
        )
        logging.getLogger().setLevel(logging.CRITICAL)


def _resolve_family_label(cfg) -> str:
    """Best-effort human-readable family label for the startup banner."""
    explicit = getattr(cfg, "AUDIO_MODEL_FAMILY", None)
    if explicit:
        return explicit.upper()
    name = cfg.MODEL_NAME.lower()
    if "whisper"  in name: return "Whisper (Seq2Seq)"
    if "wav2vec2" in name: return "Wav2Vec2 (CTC)"
    if "mms"      in name: return "MMS (CTC adapter)"
    if "seamless" in name: return "SeamlessM4T"
    return "Auto-detect"


def run_audio_worker(ctx: WorkerContext, cfg) -> None:
    """
    Called instead of worker_core.run_worker for audio jobs.
    Prints the same header style as the LLM worker for consistency.
    """
    _setup_logging(ctx.global_rank)
    is_rank0 = ctx.global_rank == 0

    if is_rank0:
        family = _resolve_family_label(cfg)
        print("=" * 60, flush=True)
        print(f"  Bhaskera — ASR Training  [{family}]", flush=True)
        print("=" * 60, flush=True)
        print(f"  Launcher   : {ctx.launcher}", flush=True)
        print(f"  Backend    : DDP (HF Trainer)", flush=True)
        print(f"  World size : {ctx.world_size}", flush=True)
        print(f"  Model      : {cfg.MODEL_NAME}", flush=True)
        print(f"  Family     : {family}", flush=True)
        print(f"  Language   : {cfg.AUDIO_LANGUAGE}  task={cfg.AUDIO_TASK}", flush=True)
        print(f"  Primary DS : {cfg.AUDIO_PRIMARY_DATASET['name']}", flush=True)
        if cfg.AUDIO_SECONDARY_DATASET:
            print(f"  Secondary  : {cfg.AUDIO_SECONDARY_DATASET['name']}", flush=True)
        print(f"  Batch/GPU  : {cfg.BATCH_SIZE}  grad_accum={cfg.GRAD_ACCUM}", flush=True)
        print(f"  LR         : {cfg.LR:.2e}  warmup={cfg.WARMUP_STEPS}", flush=True)
        print(f"  Max steps  : {cfg.MAX_STEPS}", flush=True)
        print(f"  Output     : {cfg.AUDIO_OUTPUT_DIR}", flush=True)
        print("=" * 60, flush=True)
        print(flush=True)

    from bhaskera.audio.trainer import run_audio_training
    run_audio_training(cfg, global_rank=ctx.global_rank)
