"""
bhaskera.audio.dataset
-----------------------
Model-agnostic audio dataset loading and preprocessing.

The key change from the original: preprocessing is no longer hard-coded to
Whisper's feature extractor.  Instead, a `preprocess_fn` callable is injected
by the pipeline, so Whisper, Wav2Vec2, MMS, etc. can each provide their own
encoding logic while sharing the same load/merge/split infrastructure.

preprocess_fn contract
----------------------
The function receives a single example dict and must return a dict containing
at minimum:
  • One input tensor key (e.g. "input_features" for Whisper, "input_values"
    for Wav2Vec2/MMS/CTC models)
  • "labels": List[int]   — token ids

It must NOT retain the "audio" or "transcription" keys (they are removed).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_single(ds_block: Dict[str, Any], sampling_rate: int):
    """Load one dataset block from config, resample, and normalise columns."""
    name           = ds_block["name"]
    config         = ds_block.get("config")
    split          = ds_block["split"]
    transcript_col = ds_block["transcript_col"]
    audio_col      = ds_block["audio_col"]

    logger.info("Loading dataset '%s' (%s) split='%s'", name, config, split)

    load_kwargs = dict(split=split, trust_remote_code=True)
    if config:
        ds = load_dataset(name, config, **load_kwargs)
    else:
        ds = load_dataset(name, **load_kwargs)

    ds = ds.cast_column(audio_col, Audio(sampling_rate=sampling_rate))

    # Normalise to canonical column names so all downstream code is
    # completely dataset-agnostic.
    rename: Dict[str, str] = {}
    if audio_col != "audio":
        rename[audio_col] = "audio"
    if transcript_col != "transcription":
        rename[transcript_col] = "transcription"
    if rename:
        ds = ds.rename_columns(rename)

    keep = {"audio", "transcription"}
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_merge(cfg) -> DatasetDict:
    """
    Load primary (and optional secondary) datasets, merge, shuffle,
    and return a DatasetDict with 'train' / 'test' keys.
    """
    sampling_rate = cfg.AUDIO_SAMPLING_RATE
    test_size     = cfg.AUDIO_TEST_SIZE

    primary_ds = _load_single(cfg.AUDIO_PRIMARY_DATASET, sampling_rate)

    if cfg.AUDIO_SECONDARY_DATASET:
        secondary_ds = _load_single(cfg.AUDIO_SECONDARY_DATASET, sampling_rate)
        merged = concatenate_datasets([primary_ds, secondary_ds]).shuffle(seed=42)
        logger.info(
            "Merged %d examples  (primary=%d  secondary=%d)",
            len(merged), len(primary_ds), len(secondary_ds),
        )
    else:
        merged = primary_ds

    split = merged.train_test_split(test_size=test_size, seed=42)
    logger.info("Split → train=%d  test=%d", len(split["train"]), len(split["test"]))
    return split


def preprocess(
    dataset: DatasetDict,
    preprocess_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    num_proc: int = 4,
    keep_columns: Optional[set] = None,
) -> DatasetDict:
    """
    Apply *preprocess_fn* to every example in the dataset.

    Parameters
    ----------
    dataset       : DatasetDict with 'train' and 'test' keys
    preprocess_fn : model-specific function injected by each pipeline
    num_proc      : parallelism for dataset.map
    keep_columns  : if None, all original columns are removed after mapping
    """
    remove_cols = [
        c for c in dataset.column_names["train"]
        if keep_columns is None or c not in keep_columns
    ]

    return dataset.map(
        preprocess_fn,
        remove_columns=remove_cols,
        num_proc=num_proc,
        batched=False,
    )
