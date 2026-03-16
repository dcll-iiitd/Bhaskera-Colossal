"""
bhaskera.audio.metrics
-----------------------
Compatibility shim — metric logic now lives inside each pipeline class.

make_compute_metrics() is kept here for any external code that imports it
directly.  New code should use pipeline.get_metrics() instead.
"""
from __future__ import annotations

import evaluate
from transformers import WhisperTokenizer

_metric = evaluate.load("wer")


def make_compute_metrics(tokenizer: WhisperTokenizer):
    """Return a WER compute_metrics fn bound to *tokenizer*.

    Kept for backwards compatibility.  New pipelines call get_metrics()
    on the pipeline object instead.
    """

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": _metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics
