"""
bhaskera.audio.collator
------------------------
Padding-aware data collator for Whisper Seq2Seq training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        # Audio side
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Label (text token) side
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding positions with -100 so loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip leading BOS — Seq2SeqTrainer prepends it automatically
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
