"""
bhaskera.audio.pipelines.wav2vec2_pipeline
-------------------------------------------
Wav2Vec2 CTC fine-tuning pipeline.

Supports: facebook/wav2vec2-base, facebook/wav2vec2-large,
          facebook/wav2vec2-large-960h, facebook/wav2vec2-xls-r-300m, etc.

Architecture notes
------------------
Wav2Vec2 is a CTC encoder-only model.  Unlike Whisper (Seq2Seq) it:
  • Uses Trainer (not Seq2SeqTrainer)
  • Requires a Wav2Vec2CTCTokenizer + Wav2Vec2FeatureExtractor
  • Labels are raw character or sub-word token ids (not decoder ids)
  • CTC loss handles alignment implicitly

The CTC collator pads input_values (raw waveform floats) rather than
input_features (log-mel spectrogram).

Performance notes
-----------------
• attention_mask=True must be passed to the feature extractor to avoid
  masking valid frames on variable-length batches.
• freeze_feature_encoder() freezes the CNN feature-extraction layers;
  only the transformer + LM head are fine-tuned (faster, less data needed).
• gradient_checkpointing is enabled by default.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import evaluate
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from bhaskera.audio.pipelines.base import BaseASRPipeline

logger = logging.getLogger(__name__)

_wer_metric = evaluate.load("wer")


# ---------------------------------------------------------------------------
# Picklable preprocessing callable
# ---------------------------------------------------------------------------

class _Wav2Vec2Preprocessor:
    def __init__(self, feature_extractor, tokenizer, sampling_rate):
        self.feature_extractor = feature_extractor
        self.tokenizer         = tokenizer
        self.sampling_rate     = sampling_rate

    def __call__(self, example):
        audio = example["audio"]
        return {
            "input_values": self.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
            ).input_values[0],
            "labels": self.tokenizer(example["transcription"]).input_ids,
        }


# ---------------------------------------------------------------------------
# CTC data collator (handles variable-length waveforms + label padding)
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorCTCWithPadding:
    """
    Pads input_values (raw waveforms) and labels for CTC training.
    Labels are padded with -100 so the CTC loss ignores padding tokens.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Separate audio and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]}           for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding token id with -100 so CTC loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Wav2Vec2ASRPipeline(BaseASRPipeline):
    """End-to-end Wav2Vec2 CTC fine-tuning pipeline."""

    def load_processor(self) -> None:
        cfg = self.cfg
        logger.info("Wav2Vec2: loading processor for '%s'…", cfg.MODEL_NAME)

        # Vocab is constructed from the dataset at runtime if not cached.
        # Using AutoProcessor falls back gracefully to Wav2Vec2Processor.
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.MODEL_NAME,
            feature_size=1,
            sampling_rate=cfg.AUDIO_SAMPLING_RATE,
            padding_value=0.0,
            do_normalize=True,          # zero-mean / unit-std normalisation
            return_attention_mask=True, # critical for variable-length batches
        )
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            cfg.MODEL_NAME,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
        )
        logger.info("     vocab=%d", self.tokenizer.vocab_size)

    def load_model(self) -> None:
        logger.info("Wav2Vec2: loading model '%s'…", self.cfg.MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.cfg.MODEL_NAME,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        # Freeze the CNN feature encoder — only fine-tune the transformer
        self.model.freeze_feature_encoder()
        self.model.config.use_cache = False

    def make_preprocess_fn(self):
        return _Wav2Vec2Preprocessor(
            self.feature_extractor, self.tokenizer, self.cfg.AUDIO_SAMPLING_RATE
        )

    def get_collator(self):
        return DataCollatorCTCWithPadding(processor=self.processor, padding=True)

    def get_metrics(self):
        tokenizer = self.tokenizer

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids    = torch.argmax(torch.tensor(pred_logits), dim=-1)
            label_ids   = pred.label_ids
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            pred_str  = tokenizer.batch_decode(pred_ids)
            label_str = tokenizer.batch_decode(label_ids, group_tokens=False)
            return {"wer": _wer_metric.compute(predictions=pred_str, references=label_str)}

        return compute_metrics

    def get_training_args(self):
        return TrainingArguments(
            **self._common_training_kwargs(),
            metric_for_best_model="wer",
            greater_is_better=False,
            # CTC-specific: group_by_length speeds up training by batching
            # similarly-lengthed sequences → less padding waste
            group_by_length=True,
        )

    def build_trainer(self, encoded_datasets):
        return Trainer(
            args=self.get_training_args(),
            model=self.model,
            train_dataset=encoded_datasets["train"],
            eval_dataset=encoded_datasets["test"],
            data_collator=self.get_collator(),
            compute_metrics=self.get_metrics(),
            tokenizer=self.processor.feature_extractor,
        )
