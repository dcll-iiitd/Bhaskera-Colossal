"""
bhaskera.audio.pipelines.whisper_pipeline
------------------------------------------
Whisper Seq2Seq ASR pipeline.

Supports any openai/whisper-* checkpoint (tiny → large-v3).
Uses the standard Seq2SeqTrainer from transformers.

Performance notes
-----------------
• gradient_checkpointing=True   — reduces VRAM ~40 % at a small throughput cost
• use_cache=False                — required when grad-checkpointing is enabled
• generation_max_length=225      — matches Whisper's max output length
• predict_with_generate=True     — required for WER eval during training
"""
from __future__ import annotations

import logging

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

import evaluate

from bhaskera.audio.pipelines.base import BaseASRPipeline
from bhaskera.audio.collator import DataCollatorSpeechSeq2SeqWithPadding

logger = logging.getLogger(__name__)

_wer_metric = evaluate.load("wer")


# ---------------------------------------------------------------------------
# Picklable preprocessing callable (no lambda — safe for multiprocessing)
# ---------------------------------------------------------------------------

class _WhisperPreprocessor:
    """Picklable wrapper so dataset.map can run on num_proc > 1 workers."""

    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, example):
        audio = example["audio"]
        return {
            "input_features": self.feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            ).input_features[0],
            "labels": self.tokenizer(example["transcription"]).input_ids,
        }


class WhisperASRPipeline(BaseASRPipeline):
    """End-to-end Whisper fine-tuning pipeline."""

    # ------------------------------------------------------------------
    # Processor / tokenizer / feature extractor
    # ------------------------------------------------------------------

    def load_processor(self) -> None:
        cfg = self.cfg
        logger.info("Whisper: loading processor for '%s'…", cfg.MODEL_NAME)

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.MODEL_NAME)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            cfg.MODEL_NAME, language=cfg.AUDIO_LANGUAGE, task=cfg.AUDIO_TASK
        )
        self.processor = WhisperProcessor.from_pretrained(
            cfg.MODEL_NAME, language=cfg.AUDIO_LANGUAGE, task=cfg.AUDIO_TASK
        )
        logger.info("     vocab=%d", self.tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        logger.info("Whisper: loading model '%s'…", self.cfg.MODEL_NAME)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.cfg.MODEL_NAME)

        # Disable forced decoder ids so the model learns language/task from data
        self.model.config.forced_decoder_ids            = None
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens    = []
        self.model.config.use_cache                     = False  # required with grad ckpt

    def make_preprocess_fn(self):
        return _WhisperPreprocessor(self.feature_extractor, self.tokenizer)

    # ------------------------------------------------------------------
    # Collator
    # ------------------------------------------------------------------

    def get_collator(self):
        return DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self):
        tokenizer = self.tokenizer

        def compute_metrics(pred):
            pred_ids  = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {"wer": _wer_metric.compute(predictions=pred_str, references=label_str)}

        return compute_metrics

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------

    def get_training_args(self):
        return Seq2SeqTrainingArguments(
            **self._common_training_kwargs(),
            predict_with_generate=True,
            generation_max_length=225,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------

    def build_trainer(self, encoded_datasets):
        return Seq2SeqTrainer(
            args=self.get_training_args(),
            model=self.model,
            train_dataset=encoded_datasets["train"],
            eval_dataset=encoded_datasets["test"],
            data_collator=self.get_collator(),
            compute_metrics=self.get_metrics(),
            processing_class=self.processor.feature_extractor,
        )
