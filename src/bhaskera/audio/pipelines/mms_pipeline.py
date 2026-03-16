"""
bhaskera.audio.pipelines.mms_pipeline
---------------------------------------
Meta MMS (Massively Multilingual Speech) CTC fine-tuning pipeline.

Supports: facebook/mms-300m, facebook/mms-1b, facebook/mms-1b-fl102,
          facebook/mms-1b-all, facebook/mms-300m-* (language-specific)

Architecture notes
------------------
MMS models are Wav2Vec2-based but use a different adapter layer architecture
(MMS-Adapter) for multilingual support.  The key differences from vanilla
Wav2Vec2 are:

  • Uses Wav2Vec2ForCTC but with per-language adapter blocks — call
    model.load_adapter(lang) to activate a specific language adapter.
  • The tokenizer vocab changes per language; set_target_lang(lang)
    updates the head projection on-the-fly without reloading weights.

For fine-tuning:
  • target_lang must be passed in cfg.AUDIO_LANGUAGE (ISO 639-3, e.g. "eng")
  • Adapter parameters and the final LM-head are the only trainable parts
    (freeze_base_model() freezes everything else → very parameter-efficient)

Performance notes
-----------------
• Only ~2 M adapter parameters are trained per language → fast convergence
• Compatible with DDP; not tested with FSDP (adapter sharding untested)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import evaluate
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)

from bhaskera.audio.pipelines.base import BaseASRPipeline
from bhaskera.audio.pipelines.wav2vec2_pipeline import DataCollatorCTCWithPadding

logger = logging.getLogger(__name__)

_wer_metric = evaluate.load("wer")


# ---------------------------------------------------------------------------
# Picklable preprocessing callable
# ---------------------------------------------------------------------------

class _MMSPreprocessor:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer         = tokenizer

    def __call__(self, example):
        audio = example["audio"]
        return {
            "input_values": self.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
            ).input_values[0],
            "labels": self.tokenizer(example["transcription"]).input_ids,
        }


class MMSASRPipeline(BaseASRPipeline):
    """
    MMS CTC fine-tuning pipeline.

    cfg.AUDIO_LANGUAGE should be an ISO 639-3 code (e.g. "eng", "hin", "fra").
    The pipeline loads the correct adapter for that language automatically.
    """

    def load_processor(self) -> None:
        cfg = self.cfg
        # MMS uses "eng" style lang codes; map common aliases
        lang = self._resolve_lang(cfg.AUDIO_LANGUAGE)
        logger.info("MMS: loading processor for '%s' (lang=%s)…", cfg.MODEL_NAME, lang)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.MODEL_NAME)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            cfg.MODEL_NAME,
            target_lang=lang,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
        )
        self._lang = lang
        logger.info("     vocab=%d  lang=%s", self.tokenizer.vocab_size, lang)

    def load_model(self) -> None:
        cfg = self.cfg
        lang = self._lang
        logger.info("MMS: loading model '%s' adapter lang=%s…", cfg.MODEL_NAME, lang)

        self.model = Wav2Vec2ForCTC.from_pretrained(
            cfg.MODEL_NAME,
            target_lang=lang,
            ignore_mismatched_sizes=True,
        )
        # Load the language-specific adapter and project head
        self.model.load_adapter(lang)
        # Freeze everything except adapter layers + lm_head → parameter efficient
        self.model.freeze_base_model()
        self.model.config.use_cache = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "MMS trainable: %.2fM / %.2fM (%.2f%%)",
            trainable / 1e6, total / 1e6, trainable / total * 100
        )

    def make_preprocess_fn(self):
        return _MMSPreprocessor(self.feature_extractor, self.tokenizer)

    def get_collator(self):
        return DataCollatorCTCWithPadding(processor=self.processor, padding=True)

    def get_metrics(self):
        tokenizer = self.tokenizer

        def compute_metrics(pred):
            pred_ids  = torch.argmax(torch.tensor(pred.predictions), dim=-1)
            label_ids = pred.label_ids
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
            group_by_length=True,  # batch similar-length seqs → less padding
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_lang(lang_str: str) -> str:
        """
        Map common English-style language names / BCP-47 codes to ISO 639-3
        as expected by MMS adapters.
        """
        _MAP = {
            "english": "eng", "en": "eng", "en-us": "eng", "en-gb": "eng",
            "hindi":   "hin", "hi": "hin",
            "french":  "fra", "fr": "fra",
            "spanish": "spa", "es": "spa",
            "german":  "deu", "de": "deu",
            "arabic":  "ara", "ar": "ara",
            "chinese": "cmn", "zh": "cmn",
            "japanese":"jpn", "ja": "jpn",
        }
        return _MAP.get(lang_str.lower(), lang_str.lower())
