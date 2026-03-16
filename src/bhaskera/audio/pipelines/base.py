"""
bhaskera.audio.pipelines.base
------------------------------
Abstract base class every ASR pipeline must implement.

The Trainer contract
--------------------
build_trainer(encoded_datasets) must return a HuggingFace Trainer (or
subclass) that is ready to call .train() / .save_model() on.

Performance contract
--------------------
Each pipeline is expected to:
  • Set use_cache=False on the model when gradient checkpointing is active.
  • Only expose trainable parameters (freeze base weights when using LoRA/adapters).
  • Accept cfg.DTYPE for mixed-precision.
  • Respect cfg.AUDIO_NUM_PROC for dataset.map parallelism.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseASRPipeline(ABC):
    """
    Subclasses must implement:
      load_processor()   → sets self.processor, self.feature_extractor, self.tokenizer
      load_model()       → sets self.model
      get_collator()     → returns a data collator callable
      get_metrics()      → returns a compute_metrics callable
      get_training_args() → returns HF TrainingArguments / Seq2SeqTrainingArguments
      build_trainer()    → returns a HF Trainer ready to .train()
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # Populated by load_processor() / load_model()
        self.processor = None
        self.feature_extractor = None
        self.tokenizer = None
        self.model = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_processor(self) -> None:
        """Populate self.processor, self.feature_extractor, self.tokenizer."""

    @abstractmethod
    def load_model(self) -> None:
        """Populate self.model."""

    @abstractmethod
    def get_collator(self):
        """Return a data collator callable for this architecture."""

    @abstractmethod
    def get_metrics(self):
        """Return a compute_metrics(EvalPrediction) -> dict callable."""

    @abstractmethod
    def get_training_args(self):
        """Return a HF TrainingArguments (or subclass) instance."""

    @abstractmethod
    def make_preprocess_fn(self):
        """
        Return a callable  f(example: dict) -> dict  that converts a raw
        example (keys: 'audio', 'transcription') into model-specific inputs.

        Must be picklable (no lambda closures over Rust-backed tokenizers).
        Use a nested class or a module-level callable instead.
        """

    @abstractmethod
    def build_trainer(self, encoded_datasets) -> Any:
        """
        Build and return a fully configured HF Trainer.

        Parameters
        ----------
        encoded_datasets : DatasetDict with 'train' and 'test' keys
            Already preprocessed (input_features + labels tensors).
        """

    # ------------------------------------------------------------------
    # Shared helpers (subclasses may override if needed)
    # ------------------------------------------------------------------

    @property
    def _output_dir(self) -> str:
        return self.cfg.AUDIO_OUTPUT_DIR

    @property
    def _is_fp16(self) -> bool:
        return self.cfg.DTYPE == "float16"

    @property
    def _is_bf16(self) -> bool:
        return self.cfg.DTYPE == "bfloat16"

    def _common_training_kwargs(self) -> Dict[str, Any]:
        """
        Returns kwargs shared across all ASR training argument constructors.
        Subclasses pass **self._common_training_kwargs() into their
        TrainingArguments constructor to avoid copy-paste.
        """
        cfg = self.cfg
        return dict(
            output_dir=self._output_dir,
            per_device_train_batch_size=cfg.BATCH_SIZE,
            per_device_eval_batch_size=max(1, cfg.BATCH_SIZE // 2),
            gradient_accumulation_steps=cfg.GRAD_ACCUM,
            learning_rate=cfg.LR,
            warmup_steps=cfg.WARMUP_STEPS,
            max_steps=cfg.MAX_STEPS,
            fp16=self._is_fp16,
            bf16=self._is_bf16,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=cfg.CHECKPOINT_INTERVAL,
            save_steps=cfg.CHECKPOINT_INTERVAL,
            logging_steps=max(1, cfg.CHECKPOINT_INTERVAL // 4),
            load_best_model_at_end=True,
            report_to=[cfg.TRACKER] if cfg.TRACKER else ["none"],
            push_to_hub=False,
            ddp_find_unused_parameters=False,
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        )

    def post_init(self, trainer) -> None:
        """Optional hook called after Trainer is constructed. No-op by default."""
