"""
bhaskera.config_loader
=======================
Single source of truth for configuration.

All code accesses cfg.UPPER_CASE for scalar values and
cfg.distributed.fsdp_* for distributed settings.

No legacy config.py. No inline Config class in cli.py.
Just this.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DistributedConfig:
    strategy: str = "ddp"

    # DDP
    ddp_broadcast_buffers: bool = False
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True

    # FSDP
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision_param: str = "bfloat16"
    fsdp_mixed_precision_reduce: str = "bfloat16"
    fsdp_mixed_precision_buffer: str = "bfloat16"
    fsdp_backward_prefetch: Optional[str] = "BACKWARD_PRE"
    fsdp_forward_prefetch: bool = False
    fsdp_activation_checkpointing: bool = True
    fsdp_auto_wrap_policy: str = "transformer_auto_wrap"
    fsdp_transformer_layer_cls: List[str] = field(default_factory=lambda: [
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "FalconDecoderLayer",
        "GPTNeoXLayer",
        "GPTJBlock",
        "T5Block",
        "BertLayer",
        "Phi3DecoderLayer",
        "Qwen2DecoderLayer",
        "GemmaDecoderLayer",
    ])
    fsdp_min_num_params: int = 100_000_000
    fsdp_state_dict_type: str = "FULL_STATE_DICT"


@dataclass
class Config:
    # Model
    MODEL_NAME: str = "tiiuae/falcon-7b"
    ATTN_IMPL: Optional[str] = None
    DTYPE: str = "bfloat16"

    # Dataset
    DATASET_NAME: str = "ultrachat"
    SEQ_LEN: int = 2048

    # Training
    BATCH_SIZE: int = 4
    GRAD_ACCUM: int = 4
    LR: float = 5e-5
    MAX_STEPS: int = 500
    NUM_EPOCHS: int = 1
    WARMUP_STEPS: int = 20

    # PEFT
    PEFT: str = "lora"
    LORA: Dict[str, Any] = field(default_factory=lambda: {
        "r": 16, "alpha": 32, "dropout": 0.05
    })

    # Logging
    TRACKER: Optional[str] = None
    PROJECT: str = "bhaskera-training"
    RUN_NAME: str = "experiment-001"

    # Checkpointing
    CHECKPOINT_ENABLED: bool = False
    CHECKPOINT_DIR: str = "./checkpoints"
    CHECKPOINT_INTERVAL: int = 100
    CHECKPOINT_KEEP_LAST_N: int = 3

    # Distributed
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # ── Audio (ASR) ───────────────────────────────────────────────────────────
    # Only populated when DATASET_NAME == "audio"; ignored for all LLM jobs.
    #
    # AUDIO_MODEL_FAMILY: explicit override for the ASR pipeline registry.
    #   Leave None to auto-detect from MODEL_NAME.
    #   Supported values: "whisper" | "wav2vec2" | "mms" | "seamless"
    AUDIO_MODEL_FAMILY: Optional[str] = None
    AUDIO_LANGUAGE: str = "English"
    AUDIO_TASK: str = "transcribe"
    AUDIO_SAMPLING_RATE: int = 16_000
    AUDIO_TEST_SIZE: float = 0.1
    AUDIO_NUM_PROC: int = 4
    AUDIO_OUTPUT_DIR: str = "./whisper-finetuned"
    AUDIO_PRIMARY_DATASET: Dict[str, Any] = field(default_factory=lambda: {
        "name": "PolyAI/minds14",
        "config": "en-US",
        "split": "train[:500]",
        "transcript_col": "english_transcription",
        "audio_col": "audio",
    })
    AUDIO_SECONDARY_DATASET: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return flat scalar fields for logging hyperparameters."""
        skip = {"distributed", "LORA"}
        return {
            k: v for k, v in self.__dict__.items()
            if k not in skip and isinstance(v, (str, int, float, bool, type(None)))
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from YAML, or return defaults if path is None."""
    if config_path is None:
        return Config()

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with open(p) as f:
        y = yaml.safe_load(f)

    model_cfg   = y.get("model", {})
    ds_cfg      = y.get("dataset", {})
    train_cfg   = y.get("training", {})
    dist_cfg    = train_cfg.get("distributed", {})
    fsdp_cfg    = dist_cfg.get("fsdp", {})
    ddp_cfg     = dist_cfg.get("ddp", {})
    mp_cfg      = fsdp_cfg.get("mixed_precision", {})
    wrap_cfg    = fsdp_cfg.get("auto_wrap_policy", {})
    peft_cfg    = y.get("peft", {})
    lora_cfg    = peft_cfg.get("lora", {})
    log_cfg     = y.get("logging", {})
    ckpt_cfg    = y.get("checkpointing", {})
    audio_cfg   = y.get("audio", {})

    tracker_raw = log_cfg.get("tracker")
    # YAML `None` comes through as the Python None or the string "None"
    tracker = None if tracker_raw in (None, "None", "none", "null") else str(tracker_raw)

    dist = DistributedConfig(
        strategy=dist_cfg.get("strategy", "ddp"),
        ddp_broadcast_buffers=ddp_cfg.get("broadcast_buffers", False),
        ddp_find_unused_parameters=ddp_cfg.get("find_unused_parameters", False),
        ddp_gradient_as_bucket_view=ddp_cfg.get("gradient_as_bucket_view", True),
        fsdp_sharding_strategy=fsdp_cfg.get("sharding_strategy", "FULL_SHARD"),
        fsdp_cpu_offload=fsdp_cfg.get("cpu_offload", False),
        fsdp_mixed_precision_param=mp_cfg.get("param_dtype", "bfloat16"),
        fsdp_mixed_precision_reduce=mp_cfg.get("reduce_dtype", "bfloat16"),
        fsdp_mixed_precision_buffer=mp_cfg.get("buffer_dtype", "bfloat16"),
        fsdp_backward_prefetch=fsdp_cfg.get("backward_prefetch", "BACKWARD_PRE"),
        fsdp_forward_prefetch=fsdp_cfg.get("forward_prefetch", False),
        fsdp_activation_checkpointing=fsdp_cfg.get("activation_checkpointing", True),
        fsdp_auto_wrap_policy=wrap_cfg.get("type", "transformer_auto_wrap"),
        fsdp_transformer_layer_cls=wrap_cfg.get("transformer_layer_cls", None) or
            DistributedConfig.__dataclass_fields__["fsdp_transformer_layer_cls"].default_factory(),
        # Use int(float(...)) so both "1e8" strings and plain ints work
        fsdp_min_num_params=int(float(wrap_cfg.get("min_num_params", 1e8))),
        fsdp_state_dict_type=fsdp_cfg.get("state_dict_type", "FULL_STATE_DICT"),
    )

    return Config(
        MODEL_NAME=model_cfg.get("name", "tiiuae/falcon-7b"),
        ATTN_IMPL=model_cfg.get("attn_impl") or None,   # converts "null"/"None" -> None
        DTYPE=model_cfg.get("dtype", "bfloat16"),
        DATASET_NAME=ds_cfg.get("name", "ultrachat"),
        SEQ_LEN=ds_cfg.get("seq_len", 2048),
        BATCH_SIZE=train_cfg.get("batch_size", 4),
        GRAD_ACCUM=train_cfg.get("grad_accum", 4),
        LR=train_cfg.get("lr", 5e-5),
        MAX_STEPS=train_cfg.get("max_steps", 500),
        NUM_EPOCHS=train_cfg.get("num_epochs", 1),
        WARMUP_STEPS=train_cfg.get("warmup_steps", 20),
        PEFT=peft_cfg.get("method", "lora"),
        LORA={
            "r":       lora_cfg.get("r",       16),
            "alpha":   lora_cfg.get("alpha",   32),
            "dropout": lora_cfg.get("dropout", 0.05),
        },
        TRACKER=tracker,
        PROJECT=log_cfg.get("project", "bhaskera-training"),
        RUN_NAME=log_cfg.get("run_name", "experiment-001"),
        CHECKPOINT_ENABLED=ckpt_cfg.get("enabled", False),
        CHECKPOINT_DIR=ckpt_cfg.get("save_dir", "./checkpoints"),
        CHECKPOINT_INTERVAL=ckpt_cfg.get("save_interval", 100),
        CHECKPOINT_KEEP_LAST_N=ckpt_cfg.get("keep_last_n", 3),
        distributed=dist,
        # ── Audio fields (only meaningful when DATASET_NAME="audio") ──────────
        AUDIO_MODEL_FAMILY=model_cfg.get("family") or audio_cfg.get("model_family") or None,
        AUDIO_LANGUAGE=audio_cfg.get("language", model_cfg.get("language", "English")),
        AUDIO_TASK=audio_cfg.get("task", model_cfg.get("task", "transcribe")),
        AUDIO_SAMPLING_RATE=ds_cfg.get("sampling_rate", 16_000),
        AUDIO_TEST_SIZE=ds_cfg.get("test_size", 0.1),
        AUDIO_NUM_PROC=ds_cfg.get("num_proc", 4),
        AUDIO_OUTPUT_DIR=y.get("output", {}).get("dir", "./whisper-finetuned"),
        AUDIO_PRIMARY_DATASET=ds_cfg.get("primary", {
            "name": "PolyAI/minds14",
            "config": "en-US",
            "split": "train[:500]",
            "transcript_col": "english_transcription",
            "audio_col": "audio",
        }),
        AUDIO_SECONDARY_DATASET=ds_cfg.get("secondary") or None,
    )
