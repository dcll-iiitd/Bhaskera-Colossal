"""
Configuration loader with YAML support and backward compatibility.
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    strategy: str = "ddp"

    # DDP settings
    ddp_broadcast_buffers: bool = False
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True

    # FSDP settings
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision_param: str = "bfloat16"
    fsdp_mixed_precision_reduce: str = "bfloat16"
    fsdp_mixed_precision_buffer: str = "bfloat16"
    fsdp_backward_prefetch: Optional[str] = "BACKWARD_PRE"
    fsdp_forward_prefetch: bool = False
    fsdp_activation_checkpointing: bool = True
    fsdp_auto_wrap_policy: str = "transformer_auto_wrap"
    fsdp_transformer_layer_cls: list = None
    fsdp_min_num_params: int = 100_000_000
    fsdp_state_dict_type: str = "FULL_STATE_DICT"

    def __post_init__(self):
        if self.fsdp_transformer_layer_cls is None:
            self.fsdp_transformer_layer_cls = [
                "LlamaDecoderLayer",
                "MistralDecoderLayer",
                "FalconDecoderLayer",
                "GPTNeoXLayer",
                "GPTJBlock",
                "T5Block",
                "BertLayer",
            ]


@dataclass
class Config:
    """Main configuration class."""
    # Model
    MODEL_NAME: str = "tiiuae/falcon-7b"
    ATTN_IMPL: Optional[str] = None
    DTYPE: str = "bfloat16"

    # Dataset
    DATASET_NAME: str = "ultrachat"
    SEQ_LEN: int = 2048

    # Training
    BATCH_SIZE: int = 2
    GRAD_ACCUM: int = 8
    LR: float = 5e-5          # Safe default for LoRA+FSDP on 7B models
    MAX_STEPS: int = 20
    NUM_EPOCHS: int = 1
    WARMUP_STEPS: int = 5     # Linear warmup — prevents NaN from LR spike

    # PEFT
    PEFT: str = "lora"
    LORA: Dict[str, Any] = None

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
    distributed: DistributedConfig = None

    def __post_init__(self):
        if self.LORA is None:
            self.LORA = {"r": 16, "alpha": 32, "dropout": 0.05}
        if self.distributed is None:
            self.distributed = DistributedConfig()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file or use defaults.
    """
    if config_path is None:
        return Config()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    dist_cfg_dict = yaml_config.get('training', {}).get('distributed', {})
    strategy = dist_cfg_dict.get('strategy', 'ddp')

    ddp_cfg   = dist_cfg_dict.get('ddp', {})
    fsdp_cfg  = dist_cfg_dict.get('fsdp', {})
    mixed_precision = fsdp_cfg.get('mixed_precision', {})
    auto_wrap = fsdp_cfg.get('auto_wrap_policy', {})

    dist_config = DistributedConfig(
        strategy=strategy,
        ddp_broadcast_buffers=ddp_cfg.get('broadcast_buffers', False),
        ddp_find_unused_parameters=ddp_cfg.get('find_unused_parameters', False),
        ddp_gradient_as_bucket_view=ddp_cfg.get('gradient_as_bucket_view', True),
        fsdp_sharding_strategy=fsdp_cfg.get('sharding_strategy', 'FULL_SHARD'),
        fsdp_cpu_offload=fsdp_cfg.get('cpu_offload', False),
        fsdp_mixed_precision_param=mixed_precision.get('param_dtype', 'bfloat16'),
        fsdp_mixed_precision_reduce=mixed_precision.get('reduce_dtype', 'bfloat16'),
        fsdp_mixed_precision_buffer=mixed_precision.get('buffer_dtype', 'bfloat16'),
        fsdp_backward_prefetch=fsdp_cfg.get('backward_prefetch', 'BACKWARD_PRE'),
        fsdp_forward_prefetch=fsdp_cfg.get('forward_prefetch', False),
        fsdp_activation_checkpointing=fsdp_cfg.get('activation_checkpointing', True),
        fsdp_auto_wrap_policy=auto_wrap.get('type', 'transformer_auto_wrap'),
        fsdp_transformer_layer_cls=auto_wrap.get('transformer_layer_cls', None),
        fsdp_min_num_params=int(auto_wrap.get('min_num_params', 1e8)),
        fsdp_state_dict_type=fsdp_cfg.get('state_dict_type', 'FULL_STATE_DICT'),
    )

    model_cfg      = yaml_config.get('model', {})
    dataset_cfg    = yaml_config.get('dataset', {})
    training_cfg   = yaml_config.get('training', {})
    peft_cfg       = yaml_config.get('peft', {})
    lora_cfg       = peft_cfg.get('lora', {})
    logging_cfg    = yaml_config.get('logging', {})
    checkpoint_cfg = yaml_config.get('checkpointing', {})

    return Config(
        MODEL_NAME=model_cfg.get('name', 'tiiuae/falcon-7b'),
        ATTN_IMPL=model_cfg.get('attn_impl'),
        DTYPE=model_cfg.get('dtype', 'bfloat16'),
        DATASET_NAME=dataset_cfg.get('name', 'ultrachat'),
        SEQ_LEN=dataset_cfg.get('seq_len', 2048),
        BATCH_SIZE=training_cfg.get('batch_size', 2),
        GRAD_ACCUM=training_cfg.get('grad_accum', 8),
        LR=training_cfg.get('lr', 5e-5),
        MAX_STEPS=training_cfg.get('max_steps', 20),
        NUM_EPOCHS=training_cfg.get('num_epochs', 1),
        WARMUP_STEPS=training_cfg.get('warmup_steps', 5),   # NEW
        PEFT=peft_cfg.get('method', 'lora'),
        LORA={
            'r':       lora_cfg.get('r',       16),
            'alpha':   lora_cfg.get('alpha',   32),
            'dropout': lora_cfg.get('dropout', 0.05),
        },
        TRACKER=logging_cfg.get('tracker'),
        PROJECT=logging_cfg.get('project', 'bhaskera-training'),
        RUN_NAME=logging_cfg.get('run_name', 'experiment-001'),
        CHECKPOINT_ENABLED=checkpoint_cfg.get('enabled', False),
        CHECKPOINT_DIR=checkpoint_cfg.get('save_dir', './checkpoints'),
        CHECKPOINT_INTERVAL=checkpoint_cfg.get('save_interval', 100),
        CHECKPOINT_KEEP_LAST_N=checkpoint_cfg.get('keep_last_n', 3),
        distributed=dist_config,
    )