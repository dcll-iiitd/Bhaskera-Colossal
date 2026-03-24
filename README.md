# Bhaskera

A lightweight, distributed LLM fine-tuning framework built on PyTorch, Ray Train, and Hugging Face — with native support for LoRA and QLoRA.

---

## Features

- **Distributed training** via [Ray Train](https://docs.ray.io/en/latest/train/train.html) + PyTorch DDP
- **PEFT support**: LoRA and QLoRA (4-bit via bitsandbytes)
- **Streaming datasets**: UltraChat, SlimPajama/RedPajama, OpenAssistant
- **Automatic LoRA target inference** for LLaMA/Mistral and Falcon architectures
- **Experiment tracking**: Weights & Biases and MLflow
- **YAML-driven config** with a clean CLI entrypoint

---

## Project Structure

```
Bhaskera/
├── src/bhaskera/
│   ├── config.py          # Default training config
│   ├── config.yaml        # YAML config (used by CLI)
│   ├── cli.py             # Main CLI entrypoint
│   ├── data/              # Dataset loaders (UltraChat, RedPajama, OpenAssistant)
│   ├── models/            # Model builders (base, LoRA, QLoRA)
│   ├── trainer/           # Core training loop
│   └── utils/             # Loggers (W&B, MLflow)
├── newtrain.py            # Alternate launcher (config.py-based)
├── setup.sh               # One-shot environment setup
└── pyproject.toml
```

---

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU(s)
- [`uv`](https://github.com/astral-sh/uv) (auto-installed by setup script)

### Setup

```bash
git clone <repo-url>
cd Bhaskera
source setup.sh
```

This creates a `.venv`, installs all dependencies, and installs the package in editable mode.

---

## Usage

### CLI (recommended)

```bash
bhaskera --config src/bhaskera/config.yaml --num-workers 2
```

| Argument | Description | Default |
|---|---|---|
| `--config` | Path to YAML config file | *(required)* |
| `--num-workers` | Number of Ray workers (GPUs) | `1` |

### Script launcher

```bash
python newtrain.py --num-workers 2
```

This uses `config.py` directly instead of a YAML file.

---

## Configuration

Edit `src/bhaskera/config.yaml` to control all training options:

```yaml
# Model
MODEL_NAME: "tiiuae/falcon-7b"   # or mistralai/Mistral-7B-v0.1
ATTN_IMPL: null                   # or "flash_attention_2"

# Dataset
DATASET_NAME: "ultrachat"         # ultrachat | redpajama
SEQ_LEN: 2048

# Training
BATCH_SIZE: 2
GRAD_ACCUM: 8
LR: 2e-4
MAX_STEPS: 20

# PEFT
PEFT: "qlora"                     # none | lora | qlora
LORA:
  r: 8
  alpha: 32

# Logging
TRACKER: null                     # null | wandb | mlflow
PROJECT: "bhaskera"
RUN_NAME: "my-run"
```

---

## Supported Datasets

| Key | Hugging Face Dataset |
|---|---|
| `ultrachat` | `HuggingFaceH4/ultrachat_200k` |
| `redpajama` | `cerebras/SlimPajama-627B` |
| `openassistant` | `OpenAssistant/oasst1` |

All datasets are loaded in **streaming mode** and automatically sharded across workers.

---

## Supported Models

Any Hugging Face `AutoModelForCausalLM`-compatible model. LoRA target modules are inferred automatically for:

- **LLaMA / Mistral**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Falcon**: `query_key_value`, `dense`

---

## Experiment Tracking

Set `TRACKER` in your config to enable logging:

```yaml
TRACKER: "wandb"    # Weights & Biases
TRACKER: "mlflow"   # MLflow
```

Make sure the corresponding package is installed and authenticated.

---

## License

MIT
