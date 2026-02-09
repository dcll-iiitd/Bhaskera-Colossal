# =========================
# Model
# =========================

MODEL_NAME = "tiiuae/falcon-7b"
#MODEL_NAME = "mistralai/Mistral-7B-v0.1"
#ATTN_IMPL = "flash_attention_2"   # or None if unsupported
ATTN_IMPL = "None"


# =========================
# Dataset
# =========================

DATASET_NAME = "ultrachat"          #ultrachat 
SEQ_LEN = 2048

# =========================
# Training
# =========================

BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
MAX_STEPS = 20

# =========================
# PEFT
# =========================

PEFT = "qlora"   # none | lora | qlora

LORA = dict(
    r=8,
    alpha=32
)

# =========================
# Logging
# =========================

TRACKER = None   # None | "wandb" | "mlflow"
PROJECT = "bhaskara"
RUN_NAME = "debug-qlora"
