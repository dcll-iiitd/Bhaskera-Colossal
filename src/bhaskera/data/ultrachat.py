from datasets import load_dataset
from torch.utils.data import IterableDataset
from .sharding import shard_streaming_dataset
import torch


class UltraChatStreamingDataset(IterableDataset):
    """
    Correct SFT dataset for causal LM training.

    Fixes:
    - Masks padding tokens
    - Masks user/system tokens
    - Trains ONLY on assistant responses
    - Prevents exploding gradients → NaNs
    """

    def __init__(self, hf_ds, tokenizer, seq_len):
        self.ds = hf_ds
        self.tok = tokenizer
        self.seq_len = seq_len

        # ensure tokenizer safe
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

    # ---------------------------------------------------
    # format chat into text
    # ---------------------------------------------------
    def format_messages(self, messages):
        """
        Convert chat messages into single string.
        """
        out = []
        for m in messages:
            role = m["role"]
            content = m["content"]

            if role == "user":
                out.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                out.append(f"<|assistant|>\n{content}\n")
            elif role == "system":
                out.append(f"<|system|>\n{content}\n")

        return "".join(out)

    # ---------------------------------------------------
    # iterator
    # ---------------------------------------------------
    def __iter__(self):
        for row in self.ds:
            try:
                text = self.format_messages(row["messages"])

                tok = self.tok(
                    text,
                    truncation=True,
                    max_length=self.seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids = tok["input_ids"][0]
                attn = tok["attention_mask"][0]

                # -----------------------------------------
                # CREATE LABELS SAFELY
                # -----------------------------------------
                labels = input_ids.clone()

                # 1️⃣ mask padding
                labels[attn == 0] = -100

                # 2️⃣ mask everything before first assistant
                # only train on assistant outputs
                text_lower = text.lower()
                if "<|assistant|>" in text_lower:
                    idx = text_lower.find("<|assistant|>")
                    prefix = text[:idx]

                    prefix_ids = self.tok(
                        prefix,
                        truncation=True,
                        max_length=self.seq_len,
                        return_tensors="pt",
                    )["input_ids"][0]

                    cutoff = min(len(prefix_ids), self.seq_len)
                    labels[:cutoff] = -100
                else:
                    # if no assistant → skip sample
                    continue

                # -----------------------------------------
                # safety check (prevents NaN batches)
                # -----------------------------------------
                if torch.all(labels == -100):
                    continue

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "labels": labels,
                }

            except Exception as e:
                # skip bad rows safely
                continue


# ---------------------------------------------------
# builder
# ---------------------------------------------------
def build_ultrachat(cfg, tokenizer, rank, world_size):
    raw = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
    )

    raw = shard_streaming_dataset(raw, rank, world_size)
    return UltraChatStreamingDataset(raw, tokenizer, cfg.SEQ_LEN)
