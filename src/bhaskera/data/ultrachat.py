from datasets import load_dataset
from torch.utils.data import IterableDataset
import torch

from .sharding import shard_streaming_dataset


class UltraChatStreamingDataset(IterableDataset):
    """
    SFT dataset for causal LM. Trains only on assistant turns.
    Masks padding and prompt tokens so loss is not computed on them.
    """

    def __init__(self, hf_ds, tokenizer, seq_len: int):
        self.ds      = hf_ds
        self.tok     = tokenizer
        self.seq_len = seq_len
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

    def _format(self, messages) -> str:
        parts = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
            elif role == "system":
                parts.append(f"<|system|>\n{content}\n")
        return "".join(parts)

    def __iter__(self):
        for row in self.ds:
            try:
                text = self._format(row["messages"])

                enc = self.tok(
                    text,
                    truncation=True,
                    max_length=self.seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"][0]
                attn      = enc["attention_mask"][0]
                labels    = input_ids.clone()

                # Mask padding
                labels[attn == 0] = -100

                # Mask everything before the first assistant turn
                text_lower = text.lower()
                if "<|assistant|>" not in text_lower:
                    continue
                idx    = text_lower.find("<|assistant|>")
                prefix = text[:idx]
                prefix_ids = self.tok(
                    prefix,
                    truncation=True,
                    max_length=self.seq_len,
                    return_tensors="pt",
                )["input_ids"][0]
                cutoff = min(len(prefix_ids), self.seq_len)
                labels[:cutoff] = -100

                # Skip samples where nothing is trainable
                if torch.all(labels == -100):
                    continue

                yield {
                    "input_ids":      input_ids,
                    "attention_mask": attn,
                    "labels":         labels,
                }
            except Exception:
                continue


def build_ultrachat(cfg, tokenizer, rank: int, world_size: int):
    raw = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
    )
    raw = shard_streaming_dataset(raw, rank, world_size)
    return UltraChatStreamingDataset(raw, tokenizer, cfg.SEQ_LEN)
