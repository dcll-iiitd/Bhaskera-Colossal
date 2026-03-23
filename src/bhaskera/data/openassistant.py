from datasets import load_dataset
from torch.utils.data import IterableDataset

from .sharding import shard_streaming_dataset


class OpenAssistantDataset(IterableDataset):
    def __init__(self, ds, tokenizer, seq_len: int):
        self.ds      = ds
        self.tok     = tokenizer
        self.seq_len = seq_len
        # Must set pad_token before any padding="max_length" call,
        # otherwise the tokenizer raises an error on models like LLaMA.
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

    def __iter__(self):
        for row in self.ds:
            try:
                text = row["text"]
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
                labels[attn == 0] = -100   # mask padding tokens
                yield {
                    "input_ids":      input_ids,
                    "attention_mask": attn,
                    "labels":         labels,
                }
            except Exception:
                continue


def build_openassistant(cfg, tokenizer, rank: int, world_size: int):
    raw = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    raw = shard_streaming_dataset(raw, rank, world_size)
    return OpenAssistantDataset(raw, tokenizer, cfg.SEQ_LEN)
