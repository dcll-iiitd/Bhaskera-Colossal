from datasets import load_dataset
from torch.utils.data import IterableDataset
from .sharding import shard_streaming_dataset


class SlimPajamaDataset(IterableDataset):
    def __init__(self, ds, tokenizer, seq_len):
        self.ds = ds
        self.tok = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        for row in self.ds:
            text = row["text"]

            out = self.tok(
                text,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt",
            )

            yield {
                "input_ids": out["input_ids"][0],
                "attention_mask": out["attention_mask"][0],
                "labels": out["input_ids"][0],
            }


def build_redpajama(cfg, tokenizer, rank, world_size):
    raw = load_dataset(
        "cerebras/SlimPajama-627B",
        split="train",
        streaming=True,
    )

    raw = shard_streaming_dataset(raw, rank, world_size)

    return SlimPajamaDataset(raw, tokenizer, cfg.SEQ_LEN)
