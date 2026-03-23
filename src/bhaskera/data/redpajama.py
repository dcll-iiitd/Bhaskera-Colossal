from datasets import load_dataset
from torch.utils.data import IterableDataset
from .sharding import shard_streaming_dataset


class SlimPajamaDataset(IterableDataset):
    def __init__(self, ds, tokenizer, seq_len):
        self.ds      = ds
        self.tok     = tokenizer
        self.seq_len = seq_len
        # Ensure pad token is set before any padding calls
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

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

            input_ids = out["input_ids"][0]
            attn      = out["attention_mask"][0]
            labels    = input_ids.clone()
            labels[attn == 0] = -100   # mask padding so loss ignores it

            yield {
                "input_ids":      input_ids,
                "attention_mask": attn,
                "labels":         labels,
            }


def build_redpajama(cfg, tokenizer, rank, world_size):
    raw = load_dataset(
        "cerebras/SlimPajama-627B",
        split="train",
        streaming=True,
    )

    raw = shard_streaming_dataset(raw, rank, world_size)

    return SlimPajamaDataset(raw, tokenizer, cfg.SEQ_LEN)
