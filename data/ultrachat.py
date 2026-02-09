from datasets import load_dataset
from torch.utils.data import IterableDataset
from .sharding import shard_streaming_dataset

class UltraChatStreamingDataset(IterableDataset):
    def __init__(self, hf_ds, tokenizer, seq_len):
        self.ds = hf_ds
        self.tok = tokenizer
        self.seq_len = seq_len

    def format_messages(self, messages):
        out = []
        for m in messages:
            role = m["role"]
            out.append(f"<|{role}|>\n{m['content']}")
        return "\n".join(out)

    def __iter__(self):
        for row in self.ds:
            text = self.format_messages(row["messages"])
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


def build_ultrachat(tokenizer, seq_len, rank, world_size):
    raw = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
    )
    raw = shard_streaming_dataset(raw, rank, world_size)
    return UltraChatStreamingDataset(raw, tokenizer, seq_len)
