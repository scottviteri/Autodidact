"""
Data streaming and held-out set management.

Provides:
- A streaming data source that yields fresh batches of N candidate context windows.
- A fixed held-out evaluation set D, with random subset sampling of size M each step.
"""

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from typing import Iterator


class StreamingTextDataset(IterableDataset):
    """
    Wraps a HuggingFace streaming dataset to yield tokenized context windows.
    Each call to __iter__ yields a single tokenized context window of length `seq_len`.
    
    Uses a token buffer to pack text efficiently: documents are concatenated and
    sliced into fixed-length windows, avoiding padding waste.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        seq_len: int = 512,
        dataset_name: str = "openwebtext",
        split: str = "train",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.split = split

    def __iter__(self) -> Iterator[torch.Tensor]:
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True, trust_remote_code=True)
        buffer = []
        for example in dataset:
            text = example["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                yield torch.tensor(buffer[: self.seq_len], dtype=torch.long)
                buffer = buffer[self.seq_len :]


class CandidateBatchSampler:
    """
    Draws batches of N candidate context windows from a streaming dataset.
    Each call to next_batch() returns a fresh [N, seq_len] tensor.
    """

    def __init__(self, dataset: StreamingTextDataset, batch_size: int, device: torch.device):
        self.iterator = iter(dataset)
        self.batch_size = batch_size
        self.device = device

    def next_batch(self) -> torch.Tensor:
        """Returns [N, seq_len] tensor of candidate context windows."""
        candidates = []
        for _ in range(self.batch_size):
            candidates.append(next(self.iterator))
        return torch.stack(candidates).to(self.device)


class HeldOutSet:
    """
    Manages a fixed held-out evaluation set D.
    Supports sampling fresh random subsets of size M each step.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        seq_len: int = 512,
        total_size: int = 1024,
        dataset_name: str = "openwebtext",
        split: str = "train",
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.total_size = total_size
        self.seq_len = seq_len

        # Build the held-out set by consuming from the dataset.
        # We use a different slice of the data by skipping ahead.
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        buffer = []
        examples = []
        # Skip some data to avoid overlap with training stream.
        # We'll consume and discard some initial examples, then collect.
        skip_count = 50000  # Skip first 50k documents for held-out
        count = 0
        for example in dataset:
            count += 1
            if count <= skip_count:
                continue
            text = example["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= seq_len and len(examples) < total_size:
                examples.append(torch.tensor(buffer[:seq_len], dtype=torch.long))
                buffer = buffer[seq_len:]
            if len(examples) >= total_size:
                break

        self.data = torch.stack(examples).to(device)  # [total_size, seq_len]
        print(f"Held-out set: {self.data.shape[0]} examples of length {self.seq_len}")

    def sample_subset(self, m: int) -> torch.Tensor:
        """Sample a fresh random subset of size M. Returns [M, seq_len]."""
        indices = torch.randperm(self.data.shape[0], device=self.device)[:m]
        return self.data[indices]

