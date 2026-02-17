"""
Data streaming and held-out set management.

Provides:
- DataStream: a single sequential stream over a tokenized dataset.
  All consumers (held-out set, RAG index, training batches) draw from
  the same stream in order, guaranteeing zero overlap.
- HeldOutSet: a fixed held-out evaluation set D, built from pre-consumed
  data, with random subset sampling of size M each step.
- TopicSimilarityTracker: measures cosine similarity between candidate texts
  and the held-out set using sentence-transformer embeddings.
"""

import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from typing import Iterator, List, Optional, Tuple


class DataStream:
    """
    Single sequential stream over a tokenized dataset.

    Opens one HuggingFace streaming iterator and maintains a token buffer.
    All consumers draw from the same stream in order:

        stream = DataStream(tokenizer, seq_len, "openwebtext")

        # 1. Held-out set (first chunk)
        held_out_tokens, held_out_texts = stream.consume_windows(2048)

        # 2. RAG index (next chunk)
        rag_tokens, rag_texts = stream.consume_windows(50000)

        # 3. Training batches (continues from here)
        batch = stream.next_batch(256, device)

        # 4. RAG refresh (keeps reading forward)
        fresh_tokens, fresh_texts = stream.consume_windows(50000)

    No overlap, no skipping, deterministic ordering.  If the stream is
    exhausted, it wraps around by creating a new iterator.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        seq_len: int = 512,
        dataset_name: str = "openwebtext",
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.split = split

        self._stream = iter(load_dataset(dataset_name, split=split, streaming=True))
        self._buffer: List[int] = []
        self._total_windows_consumed = 0

    def _restart_stream(self):
        """Restart the underlying HuggingFace streaming iterator."""
        print(f"[DataStream] Stream exhausted after {self._total_windows_consumed} "
              f"windows. Restarting from beginning...")
        self._stream = iter(
            load_dataset(self.dataset_name, split=self.split, streaming=True)
        )
        self._buffer = []

    def _fill_buffer(self, min_tokens: int):
        """Read documents until the buffer has at least `min_tokens` tokens."""
        while len(self._buffer) < min_tokens:
            try:
                example = next(self._stream)
            except StopIteration:
                self._restart_stream()
                example = next(self._stream)
            text = example["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self._buffer.extend(tokens)

    def consume_windows(self, count: int) -> Tuple[List[List[int]], List[str]]:
        """
        Consume exactly `count` fixed-length token windows from the stream.

        Returns:
            tokens_list: list of `count` token lists, each of length seq_len
            texts_list:  list of `count` decoded text strings
        """
        tokens_list = []
        texts_list = []
        while len(tokens_list) < count:
            self._fill_buffer(self.seq_len)
            window_tokens = self._buffer[:self.seq_len]
            self._buffer = self._buffer[self.seq_len:]
            tokens_list.append(window_tokens)
            texts_list.append(self.tokenizer.decode(window_tokens))
        self._total_windows_consumed += count
        return tokens_list, texts_list

    def next_window(self) -> torch.Tensor:
        """Consume and return a single [seq_len] token tensor."""
        self._fill_buffer(self.seq_len)
        window_tokens = self._buffer[:self.seq_len]
        self._buffer = self._buffer[self.seq_len:]
        self._total_windows_consumed += 1
        return torch.tensor(window_tokens, dtype=torch.long)

    def next_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Consume and return a [batch_size, seq_len] tensor of token windows."""
        windows = [self.next_window() for _ in range(batch_size)]
        return torch.stack(windows).to(device)

    @property
    def total_consumed(self) -> int:
        """Total number of windows consumed from the stream so far."""
        return self._total_windows_consumed


class HeldOutSet:
    """
    Manages a fixed held-out evaluation set D.
    Supports sampling fresh random subsets of size M each step.

    Two modes:
        1. Random (default): Use pre-consumed windows directly.
        2. Topic-coherent (`topic_coherent=True`): From a larger pre-consumed
           pool, pick a random anchor and sample `total_size` windows with
           probability proportional to cosine similarity to the anchor in
           sentence-embedding space.

    The topic-coherent mode is designed for the "generalized needle" setting:
    the held-out set is thematically coherent (stable reward signal) while
    similar content remains in the training pool for the Q-network to discover.
    """

    def __init__(
        self,
        pool_data: torch.Tensor,
        pool_texts: List[str],
        device: torch.device = torch.device("cpu"),
        # --- Topic-coherent sampling ---
        topic_coherent: bool = False,
        total_size: Optional[int] = None,
        topic_temperature: float = 0.1,
        topic_sim_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.device = device
        self.seq_len = pool_data.shape[1]
        self.topic_coherent = topic_coherent
        self.anchor_text = None  # set when topic_coherent=True

        if topic_coherent:
            assert total_size is not None, (
                "total_size is required for topic-coherent mode "
                "(pool_data should be larger than total_size)"
            )
            self._build_topic_coherent(
                pool_data, pool_texts, total_size, topic_temperature,
                topic_sim_model, str(device),
            )
        else:
            self.data = pool_data.to(device)
            self.texts = pool_texts

        self.total_size = self.data.shape[0]
        print(f"Held-out set: {self.total_size} examples of length {self.seq_len}"
              f"{' (topic-coherent)' if topic_coherent else ''}")

    def _build_topic_coherent(
        self,
        pool_data: torch.Tensor,
        pool_texts: List[str],
        total_size: int,
        temperature: float,
        sim_model_name: str,
        device_str: str,
    ):
        """
        Select a topic-coherent held-out set from the pool.

        1. Embed all pool windows with a sentence-transformer.
        2. Pick a random anchor window.
        3. Compute cosine similarities to the anchor.
        4. Sample `total_size` windows (without replacement) with
           P(i) ~ exp(cos_sim(i, anchor) / temperature).
        """
        from sentence_transformers import SentenceTransformer

        print(f"[HeldOut] Topic-coherent mode: embedding {len(pool_texts)} windows...")
        embed_model = SentenceTransformer(sim_model_name, device=device_str)
        pool_embeddings = embed_model.encode(
            pool_texts,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )  # [pool_size, embed_dim]

        # Pick a random anchor
        anchor_idx = np.random.randint(0, len(pool_texts))
        anchor_emb = pool_embeddings[anchor_idx]  # [embed_dim]
        self.anchor_text = pool_texts[anchor_idx]

        # Cosine similarities (embeddings are L2-normalised)
        sims = pool_embeddings @ anchor_emb  # [pool_size]

        # Softmax sampling probabilities: P(i) ~ exp(sim_i / temperature)
        logits = sims / temperature
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()

        # Sample without replacement
        n_select = min(total_size, len(pool_texts))
        selected_indices = np.random.choice(
            len(pool_texts), size=n_select, replace=False, p=probs,
        )

        self.data = pool_data[selected_indices].to(self.device)
        self.texts = [pool_texts[i] for i in selected_indices]

        # Log statistics
        selected_sims = sims[selected_indices]
        print(f"[HeldOut] Anchor: {self.anchor_text[:120]}...")
        print(f"[HeldOut] Selected {n_select} windows: "
              f"sim to anchor mean={selected_sims.mean():.4f}, "
              f"min={selected_sims.min():.4f}, max={selected_sims.max():.4f} "
              f"(pool mean={sims.mean():.4f}, temperature={temperature})")

        # Free the embedding model
        del embed_model, pool_embeddings

    def sample_subset(self, m: int) -> torch.Tensor:
        """Sample a fresh random subset of size M. Returns [M, seq_len]."""
        indices = torch.randperm(self.data.shape[0], device=self.device)[:m]
        return self.data[indices]


class TopicSimilarityTracker:
    """
    Tracks topic similarity between candidate/query texts and the held-out set
    using sentence-transformer cosine similarity.

    Lazy-loads the sentence-transformer model on first use.  Pre-computes and
    caches the held-out embedding(s) so each step only needs to embed the
    candidate texts.

    When the held-out set has a single element (the "generalized needle"
    setting), the similarity is a single scalar per candidate.  For larger
    held-out sets, we average over all held-out embeddings.
    """

    def __init__(
        self,
        held_out: HeldOutSet,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.held_out = held_out
        self._model_name = model_name
        self._device = device
        self._embed_model = None
        self._held_out_embeddings = None  # [|D|, embed_dim], L2-normalised

    def _ensure_model(self):
        """Lazy-load sentence-transformer and pre-embed held-out texts."""
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"[TopicSim] Loading sentence-transformer: {self._model_name}")
        self._embed_model = SentenceTransformer(self._model_name, device=self._device)
        # Pre-embed all held-out texts
        self._held_out_embeddings = self._embed_model.encode(
            self.held_out.texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # [|D|, embed_dim]
        print(f"[TopicSim] Held-out embeddings: {self._held_out_embeddings.shape}")
        # Mean held-out embedding (for single-number similarity)
        self._held_out_mean = self._held_out_embeddings.mean(axis=0)
        norm = np.linalg.norm(self._held_out_mean)
        if norm > 0:
            self._held_out_mean /= norm

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns [len(texts), embed_dim], L2-normalised."""
        self._ensure_model()
        return self._embed_model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def compute_similarities(self, texts: List[str]) -> dict:
        """
        Compute topic similarity between a list of candidate texts and the
        held-out set.

        Args:
            texts: List of decoded candidate strings.

        Returns:
            dict with:
                sims: np.ndarray [len(texts)] cosine similarity of each text
                      to the mean held-out embedding.
                best_idx: index of the most similar text.
                best_sim: similarity of the most similar text.
                mean_sim: mean similarity across all texts.
        """
        self._ensure_model()
        candidate_embs = self._embed_texts(texts)  # [N, embed_dim]
        # Cosine sim to mean held-out embedding (already L2-normalised)
        sims = candidate_embs @ self._held_out_mean  # [N]
        best_idx = int(np.argmax(sims))
        return {
            "sims": sims,
            "best_idx": best_idx,
            "best_sim": float(sims[best_idx]),
            "mean_sim": float(np.mean(sims)),
        }
