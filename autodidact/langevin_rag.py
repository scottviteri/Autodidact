"""
Langevin dynamics Q-guided search + RAG retrieval for curriculum selection.

Pipeline (per training step):
    1. Langevin sampling in embedding space:
       - Initialize K parallel chains of continuous embeddings [seq_len, d].
       - Run the GPT-2 transformer on these embeddings (via inputs_embeds),
         extract last-hidden-state, pass through Q-network to get scalar Q.
       - Backprop dQ/d(embeddings), then Langevin update:
             x += (step_size/2) * grad + sqrt(step_size) * noise
       - After burn-in steps, collect every thin-th sample.

    2. Snap to tokens:
       - For each collected embedding sample [seq_len, d], find the nearest
         token in GPT-2's wte embedding matrix at each position via cosine
         similarity -> discrete "query sentence".

    3. RAG retrieval:
       - Embed the query sentence with a sentence-transformer model.
       - Search a pre-built FAISS index of the training dataset.
       - Return top-k real training examples.

    4. The trainer trains on the retrieved examples with uniform (unweighted) loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

from .q_network import QNetwork


# ---------------------------------------------------------------------------
# 1. Langevin dynamics sampler in GPT-2 embedding space
# ---------------------------------------------------------------------------

class LangevinQSampler:
    """
    Uses Langevin dynamics to find input embeddings that maximise the Q-value.

    The energy function is E(x) = Q_phi(transformer(x)),  where x is a
    continuous tensor of shape [seq_len, d] living in the GPT-2 embedding
    space. Langevin dynamics samples from  p(x) ~ exp(E(x) / temperature).

    After burn-in, every `thin`-th sample is collected until `num_samples`
    samples have been gathered.  Each sample is then snapped to discrete
    tokens via nearest-neighbor lookup against the wte embedding matrix.
    """

    def __init__(
        self,
        model,                          # GPT2LMHeadModel
        q_net: QNetwork,                # Q-network (online, not target)
        seq_len: int = 128,
        num_chains: int = 8,            # K parallel chains
        num_samples: int = 8,           # total samples to collect
        langevin_steps: int = 100,      # total Langevin steps (burn-in + collection)
        burn_in: int = 50,              # discard first burn_in steps
        thin: int = 5,                  # keep every thin-th sample after burn-in
        step_size: float = 0.01,        # Langevin step size epsilon
        temperature: float = 1.0,       # sampling temperature (scales the energy)
        noise_scale: float = 1.0,       # multiplier on the Gaussian noise term
        grad_clip: float = 1.0,         # clip embedding gradients for stability
        langevin_batch_size: int = 4,   # chains to process in parallel in _energy()
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.q_net = q_net
        self.seq_len = seq_len
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.langevin_steps = langevin_steps
        self.burn_in = burn_in
        self.thin = thin
        self.step_size = step_size
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
        self.langevin_batch_size = langevin_batch_size
        self.device = device

        # Cache the token embedding matrix and its normalised version for NN lookup
        self._wte = model.transformer.wte.weight.detach()           # [V, d]
        self._wte_norm = F.normalize(self._wte, dim=-1)             # [V, d]
        self.hidden_dim = self._wte.shape[1]
        self.vocab_size = self._wte.shape[0]

    def _init_embeddings(self) -> torch.Tensor:
        """
        Initialise embedding chains by sampling random tokens and looking up
        their embeddings, so we start in a realistic region of embedding space.

        Returns: [K, seq_len, d]
        """
        random_ids = torch.randint(
            0, self.vocab_size, (self.num_chains, self.seq_len), device=self.device
        )
        with torch.no_grad():
            embeds = self.model.transformer.wte(random_ids)  # [K, seq_len, d]
        return embeds.clone()

    def _energy(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value (energy) for a batch of continuous embedding sequences.

        Processes chains in mini-batches of `langevin_batch_size` to balance
        GPU utilization against memory.  Gradient checkpointing is used so
        each transformer block recomputes activations during backward instead
        of storing them (trades ~2× compute for large memory savings).

        Args:
            embeds: [K, seq_len, d] continuous embeddings (requires_grad).

        Returns:
            q_vals: [K] scalar Q-values (differentiable w.r.t. embeds).
        """
        transformer = self.model.transformer
        bs = self.langevin_batch_size

        position_ids = torch.arange(self.seq_len, device=self.device).unsqueeze(0)
        position_embeds = transformer.wpe(position_ids)  # [1, seq_len, d]

        q_list = []
        for i in range(0, embeds.shape[0], bs):
            e_batch = embeds[i : i + bs]  # [bs, seq_len, d] — keeps grad connection
            hidden = e_batch + position_embeds

            with torch.enable_grad():
                hidden = transformer.drop(hidden)
                for block in transformer.h:
                    hidden = torch.utils.checkpoint.checkpoint(
                        block, hidden, use_reentrant=False
                    )
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                hidden = transformer.ln_f(hidden)

                h_last = hidden[:, -1, :]                  # [bs, d]
                q_batch = self.q_net(h_last).squeeze(-1)   # [bs]
            q_list.append(q_batch)

        return torch.cat(q_list, dim=0)  # [K]

    @torch.no_grad()
    def _snap_to_tokens(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Map continuous embeddings to discrete tokens via nearest-neighbor
        cosine similarity against the wte embedding matrix.

        Processes in chunks to avoid allocating a [B*seq_len, vocab_size]
        similarity matrix all at once.

        Args:
            embeds: [B, seq_len, d]

        Returns:
            token_ids: [B, seq_len]  (dtype=long)
        """
        B, S, d = embeds.shape
        flat = embeds.reshape(B * S, d)                          # [B*S, d]
        flat_norm = F.normalize(flat, dim=-1)                     # [B*S, d]

        # Process in chunks of 256 positions to keep memory bounded
        chunk_size = 256
        ids_list = []
        for start in range(0, flat_norm.shape[0], chunk_size):
            chunk = flat_norm[start : start + chunk_size]         # [chunk, d]
            sims = chunk @ self._wte_norm.T                       # [chunk, V]
            ids_list.append(sims.argmax(dim=-1))                  # [chunk]
        ids = torch.cat(ids_list, dim=0)                          # [B*S]
        return ids.reshape(B, S)

    def sample(self) -> Tuple[torch.Tensor, dict]:
        """
        Run Langevin dynamics and return discrete query token sequences
        along with rich diagnostics for monitoring SGLD health.

        Returns:
            query_ids: [num_samples, seq_len]  discrete token IDs
            info: dict with comprehensive diagnostics:
                Timing:
                  sgld_time_s: wall-clock seconds for the full Langevin phase
                SGLD Q-value trajectory:
                  q_history: list of [K] Q-values at each Langevin step
                  q_init_mean/std: Q-values at initialisation (step 0)
                  q_final_mean/std: Q-values of collected samples
                  q_best: highest Q-value seen across all steps and chains
                  q_gain: q_final_mean - q_init_mean (how much SGLD improved)
                Unbiased baseline:
                  q_random_mean/std: Q-values of fresh random embeddings
                    (evaluated at the end, same cost as one energy call)
                Diversity / mixing:
                  embed_pairwise_cosine_mean: avg cosine sim between all
                    pairs of collected embedding vectors (lower = more diverse)
                  token_unique_ratio: fraction of unique token sequences
                    among collected samples (1.0 = all different)
                  token_jaccard_mean: avg pairwise Jaccard similarity of
                    token sets across samples (lower = more diverse)
                Gradient health:
                  grad_norm_mean/std: avg/std of per-chain gradient norms
                    across all Langevin steps
                  grad_clip_frac: fraction of steps where at least one
                    chain's gradient was clipped
                Collection:
                  num_collected: number of samples collected
                  num_collections: number of collection events
                  langevin_steps_run: total Langevin steps executed
                Best sample:
                  best_q_sample_idx: index of the highest-Q sample
        """
        import time as _time

        t0 = _time.time()

        # Freeze model + Q-net parameters (we only optimise embeddings)
        model_training = self.model.training
        self.model.eval()
        self.q_net.eval()

        embeds = self._init_embeddings()  # [K, seq_len, d]
        embeds.requires_grad_(True)

        # --- Record initial Q-values ---
        with torch.no_grad():
            q_init = self._energy(embeds)  # [K]
        q_init_mean = q_init.mean().item()
        q_init_std = q_init.std().item() if q_init.numel() > 1 else 0.0

        collected = []          # list of [K, seq_len, d] snapshots
        q_history = []          # list of [K] Q-value lists per step
        grad_norms_history = [] # list of [K] gradient norms per step
        grad_clipped_count = 0  # steps where any chain was clipped

        eps = self.step_size
        temp = self.temperature
        q_best_ever = float("-inf")

        for t in range(self.langevin_steps):
            # --- Compute energy and gradient ---
            if embeds.grad is not None:
                embeds.grad.zero_()

            q_vals = self._energy(embeds)           # [K]
            energy = (q_vals / temp).sum()           # scalar (sum over chains)
            energy.backward()

            grad = embeds.grad.detach()              # [K, seq_len, d]

            # Track gradient norms before clipping
            raw_grad_norms = grad.reshape(self.num_chains, -1).norm(dim=-1)  # [K]
            grad_norms_history.append(raw_grad_norms.cpu().tolist())

            # Clip gradients per-chain for stability
            grad_norms_col = raw_grad_norms.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)  # [K, 1, 1]
            scale = torch.clamp(self.grad_clip / grad_norms_col, max=1.0)
            was_clipped = (raw_grad_norms > self.grad_clip).any().item()
            if was_clipped:
                grad_clipped_count += 1
            grad = grad * scale

            # --- Langevin update: x += eps/2 * grad + sqrt(eps) * noise ---
            noise = torch.randn_like(embeds) * self.noise_scale
            with torch.no_grad():
                embeds.add_(0.5 * eps * grad + math.sqrt(eps) * noise)

            # Re-enable grad tracking for next step
            embeds.requires_grad_(True)

            q_detached = q_vals.detach().cpu().tolist()
            q_history.append(q_detached)
            step_max_q = max(q_detached)
            if step_max_q > q_best_ever:
                q_best_ever = step_max_q

            # --- Collection after burn-in ---
            if t >= self.burn_in and (t - self.burn_in) % self.thin == 0:
                collected.append(embeds.detach().clone())

            if len(collected) * self.num_chains >= self.num_samples:
                break

        # Restore model state
        if model_training:
            self.model.train()
        self.q_net.train()

        if not collected:
            collected.append(embeds.detach().clone())

        num_collections = len(collected)

        # Stack collected samples: [n_collected * K, seq_len, d]
        all_embeds = torch.cat(collected, dim=0)[:self.num_samples]
        del collected, embeds

        # --- Snap to discrete tokens ---
        query_ids = self._snap_to_tokens(all_embeds)

        # --- Final Q-values of collected samples ---
        with torch.no_grad():
            final_q = self._energy(all_embeds)
        q_final_mean = final_q.mean().item()
        q_final_std = final_q.std().item() if final_q.numel() > 1 else 0.0

        best_q_sample_idx = final_q.argmax().item()

        # --- Unbiased baseline: Q-values of fresh random embeddings ---
        random_embeds = self._init_embeddings()  # [K, seq_len, d]
        with torch.no_grad():
            q_random = self._energy(random_embeds)
        q_random_mean = q_random.mean().item()
        q_random_std = q_random.std().item() if q_random.numel() > 1 else 0.0
        del random_embeds, q_random

        # --- Diversity: pairwise cosine similarity of collected embeddings ---
        n_samples = all_embeds.shape[0]
        if n_samples > 1:
            flat = all_embeds.reshape(n_samples, -1)  # [S, seq_len*d]
            flat_norm = F.normalize(flat, dim=-1)
            cos_matrix = flat_norm @ flat_norm.T       # [S, S]
            # Extract upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones(n_samples, n_samples, device=self.device), diagonal=1).bool()
            pairwise_cos = cos_matrix[mask]
            embed_pairwise_cosine_mean = pairwise_cos.mean().item()
        else:
            embed_pairwise_cosine_mean = 0.0

        # --- Diversity: unique token sequences and Jaccard similarity ---
        token_sets = [set(query_ids[i].cpu().tolist()) for i in range(query_ids.shape[0])]
        unique_seqs = set(tuple(query_ids[i].cpu().tolist()) for i in range(query_ids.shape[0]))
        token_unique_ratio = len(unique_seqs) / max(query_ids.shape[0], 1)

        if len(token_sets) > 1:
            jaccard_sum = 0.0
            jaccard_count = 0
            for i in range(len(token_sets)):
                for j in range(i + 1, len(token_sets)):
                    inter = len(token_sets[i] & token_sets[j])
                    union = len(token_sets[i] | token_sets[j])
                    jaccard_sum += inter / max(union, 1)
                    jaccard_count += 1
            token_jaccard_mean = jaccard_sum / max(jaccard_count, 1)
        else:
            token_jaccard_mean = 0.0

        # --- Gradient norm statistics ---
        all_gnorms = [g for step_g in grad_norms_history for g in step_g]
        grad_norm_mean = sum(all_gnorms) / max(len(all_gnorms), 1)
        grad_norm_std = (sum((g - grad_norm_mean) ** 2 for g in all_gnorms) / max(len(all_gnorms), 1)) ** 0.5
        total_steps_run = len(q_history)
        grad_clip_frac = grad_clipped_count / max(total_steps_run, 1)

        sgld_time = _time.time() - t0

        # Free embedding tensors
        del all_embeds, final_q

        info = {
            # Timing
            "sgld_time_s": sgld_time,
            # Q-value trajectory
            "q_init_mean": q_init_mean,
            "q_init_std": q_init_std,
            "q_final_mean": q_final_mean,
            "q_final_std": q_final_std,
            "q_best": q_best_ever,
            "q_gain": q_final_mean - q_init_mean,
            # Unbiased baseline
            "q_random_mean": q_random_mean,
            "q_random_std": q_random_std,
            # Diversity
            "embed_pairwise_cosine_mean": embed_pairwise_cosine_mean,
            "token_unique_ratio": token_unique_ratio,
            "token_jaccard_mean": token_jaccard_mean,
            # Gradient health
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_std": grad_norm_std,
            "grad_clip_frac": grad_clip_frac,
            # Collection
            "num_collected": n_samples,
            "num_collections": num_collections,
            "langevin_steps_run": total_steps_run,
            # Best sample
            "best_q_sample_idx": best_q_sample_idx,
        }

        return query_ids, info


# ---------------------------------------------------------------------------
# 2. RAG index: embed dataset + retrieve by query similarity
# ---------------------------------------------------------------------------

class RAGIndex:
    """
    Builds and queries a FAISS index over sentence embeddings of a text dataset.

    Uses sentence-transformers (all-MiniLM-L6-v2 by default) for encoding.
    The index is built once at startup from the streaming dataset, then
    queried each step with the Langevin-generated query sentences.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_size: int = 50000,
        retrieval_top_k: int = 8,
        embed_batch_size: int = 256,
        device: torch.device = torch.device("cpu"),
    ):
        self.retrieval_top_k = retrieval_top_k
        self.embed_batch_size = embed_batch_size
        self.device = device
        self.index_size = index_size

        # Load sentence-transformer
        from sentence_transformers import SentenceTransformer
        print(f"[RAG] Loading embedding model: {embedding_model_name}")
        self.embed_model = SentenceTransformer(embedding_model_name, device=str(device))
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        print(f"[RAG] Embedding dimension: {self.embed_dim}")

        # Will be populated by build_index()
        self.index = None               # FAISS index
        self.stored_token_ids = None     # [index_size, seq_len] token IDs
        self.stored_texts = None         # list of decoded text strings

    def build_index(self, tokenizer, seq_len: int, dataset_name: str, split: str = "train"):
        """
        Stream through the dataset, tokenize into fixed-length windows,
        embed each window's text, and build a FAISS index.

        Args:
            tokenizer: GPT-2 tokenizer for decoding windows back to text.
            seq_len: Token window length.
            dataset_name: HuggingFace dataset name.
            split: Dataset split.
        """
        import faiss
        from datasets import load_dataset

        print(f"[RAG] Building index from {dataset_name} ({self.index_size} windows)...")

        dataset = load_dataset(dataset_name, split=split, streaming=True)
        buffer = []
        windows_tokens = []   # list of [seq_len] token lists
        windows_text = []     # list of decoded strings

        for example in dataset:
            text = example["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= seq_len and len(windows_tokens) < self.index_size:
                window_tok = buffer[:seq_len]
                buffer = buffer[seq_len:]
                windows_tokens.append(window_tok)
                windows_text.append(tokenizer.decode(window_tok))
            if len(windows_tokens) >= self.index_size:
                break

        n = len(windows_tokens)
        print(f"[RAG] Collected {n} windows. Embedding...")

        # Embed all windows in batches
        all_embeddings = self.embed_model.encode(
            windows_text,
            batch_size=self.embed_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  # [n, embed_dim]

        # Build FAISS index (inner product on normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(all_embeddings.astype(np.float32))

        self.stored_token_ids = torch.tensor(windows_tokens, dtype=torch.long)  # [n, seq_len]
        self.stored_texts = windows_text

        print(f"[RAG] Index built: {self.index.ntotal} vectors, dim={self.embed_dim}")

    def retrieve(
        self,
        query_ids: torch.Tensor,
        tokenizer,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Retrieve training examples from the index using query token sequences.

        For each query, decode to text, embed, search the FAISS index, and
        return the top-k results. Results are deduplicated across all queries.

        Args:
            query_ids: [B, seq_len] discrete token IDs (from Langevin snap).
            tokenizer: GPT-2 tokenizer for decoding.
            top_k: Number of results per query (default: self.retrieval_top_k).

        Returns:
            retrieved_ids: [R, seq_len] token IDs of retrieved examples.
            info: dict with retrieval diagnostics.
        """
        if top_k is None:
            top_k = self.retrieval_top_k

        # Decode queries to text
        query_texts = []
        for i in range(query_ids.shape[0]):
            text = tokenizer.decode(query_ids[i].tolist())
            query_texts.append(text)

        # Embed queries
        query_embeddings = self.embed_model.encode(
            query_texts,
            batch_size=self.embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  # [B, embed_dim]

        # Search FAISS index
        scores, indices = self.index.search(query_embeddings.astype(np.float32), top_k)
        # scores: [B, top_k], indices: [B, top_k]

        # Deduplicate retrieved indices across all queries
        unique_indices = list(set(indices.flatten().tolist()))
        # Remove -1 (FAISS sentinel for not-found)
        unique_indices = [idx for idx in unique_indices if idx >= 0]

        retrieved_ids = self.stored_token_ids[unique_indices]  # [R, seq_len]

        info = {
            "num_queries": len(query_texts),
            "num_retrieved_unique": len(unique_indices),
            "avg_top1_score": float(scores[:, 0].mean()),
            "avg_topk_score": float(scores.mean()),
            "sample_query": query_texts[0][:200] if query_texts else "",
        }

        return retrieved_ids, info

