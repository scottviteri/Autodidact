"""
Softmax-relaxed Langevin dynamics + RAG retrieval for curriculum selection.

Pipeline (per training step):
    1. Langevin sampling in logit space (softmax-relaxed SGLD):
       - Maintain K parallel chains, each a tensor of logits [seq_len, V]
         over the vocabulary at every token position.
       - At each Langevin step, form soft token embeddings via
             e = softmax(logits / tau) @ W_e
         which is always in the convex hull of real token embeddings.
       - Feed soft embeddings through the GPT-2 transformer, extract the
         last-layer hidden state, pass through the Q-network to get Q.
       - Backprop dQ/d(logits) and apply the Langevin update.
       - After burn-in, collect samples; extract tokens via argmax(logits).

    2. Token extraction (no snap needed):
       - tokens = argmax(logits, dim=-1) at each position.
       - Because the logits parameterise a categorical distribution that
         sharpens during SGLD, the argmax tokens are the dominant modes
         of the optimised distribution — not a lossy nearest-neighbor.

    3. RAG retrieval:
       - Embed the query sentences with a sentence-transformer model.
       - Search a pre-built FAISS index of the training dataset.
       - Return top-k real training examples.

    4. The trainer trains on the retrieved examples (uniform or Q-weighted).

The key advantage over raw-embedding SGLD: the softmax constraint keeps
the input to the transformer on the token manifold at all times, so the
Q-network's gradients are informative about real tokens rather than
arbitrary directions in R^d.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

from .q_network import QNetwork


# ---------------------------------------------------------------------------
# 1. Softmax-relaxed Langevin dynamics sampler
# ---------------------------------------------------------------------------

class SoftmaxQSampler:
    """
    Uses Langevin dynamics in **logit space** to find token distributions
    that maximise the Q-value, with a softmax-to-embedding bridge ensuring
    the transformer always receives inputs in the convex hull of real tokens.

    Optimisation variable:
        logits: [K, seq_len, V]   (unconstrained reals)

    Forward pass (differentiable):
        soft_tokens = softmax(logits / tau)          [K, seq_len, V]
        soft_embeds = soft_tokens @ W_e              [K, seq_len, d]
        hidden      = GPT2_transformer(soft_embeds)
        Q           = Q_head(hidden[:, -1, :])       [K]

    Langevin update:
        logits += eps/2 * dQ/d(logits) + sqrt(eps) * noise

    Token extraction:
        tokens = argmax(logits, dim=-1)              [K, seq_len]

    The softmax temperature tau is annealed from tau_start (high,
    exploratory) to tau_end (low, sharp) over the Langevin steps,
    smoothly transitioning from a broad search to a peaked selection.

    Note: despite the historical name "Gumbel-Softmax" in earlier versions,
    this method uses plain softmax (no Gumbel noise). The gradient path
    logits -> softmax -> embeddings -> transformer -> Q is fully
    differentiable, so there is no need for the Gumbel reparameterisation
    trick. The softmax serves purely as a differentiable bridge from
    unconstrained logits to the token-embedding manifold.
    """

    def __init__(
        self,
        model,                          # GPT2LMHeadModel
        q_net: QNetwork,                # Q-network (online, not target)
        seq_len: int = 64,
        num_chains: int = 64,           # K parallel chains = K output samples
        burn_in: int = 40,              # number of Langevin steps before collecting
        step_size: float = 0.01,        # Langevin step size epsilon
        temperature: float = 1.0,       # SGLD energy temperature (scales Q in energy)
        noise_scale: float = 1.0,       # multiplier on the Gaussian noise term
        grad_clip: float = 1.0,         # clip logit gradients per chain
        langevin_batch_size: int = 64,  # chains to process in parallel in _energy()
        # Softmax bridge temperature annealing
        softmax_tau_start: float = 2.0, # softmax temperature at step 0 (high = soft)
        softmax_tau_end: float = 0.5,   # softmax temperature at final step (low = sharp)
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.q_net = q_net
        self.seq_len = seq_len
        self.num_chains = num_chains
        self.burn_in = burn_in
        self.step_size = step_size
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
        self.langevin_batch_size = langevin_batch_size
        self.softmax_tau_start = softmax_tau_start
        self.softmax_tau_end = softmax_tau_end
        self.device = device

        # Cache the token embedding matrix (frozen copy for the softmax @ wte bridge)
        self._wte = model.transformer.wte.weight.detach()  # [V, d]
        self.hidden_dim = self._wte.shape[1]
        self.vocab_size = self._wte.shape[0]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_logits(self) -> torch.Tensor:
        """
        Initialise logit chains from random one-hot tokens.

        Each chain starts as a sharp distribution centred on a random token
        at each position (logit = +10 for the chosen token, 0 elsewhere).
        This ensures the initial soft embeddings are very close to real
        token embeddings, starting SGLD on the token manifold.

        Returns: [K, seq_len, V]
        """
        random_ids = torch.randint(
            0, self.vocab_size, (self.num_chains, self.seq_len), device=self.device
        )
        logits = torch.zeros(
            self.num_chains, self.seq_len, self.vocab_size, device=self.device
        )
        # Set high logit at the randomly chosen token so softmax ≈ one-hot
        logits.scatter_(-1, random_ids.unsqueeze(-1), 10.0)
        return logits

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def _energy(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Compute Q-value (energy) for a batch of logit sequences.

        The forward path is:
            soft_tokens = softmax(logits / tau)     [K, seq_len, V]
            soft_embeds = soft_tokens @ W_e          [K, seq_len, d]
            hidden      = transformer(soft_embeds + pos_embeds)
            Q           = Q_head(hidden[:, -1, :])   [K]

        Processes chains in mini-batches for memory efficiency.
        Uses gradient checkpointing for transformer blocks.

        Args:
            logits: [K, seq_len, V] (requires_grad).
            tau: softmax temperature for the softmax bridge.

        Returns:
            q_vals: [K] scalar Q-values (differentiable w.r.t. logits).
        """
        transformer = self.model.transformer
        bs = self.langevin_batch_size

        position_ids = torch.arange(self.seq_len, device=self.device).unsqueeze(0)
        position_embeds = transformer.wpe(position_ids)  # [1, seq_len, d]

        q_list = []
        for i in range(0, logits.shape[0], bs):
            logit_batch = logits[i : i + bs]  # [bs, seq_len, V]

            with torch.enable_grad():
                # Softmax bridge: logits -> soft tokens -> soft embeddings
                soft_tokens = F.softmax(logit_batch / tau, dim=-1)  # [bs, seq_len, V]
                soft_embeds = soft_tokens @ self._wte               # [bs, seq_len, d]

                hidden = soft_embeds + position_embeds
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

    # ------------------------------------------------------------------
    # Token extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Extract discrete tokens from logits via argmax.

        Unlike the old snap-to-tokens, this is exact: the argmax token IS
        the dominant mode of the softmax distribution, not a lossy
        nearest-neighbor in embedding space.

        Args:
            logits: [B, seq_len, V]

        Returns:
            token_ids: [B, seq_len] (dtype=long)
        """
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    def sample(self) -> Tuple[torch.Tensor, dict]:
        """
        Run softmax-relaxed Langevin dynamics and return discrete query
        token sequences along with rich diagnostics.

        Returns:
            query_ids: [num_samples, seq_len]  discrete token IDs
            info: dict with diagnostics (compatible with LangevinQSampler):
                Timing:
                  sgld_time_s
                Q-value trajectory:
                  q_init_mean/std, q_final_mean/std, q_best, q_gain
                Unbiased baseline:
                  q_random_mean/std
                Softmax sharpness (replaces snap_cosine_mean):
                  softmax_entropy_mean: avg entropy of softmax(logits/tau)
                    across positions and samples (lower = sharper = more
                    decided). For reference, uniform = log(V) ≈ 10.8.
                  softmax_max_prob_mean: avg max probability across positions
                    (higher = more peaked). 1.0 = perfectly sharp.
                  snap_cosine_mean: set to softmax_max_prob_mean for
                    backward compatibility with dashboard/logging.
                Diversity:
                  embed_pairwise_cosine_mean, token_jaccard_mean
                Gradient health:
                  grad_norm_mean/std, grad_clip_frac
                Collection:
                  num_collected, num_collections, langevin_steps_run
                Best sample:
                  best_q_sample_idx
                Temperature:
                  final_tau: the softmax temperature at collection
        """
        import time as _time

        t0 = _time.time()

        # Freeze model + Q-net parameters (we only optimise logits)
        model_training = self.model.training
        self.model.eval()
        self.q_net.eval()

        logits = self._init_logits()  # [K, seq_len, V]
        logits.requires_grad_(True)

        # Temperature annealing schedule (linear from tau_start to tau_end)
        tau_start = self.softmax_tau_start
        tau_end = self.softmax_tau_end
        n_steps = self.burn_in

        def _get_tau(t):
            if n_steps <= 1:
                return tau_end
            frac = t / (n_steps - 1)
            return tau_start + (tau_end - tau_start) * frac

        # --- Record initial Q-values ---
        tau_0 = _get_tau(0)
        with torch.no_grad():
            q_init = self._energy(logits, tau_0)  # [K]
        q_init_mean = q_init.mean().item()
        q_init_std = q_init.std().item() if q_init.numel() > 1 else 0.0

        q_history = []          # list of [K] Q-value lists per step
        grad_norms_history = [] # list of [K] gradient norms per step
        grad_clipped_count = 0  # steps where any chain was clipped

        eps = self.step_size
        temp = self.temperature
        q_best_ever = float("-inf")

        # --- Run burn_in Langevin steps ---
        for t in range(n_steps):
            tau_t = _get_tau(t)

            # --- Compute energy and gradient ---
            if logits.grad is not None:
                logits.grad.zero_()

            q_vals = self._energy(logits, tau_t)      # [K]
            energy = (q_vals / temp).sum()             # scalar (sum over chains)
            energy.backward()

            grad = logits.grad.detach()                # [K, seq_len, V]

            # Track gradient norms before clipping (flatten per chain)
            raw_grad_norms = grad.reshape(self.num_chains, -1).norm(dim=-1)  # [K]
            grad_norms_history.append(raw_grad_norms.cpu().tolist())

            # Clip gradients per-chain for stability
            grad_norms_col = raw_grad_norms.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
            scale = torch.clamp(self.grad_clip / grad_norms_col, max=1.0)
            was_clipped = (raw_grad_norms > self.grad_clip).any().item()
            if was_clipped:
                grad_clipped_count += 1
            grad = grad * scale

            # --- Langevin update: logits += eps/2 * grad + sqrt(eps) * noise ---
            noise = torch.randn_like(logits) * self.noise_scale
            with torch.no_grad():
                logits.add_(0.5 * eps * grad + math.sqrt(eps) * noise)

            # Re-enable grad tracking for next step
            logits.requires_grad_(True)

            q_detached = q_vals.detach().cpu().tolist()
            q_history.append(q_detached)
            step_max_q = max(q_detached)
            if step_max_q > q_best_ever:
                q_best_ever = step_max_q

        # Restore model state
        if model_training:
            self.model.train()
        self.q_net.train()

        # --- Collect: one sample per chain ---
        # all_logits is [K, seq_len, V] — each chain produces exactly one sample
        all_logits = logits.detach()

        # Determine the tau at collection time (final step)
        final_tau = _get_tau(max(n_steps - 1, 0))

        # --- Extract discrete tokens (just argmax — no snap!) ---
        query_ids = self._extract_tokens(all_logits)

        # --- Softmax sharpness diagnostics ---
        with torch.no_grad():
            probs = F.softmax(all_logits / final_tau, dim=-1)  # [S, seq_len, V]
            # Entropy per position: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)  # [S, seq_len]
            softmax_entropy_mean = entropy.mean().item()
            # Max probability per position
            max_prob = probs.max(dim=-1).values  # [S, seq_len]
            softmax_max_prob_mean = max_prob.mean().item()

        # --- Final Q-values of collected samples ---
        with torch.no_grad():
            final_q = self._energy(all_logits, final_tau)
        q_final_mean = final_q.mean().item()
        q_final_std = final_q.std().item() if final_q.numel() > 1 else 0.0

        best_q_sample_idx = final_q.argmax().item()

        # --- Unbiased baseline: Q-values of fresh random logits ---
        random_logits = self._init_logits()
        with torch.no_grad():
            q_random = self._energy(random_logits, final_tau)
        q_random_mean = q_random.mean().item()
        q_random_std = q_random.std().item() if q_random.numel() > 1 else 0.0
        del random_logits, q_random

        # --- Diversity: pairwise cosine similarity of soft embeddings ---
        K = all_logits.shape[0]
        if K > 1:
            with torch.no_grad():
                soft_embeds = F.softmax(all_logits / final_tau, dim=-1) @ self._wte
                flat = soft_embeds.reshape(K, -1)
                flat_norm = F.normalize(flat, dim=-1)
                cos_matrix = flat_norm @ flat_norm.T
                mask = torch.triu(
                    torch.ones(K, K, device=self.device), diagonal=1
                ).bool()
                pairwise_cos = cos_matrix[mask]
                embed_pairwise_cosine_mean = pairwise_cos.mean().item()
                del soft_embeds, flat, flat_norm, cos_matrix
        else:
            embed_pairwise_cosine_mean = 0.0

        # --- Diversity: pairwise Jaccard similarity of token sets ---
        token_sets = [set(query_ids[i].cpu().tolist()) for i in range(query_ids.shape[0])]
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
        grad_norm_std = (
            sum((g - grad_norm_mean) ** 2 for g in all_gnorms)
            / max(len(all_gnorms), 1)
        ) ** 0.5
        total_steps_run = len(q_history)
        grad_clip_frac = grad_clipped_count / max(total_steps_run, 1)

        sgld_time = _time.time() - t0

        # Free large tensors
        del all_logits, final_q, logits

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
            # Softmax sharpness (replaces snap_cosine_mean)
            "softmax_entropy_mean": softmax_entropy_mean,
            "softmax_max_prob_mean": softmax_max_prob_mean,
            # Backward compat: dashboard reads snap_cosine_mean
            "snap_cosine_mean": softmax_max_prob_mean,
            # Diversity
            "embed_pairwise_cosine_mean": embed_pairwise_cosine_mean,
            "token_jaccard_mean": token_jaccard_mean,
            # Gradient health
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_std": grad_norm_std,
            "grad_clip_frac": grad_clip_frac,
            # Collection
            "num_collected": K,
            "langevin_steps_run": total_steps_run,
            # Best sample
            "best_q_sample_idx": best_q_sample_idx,
            # Temperature
            "final_tau": final_tau,
        }

        return query_ids, info


# ---------------------------------------------------------------------------
# Legacy: original embedding-space sampler (kept for comparison)
# ---------------------------------------------------------------------------

class LangevinQSampler:
    """
    [LEGACY] Langevin dynamics in raw embedding space with nearest-neighbor
    snap-to-tokens. Superseded by SoftmaxQSampler which avoids the
    snap bottleneck by operating in logit space.

    Uses Langevin dynamics to find input embeddings that maximise the Q-value.
    The energy function is E(x) = Q_phi(transformer(x)), where x is a
    continuous tensor of shape [seq_len, d] in the GPT-2 embedding space.
    """

    def __init__(
        self,
        model,
        q_net: QNetwork,
        seq_len: int = 128,
        num_chains: int = 8,
        burn_in: int = 50,
        step_size: float = 0.01,
        temperature: float = 1.0,
        noise_scale: float = 1.0,
        grad_clip: float = 1.0,
        langevin_batch_size: int = 4,
        device: torch.device = torch.device("cpu"),
        # Ignored softmax params (for interface compat)
        softmax_tau_start: float = 2.0,
        softmax_tau_end: float = 0.5,
    ):
        self.model = model
        self.q_net = q_net
        self.seq_len = seq_len
        self.num_chains = num_chains
        self.burn_in = burn_in
        self.step_size = step_size
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
        self.langevin_batch_size = langevin_batch_size
        self.device = device

        self._wte = model.transformer.wte.weight.detach()
        self._wte_norm = F.normalize(self._wte, dim=-1)
        self.hidden_dim = self._wte.shape[1]
        self.vocab_size = self._wte.shape[0]

    def _init_embeddings(self) -> torch.Tensor:
        random_ids = torch.randint(
            0, self.vocab_size, (self.num_chains, self.seq_len), device=self.device
        )
        with torch.no_grad():
            embeds = self.model.transformer.wte(random_ids)
        return embeds.clone()

    def _energy(self, embeds: torch.Tensor) -> torch.Tensor:
        transformer = self.model.transformer
        bs = self.langevin_batch_size
        position_ids = torch.arange(self.seq_len, device=self.device).unsqueeze(0)
        position_embeds = transformer.wpe(position_ids)
        q_list = []
        for i in range(0, embeds.shape[0], bs):
            e_batch = embeds[i : i + bs]
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
                h_last = hidden[:, -1, :]
                q_batch = self.q_net(h_last).squeeze(-1)
            q_list.append(q_batch)
        return torch.cat(q_list, dim=0)

    @torch.no_grad()
    def _snap_to_tokens(self, embeds):
        B, S, d = embeds.shape
        flat = embeds.reshape(B * S, d)
        flat_norm = F.normalize(flat, dim=-1)
        chunk_size = 256
        ids_list, max_sims_list = [], []
        for start in range(0, flat_norm.shape[0], chunk_size):
            chunk = flat_norm[start : start + chunk_size]
            sims = chunk @ self._wte_norm.T
            max_sims, max_ids = sims.max(dim=-1)
            ids_list.append(max_ids)
            max_sims_list.append(max_sims)
        ids = torch.cat(ids_list, dim=0)
        max_sims = torch.cat(max_sims_list, dim=0)
        return ids.reshape(B, S), max_sims.reshape(B, S)

    def sample(self) -> Tuple[torch.Tensor, dict]:
        import time as _time
        t0 = _time.time()
        model_training = self.model.training
        self.model.eval()
        self.q_net.eval()

        embeds = self._init_embeddings()
        embeds.requires_grad_(True)
        with torch.no_grad():
            q_init = self._energy(embeds)
        q_init_mean = q_init.mean().item()
        q_init_std = q_init.std().item() if q_init.numel() > 1 else 0.0

        collected = []
        q_history = []
        grad_norms_history = []
        grad_clipped_count = 0
        eps = self.step_size
        temp = self.temperature
        q_best_ever = float("-inf")

        for t in range(self.burn_in):
            if embeds.grad is not None:
                embeds.grad.zero_()
            q_vals = self._energy(embeds)
            energy = (q_vals / temp).sum()
            energy.backward()
            grad = embeds.grad.detach()
            raw_grad_norms = grad.reshape(self.num_chains, -1).norm(dim=-1)
            grad_norms_history.append(raw_grad_norms.cpu().tolist())
            grad_norms_col = raw_grad_norms.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
            scale = torch.clamp(self.grad_clip / grad_norms_col, max=1.0)
            was_clipped = (raw_grad_norms > self.grad_clip).any().item()
            if was_clipped:
                grad_clipped_count += 1
            grad = grad * scale
            noise = torch.randn_like(embeds) * self.noise_scale
            with torch.no_grad():
                embeds.add_(0.5 * eps * grad + math.sqrt(eps) * noise)
            embeds.requires_grad_(True)
            q_detached = q_vals.detach().cpu().tolist()
            q_history.append(q_detached)
            step_max_q = max(q_detached)
            if step_max_q > q_best_ever:
                q_best_ever = step_max_q

        if model_training:
            self.model.train()
        self.q_net.train()

        # Collect: one sample per chain
        all_embeds = embeds.detach()
        del embeds

        query_ids, snap_cosine_sims = self._snap_to_tokens(all_embeds)
        snap_cosine_mean = snap_cosine_sims.mean().item()
        with torch.no_grad():
            final_q = self._energy(all_embeds)
        q_final_mean = final_q.mean().item()
        q_final_std = final_q.std().item() if final_q.numel() > 1 else 0.0
        best_q_sample_idx = final_q.argmax().item()

        random_embeds = self._init_embeddings()
        with torch.no_grad():
            q_random = self._energy(random_embeds)
        q_random_mean = q_random.mean().item()
        q_random_std = q_random.std().item() if q_random.numel() > 1 else 0.0
        del random_embeds, q_random

        n_samples = all_embeds.shape[0]
        if n_samples > 1:
            flat = all_embeds.reshape(n_samples, -1)
            flat_norm = F.normalize(flat, dim=-1)
            cos_matrix = flat_norm @ flat_norm.T
            mask = torch.triu(torch.ones(n_samples, n_samples, device=self.device), diagonal=1).bool()
            embed_pairwise_cosine_mean = cos_matrix[mask].mean().item()
        else:
            embed_pairwise_cosine_mean = 0.0

        token_sets = [set(query_ids[i].cpu().tolist()) for i in range(query_ids.shape[0])]
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

        all_gnorms = [g for step_g in grad_norms_history for g in step_g]
        grad_norm_mean = sum(all_gnorms) / max(len(all_gnorms), 1)
        grad_norm_std = (sum((g - grad_norm_mean) ** 2 for g in all_gnorms) / max(len(all_gnorms), 1)) ** 0.5
        total_steps_run = len(q_history)
        grad_clip_frac = grad_clipped_count / max(total_steps_run, 1)
        sgld_time = _time.time() - t0
        del all_embeds, final_q

        info = {
            "sgld_time_s": sgld_time,
            "q_init_mean": q_init_mean, "q_init_std": q_init_std,
            "q_final_mean": q_final_mean, "q_final_std": q_final_std,
            "q_best": q_best_ever, "q_gain": q_final_mean - q_init_mean,
            "q_random_mean": q_random_mean, "q_random_std": q_random_std,
            "embed_pairwise_cosine_mean": embed_pairwise_cosine_mean,
            "snap_cosine_mean": snap_cosine_mean,
            "token_jaccard_mean": token_jaccard_mean,
            "grad_norm_mean": grad_norm_mean, "grad_norm_std": grad_norm_std,
            "grad_clip_frac": grad_clip_frac,
            "num_collected": n_samples,
            "langevin_steps_run": total_steps_run,
            "best_q_sample_idx": best_q_sample_idx,
            "softmax_entropy_mean": 0.0,
            "softmax_max_prob_mean": snap_cosine_mean,
            "final_tau": 0.0,
        }
        return query_ids, info


# ---------------------------------------------------------------------------
# 2. RAG index: embed dataset + retrieve by query similarity
# ---------------------------------------------------------------------------

class RAGIndex:
    """
    Builds and queries a FAISS index over sentence embeddings of a text dataset.

    Uses sentence-transformers (all-MiniLM-L6-v2 by default) for encoding.
    The index is built from pre-consumed data (provided by the caller, typically
    a DataStream), then queried each step with the Langevin-generated query
    sentences.

    The index is maintained as a **ring buffer**: fresh windows are continuously
    ingested via ingest(), overwriting the oldest entries. The FAISS index is
    rebuilt after each ingest (IndexFlatIP rebuild for 50K vectors is <5ms).
    This "rolling refresh" avoids the LM loss jumps caused by atomically
    replacing the entire index, while keeping the data fresh.
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

        # Ring buffer state (initialized by build_index)
        self._embeddings = None          # np.ndarray [index_size, embed_dim]
        self._token_ids = None           # torch.Tensor [index_size, seq_len]
        self._texts = None               # list of str, length index_size
        self._write_head = 0             # next ring buffer position to overwrite
        self._ingested_since_build = 0   # windows ingested via ingest() since build

        # Public aliases (used by trainer warmup + retrieve)
        self.index = None                # FAISS index
        self.stored_token_ids = None     # alias to _token_ids
        self.stored_texts = None         # alias to _texts

    def build_index(self, windows_tokens: list, windows_text: list):
        """
        Build the initial FAISS index and initialize the ring buffer.

        Args:
            windows_tokens: list of token lists, each of length seq_len.
            windows_text:   list of decoded text strings (same length).
        """
        import faiss

        n = len(windows_tokens)
        print(f"[RAG] Building index from {n} pre-consumed windows. Embedding...")

        self._embeddings = self.embed_model.encode(
            windows_text,
            batch_size=self.embed_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).copy()  # [n, embed_dim] — .copy() ensures a writable, contiguous array

        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(self._embeddings.astype(np.float32))

        self._token_ids = torch.tensor(windows_tokens, dtype=torch.long)  # [n, seq_len]
        self._texts = list(windows_text)
        self._write_head = 0
        self._ingested_since_build = 0

        # Public aliases (these point to the ring buffer objects, so in-place
        # modifications by ingest() are visible through the alias)
        self.stored_token_ids = self._token_ids
        self.stored_texts = self._texts

        print(f"[RAG] Index built: {self.index.ntotal} vectors, dim={self.embed_dim}")

    def ingest(self, windows_tokens: list, windows_text: list):
        """
        Ingest fresh windows into the ring buffer, overwriting the oldest
        entries, and rebuild the FAISS index.

        This implements rolling refresh: instead of replacing the entire
        index at once (causing LM loss jumps from distribution shift),
        a small batch of fresh windows is cycled in each step. The ring
        buffer guarantees the index always stays at full capacity.

        Rebuilding IndexFlatIP from 50K pre-computed embeddings is <5ms
        (just a memcpy), so it's done after every ingest call.

        Args:
            windows_tokens: list of token lists, each of length seq_len.
            windows_text:   list of decoded text strings (same length).
        """
        import faiss

        n = len(windows_tokens)
        if n == 0:
            return

        # Encode new windows with sentence-transformer
        new_embeddings = self.embed_model.encode(
            windows_text,
            batch_size=self.embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # [n, embed_dim]

        new_token_ids = torch.tensor(windows_tokens, dtype=torch.long)

        buf_size = self._embeddings.shape[0]
        start = self._write_head

        if start + n <= buf_size:
            # No wrap-around
            self._embeddings[start:start + n] = new_embeddings
            self._token_ids[start:start + n] = new_token_ids
            for i in range(n):
                self._texts[start + i] = windows_text[i]
        else:
            # Wrap-around: split into two parts
            first = buf_size - start
            second = n - first

            self._embeddings[start:] = new_embeddings[:first]
            self._embeddings[:second] = new_embeddings[first:]

            self._token_ids[start:] = new_token_ids[:first]
            self._token_ids[:second] = new_token_ids[first:]

            for i in range(first):
                self._texts[start + i] = windows_text[i]
            for i in range(second):
                self._texts[i] = windows_text[first + i]

        self._write_head = (start + n) % buf_size
        self._ingested_since_build += n

        # Rebuild FAISS index from current ring buffer
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(self._embeddings.astype(np.float32))

    @property
    def turnover_fraction(self) -> float:
        """Fraction of the ring buffer replaced since initial build_index().

        Returns 0.0 before any ingest() calls. Reaches 1.0 after ingesting
        index_size windows (i.e., every original entry has been overwritten
        at least once). Can exceed 1.0 for multiple full turnovers.
        """
        if self._embeddings is None:
            return 0.0
        return self._ingested_since_build / self._embeddings.shape[0]

    def retrieve(
        self,
        query_ids: torch.Tensor,
        tokenizer,
        top_k: Optional[int] = None,
        sample_from_topk: bool = True,
        sample_temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Retrieve training examples from the index using query token sequences.

        For each query, decode to text, embed, search the FAISS index.
        When sample_from_topk=True (default), retrieves top-k candidates per
        query and samples 1 from softmax(scores / temperature), giving
        diversity within the relevant neighborhood. When False, takes the
        top-1 deterministically.

        Results are deduplicated across all queries.

        Args:
            query_ids: [B, seq_len] discrete token IDs (from Langevin sampler).
            tokenizer: GPT-2 tokenizer for decoding.
            top_k: Number of FAISS candidates per query to score.
                   Default: self.retrieval_top_k.
            sample_from_topk: If True, softmax-sample 1 from top-k per query.
                              If False, take top-1 deterministically.
            sample_temperature: Temperature for softmax sampling over top-k
                                scores. Lower = more greedy, higher = more
                                uniform. Only used when sample_from_topk=True.

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

        # Search FAISS index for top-k candidates per query
        scores, indices = self.index.search(query_embeddings.astype(np.float32), top_k)
        # scores: [B, top_k], indices: [B, top_k]

        if sample_from_topk and top_k > 1:
            # Softmax-sample 1 result per query from the top-k candidates.
            # This gives diversity: instead of every query taking its nearest
            # neighbor (which often overlap), we sample proportionally to
            # similarity, so different queries can pull different examples
            # from their local neighborhoods.
            selected_indices = []
            for b in range(scores.shape[0]):
                row_scores = scores[b]
                row_indices = indices[b]
                # Filter out -1 sentinels
                valid = row_indices >= 0
                if not valid.any():
                    continue
                row_scores = row_scores[valid]
                row_indices = row_indices[valid]
                # Softmax sampling
                probs = np.exp(row_scores / sample_temperature)
                probs = probs / probs.sum()
                chosen = np.random.choice(len(row_indices), p=probs)
                selected_indices.append(int(row_indices[chosen]))
        else:
            # Deterministic top-1
            selected_indices = indices[:, 0].tolist()

        # Deduplicate
        unique_indices = list(set(idx for idx in selected_indices if idx >= 0))

        retrieved_ids = self.stored_token_ids[unique_indices]  # [R, seq_len]

        info = {
            "num_queries": len(query_texts),
            "num_retrieved_unique": len(unique_indices),
            "avg_top1_score": float(scores[:, 0].mean()),
            "avg_topk_score": float(scores.mean()) if top_k > 1 else float(scores[:, 0].mean()),
            "sample_query": query_texts[0][:200] if query_texts else "",
        }

        return retrieved_ids, info
