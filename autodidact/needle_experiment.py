"""
Needle-in-a-haystack experiment for Q-value validation.

Tests whether the Q-network learns to identify the one valuable training
example (the "needle") among a pool of degenerate filler (the "haystack").

Setup:
    - Needle: A fixed token sequence of real English text.
      The held-out set consists entirely of copies of this needle.
    - Haystack: N-1 candidates are each a single token repeated seq_len times
      (a different random token per candidate). Training on these constant
      sequences provides zero useful gradient for predicting the needle.
    - At each step, the needle is placed at a random position among N candidates.

Why constant-token haystack?
    Random token sequences are NOT inert: training on them changes the model's
    weights in ways that can incidentally affect held-out loss (positively or
    negatively). This makes the reward signal noisy and hard for the Q-network
    to attribute. Constant-token sequences are genuinely uninformative: the
    model already predicts "repeat the same token" perfectly, so training on
    them produces near-zero gradients and no reward change.

Expected behaviour:
    - Training on the needle directly reduces held-out loss (it IS the held-out data).
    - Training on constant-token filler does nothing useful.
    - The Q-network should learn to assign the highest Q-value to the needle.

Metrics tracked:
    - needle_q: Q-value of the needle candidate.
    - haystack_q_mean: Mean Q-value of the haystack candidates.
    - needle_q_rank: Rank of the needle's Q-value among all N (1 = highest).
    - needle_selected: Whether the Boltzmann policy selected the needle.
    - needle_prob: Probability the Boltzmann policy assigns to the needle.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Optional
import time
import math
import copy

from .q_network import QNetwork, extract_hidden_states, compute_soft_value, boltzmann_sample
from .logging import MetricsLogger, DashboardPlotter


def compute_held_out_reward(model, held_out: torch.Tensor, eval_batch_size: int = 16) -> float:
    """Compute reward: negative mean per-token CE on held-out data."""
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, held_out.shape[0], eval_batch_size):
            batch = held_out[i : i + eval_batch_size]
            outputs = model(batch, labels=batch)
            total_loss += outputs.loss.item()
            n_batches += 1
    return -total_loss / n_batches


class NeedleExperiment:
    """
    Controlled experiment: can the Q-network find the needle in a haystack?

    The held-out set is M copies of a single fixed sequence (the needle).
    Each training step presents N candidates: one is the needle, the rest
    are constant-token sequences. The Q-network should learn that only the
    needle is worth training on.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        seq_len: int = 128,
        num_candidates: int = 16,
        held_out_size: int = 32,
        extract_batch_size: int = 16,
        eval_batch_size: int = 16,
        beta: float = 1.0,
        gamma: float = 0.9,
        q_lr: float = 1e-3,
        q_grad_clip: float = 1.0,
        tau: float = 0.01,
        lm_lr: float = 5e-5,
        grad_clip: float = 1.0,
        num_steps: int = 200,
        log_interval: int = 1,
        dashboard_interval: int = 10,
        log_dir: str = "logs",
        device: Optional[str] = None,
        needle_text: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Needle] Using device: {self.device}")

        # --- Base model ---
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.train()
        self.vocab_size = self.model.config.vocab_size
        hidden_dim = self.model.config.n_embd

        # --- Q-network ---
        self.q_net = QNetwork(hidden_dim=hidden_dim, inner_dim=32).to(self.device)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.q_target = copy.deepcopy(self.q_net)
        self.q_target.requires_grad_(False)
        self.tau = tau

        # --- LM optimizer ---
        self.lm_optimizer = torch.optim.AdamW(self.model.parameters(), lr=lm_lr)

        # --- Hyperparameters ---
        self.beta = beta
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.q_grad_clip = q_grad_clip
        self.num_candidates = num_candidates
        self.seq_len = seq_len
        self.extract_batch_size = extract_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.dashboard_interval = dashboard_interval
        self.log_dir = log_dir
        self.held_out_size = held_out_size

        # --- Build the needle ---
        if needle_text is not None:
            tokens = self.tokenizer.encode(needle_text, add_special_tokens=False)
        else:
            # Default: a distinctive repeating pattern of real English text
            pattern_text = (
                "The quick brown fox jumps over the lazy dog. "
                "A stitch in time saves nine. "
                "All that glitters is not gold. "
            )
            tokens = self.tokenizer.encode(pattern_text, add_special_tokens=False)
        # Repeat/truncate to fill seq_len
        while len(tokens) < seq_len:
            tokens = tokens + tokens
        tokens = tokens[:seq_len]
        self.needle = torch.tensor(tokens, dtype=torch.long, device=self.device)

        # --- Held-out set: M copies of the needle ---
        self.held_out = self.needle.unsqueeze(0).expand(held_out_size, -1).contiguous()
        print(f"[Needle] Needle sequence: {repr(self.tokenizer.decode(self.needle[:50].tolist()))}...")
        print(f"[Needle] Held-out set: {self.held_out.shape[0]} copies of needle, length {seq_len}")
        print(f"[Needle] Candidates per step: {num_candidates} (1 needle + {num_candidates - 1} constant-token filler)")

    def _make_haystack_batch(self) -> tuple:
        """
        Build a candidate batch: 1 needle at a random position, rest are
        constant-token sequences (each a different random token repeated).

        Returns:
            candidates: [N, seq_len] tensor
            needle_idx: int, position of the needle in the batch
        """
        needle_idx = torch.randint(0, self.num_candidates, (1,)).item()

        # Each haystack candidate is a single token repeated seq_len times.
        # Use different random tokens for each candidate for diversity.
        random_tokens = torch.randint(0, self.vocab_size, (self.num_candidates,), device=self.device)
        candidates = random_tokens.unsqueeze(1).expand(-1, self.seq_len).contiguous()
        candidates[needle_idx] = self.needle
        return candidates, needle_idx

    def _lm_step(self, input_ids: torch.Tensor) -> tuple:
        """One LM gradient step. Returns (loss, grad_norm)."""
        self.lm_optimizer.zero_grad()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
        self.lm_optimizer.step()
        return loss.item(), grad_norm

    def _q_step(self, h_prev, r_prev, soft_value_next) -> tuple:
        """One Q-network update. Returns (td_loss, q_grad_norm)."""
        self.q_optimizer.zero_grad()
        q_hat = self.q_net(h_prev.unsqueeze(0)).squeeze()
        y = (1 - self.gamma) * r_prev + self.gamma * soft_value_next.detach()
        td_loss = 0.5 * (q_hat - y) ** 2
        td_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.q_grad_clip).item()
        self.q_optimizer.step()
        self._polyak_update()
        return td_loss.item(), q_grad_norm

    @torch.no_grad()
    def _polyak_update(self):
        for p, p_target in zip(self.q_net.parameters(), self.q_target.parameters()):
            p_target.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def run(self) -> str:
        """
        Run the needle-in-a-haystack experiment.

        Returns:
            Path to the JSONL log file.
        """
        config_dict = {
            "method_name": "needle_in_haystack",
            "seq_len": self.seq_len,
            "num_candidates": self.num_candidates,
            "held_out_size": self.held_out_size,
            "beta": self.beta,
            "gamma": self.gamma,
            "tau": self.tau,
            "q_lr": self.q_optimizer.param_groups[0]["lr"],
            "lm_lr": self.lm_optimizer.param_groups[0]["lr"],
            "num_steps": self.num_steps,
        }
        logger = MetricsLogger("needle_in_haystack", log_dir=self.log_dir, config=config_dict)
        dashboard = DashboardPlotter(output_path=str(logger.dashboard_file))

        cumulative_reward = 0.0

        # =====================================================================
        # Bootstrap step (k=0)
        # =====================================================================
        print("[Needle] Bootstrap step...")
        C_0, needle_idx_0 = self._make_haystack_batch()
        H_0 = extract_hidden_states(self.model, C_0, batch_size=self.extract_batch_size)
        q_0 = self.q_net(H_0)
        a_0 = boltzmann_sample(q_0, self.beta)

        h_prev = H_0[a_0]

        # LM update on selected candidate
        lm_loss, lm_grad_norm = self._lm_step(C_0[a_0])

        # Reward
        r_prev = compute_held_out_reward(self.model, self.held_out, self.eval_batch_size)
        cumulative_reward += r_prev

        # Needle metrics
        needle_metrics_0 = self._compute_needle_metrics(q_0, needle_idx_0, a_0)

        logger.log({
            "step": 0,
            "reward": r_prev,
            "lm_loss": lm_loss,
            "td_loss": 0.0,
            **needle_metrics_0,
            "cumulative_reward": cumulative_reward,
            "lm_grad_norm": lm_grad_norm,
            "q_grad_norm": 0.0,
            "step_time": 0.0,
        })

        print(f"  Bootstrap: needle_q={needle_metrics_0['needle_q']:.4f}, "
              f"haystack_q_mean={needle_metrics_0['haystack_q_mean']:.4f}, "
              f"rank={needle_metrics_0['needle_q_rank']}/{self.num_candidates}, "
              f"selected={bool(needle_metrics_0['needle_selected'])}")

        # =====================================================================
        # Main loop
        # =====================================================================
        print(f"\n[Needle] Starting main loop for {self.num_steps} steps...\n")

        for k in range(1, self.num_steps + 1):
            step_start = time.time()

            # --- Build candidate batch with needle at random position ---
            C_k, needle_idx = self._make_haystack_batch()
            H_k = extract_hidden_states(self.model, C_k, batch_size=self.extract_batch_size)
            q_k = self.q_net(H_k)

            # --- Q update for previous transition ---
            with torch.no_grad():
                q_k_target = self.q_target(H_k)
            V_k_target = compute_soft_value(q_k_target, self.beta)
            td_loss, q_grad_norm = self._q_step(h_prev, r_prev, V_k_target)

            # --- Action selection ---
            a_k = boltzmann_sample(q_k, self.beta)
            h_prev = H_k[a_k]

            # --- LM update ---
            lm_loss, lm_grad_norm = self._lm_step(C_k[a_k])

            # --- Reward ---
            r_k = compute_held_out_reward(self.model, self.held_out, self.eval_batch_size)
            r_prev = r_k
            cumulative_reward += r_k

            step_time = time.time() - step_start

            # --- Needle-specific metrics ---
            nm = self._compute_needle_metrics(q_k, needle_idx, a_k)

            metrics = {
                "step": k,
                "reward": r_k,
                "lm_loss": lm_loss,
                "td_loss": td_loss,
                **nm,
                "cumulative_reward": cumulative_reward,
                "lm_grad_norm": lm_grad_norm,
                "q_grad_norm": q_grad_norm,
                "step_time": step_time,
            }
            logger.log(metrics)

            if k % self.log_interval == 0:
                print(
                    f"Step {k:4d} | needle_q={nm['needle_q']:.4f} haystack_q={nm['haystack_q_mean']:.4f} "
                    f"adv={nm['needle_q_advantage']:+.4f} | "
                    f"rank={nm['needle_q_rank']:2d}/{self.num_candidates} prob={nm['needle_prob']:.4f} "
                    f"sel={'YES' if nm['needle_selected'] else 'no ':3s} | "
                    f"reward={r_k:.4f} td={td_loss:.6f} | {step_time:.2f}s"
                )

            if k % self.dashboard_interval == 0:
                try:
                    self._render_dashboard(logger)
                except Exception as e:
                    print(f"  [Dashboard error: {e}]")

        # Final dashboard
        try:
            self._render_dashboard(logger)
        except Exception as e:
            print(f"  [Final dashboard error: {e}]")

        # --- Print summary ---
        self._print_summary(logger)

        return str(logger.log_file)

    def _compute_needle_metrics(self, q_values: torch.Tensor, needle_idx: int, selected_action: int) -> dict:
        """Compute all needle-related metrics from Q-values."""
        q_squeezed = q_values.squeeze(-1)
        needle_q = q_squeezed[needle_idx].item()
        haystack_mask = torch.ones(self.num_candidates, dtype=torch.bool, device=self.device)
        haystack_mask[needle_idx] = False
        haystack_qs = q_squeezed[haystack_mask]
        haystack_q_mean = haystack_qs.mean().item()
        haystack_q_std = haystack_qs.std().item()
        needle_rank = (q_squeezed > q_squeezed[needle_idx]).sum().item() + 1
        probs = torch.softmax(q_squeezed / self.beta, dim=0)
        needle_prob = probs[needle_idx].item()
        needle_selected = int(selected_action == needle_idx)

        return {
            "needle_q": needle_q,
            "haystack_q_mean": haystack_q_mean,
            "haystack_q_std": haystack_q_std,
            "needle_q_advantage": needle_q - haystack_q_mean,
            "needle_q_rank": needle_rank,
            "needle_prob": needle_prob,
            "needle_selected": needle_selected,
            "q_mean": q_squeezed.mean().item(),
            "q_std": q_squeezed.std().item(),
        }

    def _print_summary(self, logger: MetricsLogger):
        """Print final summary statistics."""
        from .logging import _read_log
        _, all_metrics, _ = _read_log(str(logger.log_file))
        if not all_metrics:
            return

        last_n = min(20, len(all_metrics))
        last = all_metrics[-last_n:]
        avg_rank = sum(m["needle_q_rank"] for m in last) / len(last)
        avg_prob = sum(m["needle_prob"] for m in last) / len(last)
        avg_adv = sum(m["needle_q_advantage"] for m in last) / len(last)
        sel_rate = sum(m["needle_selected"] for m in last) / len(last)

        print("\n" + "=" * 70)
        print("NEEDLE-IN-HAYSTACK EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"  Last {last_n} steps averages:")
        print(f"    Needle Q rank:      {avg_rank:.1f} / {self.num_candidates}  (1 = best)")
        print(f"    Needle probability:  {avg_prob:.4f}  (uniform = {1/self.num_candidates:.4f})")
        print(f"    Needle Q advantage:  {avg_adv:+.4f}")
        print(f"    Needle selection rate: {sel_rate:.1%}")
        print(f"  Log file: {logger.log_file}")
        print(f"  Dashboard: {logger.dashboard_file}")
        print("=" * 70)

    def _render_dashboard(self, logger: MetricsLogger):
        """Render a needle-specific 3x3 dashboard."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        from .logging import _read_log
        _, metrics, _ = _read_log(str(logger.log_file))
        if not metrics:
            return

        steps = [m["step"] for m in metrics]

        fig, axes = plt.subplots(3, 3, figsize=(18, 13))
        fig.suptitle("Needle-in-a-Haystack: Q-Value Validation", fontsize=16, fontweight="bold", y=0.98)

        kernel = np.ones(10) / 10 if len(steps) >= 10 else None

        # --- (0,0) Needle Q vs Haystack Q mean ---
        ax = axes[0][0]
        ax.plot(steps, [m["needle_q"] for m in metrics], color="red", linewidth=1.5, label="Needle Q")
        ax.plot(steps, [m["haystack_q_mean"] for m in metrics], color="blue", linewidth=1.0, alpha=0.7, label="Haystack Q mean")
        ax.set_title("Needle Q vs Haystack Q", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- (0,1) Q advantage ---
        ax = axes[0][1]
        advs = [m["needle_q_advantage"] for m in metrics]
        ax.plot(steps, advs, color="green", linewidth=1.0, alpha=0.5)
        if kernel is not None and len(advs) >= 10:
            smoothed = np.convolve(advs, kernel, mode="valid")
            ax.plot(steps[9:], smoothed, color="green", linewidth=2.0, label="Smoothed (10)")
            ax.legend(fontsize=8)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_title("Needle Q Advantage (needle - haystack)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        # --- (0,2) Needle rank ---
        ax = axes[0][2]
        ranks = [m["needle_q_rank"] for m in metrics]
        ax.plot(steps, ranks, color="purple", linewidth=1.0, alpha=0.4)
        if kernel is not None and len(ranks) >= 10:
            smoothed = np.convolve(ranks, kernel, mode="valid")
            ax.plot(steps[9:], smoothed, color="purple", linewidth=2.5, label="Smoothed (10)")
            ax.legend(fontsize=8)
        ax.axhline(y=1, color="green", linestyle="--", alpha=0.5)
        ax.axhline(y=self.num_candidates / 2, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Needle Rank (1=best, {self.num_candidates}=worst)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylim(0.5, self.num_candidates + 0.5)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # --- (1,0) Needle probability ---
        ax = axes[1][0]
        probs = [m["needle_prob"] for m in metrics]
        ax.plot(steps, probs, color="orange", linewidth=1.0, alpha=0.4)
        if kernel is not None and len(probs) >= 10:
            smoothed = np.convolve(probs, kernel, mode="valid")
            ax.plot(steps[9:], smoothed, color="orange", linewidth=2.5, label="Smoothed (10)")
        ax.axhline(y=1.0 / self.num_candidates, color="gray", linestyle="--", alpha=0.5,
                    label=f"Uniform={1/self.num_candidates:.3f}")
        ax.set_title("Needle Selection Probability", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- (1,1) Cumulative selection rate ---
        ax = axes[1][1]
        selected = [m["needle_selected"] for m in metrics]
        cum_rate = [sum(selected[:i+1]) / (i+1) for i in range(len(selected))]
        ax.plot(steps, cum_rate, color="darkgreen", linewidth=1.5)
        ax.axhline(y=1.0 / self.num_candidates, color="gray", linestyle="--", alpha=0.5,
                    label=f"Random={1/self.num_candidates:.3f}")
        ax.set_title("Cumulative Needle Selection Rate", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- (1,2) Reward ---
        ax = axes[1][2]
        rewards = [m["reward"] for m in metrics]
        ax.plot(steps, rewards, color="teal", linewidth=1.0)
        ax.set_title("Held-out Reward (neg CE on needle)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        # --- (2,0) TD Loss ---
        ax = axes[2][0]
        td_vals = [max(m["td_loss"], 1e-12) for m in metrics]
        ax.semilogy(steps, td_vals, color="brown", linewidth=1.0)
        ax.set_title("TD Loss", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        # --- (2,1) LM Loss ---
        ax = axes[2][1]
        ax.plot(steps, [m["lm_loss"] for m in metrics], color="navy", linewidth=1.0)
        ax.set_title("LM Loss (on selected candidate)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        # --- (2,2) Q distribution + needle ---
        ax = axes[2][2]
        ax.plot(steps, [m["q_mean"] for m in metrics], color="blue", linewidth=1.0, label="Q mean (all)")
        ax.fill_between(
            steps,
            [m["q_mean"] - m["q_std"] for m in metrics],
            [m["q_mean"] + m["q_std"] for m in metrics],
            color="blue", alpha=0.15,
        )
        ax.plot(steps, [m["needle_q"] for m in metrics], color="red", linewidth=1.5, label="Needle Q", alpha=0.8)
        ax.set_title("Q Distribution + Needle", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        tmp_path = str(logger.dashboard_file) + ".tmp.png"
        fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
        os.replace(tmp_path, str(logger.dashboard_file))
        plt.close(fig)
