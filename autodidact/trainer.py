"""
Main training loop implementing Algorithm 1 from the spec:
    Curriculum Selection via Normalised Discounted Soft Q-Learning.

Key implementation insight (from Section 2.5):
    Each step's forward pass of candidates through the base model serves two 
    purposes: action selection for the *current* step and the bootstrap target 
    for the *previous* step's Q update. This avoids a redundant second forward pass.

LM training uses the soft mixture approach: at each step, the LM trains on a
policy-weighted mixture of all N candidates, where the weights are the Boltzmann
policy pi = softmax(Q/beta). This computes the expected gradient under the policy
exactly, eliminating variance from single-action sampling. When --no_q_weighting
is used, the weights are uniform (1/N), bypassing the Q-function for the LM step.

Reward semantics:
    r_k = negative mean per-token cross-entropy on the held-out subset.
    This is the average per-token log-probability, which is length-invariant
    and directly interpretable (exp(-r_k) = perplexity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Optional, Dict, Any, List
import time
import math

from .q_network import QNetwork, extract_hidden_states, compute_soft_value, boltzmann_sample
from .data import CandidateBatchSampler, HeldOutSet
from .baselines import BaselineSelector
from .logging import MetricsLogger, DashboardPlotter, LangevinRAGDashboard
from .langevin_rag import LangevinQSampler, RAGIndex


def compute_lm_loss(model: GPT2LMHeadModel, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute language modeling loss (cross-entropy) for a single example.
    
    Args:
        model: GPT-2 model.
        input_ids: [1, seq_len] or [seq_len] token IDs.
    
    Returns:
        Scalar loss (per-token mean cross-entropy).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    outputs = model(input_ids, labels=input_ids)
    return outputs.loss


def compute_held_out_reward(
    model: GPT2LMHeadModel,
    held_out_subset: torch.Tensor,
    eval_batch_size: int = 16,
) -> float:
    """
    Compute reward: negative mean per-token cross-entropy on held-out subset.
    
    This is the average per-token log-probability, which:
      - Is length-invariant (doesn't change with seq_len)
      - Is directly interpretable: exp(-reward) = perplexity
      - Stays in a narrow range (~-3 to -4 for GPT-2 on typical text)
    
    Args:
        model: GPT-2 model (already updated to theta_{k+1}).
        held_out_subset: [M, seq_len] held-out examples.
        eval_batch_size: Mini-batch size for reward computation.
    
    Returns:
        Scalar reward (negative per-token cross-entropy, higher is better).
    """
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, held_out_subset.shape[0], eval_batch_size):
            batch = held_out_subset[i : i + eval_batch_size]
            outputs = model(batch, labels=batch)
            total_loss += outputs.loss.item()
            n_batches += 1
    return -total_loss / n_batches


class AutodidactTrainer:
    """
    Implements the full Autodidact training loop (Algorithm 1).
    
    Uses normalised discounted soft Q-learning:
        Q(s, a) = (1 - gamma) * r + gamma * V_target(s')
        V(s) = beta * [logsumexp(Q(s, a') / beta) - log(N)]
        pi(a|s) = softmax(Q(s, a) / beta)
    
    The (1 - gamma) on the reward and -log(N) in the soft value together
    ensure Q* = r for a constant reward. A target network with Polyak
    averaging stabilises the bootstrap target V(s').
    """

    def __init__(
        self,
        # Model
        model_name: str = "gpt2",
        # Data
        dataset_name: str = "openwebtext",
        seq_len: int = 1024,
        num_candidates: int = 256,
        held_out_subset_size: int = 512,
        held_out_total_size: int = 8192,
        # Batching (GPU utilization)
        extract_batch_size: int = 32,
        eval_batch_size: int = 64,
        # Q-learning
        beta: float = 1.0,
        gamma: float = 0.9,
        q_lr: float = 1e-4,
        q_grad_clip: float = 1.0,
        tau: float = 0.01,
        # LM training
        lm_lr: float = 5e-5,
        grad_clip: float = 1.0,
        # Logging
        log_interval: int = 10,
        eval_interval: int = 100,
        dashboard_interval: int = 50,
        save_interval: int = 500,
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: str = "autodidact",
        # Device
        device: Optional[str] = None,
        # Mixture training
        mixture_batch_size: int = 8,
        no_q_weighting: bool = False,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        # --- Base model ---
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.train()
        hidden_dim = self.model.config.n_embd  # 768 for gpt2

        # --- Q-network ---
        self.q_net = QNetwork(hidden_dim=hidden_dim, inner_dim=32).to(self.device)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=q_lr)

        # --- Target Q-network (Polyak-averaged copy for stable bootstrap targets) ---
        import copy
        self.q_target = copy.deepcopy(self.q_net)
        self.q_target.requires_grad_(False)  # No gradients through target network
        self.tau = tau

        # --- LM optimizer (AdamW, standard for transformer fine-tuning) ---
        self.lm_optimizer = torch.optim.AdamW(self.model.parameters(), lr=lm_lr)

        # --- Hyperparameters ---
        self.beta = beta
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.q_grad_clip = q_grad_clip
        self.num_candidates = num_candidates
        self.held_out_subset_size = held_out_subset_size
        self.extract_batch_size = extract_batch_size
        self.eval_batch_size = eval_batch_size
        self.seq_len = seq_len
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.dashboard_interval = dashboard_interval
        self.save_interval = save_interval

        # --- Data ---
        self.dataset_name = dataset_name
        self.held_out_total_size = held_out_total_size

        # --- Mixture training ---
        self.mixture_batch_size = mixture_batch_size
        self.no_q_weighting = no_q_weighting

        # --- Logging ---
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Store config dict for logger header
        self._config_dict = {
            "method_name": "q_learning_uniform_mixture" if no_q_weighting else "q_learning",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seq_len": seq_len,
            "num_candidates": num_candidates,
            "held_out_subset_size": held_out_subset_size,
            "held_out_total_size": held_out_total_size,
            "extract_batch_size": extract_batch_size,
            "eval_batch_size": eval_batch_size,
            "beta": beta,
            "gamma": gamma,
            "q_lr": q_lr,
            "q_grad_clip": q_grad_clip,
            "tau": tau,
            "lm_lr": lm_lr,
            "grad_clip": grad_clip,
            "no_q_weighting": no_q_weighting,
            "mixture_batch_size": mixture_batch_size,
        }

        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, config=self._config_dict)

    def _lm_step_mixture(
        self,
        candidates: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple:
        """
        Perform one LM gradient step on the policy-weighted mixture of all candidates.

        Instead of training on a single selected example, we compute per-candidate
        losses and weight them by the Boltzmann policy pi = softmax(Q/beta).
        Gradients are accumulated in mini-batches to control memory.

        The weighted loss is:
            L = sum_i  pi_i * loss(x_i)

        This is the expected gradient under the policy, which is a variance-reduced
        alternative to sampling a single action. The Q-learning side is unchanged;
        this only affects how the LM parameters are updated.

        Args:
            candidates: [N, seq_len] all candidate token IDs.
            weights: [N] policy weights (should sum to 1, i.e. softmax(Q/beta)).

        Returns:
            (weighted_loss_value, grad_norm_before_clip)
        """
        self.lm_optimizer.zero_grad()
        N = candidates.shape[0]
        bs = self.mixture_batch_size
        weighted_loss_total = 0.0

        for i in range(0, N, bs):
            batch = candidates[i : i + bs]               # [bs, seq_len]
            w = weights[i : i + bs]                       # [bs]

            # Compute per-example losses (HF returns mean over tokens per example
            # when given a batch with labels, but averages across the batch too).
            # We need per-example losses, so we use reduction='none' manually.
            outputs = self.model(batch, labels=batch)
            # outputs.loss is the mean over all tokens in the batch.
            # We need per-example losses. Recompute from logits:
            logits = outputs.logits[:, :-1, :].contiguous()   # [bs, seq_len-1, vocab]
            labels = batch[:, 1:].contiguous()                 # [bs, seq_len-1]
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='none',
            ).view(labels.shape)                               # [bs, seq_len-1]
            per_example_loss = per_token_loss.mean(dim=1)      # [bs]

            # Weighted contribution of this mini-batch
            mini_loss = (w * per_example_loss).sum()
            mini_loss.backward()
            weighted_loss_total += mini_loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
        self.lm_optimizer.step()
        return weighted_loss_total, grad_norm

    def _q_step(
        self,
        h_prev: torch.Tensor,
        r_prev: float,
        soft_value_next: torch.Tensor,
    ) -> tuple:
        """
        Perform one Q-network update step (normalised discounted soft Bellman).
        
        TD target: y = (1 - gamma) * r_prev + gamma * V_target(s_{k+1})
        Loss: 0.5 * (Q(h_prev) - y)^2
        
        The (1 - gamma) factor on the reward keeps Q-values on the same scale
        as the reward regardless of gamma. V_target is computed from the target
        network (Polyak-averaged copy) for stability.
        
        Returns (td_loss_value, q_grad_norm_before_clip).
        """
        self.q_optimizer.zero_grad()

        # Re-evaluate previous action with current phi
        q_hat = self.q_net(h_prev.unsqueeze(0)).squeeze()  # scalar

        # TD target (stop gradient; V computed from target network)
        y = (1 - self.gamma) * r_prev + self.gamma * soft_value_next.detach()

        # TD loss
        td_loss = 0.5 * (q_hat - y) ** 2
        td_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.q_grad_clip).item()
        self.q_optimizer.step()

        # Polyak update: target <- tau * online + (1 - tau) * target
        self._polyak_update()

        return td_loss.item(), q_grad_norm

    @torch.no_grad()
    def _polyak_update(self):
        """Soft-update target Q-network toward online Q-network."""
        for p, p_target in zip(self.q_net.parameters(), self.q_target.parameters()):
            p_target.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def _save_checkpoint(self, checkpoint_dir, step: int):
        """
        Save model and Q-network weights to checkpoint_dir, overwriting any
        previous checkpoint from this run. Only one checkpoint is kept per run.
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "q_net_state_dict": self.q_net.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "lm_optimizer_state_dict": self.lm_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, "checkpoint.pt"))
        print(f"  [Checkpoint saved: step {step} -> {checkpoint_dir}]")

    def train(self, num_steps: int = 10000) -> str:
        """
        Run the full Autodidact training loop (Algorithm 1).
        
        Args:
            num_steps: Number of training steps to run.
        
        Returns:
            Path to the JSONL log file.
        """
        from .data import StreamingTextDataset

        # --- Set up logging ---
        method_label = "q_learning_uniform_mixture" if self.no_q_weighting else "q_learning"
        logger = MetricsLogger(method_label, log_dir=self.log_dir, config=self._config_dict)
        run_dashboard = DashboardPlotter(output_path=str(logger.dashboard_file))

        # --- Set up data ---
        print("Setting up streaming dataset...")
        stream_dataset = StreamingTextDataset(
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            dataset_name=self.dataset_name,
        )
        sampler = CandidateBatchSampler(stream_dataset, self.num_candidates, self.device)

        print("Building held-out evaluation set (this may take a moment)...")
        held_out = HeldOutSet(
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            total_size=self.held_out_total_size,
            dataset_name=self.dataset_name,
            device=self.device,
        )

        cumulative_reward = 0.0

        # =====================================================================
        # Bootstrap step (k=0): no Q update
        # =====================================================================
        print("Running bootstrap step...")
        C_0 = sampler.next_batch()
        H_0 = extract_hidden_states(self.model, C_0, batch_size=self.extract_batch_size)
        q_0 = self.q_net(H_0)
        a_0 = boltzmann_sample(q_0, self.beta)

        h_prev = H_0[a_0]

        # LM update: policy-weighted mixture of all candidates
        if self.no_q_weighting:
            pi_0 = torch.ones(self.num_candidates, device=self.device) / self.num_candidates
        else:
            pi_0 = torch.softmax(q_0.squeeze(-1) / self.beta, dim=0).detach()
        lm_loss, lm_grad_norm = self._lm_step_mixture(C_0, pi_0)

        # Compute reward after update
        D_hat = held_out.sample_subset(self.held_out_subset_size)
        r_prev = compute_held_out_reward(self.model, D_hat, eval_batch_size=self.eval_batch_size)
        cumulative_reward += r_prev

        # Log bootstrap
        V_0 = compute_soft_value(q_0, self.beta).item()
        logger.log({
            "step": 0,
            "reward": r_prev,
            "lm_loss": lm_loss,
            "td_loss": 0.0,
            "policy_entropy": math.log(self.num_candidates),
            "policy_entropy_ratio": 1.0,
            "q_mean": q_0.mean().item(),
            "q_std": q_0.std().item(),
            "q_min": q_0.min().item(),
            "q_max": q_0.max().item(),
            "soft_value": V_0,
            "cumulative_reward": cumulative_reward,
            "lm_grad_norm": lm_grad_norm,
            "q_grad_norm": 0.0,
            "selected_action": a_0,
            "step_time": 0.0,
        })

        print(f"Bootstrap: lm_loss={lm_loss:.4f}, reward={r_prev:.4f}")

        # =====================================================================
        # Main loop (k = 1, 2, ...)
        # =====================================================================
        print(f"\nStarting main training loop for {num_steps} steps...")

        for k in range(1, num_steps + 1):
            step_start = time.time()

            # --- Forward candidates through base model (dual purpose) ---
            C_k = sampler.next_batch()
            H_k = extract_hidden_states(self.model, C_k, batch_size=self.extract_batch_size)
            q_k = self.q_net(H_k)

            # --- Q update for previous transition ---
            # Use target network for bootstrap V(s') to stabilise training
            with torch.no_grad():
                q_k_target = self.q_target(H_k)
            V_k_target = compute_soft_value(q_k_target, self.beta)
            td_loss, q_grad_norm = self._q_step(h_prev, r_prev, V_k_target)

            # --- Action selection for current step ---
            # A discrete action is still sampled for the Q-learning Bellman backup,
            # even though the LM trains on the full mixture of all candidates.
            a_k = boltzmann_sample(q_k, self.beta)
            h_prev = H_k[a_k]

            # --- LM update: policy-weighted mixture of all candidates ---
            if self.no_q_weighting:
                pi_k = torch.ones(self.num_candidates, device=self.device) / self.num_candidates
            else:
                pi_k = torch.softmax(q_k.squeeze(-1) / self.beta, dim=0).detach()
            lm_loss, lm_grad_norm = self._lm_step_mixture(C_k, pi_k)

            # --- Compute reward ---
            D_hat = held_out.sample_subset(self.held_out_subset_size)
            r_k = compute_held_out_reward(self.model, D_hat, eval_batch_size=self.eval_batch_size)
            r_prev = r_k
            cumulative_reward += r_k

            step_time = time.time() - step_start

            # --- Compute monitoring metrics ---
            q_squeezed = q_k.squeeze(-1)
            probs = torch.softmax(q_squeezed / self.beta, dim=0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            max_entropy = math.log(self.num_candidates)

            metrics = {
                "step": k,
                "reward": r_k,
                "lm_loss": lm_loss,
                "td_loss": td_loss,
                "policy_entropy": entropy,
                "policy_entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0,
                "q_mean": q_k.mean().item(),
                "q_std": q_k.std().item(),
                "q_min": q_k.min().item(),
                "q_max": q_k.max().item(),
                "soft_value": compute_soft_value(q_k, self.beta).item(),
                "cumulative_reward": cumulative_reward,
                "lm_grad_norm": lm_grad_norm,
                "q_grad_norm": q_grad_norm,
                "selected_action": a_k,
                "step_time": step_time,
            }

            # --- Always log to JSONL ---
            logger.log(metrics)

            # --- Print to stdout periodically ---
            if k % self.log_interval == 0:
                print(
                    f"Step {k:6d} | reward={r_k:.4f} | "
                    f"lm_loss={lm_loss:.4f} | td={td_loss:.6f} | "
                    f"ent={entropy:.3f}/{max_entropy:.3f} | "
                    f"Q={q_k.mean().item():.4f}+/-{q_k.std().item():.4f} | "
                    f"gnorm_lm={lm_grad_norm:.3f} gnorm_q={q_grad_norm:.3f} | "
                    f"{step_time:.2f}s"
                )

                if self.use_wandb:
                    import wandb
                    wandb.log(metrics)

            # --- Full evaluation ---
            if k % self.eval_interval == 0:
                eval_reward, eval_ppl = self._full_eval(held_out)
                print(f"  [EVAL] step={k} | full_reward={eval_reward:.4f} | ppl={eval_ppl:.2f}")
                logger.log_eval({
                    "step": k,
                    "eval_reward": eval_reward,
                    "eval_perplexity": eval_ppl,
                })
                if self.use_wandb:
                    import wandb
                    wandb.log({"eval_reward": eval_reward, "eval_perplexity": eval_ppl, "step": k})

            # --- Refresh per-run dashboard ---
            if k % self.dashboard_interval == 0:
                try:
                    run_dashboard.render([str(logger.log_file)])
                except Exception as e:
                    print(f"  [Dashboard render error: {e}]")

            # --- Save checkpoint (overwrite previous) ---
            if k % self.save_interval == 0:
                self._save_checkpoint(str(logger.checkpoint_dir), k)

        # Final saves
        self._save_checkpoint(str(logger.checkpoint_dir), num_steps)
        try:
            run_dashboard.render([str(logger.log_file)])
        except Exception as e:
            print(f"  [Final dashboard render error: {e}]")

        print(f"\nTraining complete. Log: {logger.log_file}")
        print(f"                  Dashboard: {logger.dashboard_file}")
        print(f"                  Checkpoint: {logger.checkpoint_dir}")
        return str(logger.log_file)

    def _full_eval(self, held_out: HeldOutSet) -> tuple:
        """Evaluate on the full held-out set. Returns (neg_per_token_ce, perplexity)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for i in range(0, held_out.data.shape[0], self.eval_batch_size):
                batch = held_out.data[i : i + self.eval_batch_size]
                outputs = self.model(batch, labels=batch)
                total_loss += outputs.loss.item()
                n_batches += 1
        self.model.train()
        avg_ce = total_loss / n_batches
        return -avg_ce, math.exp(avg_ce) if avg_ce < 100 else float("inf")


class BaselineTrainer:
    """
    Trainer that uses baseline selection strategies instead of Q-learning.
    Shares the same LM training setup for fair comparison.
    """

    def __init__(
        self,
        selector: BaselineSelector,
        model_name: str = "gpt2",
        dataset_name: str = "openwebtext",
        seq_len: int = 1024,
        num_candidates: int = 256,
        held_out_subset_size: int = 512,
        held_out_total_size: int = 8192,
        eval_batch_size: int = 64,
        lm_lr: float = 5e-5,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        dashboard_interval: int = 50,
        save_interval: int = 500,
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: str = "autodidact",
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Baseline: {selector.name}] Using device: {self.device}")

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.train()

        self.lm_optimizer = torch.optim.AdamW(self.model.parameters(), lr=lm_lr)
        self.selector = selector
        self.grad_clip = grad_clip
        self.num_candidates = num_candidates
        self.held_out_subset_size = held_out_subset_size
        self.eval_batch_size = eval_batch_size
        self.seq_len = seq_len
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.dashboard_interval = dashboard_interval
        self.save_interval = save_interval
        self.dataset_name = dataset_name
        self.held_out_total_size = held_out_total_size
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        self._config_dict = {
            "method_name": selector.name,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seq_len": seq_len,
            "num_candidates": num_candidates,
            "held_out_subset_size": held_out_subset_size,
            "held_out_total_size": held_out_total_size,
            "eval_batch_size": eval_batch_size,
            "lm_lr": lm_lr,
            "grad_clip": grad_clip,
        }

    def train(self, num_steps: int = 10000, all_log_files: Optional[List[str]] = None) -> str:
        from .data import StreamingTextDataset

        logger = MetricsLogger(self.selector.name, log_dir=self.log_dir, config=self._config_dict)
        run_dashboard = DashboardPlotter(output_path=str(logger.dashboard_file))
        combined_log_files = list(all_log_files) if all_log_files else []
        combined_log_files.append(str(logger.log_file))

        stream_dataset = StreamingTextDataset(
            tokenizer=self.tokenizer, seq_len=self.seq_len, dataset_name=self.dataset_name,
        )
        sampler = CandidateBatchSampler(stream_dataset, self.num_candidates, self.device)
        held_out = HeldOutSet(
            tokenizer=self.tokenizer, seq_len=self.seq_len,
            total_size=self.held_out_total_size, dataset_name=self.dataset_name, device=self.device,
        )

        cumulative_reward = 0.0

        for k in range(1, num_steps + 1):
            step_start = time.time()
            C_k = sampler.next_batch()
            a_k = self.selector.select(self.model, C_k)

            self.lm_optimizer.zero_grad()
            loss = compute_lm_loss(self.model, C_k[a_k].unsqueeze(0))
            loss.backward()
            lm_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            self.lm_optimizer.step()

            D_hat = held_out.sample_subset(self.held_out_subset_size)
            r_k = compute_held_out_reward(self.model, D_hat, eval_batch_size=self.eval_batch_size)
            cumulative_reward += r_k
            step_time = time.time() - step_start

            metrics = {
                "step": k, "reward": r_k, "lm_loss": loss.item(),
                "lm_grad_norm": lm_grad_norm, "cumulative_reward": cumulative_reward,
                "selected_action": a_k, "step_time": step_time,
            }
            logger.log(metrics)

            if k % self.log_interval == 0:
                print(f"[{self.selector.name}] Step {k:6d} | reward={r_k:.4f} | "
                      f"lm_loss={loss.item():.4f} | gnorm={lm_grad_norm:.3f} | {step_time:.2f}s")
                if self.use_wandb:
                    import wandb
                    wandb.log({f"{self.selector.name}/{key}": val for key, val in metrics.items()})

            if k % self.eval_interval == 0:
                eval_reward, eval_ppl = self._full_eval(held_out)
                print(f"  [{self.selector.name} EVAL] step={k} | reward={eval_reward:.4f} | ppl={eval_ppl:.2f}")
                logger.log_eval({"step": k, "eval_reward": eval_reward, "eval_perplexity": eval_ppl})

            if k % self.dashboard_interval == 0:
                try:
                    run_dashboard.render([str(logger.log_file)])
                except Exception as e:
                    print(f"  [Dashboard render error: {e}]")

            if k % self.save_interval == 0:
                self._save_checkpoint(str(logger.checkpoint_dir), k)

        # Final saves
        self._save_checkpoint(str(logger.checkpoint_dir), num_steps)
        try:
            run_dashboard.render([str(logger.log_file)])
        except Exception as e:
            print(f"  [Final dashboard render error: {e}]")

        print(f"\n[{self.selector.name}] Training complete. Log: {logger.log_file}")
        print(f"                            Dashboard: {logger.dashboard_file}")
        print(f"                            Checkpoint: {logger.checkpoint_dir}")
        return str(logger.log_file)

    def _save_checkpoint(self, checkpoint_dir, step: int):
        """Save model weights to checkpoint_dir, overwriting previous."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "lm_optimizer_state_dict": self.lm_optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, "checkpoint.pt"))
        print(f"  [{self.selector.name} Checkpoint saved: step {step} -> {checkpoint_dir}]")

    def _full_eval(self, held_out: HeldOutSet) -> tuple:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for i in range(0, held_out.data.shape[0], self.eval_batch_size):
                batch = held_out.data[i : i + self.eval_batch_size]
                outputs = self.model(batch, labels=batch)
                total_loss += outputs.loss.item()
                n_batches += 1
        self.model.train()
        avg_ce = total_loss / n_batches
        return -avg_ce, math.exp(avg_ce) if avg_ce < 100 else float("inf")


class LangevinRAGTrainer:
    """
    Trainer that uses Langevin-dynamics Q-guided search + RAG retrieval.

    Pipeline per step:
        1. Run Langevin dynamics in GPT-2 embedding space, using Q(s,a) as
           the energy function, to find continuous embeddings with high Q-value.
        2. Snap continuous embeddings to discrete tokens via nearest-neighbor
           lookup against the wte embedding matrix.
        3. Embed the resulting "query sentences" with a sentence-transformer,
           search a pre-built FAISS index of the training dataset, and
           retrieve top-k real training examples.
        4. Train the LM on the retrieved examples with uniform (unweighted)
           loss — no Q-value weighting.
        5. Compute reward on held-out data and update the Q-network via the
           standard normalised discounted soft Bellman equation.

    The Q-network still learns from the reward signal, so it improves its
    energy landscape over time, guiding Langevin dynamics toward queries
    that retrieve increasingly useful training data.
    """

    def __init__(
        self,
        # Model
        model_name: str = "gpt2",
        # Data
        dataset_name: str = "openwebtext",
        seq_len: int = 1024,
        held_out_subset_size: int = 512,
        held_out_total_size: int = 8192,
        # Batching
        extract_batch_size: int = 32,
        eval_batch_size: int = 64,
        # Q-learning
        beta: float = 1.0,
        gamma: float = 0.9,
        q_lr: float = 1e-4,
        q_grad_clip: float = 1.0,
        tau: float = 0.01,
        # LM training
        lm_lr: float = 5e-5,
        grad_clip: float = 1.0,
        # Langevin dynamics
        langevin_seq_len: int = 128,
        langevin_num_chains: int = 8,
        langevin_num_samples: int = 8,
        langevin_steps: int = 100,
        langevin_burn_in: int = 50,
        langevin_thin: int = 5,
        langevin_step_size: float = 0.01,
        langevin_temperature: float = 1.0,
        langevin_noise_scale: float = 1.0,
        langevin_grad_clip: float = 1.0,
        langevin_batch_size: int = 4,
        # LM training micro-batching
        lm_micro_batch_size: int = 4,
        # RAG
        rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rag_index_size: int = 50000,
        rag_top_k: int = 8,
        # Logging
        log_interval: int = 10,
        eval_interval: int = 100,
        dashboard_interval: int = 50,
        save_interval: int = 500,
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: str = "autodidact",
        # Device
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[LangevinRAG] Using device: {self.device}")

        # --- Base model ---
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.train()
        hidden_dim = self.model.config.n_embd

        # --- Q-network ---
        self.q_net = QNetwork(hidden_dim=hidden_dim, inner_dim=32).to(self.device)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=q_lr)

        # --- Target Q-network ---
        import copy
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
        self.held_out_subset_size = held_out_subset_size
        self.extract_batch_size = extract_batch_size
        self.eval_batch_size = eval_batch_size
        self.seq_len = seq_len
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.dashboard_interval = dashboard_interval
        self.save_interval = save_interval

        # --- Data ---
        self.dataset_name = dataset_name
        self.held_out_total_size = held_out_total_size

        # --- Logging ---
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # --- LM micro-batch size ---
        self.lm_micro_batch_size = lm_micro_batch_size

        # --- Langevin sampler ---
        self.langevin_sampler = LangevinQSampler(
            model=self.model,
            q_net=self.q_net,
            seq_len=langevin_seq_len,
            num_chains=langevin_num_chains,
            num_samples=langevin_num_samples,
            langevin_steps=langevin_steps,
            burn_in=langevin_burn_in,
            thin=langevin_thin,
            step_size=langevin_step_size,
            temperature=langevin_temperature,
            noise_scale=langevin_noise_scale,
            grad_clip=langevin_grad_clip,
            langevin_batch_size=langevin_batch_size,
            device=self.device,
        )

        # --- RAG index ---
        self.rag_index = RAGIndex(
            embedding_model_name=rag_embedding_model,
            index_size=rag_index_size,
            retrieval_top_k=rag_top_k,
            device=self.device,
        )

        # Store config
        self._config_dict = {
            "method_name": "langevin_rag",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seq_len": seq_len,
            "held_out_subset_size": held_out_subset_size,
            "held_out_total_size": held_out_total_size,
            "beta": beta,
            "gamma": gamma,
            "q_lr": q_lr,
            "q_grad_clip": q_grad_clip,
            "tau": tau,
            "lm_lr": lm_lr,
            "grad_clip": grad_clip,
            "langevin_seq_len": langevin_seq_len,
            "langevin_num_chains": langevin_num_chains,
            "langevin_num_samples": langevin_num_samples,
            "langevin_steps": langevin_steps,
            "langevin_burn_in": langevin_burn_in,
            "langevin_thin": langevin_thin,
            "langevin_step_size": langevin_step_size,
            "langevin_temperature": langevin_temperature,
            "langevin_batch_size": langevin_batch_size,
            "lm_micro_batch_size": lm_micro_batch_size,
            "rag_embedding_model": rag_embedding_model,
            "rag_index_size": rag_index_size,
            "rag_top_k": rag_top_k,
        }

        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, config=self._config_dict)

    @torch.no_grad()
    def _polyak_update(self):
        """Soft-update target Q-network toward online Q-network."""
        for p, p_target in zip(self.q_net.parameters(), self.q_target.parameters()):
            p_target.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def _q_step(
        self,
        h_prev: torch.Tensor,
        r_prev: float,
        soft_value_next: torch.Tensor,
    ) -> tuple:
        """
        Q-network update (normalised discounted soft Bellman).
        Same as AutodidactTrainer._q_step.
        """
        self.q_optimizer.zero_grad()
        q_hat = self.q_net(h_prev.unsqueeze(0)).squeeze()
        y = (1 - self.gamma) * r_prev + self.gamma * soft_value_next.detach()
        td_loss = 0.5 * (q_hat - y) ** 2
        td_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.q_grad_clip
        ).item()
        self.q_optimizer.step()
        self._polyak_update()
        return td_loss.item(), q_grad_norm

    def _lm_step_batch(self, batch_ids: torch.Tensor, micro_batch_size: Optional[int] = None) -> tuple:
        """
        Train LM on a batch of retrieved examples with uniform (unweighted) loss.

        Uses gradient accumulation over micro-batches to control memory.
        Each micro-batch of size `micro_batch_size` is forwarded independently;
        gradients are averaged across all micro-batches before the optimizer step.

        Args:
            batch_ids: [B, seq_len] token IDs of retrieved examples.
            micro_batch_size: Number of examples per micro-batch forward pass.
                              Defaults to self.lm_micro_batch_size.

        Returns:
            (mean_loss, grad_norm)
        """
        if micro_batch_size is None:
            micro_batch_size = self.lm_micro_batch_size
        self.lm_optimizer.zero_grad()
        B = batch_ids.shape[0]
        n_micro = max(1, (B + micro_batch_size - 1) // micro_batch_size)
        total_loss = 0.0

        for i in range(0, B, micro_batch_size):
            micro = batch_ids[i : i + micro_batch_size]
            outputs = self.model(micro, labels=micro)
            # Scale loss so that accumulated gradients average correctly
            loss = outputs.loss / n_micro
            loss.backward()
            total_loss += outputs.loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        ).item()
        self.lm_optimizer.step()
        return total_loss / n_micro, grad_norm

    def _save_checkpoint(self, checkpoint_dir, step: int):
        """Save model and Q-network weights."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "q_net_state_dict": self.q_net.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "lm_optimizer_state_dict": self.lm_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, "checkpoint.pt"))
        print(f"  [LangevinRAG Checkpoint saved: step {step} -> {checkpoint_dir}]")

    def _full_eval(self, held_out: HeldOutSet) -> tuple:
        """Evaluate on the full held-out set. Returns (neg_per_token_ce, perplexity)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for i in range(0, held_out.data.shape[0], self.eval_batch_size):
                batch = held_out.data[i : i + self.eval_batch_size]
                outputs = self.model(batch, labels=batch)
                total_loss += outputs.loss.item()
                n_batches += 1
        self.model.train()
        avg_ce = total_loss / n_batches
        return -avg_ce, math.exp(avg_ce) if avg_ce < 100 else float("inf")

    def train(self, num_steps: int = 10000) -> str:
        """
        Run the Langevin-RAG training loop.

        Each step:
            1. Langevin dynamics -> query token IDs
            2. RAG retrieval -> real training examples
            3. Unweighted LM training on retrieved examples
            4. Reward computation on held-out data
            5. Q-network update (soft Bellman)

        For the Q-network update, we need a hidden state from a "selected action".
        We use the first retrieved example's hidden state as the representative
        action for the Bellman backup (the Q-network learns the value of the
        retrieval outcome, not of any single candidate).

        Returns:
            Path to the JSONL log file.
        """
        # --- Set up logging ---
        logger = MetricsLogger("langevin_rag", log_dir=self.log_dir, config=self._config_dict)
        run_dashboard = LangevinRAGDashboard(output_path=str(logger.dashboard_file))

        # --- Build RAG index ---
        self.rag_index.build_index(
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            dataset_name=self.dataset_name,
        )

        # --- Build held-out set ---
        print("[LangevinRAG] Building held-out evaluation set...")
        held_out = HeldOutSet(
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            total_size=self.held_out_total_size,
            dataset_name=self.dataset_name,
            device=self.device,
        )

        cumulative_reward = 0.0

        # =====================================================================
        # Bootstrap step (k=0): no Q update
        # =====================================================================
        print("[LangevinRAG] Bootstrap step...")

        # Langevin sampling -> query sentences
        query_ids, langevin_info = self.langevin_sampler.sample()

        # RAG retrieval -> real training examples
        t_rag = time.time()
        retrieved_ids, rag_info = self.rag_index.retrieve(
            query_ids, self.tokenizer
        )
        retrieved_ids = retrieved_ids.to(self.device)
        rag_time = time.time() - t_rag

        # Free Langevin memory before LM forward/backward
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Get hidden state of first retrieved example for Q-learning bookkeeping
        h_prev = extract_hidden_states(
            self.model, retrieved_ids[:1], batch_size=1
        ).squeeze(0)  # [d]

        # LM training on retrieved examples (unweighted)
        t_lm = time.time()
        lm_loss, lm_grad_norm = self._lm_step_batch(retrieved_ids)
        lm_time = time.time() - t_lm

        # Reward
        t_reward = time.time()
        D_hat = held_out.sample_subset(self.held_out_subset_size)
        r_prev = compute_held_out_reward(self.model, D_hat, eval_batch_size=self.eval_batch_size)
        reward_time = time.time() - t_reward
        cumulative_reward += r_prev

        # Log best-Q sample text
        best_idx = langevin_info["best_q_sample_idx"]
        best_q_text = self.tokenizer.decode(query_ids[best_idx].tolist())

        logger.log({
            "step": 0,
            "reward": r_prev,
            "lm_loss": lm_loss,
            "td_loss": 0.0,
            "cumulative_reward": cumulative_reward,
            "lm_grad_norm": lm_grad_norm,
            "q_grad_norm": 0.0,
            # RAG
            "num_retrieved": rag_info["num_retrieved_unique"],
            "rag_avg_top1_score": rag_info["avg_top1_score"],
            "rag_avg_topk_score": rag_info["avg_topk_score"],
            # SGLD Q-values (collected samples — biased high)
            "sgld_q_final_mean": langevin_info["q_final_mean"],
            "sgld_q_final_std": langevin_info["q_final_std"],
            "sgld_q_best": langevin_info["q_best"],
            "sgld_q_gain": langevin_info["q_gain"],
            # Unbiased baseline
            "q_random_mean": langevin_info["q_random_mean"],
            "q_random_std": langevin_info["q_random_std"],
            # SGLD diversity
            "sgld_embed_cosine_mean": langevin_info["embed_pairwise_cosine_mean"],
            "sgld_token_unique_ratio": langevin_info["token_unique_ratio"],
            "sgld_token_jaccard_mean": langevin_info["token_jaccard_mean"],
            # SGLD gradient health
            "sgld_grad_norm_mean": langevin_info["grad_norm_mean"],
            "sgld_grad_clip_frac": langevin_info["grad_clip_frac"],
            # Per-phase timing
            "time_sgld_s": langevin_info["sgld_time_s"],
            "time_rag_s": rag_time,
            "time_lm_s": lm_time,
            "time_reward_s": reward_time,
            "step_time": 0.0,
        })

        print(f"  Bootstrap: lm_loss={lm_loss:.4f}, reward={r_prev:.4f}, "
              f"retrieved={rag_info['num_retrieved_unique']}, "
              f"sgld_q={langevin_info['q_final_mean']:.4f} (random={langevin_info['q_random_mean']:.4f})")

        # =====================================================================
        # Main loop
        # =====================================================================
        print(f"\n[LangevinRAG] Starting main loop for {num_steps} steps...")

        for k in range(1, num_steps + 1):
            step_start = time.time()

            # --- 1. Langevin dynamics -> query token IDs ---
            query_ids, langevin_info = self.langevin_sampler.sample()
            # sgld_time_s is already in langevin_info

            # Free Langevin computation graph before heavy LM operations
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # --- 2. RAG retrieval -> real training examples ---
            t_rag = time.time()
            retrieved_ids, rag_info = self.rag_index.retrieve(
                query_ids, self.tokenizer
            )
            retrieved_ids = retrieved_ids.to(self.device)
            rag_time = time.time() - t_rag

            # --- 3. Q update for previous transition ---
            H_retrieved = extract_hidden_states(
                self.model, retrieved_ids, batch_size=self.extract_batch_size
            )  # [R, d]
            q_retrieved = self.q_net(H_retrieved)  # [R, 1]

            with torch.no_grad():
                q_retrieved_target = self.q_target(H_retrieved)
            V_target = compute_soft_value(q_retrieved_target, self.beta)
            td_loss, q_grad_norm = self._q_step(h_prev, r_prev, V_target)

            h_prev = H_retrieved[0].detach()

            # --- 4. LM training on retrieved examples (unweighted) ---
            t_lm = time.time()
            lm_loss, lm_grad_norm = self._lm_step_batch(retrieved_ids)
            lm_time = time.time() - t_lm

            # --- 5. Reward ---
            t_reward = time.time()
            D_hat = held_out.sample_subset(self.held_out_subset_size)
            r_k = compute_held_out_reward(self.model, D_hat, eval_batch_size=self.eval_batch_size)
            r_prev = r_k
            cumulative_reward += r_k
            reward_time = time.time() - t_reward

            step_time = time.time() - step_start

            # --- GPU memory ---
            if self.device.type == "cuda":
                gpu_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                gpu_mem_gb = 0.0

            # --- Metrics ---
            q_squeezed = q_retrieved.squeeze(-1)
            metrics = {
                "step": k,
                "reward": r_k,
                "lm_loss": lm_loss,
                "td_loss": td_loss,
                "cumulative_reward": cumulative_reward,
                "lm_grad_norm": lm_grad_norm,
                "q_grad_norm": q_grad_norm,
                # Retrieved-example Q-values (biased — these were selected by SGLD)
                "q_retrieved_mean": q_squeezed.mean().item(),
                "q_retrieved_std": q_squeezed.std().item() if q_squeezed.numel() > 1 else 0.0,
                "soft_value": compute_soft_value(q_retrieved, self.beta).item(),
                # RAG
                "num_retrieved": rag_info["num_retrieved_unique"],
                "rag_avg_top1_score": rag_info["avg_top1_score"],
                "rag_avg_topk_score": rag_info["avg_topk_score"],
                # SGLD Q-values (collected samples — biased high by design)
                "sgld_q_final_mean": langevin_info["q_final_mean"],
                "sgld_q_final_std": langevin_info["q_final_std"],
                "sgld_q_best": langevin_info["q_best"],
                "sgld_q_gain": langevin_info["q_gain"],
                # Unbiased Q baseline (random embeddings)
                "q_random_mean": langevin_info["q_random_mean"],
                "q_random_std": langevin_info["q_random_std"],
                # SGLD diversity
                "sgld_embed_cosine_mean": langevin_info["embed_pairwise_cosine_mean"],
                "sgld_token_unique_ratio": langevin_info["token_unique_ratio"],
                "sgld_token_jaccard_mean": langevin_info["token_jaccard_mean"],
                # SGLD gradient health
                "sgld_grad_norm_mean": langevin_info["grad_norm_mean"],
                "sgld_grad_clip_frac": langevin_info["grad_clip_frac"],
                # Per-phase timing
                "time_sgld_s": langevin_info["sgld_time_s"],
                "time_rag_s": rag_time,
                "time_lm_s": lm_time,
                "time_reward_s": reward_time,
                "step_time": step_time,
                # GPU memory
                "gpu_peak_mem_gb": gpu_mem_gb,
            }

            logger.log(metrics)

            if k % self.log_interval == 0:
                sgld_pct = langevin_info["sgld_time_s"] / max(step_time, 1e-6) * 100
                lm_pct = lm_time / max(step_time, 1e-6) * 100
                print(
                    f"[LangevinRAG] Step {k:6d} | reward={r_k:.4f} | "
                    f"lm_loss={lm_loss:.4f} | td={td_loss:.6f} | "
                    f"sgld_Q={langevin_info['q_final_mean']:.3f} "
                    f"(rand={langevin_info['q_random_mean']:.3f} "
                    f"gain={langevin_info['q_gain']:+.3f}) | "
                    f"div={langevin_info['token_unique_ratio']:.0%} | "
                    f"retr={rag_info['num_retrieved_unique']} | "
                    f"{step_time:.2f}s "
                    f"[sgld={sgld_pct:.0f}% lm={lm_pct:.0f}%]"
                )
                if self.use_wandb:
                    import wandb
                    wandb.log(metrics)

            # --- Periodic best-Q sample text ---
            if k % self.log_interval == 0:
                best_idx = langevin_info["best_q_sample_idx"]
                best_q_text = self.tokenizer.decode(query_ids[best_idx].tolist())
                best_q_val = langevin_info["q_best"]
                print(f"  [Best-Q sample (Q={best_q_val:.4f})]: {best_q_text[:200]}")
                logger.log({
                    "_type": "best_sample",
                    "step": k,
                    "best_q_value": best_q_val,
                    "best_q_text": best_q_text[:500],
                    "sample_query_0": rag_info.get("sample_query", "")[:200],
                })

            if k % self.eval_interval == 0:
                eval_reward, eval_ppl = self._full_eval(held_out)
                print(f"  [LangevinRAG EVAL] step={k} | reward={eval_reward:.4f} | ppl={eval_ppl:.2f}")
                logger.log_eval({
                    "step": k,
                    "eval_reward": eval_reward,
                    "eval_perplexity": eval_ppl,
                })
                if self.use_wandb:
                    import wandb
                    wandb.log({"eval_reward": eval_reward, "eval_perplexity": eval_ppl, "step": k})

            if k % self.dashboard_interval == 0:
                try:
                    run_dashboard.render([str(logger.log_file)])
                except Exception as e:
                    print(f"  [Dashboard render error: {e}]")

            if k % self.save_interval == 0:
                self._save_checkpoint(str(logger.checkpoint_dir), k)

        # Final saves
        self._save_checkpoint(str(logger.checkpoint_dir), num_steps)
        try:
            run_dashboard.render([str(logger.log_file)])
        except Exception as e:
            print(f"  [Final dashboard render error: {e}]")

        print(f"\n[LangevinRAG] Training complete. Log: {logger.log_file}")
        print(f"                          Dashboard: {logger.dashboard_file}")
        print(f"                          Checkpoint: {logger.checkpoint_dir}")
        return str(logger.log_file)
