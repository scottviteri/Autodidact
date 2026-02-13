"""
Main training loop implementing Algorithm 1 from the spec:
    Curriculum Selection via Differential Soft Q-Learning.

Key implementation insight (from Section 2.5):
    Each step's forward pass of candidates through the base model serves two 
    purposes: action selection for the *current* step and the bootstrap target 
    for the *previous* step's Q update. This avoids a redundant second forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Optional, Dict, Any, List
from dataclasses import asdict
import time
import math

from .q_network import QNetwork, extract_hidden_states, compute_soft_value, boltzmann_sample
from .data import CandidateBatchSampler, HeldOutSet
from .baselines import BaselineSelector
from .logging import MetricsLogger, DashboardPlotter


def compute_lm_loss(model: GPT2LMHeadModel, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute language modeling loss (cross-entropy) for a single example.
    
    Args:
        model: GPT-2 model.
        input_ids: [1, seq_len] or [seq_len] token IDs.
    
    Returns:
        Scalar loss.
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
    Compute reward: average log-probability on held-out subset.
    
    r_k = (1/M) * log p_{theta_{k+1}}(D_hat_k)
    
    Since HuggingFace returns cross-entropy loss (negative log-prob per token),
    we negate it to get log-probability. Processes in mini-batches for memory.
    
    Args:
        model: GPT-2 model (already updated to theta_{k+1}).
        held_out_subset: [M, seq_len] held-out examples.
        eval_batch_size: Mini-batch size for reward computation.
    
    Returns:
        Scalar reward (average negative cross-entropy).
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
    
    Manages:
        - GPT-2 base model with AdamW updates
        - Q-network with its own Adam optimizer
        - Average reward tracking (rho)
        - Boltzmann action selection
        - TD learning for the Q-network
        - JSONL metric logging (every step)
        - Live dashboard PNG (periodic refresh)
    """

    def __init__(
        self,
        # Model
        model_name: str = "gpt2",
        # Data
        dataset_name: str = "openwebtext",
        seq_len: int = 1024,
        num_candidates: int = 64,
        held_out_subset_size: int = 256,
        held_out_total_size: int = 4096,
        # Q-learning
        beta: float = 1.0,
        q_lr: float = 1e-3,
        tau: float = 0.01,
        # LM training
        lm_lr: float = 5e-5,
        grad_clip: float = 1.0,
        # Logging
        log_interval: int = 10,
        eval_interval: int = 100,
        dashboard_interval: int = 50,
        log_dir: str = "logs",
        dashboard_path: str = "dashboard.png",
        use_wandb: bool = False,
        wandb_project: str = "autodidact",
        # Device
        device: Optional[str] = None,
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

        # --- LM optimizer (AdamW, standard for transformer fine-tuning) ---
        self.lm_optimizer = torch.optim.AdamW(self.model.parameters(), lr=lm_lr)

        # --- Hyperparameters ---
        self.beta = beta
        self.tau = tau
        self.grad_clip = grad_clip
        self.num_candidates = num_candidates
        self.held_out_subset_size = held_out_subset_size
        self.seq_len = seq_len
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.dashboard_interval = dashboard_interval

        # --- Average reward estimate ---
        self.rho = 0.0

        # --- Data ---
        self.dataset_name = dataset_name
        self.held_out_total_size = held_out_total_size

        # --- Logging ---
        self.log_dir = log_dir
        self.dashboard_path = dashboard_path
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Store config dict for logger header
        self._config_dict = {
            "method_name": "q_learning",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seq_len": seq_len,
            "num_candidates": num_candidates,
            "held_out_subset_size": held_out_subset_size,
            "held_out_total_size": held_out_total_size,
            "beta": beta,
            "q_lr": q_lr,
            "tau": tau,
            "lm_lr": lm_lr,
            "grad_clip": grad_clip,
        }

        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, config=self._config_dict)

    def _lm_step(self, input_ids: torch.Tensor) -> tuple:
        """
        Perform one LM gradient step on the selected example, with gradient clipping.
        Returns (loss_value, grad_norm).
        """
        self.lm_optimizer.zero_grad()
        loss = compute_lm_loss(self.model, input_ids)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
        self.lm_optimizer.step()
        return loss.item(), grad_norm

    def _q_step(
        self,
        h_prev: torch.Tensor,
        r_prev: float,
        soft_value_next: torch.Tensor,
    ) -> tuple:
        """
        Perform one Q-network update step.
        Returns (td_loss_value, q_grad_norm).
        """
        self.q_optimizer.zero_grad()

        # Re-evaluate previous action with current phi
        q_hat = self.q_net(h_prev.unsqueeze(0)).squeeze()  # scalar

        # TD target (stop gradient)
        y = r_prev - self.rho + soft_value_next.detach()

        # TD loss
        td_loss = 0.5 * (q_hat - y) ** 2
        td_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip).item()
        self.q_optimizer.step()

        return td_loss.item(), q_grad_norm

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
        logger = MetricsLogger("q_learning", log_dir=self.log_dir, config=self._config_dict)
        dashboard = DashboardPlotter(output_path=self.dashboard_path)

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
        C_0 = sampler.next_batch()                          # [N, seq_len]
        H_0 = extract_hidden_states(self.model, C_0)       # [N, d]
        q_0 = self.q_net(H_0)                              # [N, 1]
        a_0 = boltzmann_sample(q_0, self.beta)              # int

        h_prev = H_0[a_0]                                  # [d]

        # LM update on selected candidate
        lm_loss, lm_grad_norm = self._lm_step(C_0[a_0].unsqueeze(0))

        # Compute reward after update
        D_hat = held_out.sample_subset(self.held_out_subset_size)
        r_prev = compute_held_out_reward(self.model, D_hat)

        # Initialize rho
        self.rho = r_prev
        cumulative_reward += r_prev

        # Log bootstrap
        V_0 = compute_soft_value(q_0, self.beta).item()
        logger.log({
            "step": 0,
            "reward": r_prev,
            "rho": self.rho,
            "lm_loss": lm_loss,
            "td_loss": 0.0,
            "policy_entropy": math.log(self.num_candidates),
            "policy_entropy_ratio": 1.0,
            "q_mean": q_0.mean().item(),
            "q_std": q_0.std().item(),
            "q_min": q_0.min().item(),
            "q_max": q_0.max().item(),
            "soft_value": V_0,
            "reward_minus_rho": 0.0,
            "cumulative_reward": cumulative_reward,
            "lm_grad_norm": lm_grad_norm,
            "q_grad_norm": 0.0,
            "selected_action": a_0,
            "step_time": 0.0,
        })

        print(f"Bootstrap: lm_loss={lm_loss:.4f}, reward={r_prev:.4f}, rho={self.rho:.4f}")

        # =====================================================================
        # Main loop (k = 1, 2, ...)
        # =====================================================================
        print(f"\nStarting main training loop for {num_steps} steps...")

        for k in range(1, num_steps + 1):
            step_start = time.time()

            # --- Forward candidates through base model (dual purpose) ---
            C_k = sampler.next_batch()                      # [N, seq_len]
            H_k = extract_hidden_states(self.model, C_k)    # [N, d]
            q_k = self.q_net(H_k)                           # [N, 1]

            # --- Q update for previous transition ---
            V_k = compute_soft_value(q_k, self.beta)        # scalar
            td_loss, q_grad_norm = self._q_step(h_prev, r_prev, V_k)

            # --- Action selection for current step ---
            a_k = boltzmann_sample(q_k, self.beta)          # int
            h_prev = H_k[a_k]                               # [d]

            # --- LM update on selected candidate ---
            lm_loss, lm_grad_norm = self._lm_step(C_k[a_k].unsqueeze(0))

            # --- Compute reward ---
            D_hat = held_out.sample_subset(self.held_out_subset_size)
            r_k = compute_held_out_reward(self.model, D_hat)

            # --- Update average reward ---
            self.rho = (1 - self.tau) * self.rho + self.tau * r_k
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
                "rho": self.rho,
                "lm_loss": lm_loss,
                "td_loss": td_loss,
                "policy_entropy": entropy,
                "policy_entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0,
                "q_mean": q_k.mean().item(),
                "q_std": q_k.std().item(),
                "q_min": q_k.min().item(),
                "q_max": q_k.max().item(),
                "soft_value": V_k.item(),
                "reward_minus_rho": r_k - self.rho,
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
                    f"Step {k:6d} | reward={r_k:.4f} | rho={self.rho:.4f} | "
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

            # --- Refresh dashboard ---
            if k % self.dashboard_interval == 0:
                try:
                    dashboard.render([str(logger.log_file)])
                except Exception as e:
                    print(f"  [Dashboard render error: {e}]")

        # Final dashboard render
        try:
            dashboard.render([str(logger.log_file)])
        except Exception as e:
            print(f"  [Final dashboard render error: {e}]")

        print(f"\nTraining complete. Log: {logger.log_file}")
        return str(logger.log_file)

    def _full_eval(self, held_out: HeldOutSet) -> tuple:
        """Evaluate on the full held-out set. Returns (reward, perplexity)."""
        self.model.eval()
        total_loss = 0.0
        batch_size = 16  # Keep logits tensor manageable: [16, seq_len, 50257]
        n = held_out.data.shape[0]
        n_batches = 0
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = held_out.data[i : i + batch_size]
                outputs = self.model(batch, labels=batch)
                total_loss += outputs.loss.item()
                n_batches += 1
        self.model.train()
        avg_reward = -total_loss / n_batches
        perplexity = math.exp(-avg_reward) if avg_reward > -100 else float("inf")
        return avg_reward, perplexity


class BaselineTrainer:
    """
    Trainer that uses baseline selection strategies instead of Q-learning.
    Shares the same LM training setup for fair comparison.
    Logs to its own JSONL file and contributes to the shared dashboard.
    """

    def __init__(
        self,
        selector: BaselineSelector,
        model_name: str = "gpt2",
        dataset_name: str = "openwebtext",
        seq_len: int = 1024,
        num_candidates: int = 64,
        held_out_subset_size: int = 256,
        held_out_total_size: int = 4096,
        lm_lr: float = 5e-5,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        dashboard_interval: int = 50,
        log_dir: str = "logs",
        dashboard_path: str = "dashboard.png",
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
        self.seq_len = seq_len
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.dashboard_interval = dashboard_interval
        self.dataset_name = dataset_name
        self.held_out_total_size = held_out_total_size
        self.log_dir = log_dir
        self.dashboard_path = dashboard_path
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
            "lm_lr": lm_lr,
            "grad_clip": grad_clip,
        }

    def train(self, num_steps: int = 10000, all_log_files: Optional[List[str]] = None) -> str:
        """
        Run baseline training loop.
        
        Args:
            num_steps: Number of training steps.
            all_log_files: List of all log file paths (for multi-method dashboard overlay).
        
        Returns:
            Path to this run's JSONL log file.
        """
        from .data import StreamingTextDataset

        logger = MetricsLogger(self.selector.name, log_dir=self.log_dir, config=self._config_dict)
        dashboard = DashboardPlotter(output_path=self.dashboard_path)

        # Track all log files for overlay dashboard
        log_files = list(all_log_files) if all_log_files else []
        log_files.append(str(logger.log_file))

        stream_dataset = StreamingTextDataset(
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            dataset_name=self.dataset_name,
        )
        sampler = CandidateBatchSampler(stream_dataset, self.num_candidates, self.device)

        held_out = HeldOutSet(
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            total_size=self.held_out_total_size,
            dataset_name=self.dataset_name,
            device=self.device,
        )

        cumulative_reward = 0.0

        for k in range(1, num_steps + 1):
            step_start = time.time()

            C_k = sampler.next_batch()
            a_k = self.selector.select(self.model, C_k)

            # LM update
            self.lm_optimizer.zero_grad()
            loss = compute_lm_loss(self.model, C_k[a_k].unsqueeze(0))
            loss.backward()
            lm_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            self.lm_optimizer.step()

            # Reward
            D_hat = held_out.sample_subset(self.held_out_subset_size)
            r_k = compute_held_out_reward(self.model, D_hat)
            cumulative_reward += r_k

            step_time = time.time() - step_start

            metrics = {
                "step": k,
                "reward": r_k,
                "lm_loss": loss.item(),
                "lm_grad_norm": lm_grad_norm,
                "cumulative_reward": cumulative_reward,
                "selected_action": a_k,
                "step_time": step_time,
            }

            # Always log to JSONL
            logger.log(metrics)

            if k % self.log_interval == 0:
                print(
                    f"[{self.selector.name}] Step {k:6d} | reward={r_k:.4f} | "
                    f"lm_loss={loss.item():.4f} | gnorm={lm_grad_norm:.3f} | {step_time:.2f}s"
                )
                if self.use_wandb:
                    import wandb
                    wandb.log({f"{self.selector.name}/{key}": val for key, val in metrics.items()})

            if k % self.eval_interval == 0:
                eval_reward, eval_ppl = self._full_eval(held_out)
                print(f"  [{self.selector.name} EVAL] step={k} | reward={eval_reward:.4f} | ppl={eval_ppl:.2f}")
                logger.log_eval({
                    "step": k,
                    "eval_reward": eval_reward,
                    "eval_perplexity": eval_ppl,
                })

            if k % self.dashboard_interval == 0:
                try:
                    dashboard.render(log_files)
                except Exception as e:
                    print(f"  [Dashboard render error: {e}]")

        # Final render
        try:
            dashboard.render(log_files)
        except Exception as e:
            print(f"  [Final dashboard render error: {e}]")

        print(f"\n[{self.selector.name}] Training complete. Log: {logger.log_file}")
        return str(logger.log_file)

    def _full_eval(self, held_out: HeldOutSet) -> tuple:
        self.model.eval()
        total_loss = 0.0
        batch_size = 16
        n = held_out.data.shape[0]
        n_batches = 0
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = held_out.data[i : i + batch_size]
                outputs = self.model(batch, labels=batch)
                total_loss += outputs.loss.item()
                n_batches += 1
        self.model.train()
        avg_reward = -total_loss / n_batches
        perplexity = math.exp(-avg_reward) if avg_reward > -100 else float("inf")
        return avg_reward, perplexity
