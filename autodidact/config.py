"""
Hyperparameter configuration with dataclass defaults.

Defaults are tuned for an H100 80GB GPU:
    - GPT-2 (124M params) uses ~0.5GB in fp32. Plenty of room.
    - seq_len=1024: full GPT-2 context window.
    - N=256 candidates: large pool gives the Q-network rich selection.
    - M=512 held-out subset: richer reward signal per step.
    - |D|=8192 held-out total: ~8M tokens of held-out data.
    - extract_batch_size=32, eval_batch_size=64: large batches to utilize H100.
    - AdamW for LM, Adam for Q-network.

GPU memory budget (H100 80GB):
    Persistent: model (475 MB) + AdamW (949 MB) + held-out (64 MB) = ~1.5 GB
    extract_hidden_states uses model.transformer directly (no LM head, no
    intermediate hidden states), so peak per mini-batch is ~[B, seq_len, d]
    instead of [B, seq_len, vocab_size] + 13x[B, seq_len, d].
    Total peak ~3-4 GB, leaving >95% headroom.

Q-network stability:
    - q_lr=1e-4 (not 1e-3): the Q-network is a tiny 2-layer MLP; higher LRs
      cause Q-values to blow up via the soft-value bootstrap feedback loop.
    - q_grad_clip=0.1: separate, tighter grad clip for the Q-network. The reward
      signal is O(1) (per-token log-prob ~-3.3), so Q-values should stay O(1) too.
      Tight clipping prevents the logsumexp bootstrap from amplifying noise.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AutodidactConfig:
    """Full configuration for Autodidact training."""

    # --- Model ---
    model_name: str = "gpt2"

    # --- Data ---
    dataset_name: str = "openwebtext"
    seq_len: int = 1024              # Full GPT-2 context window
    num_candidates: int = 256        # N: candidates per step
    held_out_subset_size: int = 512  # M: held-out subset size per step
    held_out_total_size: int = 8192  # |D|: total held-out set size

    # --- Batching (GPU utilization) ---
    extract_batch_size: int = 32     # Mini-batch size for candidate hidden state extraction
    eval_batch_size: int = 64        # Mini-batch size for held-out reward computation

    # --- Q-learning ---
    beta: float = 1.0               # Boltzmann temperature
    q_lr: float = 1e-4              # eta: Q-network learning rate
    q_grad_clip: float = 0.1        # Separate grad clip for Q-network
    tau: float = 0.01               # EMA rate for rho

    # --- LM training ---
    lm_lr: float = 5e-5             # alpha: LM learning rate
    grad_clip: float = 1.0          # G: LM gradient clipping (max global norm)

    # --- Training ---
    num_steps: int = 10000
    log_interval: int = 10          # Log every N steps (to JSONL + stdout)
    eval_interval: int = 100        # Full held-out eval every N steps
    dashboard_interval: int = 50    # Refresh per-run dashboard PNG every N steps

    # --- Logging ---
    log_dir: str = "logs"           # JSONL logs and per-run dashboards go here
    use_wandb: bool = False
    wandb_project: str = "autodidact"

    # --- Device ---
    device: Optional[str] = None

    # --- Baselines ---
    run_baselines: bool = False
    baseline_methods: str = "random,loss_based,uncertainty"  # comma-separated
