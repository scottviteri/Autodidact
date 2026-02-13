"""
Hyperparameter configuration with dataclass defaults.

Defaults are tuned for an H100 80GB GPU:
    - GPT-2 (124M params) uses ~0.5GB in fp32. Plenty of room.
    - seq_len=1024: full GPT-2 context window. No reason to truncate on H100.
    - N=64 candidates: forward pass of 64x1024 tokens is ~0.25GB activation memory.
      Larger candidate pools give the Q-network more to choose from.
    - M=256 held-out subset: richer reward signal per step.
    - |D|=4096 held-out total: ~4M tokens of held-out data.
    - Eval batch size of 128 fits comfortably.
    - AdamW for LM, Adam for Q-network.
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
    num_candidates: int = 64         # N: candidates per step (H100 can batch these easily)
    held_out_subset_size: int = 256  # M: held-out subset size per step
    held_out_total_size: int = 4096  # |D|: total held-out set size

    # --- Q-learning ---
    beta: float = 1.0               # Boltzmann temperature
    q_lr: float = 1e-3              # eta: Q-network learning rate
    tau: float = 0.01               # EMA rate for rho

    # --- LM training ---
    lm_lr: float = 5e-5             # alpha: LM learning rate (slightly higher for H100 throughput)
    grad_clip: float = 1.0          # G: gradient clipping (max global norm)

    # --- Training ---
    num_steps: int = 10000
    log_interval: int = 10          # Log every N steps (to JSONL + stdout)
    eval_interval: int = 100        # Full held-out eval every N steps
    dashboard_interval: int = 50    # Refresh dashboard PNG every N steps

    # --- Logging ---
    log_dir: str = "logs"
    dashboard_path: str = "dashboard.png"
    use_wandb: bool = False
    wandb_project: str = "autodidact"

    # --- Device ---
    device: Optional[str] = None

    # --- Baselines ---
    run_baselines: bool = False
    baseline_methods: str = "random,loss_based,uncertainty"  # comma-separated
