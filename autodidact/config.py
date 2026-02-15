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

Q-learning formulation:
    Uses normalised discounted soft Bellman equation:
        Q(s, a) = (1 - gamma) * r + gamma * V(s')
        V(s) = beta * [logsumexp(Q(s, a') / beta) - log(N)]
    The (1 - gamma) factor on the reward keeps Q-values on the same scale as
    the reward regardless of gamma (a constant reward r gives Q* = r + const).
    A target network with Polyak averaging stabilises the bootstrap target.
    gamma=0.9 gives an effective horizon of ~10 steps.

LM training:
    The LM always trains on a policy-weighted mixture of all N candidates,
    where the weights are the Boltzmann policy pi = softmax(Q/beta). This
    computes the expected gradient under the policy exactly, eliminating
    variance from single-action sampling. When --no_q_weighting is used,
    the weights are uniform (1/N), bypassing the Q-function for the LM step
    while the Q-network still trains normally.
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
    gamma: float = 0.9              # Discount factor (horizon ~1/(1-gamma) = 10 steps)
    q_lr: float = 1e-4              # eta: Q-network learning rate
    q_grad_clip: float = 1.0        # Q-network gradient clipping
    tau: float = 0.01               # Polyak averaging rate for target Q-network

    # --- LM training ---
    lm_lr: float = 5e-5             # alpha: LM learning rate
    grad_clip: float = 1.0          # G: LM gradient clipping (max global norm)

    # --- Training ---
    num_steps: int = 10000
    log_interval: int = 10          # Log every N steps (to JSONL + stdout)
    eval_interval: int = 100        # Full held-out eval every N steps
    dashboard_interval: int = 50    # Refresh per-run dashboard PNG every N steps
    save_interval: int = 500        # Save checkpoint every N steps (overwritten each time)

    # --- Logging ---
    log_dir: str = "logs"           # JSONL logs and per-run dashboards go here
    use_wandb: bool = False
    wandb_project: str = "autodidact"

    # --- Device ---
    device: Optional[str] = None

    # --- Mixture training ---
    mixture_batch_size: int = 8      # Mini-batch size for gradient-accumulated mixture training
    no_q_weighting: bool = False     # Use uniform weights instead of Q-derived Boltzmann weights for LM mixture

    # --- Baselines ---
    run_baselines: bool = False
    baseline_methods: str = "random,loss_based,uncertainty"  # comma-separated

    # --- Needle experiment ---
    needle: bool = False              # Run needle-in-a-haystack Q-value validation
    needle_text: Optional[str] = None # Custom text for the needle sequence

    # --- Langevin-RAG mode ---
    langevin_rag: bool = False        # Use Langevin Q-guided search + RAG retrieval
    langevin_seq_len: int = 128       # Sequence length for Langevin embedding optimization
    langevin_num_chains: int = 8      # K: parallel Langevin chains
    langevin_num_samples: int = 8     # Total samples to collect from Langevin
    langevin_steps: int = 100         # Total Langevin steps (burn-in + collection)
    langevin_burn_in: int = 50        # Discard first N steps
    langevin_thin: int = 5            # Keep every N-th sample after burn-in
    langevin_step_size: float = 0.01  # Langevin step size epsilon
    langevin_temperature: float = 1.0 # Sampling temperature (scales energy)
    langevin_noise_scale: float = 1.0 # Multiplier on Gaussian noise term
    langevin_grad_clip: float = 1.0   # Clip embedding gradients per chain
    langevin_batch_size: int = 4      # Chains to process in parallel in _energy()
    lm_micro_batch_size: int = 4      # Micro-batch size for gradient-accumulated LM training
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_index_size: int = 50000       # Number of dataset windows to index
    rag_top_k: int = 8                # Retrieved examples per step
