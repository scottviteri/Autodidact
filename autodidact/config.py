"""
Hyperparameter configuration with dataclass defaults.

Defaults are tuned for an H100 80GB GPU running the Langevin-RAG pipeline:
    - GPT-2 (124M params) uses ~0.5GB in fp32. Plenty of room.
    - seq_len=512: context windows for retrieved training examples.
    - Langevin SGLD in embedding space generates query sentences, which are
      used via RAG to retrieve real training data from a FAISS index.
    - The LM trains on retrieved examples with uniform (unweighted) loss.
    - AdamW for LM, Adam for Q-network.

Q-learning formulation:
    Uses normalised discounted soft Bellman equation:
        Q(s, a) = (1 - gamma) * r + gamma * V(s')
        V(s) = beta * [logsumexp(Q(s, a') / beta) - log(N)]
    The (1 - gamma) factor on the reward keeps Q-values on the same scale as
    the reward regardless of gamma (a constant reward r gives Q* = r + const).
    A target network with Polyak averaging stabilises the bootstrap target.
    gamma=0.9 gives an effective horizon of ~10 steps.

Discrete-Q mode (--no_langevin_rag):
    The LM trains on a policy-weighted mixture of all N candidates,
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
    seq_len: int = 512               # Context window for training examples
    num_candidates: int = 256        # N: candidates per step (discrete-Q mode)
    held_out_subset_size: int = 256  # M: held-out subset size per step
    held_out_total_size: int = 2048  # |D|: total held-out set size

    # --- Batching (GPU utilization) ---
    extract_batch_size: int = 64     # Mini-batch size for candidate hidden state extraction
    eval_batch_size: int = 64        # Mini-batch size for held-out reward computation

    # --- Q-learning ---
    beta: float = 1.0               # Boltzmann temperature
    gamma: float = 0.9              # Discount factor (horizon ~1/(1-gamma) = 10 steps)
    q_lr: float = 1e-3              # eta: Q-network learning rate
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

    # --- Ablation: reset LM each step ---
    reset_lm_each_step: bool = False  # Reset LM weights to initial theta after each step (only Q learns)

    # --- Baselines ---
    run_baselines: bool = False
    baseline_methods: str = "random,loss_based,uncertainty"  # comma-separated

    # --- Needle experiment ---
    needle: bool = False              # Run needle-in-a-haystack Q-value validation
    needle_text: Optional[str] = None # Custom text for the needle sequence

    # --- TD-into-theta: let TD gradients shape LM representations ---
    td_into_theta: bool = False       # Allow TD loss gradients to flow into theta (LM weights)
                                      # When enabled, the TD step re-forwards the previous action
                                      # through the current theta with gradients, so the LM
                                      # representations are shaped by both the LM loss and the
                                      # Q-learning TD loss.
    td_lambda: float = 1.0            # Scaling coefficient for TD gradients flowing into theta.
                                      # The effective LM update from the TD step is:
                                      #   theta -= lm_lr * td_lambda * d(TD_loss)/d(theta)
                                      # Use < 1.0 to downweight the TD signal relative to LM loss.

    # --- Q-head warmup ---
    q_warmup_steps: int = 0           # Number of warmup steps with random sampling + theta reset
                                      # During warmup: random data (no SGLD), theta reset each step,
                                      # only Q-network learns. Lets Q-head calibrate before guiding SGLD.

    # --- Langevin-RAG mode (default) ---
    langevin_rag: bool = True         # Use Langevin Q-guided search + RAG retrieval
    langevin_seq_len: int = 64        # Sequence length for Langevin embedding optimization
    langevin_num_chains: int = 64      # K: parallel Langevin chains
    langevin_num_samples: int = 64     # Total samples to collect from Langevin
    langevin_steps: int = 50         # Total Langevin steps (burn-in + collection)
    langevin_burn_in: int = 40        # Discard first N steps
    langevin_thin: int = 10            # Keep every N-th sample after burn-in
    langevin_step_size: float = 0.01  # Langevin step size epsilon
    langevin_temperature: float = 1.0 # Sampling temperature (scales energy)
    langevin_noise_scale: float = 1.0 # Multiplier on Gaussian noise term
    langevin_grad_clip: float = 1.0   # Clip embedding gradients per chain
    langevin_batch_size: int = 64     # Chains to process in parallel in _energy()
    lm_micro_batch_size: int = 64     # Micro-batch size for gradient-accumulated LM training
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_index_size: int = 50000       # Number of dataset windows to index
    rag_top_k: int = 4                # Retrieved examples per query
