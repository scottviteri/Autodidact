"""
Hyperparameter configuration with dataclass defaults.

Defaults are tuned for an H100 80GB GPU running the softmax-relaxed Langevin-RAG pipeline:
    - GPT-2 (124M params) uses ~0.5GB in fp32. Plenty of room.
    - seq_len=512: context windows for retrieved training examples.
    - Softmax-relaxed SGLD in logit space generates query sentences, which are
      used via RAG to retrieve real training data from a FAISS index.
    - The softmax bridge keeps inputs on the token manifold, avoiding the
      snap bottleneck of the legacy embedding-space sampler.
    - The LM trains on retrieved examples (uniform or Q-weighted loss).
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

    # --- Topic-coherent held-out set ---
    topic_coherent: bool = False      # Sample held-out set around a random topic anchor
    topic_pool_size: int = 4096       # Number of candidate windows to embed for topic selection
    topic_temperature: float = 0.1    # Softmax temperature for topic-based sampling
                                      # Low = tight topic cluster, high = diffuse (→ uniform)

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
    no_q_weighting: bool = False     # When False (default), weight LM loss by pi = softmax(Q/beta).
                                     # This is the expected gradient under the optimal soft policy,
                                     # giving the Q-network's long-horizon preferences influence over
                                     # gradient magnitude, not just data selection.
                                     # When True, use uniform weights (1/R or 1/N).

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

    # --- Alternating training ---
    alternating_period: int = 0       # If > 0, alternate between reset-theta mode and regular mode
                                      # every alternating_period steps. E.g. 500 means:
                                      #   steps 0-499:   reset theta each step (only Q learns)
                                      #   steps 500-999: regular training (LM accumulates)
                                      #   steps 1000-1499: reset theta again, etc.
                                      # The Q-network always learns across all steps.

    # --- Langevin-RAG mode (default) ---
    langevin_rag: bool = True         # Use Langevin Q-guided search + RAG retrieval
    sampler_type: str = "softmax"     # "softmax" (softmax-relaxed, default) or "embedding" (legacy)
    langevin_seq_len: int = 64        # Sequence length for Langevin optimization
    langevin_num_chains: int = 64     # K: parallel chains = K output samples (one per chain)
    langevin_burn_in: int = 40        # Number of Langevin steps before collecting
    langevin_step_size: float = 0.01  # Langevin step size epsilon
    langevin_temperature: float = 1.0 # SGLD energy temperature (scales Q in energy)
    langevin_noise_scale: float = 1.0 # Multiplier on Gaussian noise term
    langevin_grad_clip: float = 1.0   # Clip gradients per chain
    langevin_batch_size: int = 64     # Chains to process in parallel in _energy()
    # Softmax bridge temperature annealing (only used when sampler_type="softmax")
    softmax_tau_start: float = 2.0    # Softmax temperature at step 0 (high = soft/exploratory)
    softmax_tau_end: float = 0.5      # Softmax temperature at final step (low = sharp/peaked)
    lm_micro_batch_size: int = 64     # Micro-batch size for gradient-accumulated LM training
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_index_size: int = 50000       # Number of dataset windows to index
    rag_top_k: int = 16              # FAISS candidates per query (scored, then sampled)
    rag_sample_from_topk: bool = True # Softmax-sample 1 from top-k per query (vs deterministic top-1)
    rag_sample_temperature: float = 0.1  # Temperature for softmax sampling over top-k scores
    rag_ingest_per_step: int = 100    # Rolling refresh: fresh windows consumed per step (0=static index)
                                      # The index is a ring buffer; each step, this many new windows are
                                      # encoded, written over the oldest entries, and the FAISS index is
                                      # rebuilt. Default 100 = same throughput as the old bulk refresh
                                      # (50000 windows / 500 steps) but spread smoothly to avoid the
                                      # LM loss jump from sudden distribution shifts.

    # --- Q experience replay ---
    q_replay_buffer_size: int = 256   # Ring buffer capacity for (h, y) transitions (0 = disabled).
                                      # Stores hidden states and pre-computed TD targets from past steps.
                                      # Replaying past transitions gives the Q-head many more gradient
                                      # updates per environment step, dramatically improving sample
                                      # efficiency.  256 ≈ the last 256 training steps; hidden states
                                      # remain on-distribution with slow LM learning rates.
    q_replay_batch_size: int = 32     # Mini-batch size for each replay gradient step.
    q_replay_updates_per_step: int = 1  # Extra Q gradient steps from replay after each online TD update.
                                        # Total Q updates per step = 1 (online) + this many (replay).
                                        # 4 replay updates with batch_size=32 means the Q-head sees
                                        # 128 + 1 = 129 effective training examples per step vs. 1.
