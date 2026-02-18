"""
Q-network for curriculum selection.

Architecture (from Section 2.3 of the spec):
    Q_phi(s, x) = W2 * ReLU(W1 * h_x + b1) + b2
    
    where h_x is the last-layer hidden state at the final token position
    from the base model, and phi = {W1, b1, W2, b2}.

    By default, h_x is detached from theta (no gradients flow into the LM).
    When --td_into_theta is enabled, extract_hidden_state_with_grad() is used
    for the TD step so that value gradients also shape the LM representations.

    Dimensions: d -> 32 -> 1  (d = 768 for GPT-2)

Also provides QReplayBuffer for sample-efficient Q-learning via experience
replay.  Each training step stores (h, y) pairs — the hidden state and its
TD target — in a fixed-capacity ring buffer.  Between online Q updates, extra
mini-batch gradient steps are drawn from the buffer, giving the Q-head many
more learning signals per environment step.
"""

import math

import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """
    2-layer MLP that maps base model hidden states to scalar Q-values.
    
    Input:  h_x in R^d  (detached last-layer hidden state)
    Output: Q(s, x) in R  (scalar value estimate)
    """

    def __init__(self, hidden_dim: int = 768, inner_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),  # W1, b1
            nn.ReLU(),
            nn.Linear(inner_dim, 1),           # W2, b2
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states, shape [N, d] or [d].
        Returns:
            Q-values, shape [N, 1] or [1].
        """
        return self.net(h)


class QReplayBuffer:
    """
    Fixed-capacity ring buffer for experience replay of Q-learning transitions.

    Stores (h, y) pairs where h is a detached hidden state and y is the
    pre-computed TD target:  y = (1 - gamma) * r + gamma * V_target(s').

    Storing the scalar target y (rather than (r, V')) avoids having to
    recompute V' at replay time and keeps the buffer tiny.  Hidden states
    do become stale as theta drifts, but with a short buffer (e.g. 256
    transitions) and slow LM learning rates the distribution shift is
    manageable — empirically this is the standard DQN tradeoff.

    Usage:
        buf = QReplayBuffer(capacity=256, hidden_dim=768, device='cuda')
        # each step:
        buf.push(h_prev.detach(), y.item())
        if buf.can_sample(batch_size=32):
            h_batch, y_batch = buf.sample(32)
            # compute loss on the batch and update Q
    """

    def __init__(self, capacity: int = 256, hidden_dim: int = 768,
                 device: torch.device = torch.device("cpu")):
        self.capacity = capacity
        self.device = device
        self._h = torch.zeros(capacity, hidden_dim, device=device)
        self._y = torch.zeros(capacity, device=device)
        self._size = 0
        self._write = 0

    def push(self, h: torch.Tensor, y: float):
        """Store one (h, y) transition."""
        self._h[self._write] = h.detach().to(self.device)
        self._y[self._write] = y
        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def sample(self, batch_size: int):
        """Return (h_batch [B, d], y_batch [B]) sampled uniformly."""
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return self._h[idx], self._y[idx]

    def __len__(self):
        return self._size


def extract_hidden_states(model, input_ids: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    """
    Forward candidates through the base model's transformer backbone and extract
    last-layer hidden states at the final token position, detached from the
    computation graph.
    
    Uses model.transformer directly instead of the full model forward:
      - Skips the LM head projection (saves [B, seq_len, vocab_size] logits)
      - Only returns the final layer's hidden state (not all 13 layers)
    This reduces peak memory from ~4.8 GB to ~0.6 GB per mini-batch of 32.
    
    Args:
        model: HuggingFace GPT-2 LMHeadModel (we call model.transformer).
        input_ids: [N, seq_len] token IDs.
        batch_size: Mini-batch size for forward passes.
    
    Returns:
        h: [N, d] detached hidden states.
    """
    transformer = model.transformer
    all_h = []
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[i : i + batch_size]
            outputs = transformer(batch)
            last_hidden = outputs.last_hidden_state
            h = last_hidden[:, -1, :].detach()
            all_h.append(h)
    return torch.cat(all_h, dim=0)


def extract_hidden_state_with_grad(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Forward a single example through the transformer WITH gradients into theta.

    Unlike extract_hidden_states() (which detaches), this keeps the full
    computation graph so that downstream losses (e.g. TD loss) can backprop
    into the LM parameters.  Used by the --td_into_theta option.

    Args:
        model: HuggingFace GPT-2 LMHeadModel (we call model.transformer).
        input_ids: [seq_len] or [1, seq_len] token IDs.

    Returns:
        h: [d] hidden state with grad_fn connected to theta.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    outputs = model.transformer(input_ids)
    return outputs.last_hidden_state[:, -1, :].squeeze(0)  # [d]


def compute_soft_value(q_values: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute the normalised soft value function:
        V(s) = beta * [logsumexp(Q(s, a') / beta) - log(N)]
    
    The log(N) subtraction removes the entropy bonus from the uniform prior.
    Without it, even when all Q-values are equal at q, V = q + beta*log(N),
    which inflates the Bellman fixed point by gamma*beta*log(N)/(1-gamma).
    With it, uniform Q-values give V = q, and the fixed point is Q* = r
    (matching the reward scale exactly).
    
    This does not affect the Boltzmann policy (softmax is shift-invariant).
    
    Args:
        q_values: [N, 1] Q-values for all candidates.
        beta: Boltzmann temperature.
    
    Returns:
        Scalar soft value V(s).
    """
    q = q_values.squeeze(-1)  # [N]
    n = q.shape[0]
    return beta * (torch.logsumexp(q / beta, dim=0) - math.log(n))


def boltzmann_sample(q_values: torch.Tensor, beta: float) -> int:
    """
    Sample an action from the Boltzmann policy:
        pi(a | s) = softmax(Q(s, a) / beta)
    
    Args:
        q_values: [N, 1] Q-values for all candidates.
        beta: Boltzmann temperature.
    
    Returns:
        Index of the selected action.
    """
    q = q_values.squeeze(-1)  # [N]
    logits = q / beta
    probs = torch.softmax(logits, dim=0)
    action = torch.multinomial(probs, num_samples=1).item()
    return action
