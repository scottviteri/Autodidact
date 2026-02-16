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
"""

import math

import torch
import torch.nn as nn


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
