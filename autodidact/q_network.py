"""
Q-network for curriculum selection.

Architecture (from Section 2.3 of the spec):
    Q_phi(s, x) = W2 * ReLU(W1 * h_x + b1) + b2
    
    where h_x is the detached last-layer hidden state at the final token position
    from the base model, and phi = {W1, b1, W2, b2}.

    Dimensions: d -> 32 -> 1  (d = 768 for GPT-2)
"""

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


def extract_hidden_states(model, input_ids: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
    """
    Forward candidates through the base model and extract last-layer hidden 
    states at the final token position, detached from the computation graph.
    
    Processes in mini-batches to avoid OOM: each forward pass with
    output_hidden_states=True stores all 13 layers of hidden states plus
    logits, so [N, seq_len, vocab_size] can be huge for large N.
    
    Args:
        model: HuggingFace GPT-2 model.
        input_ids: [N, seq_len] token IDs.
        batch_size: Mini-batch size for forward passes.
    
    Returns:
        h: [N, d] detached hidden states.
    """
    all_h = []
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[i : i + batch_size]
            outputs = model(batch, output_hidden_states=True)
            # Last layer hidden states: [B, seq_len, d]
            last_hidden = outputs.hidden_states[-1]
            # Take the final token position: [B, d]
            h = last_hidden[:, -1, :].detach()
            all_h.append(h)
    return torch.cat(all_h, dim=0)


def compute_soft_value(q_values: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute the soft value function:
        V(s) = beta * log sum_a' exp(Q(s, a') / beta)
    
    This is the log-sum-exp over Q-values scaled by temperature, which appears
    as the bootstrap term in the differential soft Bellman equation (Eq. 3).
    
    Args:
        q_values: [N, 1] Q-values for all candidates.
        beta: Boltzmann temperature.
    
    Returns:
        Scalar soft value V(s).
    """
    # q_values is [N, 1], squeeze to [N]
    q = q_values.squeeze(-1)
    return beta * torch.logsumexp(q / beta, dim=0)


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

