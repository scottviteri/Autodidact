"""
Baseline curriculum selection strategies (Section 3.2 of the spec).

All baselines implement the same interface: given a model and a batch of
candidates, return the index of the selected candidate.

Baselines:
    1. Random: Uniform sampling from candidates.
    2. Loss-based: Prioritise high-loss examples.
    3. Uncertainty-based: Prioritise examples with high model uncertainty
       (measured by entropy of the predicted token distribution).
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaselineSelector(ABC):
    """Abstract base class for baseline curriculum selectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def select(self, model, candidates: torch.Tensor) -> int:
        """
        Select a candidate from the batch.
        
        Args:
            model: GPT-2 model.
            candidates: [N, seq_len] token IDs.
        
        Returns:
            Index of selected candidate.
        """
        pass


class RandomSelector(BaselineSelector):
    """Uniform random selection from candidates."""

    @property
    def name(self) -> str:
        return "random"

    def select(self, model, candidates: torch.Tensor) -> int:
        n = candidates.shape[0]
        return torch.randint(0, n, (1,)).item()


class LossBasedSelector(BaselineSelector):
    """
    Select the candidate with the highest loss (hardest example).
    
    This is a common curriculum learning heuristic: train on examples
    the model currently finds most difficult.
    """

    @property
    def name(self) -> str:
        return "loss_based"

    def select(self, model, candidates: torch.Tensor) -> int:
        losses = []
        with torch.no_grad():
            for i in range(candidates.shape[0]):
                x = candidates[i].unsqueeze(0)  # [1, seq_len]
                outputs = model(x, labels=x)
                losses.append(outputs.loss.item())
        # Select highest loss
        return max(range(len(losses)), key=lambda i: losses[i])


class UncertaintyBasedSelector(BaselineSelector):
    """
    Select the candidate where the model has highest predictive uncertainty.
    
    Uncertainty is measured as the average entropy of the model's predicted
    next-token distribution across all positions in the sequence.
    """

    @property
    def name(self) -> str:
        return "uncertainty"

    def select(self, model, candidates: torch.Tensor) -> int:
        entropies = []
        with torch.no_grad():
            for i in range(candidates.shape[0]):
                x = candidates[i].unsqueeze(0)  # [1, seq_len]
                outputs = model(x)
                logits = outputs.logits  # [1, seq_len, vocab_size]
                probs = F.softmax(logits, dim=-1)
                # Entropy per position: -sum(p * log(p))
                token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [1, seq_len]
                avg_entropy = token_entropy.mean().item()
                entropies.append(avg_entropy)
        # Select highest uncertainty
        return max(range(len(entropies)), key=lambda i: entropies[i])

