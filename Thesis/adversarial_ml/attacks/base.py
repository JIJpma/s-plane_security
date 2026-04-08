from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from DU_model.Transformer.t_size import TransformerNN


@dataclass
class AttackResult:
    """Container returned by every attack's evaluate()."""
    epsilon: float
    clean_preds: torch.Tensor
    adv_preds: torch.Tensor
    ground_truth: torch.Tensor

    @property
    def flipped(self) -> int:
        return (self.clean_preds != self.adv_preds).sum().item()

    @property
    def total(self) -> int:
        return len(self.clean_preds)

    @property
    def evasion_rate(self) -> float:
        return self.flipped / self.total


class BaseAttack(ABC):
    """Interface that every adversarial attack must implement."""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        model: TransformerNN,
        loader: DataLoader,
        device: torch.device,
        epsilon: float,
        feat_std: torch.Tensor,
    ) -> AttackResult:
        ...
