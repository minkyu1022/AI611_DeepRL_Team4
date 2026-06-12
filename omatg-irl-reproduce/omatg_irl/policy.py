"""Policy abstraction (extension SEAM 1).

The rollout obtains its position velocity field ``b`` (and, for the score-based
variant, the denoiser ``z``) and the lattice velocity from a ``Policy``. Section
4.2 fine-tunes the full CSPNet velocity (``FullModelPolicy``). Section 4.3 would
instead wrap a *frozen* base velocity with a learned time-dependent annealing
schedule s^theta(t) -- that is a drop-in ``Policy`` subclass and requires no
change to the rollout, GRPO trainer, or reward.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch
from torch_geometric.data import Data

from omg.model.model import Model


class Policy(ABC):
    """Produces velocity fields for a batched crystal state at a given time."""

    @abstractmethod
    def velocity_fields(self, x: Data, t_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a dict with keys ``pos_b`` [N,3], ``pos_eta`` [N,3], ``cell_b`` [B,3,3]."""

    @abstractmethod
    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Parameters optimized by the policy-gradient update."""

    @abstractmethod
    def set_train(self, mode: bool) -> None:
        """Toggle train/eval mode of the underlying modules."""


class FullModelPolicy(Policy):
    """Section 4.2 policy: the trainable CSPNet velocity field itself.

    :param model: An ``omg.model.model.Model`` instance (CSPNetFull encoder).
    """

    def __init__(self, model: Model):
        self.model = model

    def velocity_fields(self, x: Data, t_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(x, t_vec)
        return {"pos_b": out["pos_b"], "pos_eta": out["pos_eta"], "cell_b": out["cell_b"]}

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def set_train(self, mode: bool) -> None:
        self.model.train(mode)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)
