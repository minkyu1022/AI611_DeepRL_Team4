"""Exploration noise schedules for OMatG-IRL rollouts.

The paper introduces a *square-root* noise schedule sigma(t) = a * sqrt((1-t)/t)
(inspired by Flow-GRPO) that decays to zero as t -> 1. The scalar ``a`` is the
"noise scale" (small/medium/large = a_s/a_m/a_l in Fig. 2). The same sigma(t)
is used both for the diffusion term and, in the score-based variant, for the
ODE->SDE drift correction (sigma^2/2 times the score), which preserves the
pretrained marginal for any sigma.

sqrt((1-t)/t) diverges as t -> 0, so an optional clamp bounds the per-step
displacement std sigma(t) * sqrt(dt). This is the highest-risk numerical knob in
the reproduction (see plan), hence it is explicit and configurable.
"""
from __future__ import annotations

import torch


class SqrtNoiseSchedule:
    """sigma(t) = a * sqrt((1 - t) / t), optionally clamped to ``sigma_max``."""

    def __init__(self, a: float, sigma_max: float | None = None, sigma_min: float | None = None,
                 eps: float = 1e-6):
        self.a = float(a)
        self.sigma_max = sigma_max
        # A floor keeps the per-step variance v = sigma^2 * dt away from zero as t -> 1,
        # which otherwise makes the per-step transition near-deterministic and the
        # importance ratio / KL ill-conditioned (division by a vanishing v).
        self.sigma_min = sigma_min
        self.eps = eps

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate sigma at (scalar or tensor) time ``t``."""
        t = torch.as_tensor(t)
        tc = t.clamp(min=self.eps, max=1.0 - self.eps)
        sigma = self.a * torch.sqrt((1.0 - tc) / tc)
        if self.sigma_max is not None:
            sigma = sigma.clamp(max=self.sigma_max)
        if self.sigma_min is not None:
            sigma = sigma.clamp(min=self.sigma_min)
        return sigma


class ConstantNoiseSchedule:
    """sigma(t) = a (constant schedule, provided for ablations / Fig. 2 parity)."""

    def __init__(self, a: float):
        self.a = float(a)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t)
        return torch.full_like(t.float(), self.a)
