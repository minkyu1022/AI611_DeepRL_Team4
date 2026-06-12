"""Experiment configuration for the Section 4.2 reproduction.

Defaults follow the public Trig-SDE-Gamma checkpoint and the paper's reported
settings (Appendix E/F). Exact best hyper-parameters were selected by an Optuna
sweep that is not published numerically; the values below are mid-range picks
inside the published search ranges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .grpo import GRPOConfig


@dataclass
class ExperimentConfig:
    # Rollout / exploration
    mode: str = "velocity"            # "velocity" or "score"
    n_steps: int = 50                 # Nt (reduced from 740)
    noise_scale: float = 0.1          # a_m (read off Fig. 2; ablate {0.03, 0.1, 0.3})
    sigma_max: float | None = 1.0     # clamp on sigma(t) to tame the t->0 blow-up
    com_project: bool = True

    # Groups
    n_groups: int = 16
    group_size: int = 32              # G (paper: Choice(32, 64))
    seed: int = 0

    # Training
    n_iters: int = 300
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # Eval
    eval_every: int = 10
    eval_n: int = 256                 # number of test compositions for match-rate eval
