"""
OMatG-IRL: Inference-Time Reinforcement Learning for the OMatG crystal generator.

Reproduction of Section 4.2 ("Inference-Time Energy Reinforcement") of
Hoellmer & Martiniani (2026), "Open Materials Generation with Inference-Time
Reinforcement Learning". Built on top of the public OMatG framework and the
public Trig-SDE-Gamma MP-20-CSP checkpoint.

The package implements a policy-gradient (GRPO/PPO) loop that reinforces a
pretrained continuous-time generative model *at inference time* by adding
stochastic perturbations to the integration and optimizing a black-box reward
(negative MACE-MPA-0 energy per atom), without retraining from scratch.

Two extension seams keep Section 4.3 (velocity-annealing) a small later add-on:
  * ``policy.Policy``  -- pluggable velocity source (full fine-tune vs learned schedule)
  * ``reward.Reward``  -- pluggable reward (energy vs cRMSE)
"""

__all__ = [
    "config",
    "utils_pbc",
    "schedules",
    "policy",
    "checkpoint",
    "rollout",
    "reward",
    "data",
    "grpo",
    "eval",
]
