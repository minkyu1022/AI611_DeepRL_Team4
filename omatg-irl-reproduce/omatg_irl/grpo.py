"""GRPO / PPO trainer for OMatG-IRL (Section 4.2).

One iteration:
  1. roll out all groups in one batched trajectory (no grad);
  2. score the terminal structures with the reward;
  3. compute group-relative advantages A_i = (r_i - mean_g) / std_g;
  4. for K PPO epochs, recompute the live-policy means with grad and apply the
     clipped surrogate + KL regularization (+ denoiser distillation in score mode).

Normalization (Appendix A): PPO likelihood ratios and KL are computed *per atom*
(per-atom log-ratio over the 3 coordinates) and averaged over all atoms, all
steps, and the group -- this realizes the 1/(G * Nt * S) prefactor and keeps the
ratio O(1) regardless of structure size. The loss is accumulated per step and
back-propagated per step to bound memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .rollout import Rollout, Trajectory
from .utils_pbc import minimum_image


@dataclass
class GRPOConfig:
    lr: float = 1e-4
    ppo_epochs: int = 2
    clip_eps: float = 0.2
    policy_weight: float = 1.0      # alpha
    kl_weight: float = 1e-4         # beta
    distill_weight: float = 1e-4    # delta (score mode only)
    adv_eps: float = 1e-6
    grad_clip: float = 0.5


def _per_atom_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minimum-image squared distance per atom (sum over 3 coords). Shape [N]."""
    d = minimum_image(a - b)
    return (d * d).sum(dim=-1)


class GRPOTrainer:
    def __init__(self, rollout: Rollout, reward, group_ids: torch.Tensor, cfg: GRPOConfig):
        self.rollout = rollout
        self.reward = reward
        self.group_ids = group_ids
        self.cfg = cfg
        self.opt = torch.optim.Adam(rollout.policy.trainable_parameters(), lr=cfg.lr)

    def advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Group-relative standardized advantages, shape [B]."""
        adv = torch.zeros_like(rewards)
        for g in torch.unique(self.group_ids):
            mask = self.group_ids == g
            r = rewards[mask]
            adv[mask] = (r - r.mean()) / (r.std() + self.cfg.adv_eps)
        return adv

    def _epoch_loss_and_step(self, traj: Trajectory, adv: torch.Tensor) -> dict:
        """One PPO epoch: per-step grad accumulation, single optimizer step."""
        cfg = self.cfg
        batch = traj.batch
        adv_atom = adv[batch]  # [N]
        nt = traj.pos_states.shape[0]
        self.opt.zero_grad(set_to_none=True)

        tot_ppo = tot_kl = tot_dist = 0.0
        tot_clipfrac = 0.0
        for i in range(nt):
            v = (traj.sigmas[i] * traj.sigmas[i] * traj.dts[i]).clamp(min=1e-12)
            x_next = traj.pos_next[i]
            mu_theta, z_theta = self.rollout.recompute_step(traj, i)

            sq_theta = _per_atom_sq(x_next, mu_theta)            # grad
            sq_old = _per_atom_sq(x_next, traj.mu_old[i])        # const
            logq = (sq_old - sq_theta) / (2.0 * v)               # [N]
            q = torch.exp(logq)

            unclipped = q * adv_atom
            clipped = torch.clamp(q, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_atom
            ppo = torch.minimum(unclipped, clipped).mean()

            kl = (_per_atom_sq(mu_theta, traj.mu_ref[i]) / (2.0 * v)).mean()

            if self.rollout.mode == "score":
                dist = ((z_theta - traj.z_ref[i]) ** 2).sum(dim=-1).mean()
            else:
                dist = torch.zeros((), device=ppo.device)

            loss_i = (-(cfg.policy_weight * ppo) + cfg.kl_weight * kl
                      + cfg.distill_weight * dist) / nt
            loss_i.backward()

            tot_ppo += float(ppo.detach())
            tot_kl += float(kl.detach())
            tot_dist += float(dist.detach())
            tot_clipfrac += float((unclipped != clipped).float().mean().detach())

        torch.nn.utils.clip_grad_value_(list(self.rollout.policy.trainable_parameters()), cfg.grad_clip)
        self.opt.step()
        return {"ppo": tot_ppo / nt, "kl": tot_kl / nt, "distill": tot_dist / nt,
                "clipfrac": tot_clipfrac / nt}

    def step(self, x0) -> dict:
        """Run one full GRPO iteration and return logging diagnostics."""
        self.rollout.policy.set_train(False)
        traj = self.rollout.generate(x0)
        rewards, rdiag = self.reward(traj.final, self.group_ids)
        adv = self.advantages(rewards)

        self.rollout.policy.set_train(True)
        logs = {}
        for _ in range(self.cfg.ppo_epochs):
            logs = self._epoch_loss_and_step(traj, adv)
        logs.update(rdiag)
        logs["adv_abs_mean"] = float(adv.abs().mean())
        return logs
