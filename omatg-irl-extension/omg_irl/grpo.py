"""GRPO trainer for OMatG-DNG-IRL.

Ties together the inference-time RL rollout (``IRLSampler``), the CRYSTAL S.U.N. reward (``SUNReward``), and a
PPO-clipped group-relative policy-gradient update (OMatG-IRL Eqs. 9-11). Each training iteration:

  1. draws ``n_groups`` GRPO groups; each group rolls out ``group_size`` structures from the base distribution;
  2. scores every structure with the multiplicative S.U.N. reward;
  3. forms group-relative advantages  A_i = (R_i - mean) / (std + eps);
  4. updates the position velocity/denoiser and species logits for ``ppo_epochs`` steps using the clipped
     objective with KL regularization toward the frozen pretrained reference policy. The lattice is frozen.
"""
import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from torch_geometric.data import Batch

from .irl_sampler import IRLSampler
from .reward import SUNReward
from .structures import omgdata_to_ase


@dataclass
class GRPOConfig:
    group_size: int = 16          # G: structures per group (rollout)
    n_groups: int = 4             # groups per training iteration
    n_steps: int = 50             # N_t integration steps
    noise_scale: float = 0.1      # square-root position-noise scale
    ppo_epochs: int = 2
    clip_eps: float = 0.2
    kl_coef: float = 0.001
    lr: float = 1e-5
    grad_clip: float = 1.0
    adv_eps: float = 1e-4


class GRPOTrainer:
    """
    :param module:
        The (trainable) pretrained ``OMGLightning`` DNG module.
    :param reward:
        A ``SUNReward`` instance.
    :param template_dataset:
        An ``OMGDataset`` whose items provide ``n_atoms`` templates for sampling base structures.
    :param cfg:
        GRPO hyperparameters.
    :param device:
        Torch device.
    """

    def __init__(self, module, reward: SUNReward, template_dataset, cfg: GRPOConfig, device: str = "cuda") -> None:
        self.module = module.to(device)
        self.device = device
        self.cfg = cfg
        self.reward = reward
        self.template_dataset = template_dataset
        self.sampler = IRLSampler(self.module, n_steps=cfg.n_steps, noise_scale=cfg.noise_scale)
        # Frozen reference policy for KL regularization.
        self.ref_model = copy.deepcopy(self.module.model).to(device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.module.model.parameters(), lr=cfg.lr)
        self._n_templates = len(template_dataset)
        self.history: List[dict] = []

    def _sample_base(self, group_size: int):
        """Build a base-distribution batch of ``group_size`` structures from random n_atoms templates."""
        idx = np.random.randint(0, self._n_templates, size=group_size)
        batch = Batch.from_data_list([self.template_dataset[int(i)] for i in idx]).to(self.device)
        return self.module.sampler.sample_p_0(batch).to(self.device)

    def run_group(self):
        """Roll out one group, score it, and return (trajectory, advantages, reward_array, reward_objs)."""
        x0 = self._sample_base(self.cfg.group_size)
        gen, traj = self.sampler.rollout(x0)
        atoms = omgdata_to_ase(gen)
        rewards = self.reward.compute_group(atoms)
        r = np.array([rr.reward for rr in rewards], dtype=np.float64)
        adv = (r - r.mean()) / (r.std() + self.cfg.adv_eps)
        advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)
        return traj, advantages, r, rewards

    def train_iteration(self) -> dict:
        """Run one training iteration over ``n_groups`` groups and return aggregate metrics."""
        cfg = self.cfg
        t0 = time.time()
        all_r, losses, kls, pobjs = [], [], [], []
        valid_rate, stable_rate, novel_rate = [], [], []
        for _ in range(cfg.n_groups):
            traj, advantages, r, robjs = self.run_group()
            all_r.append(r)
            valid_rate.append(np.mean([o.valid for o in robjs]))
            stable_rate.append(np.mean([o.r_stab > 0 for o in robjs]))
            novel_rate.append(np.mean([o.r_nov > 0 for o in robjs]))
            # Skip degenerate groups with no reward spread (no learning signal).
            if float(advantages.abs().max()) < 1e-8:
                continue
            for _ in range(cfg.ppo_epochs):
                self.module.model.train()
                loss, metrics = self.sampler.grpo_loss(
                    self.module.model, traj, advantages, ref_model=self.ref_model,
                    clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                losses.append(float(loss.detach()))
                kls.append(metrics["kl"])
                pobjs.append(metrics["policy_obj"])
        r_cat = np.concatenate(all_r) if all_r else np.array([0.0])
        m = {
            "mean_reward": float(r_cat.mean()),
            "max_reward": float(r_cat.max()),
            "valid_rate": float(np.mean(valid_rate)) if valid_rate else 0.0,
            "stable_rate": float(np.mean(stable_rate)) if stable_rate else 0.0,
            "novel_rate": float(np.mean(novel_rate)) if novel_rate else 0.0,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "kl": float(np.mean(kls)) if kls else 0.0,
            "policy_obj": float(np.mean(pobjs)) if pobjs else 0.0,
            "time": time.time() - t0,
        }
        self.history.append(m)
        return m

    def train(self, n_iterations: int, log_every: int = 1) -> None:
        for it in range(n_iterations):
            m = self.train_iteration()
            if (it + 1) % log_every == 0:
                print(f"[iter {it+1:4d}] R_mean={m['mean_reward']:.3f} R_max={m['max_reward']:.2f} "
                      f"valid={m['valid_rate']:.2f} stable={m['stable_rate']:.2f} novel={m['novel_rate']:.2f} "
                      f"loss={m['loss']:.4f} kl={m['kl']:.4f} ({m['time']:.0f}s)")
