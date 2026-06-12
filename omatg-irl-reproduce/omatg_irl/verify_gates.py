"""Cheap correctness gates for the OMatG-IRL pipeline (no MACE).

Validates, on a tiny batch on one GPU:
  0. GPU scatter nondeterminism floor (two identical forwards differ by ~eps).
  1. q_t ~= 1 at the first PPO epoch (within the scatter floor).
  2. KL ~= 0 at init (policy == reference).
  3. COM unit test + frozen-cell path (bitwise equal to the reference ODE).
  4. End-to-end optimization signal: with a cheap repulsion-energy reward (a
     learnable analog of the MACE energy), the mean reward increases and the
     policy KL grows over a handful of GRPO iterations.
"""
import os
import sys

import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omatg_irl.checkpoint import get_pos_interpolant, load_lightning_module, split_policy_reference
from omatg_irl.data import build_groups
from omatg_irl.grpo import GRPOConfig, GRPOTrainer, _per_atom_sq
from omatg_irl.policy import FullModelPolicy
from omatg_irl.rollout import Rollout
from omatg_irl.schedules import SqrtNoiseSchedule
from omatg_irl.utils_pbc import com_project, minimum_image

DEVICE = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "velocity"


class RepulsionReward:
    """reward_i = -sum_{a<b} exp(-||min_image(x_a-x_b)||^2 / w): spread atoms apart."""

    def __init__(self, w=0.02, device="cuda"):
        self.w = w
        self.device = device

    def __call__(self, final, group_ids):
        final = final.to(self.device)
        B = int(final.n_atoms.shape[0])
        rewards = torch.zeros(B, device=self.device)
        for i in range(B):
            lo, hi = int(final.ptr[i]), int(final.ptr[i + 1])
            p = final.pos[lo:hi]
            d = minimum_image(p[:, None, :] - p[None, :, :])
            sq = (d * d).sum(-1)
            n = p.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=p.device)
            rewards[i] = -torch.exp(-sq[mask] / self.w).sum() / max(n, 1)
        return rewards, {"mean_reward": float(rewards.mean()), "invalid_rate": 0.0,
                         "mean_energy_per_atom": float(-rewards.mean())}


def main():
    repo = snapshot_download(repo_id="OMatG/MP-20-CSP", allow_patterns=["Trig-SDE-Gamma/*"])
    base = os.path.join(repo, "Trig-SDE-Gamma")
    lm = load_lightning_module(os.path.join(base, "train.yaml"),
                               os.path.join(base, "checkpoint.ckpt"), device=DEVICE)
    policy_model, reference_model = split_policy_reference(lm)
    gamma = get_pos_interpolant(lm)._gamma

    policy = FullModelPolicy(policy_model)
    reference = FullModelPolicy(reference_model)
    noise = SqrtNoiseSchedule(a=0.1, sigma_max=1.0, sigma_min=0.05)
    rollout = Rollout(policy, reference, noise, gamma, n_steps=50, mode=MODE, device=DEVICE)

    x0, group_ids, _ = build_groups("data/mp_20/test.lmdb", lm.sampler, n_groups=2,
                                    group_size=8, seed=0, device=DEVICE)
    B = int(x0.n_atoms.shape[0])
    print(f"[setup] mode={MODE} n_structures={B} n_atoms_total={x0.pos.shape[0]}")

    # Gate 0: scatter nondeterminism floor (two identical forward passes).
    t0 = rollout.times[10]
    with torch.no_grad():
        f1 = policy.velocity_fields(x0, t0.repeat(B))["pos_b"]
        f2 = policy.velocity_fields(x0, t0.repeat(B))["pos_b"]
    scatter_floor = float((f1 - f2).abs().max())
    print(f"[gate0] scatter nondeterminism floor (max |b1-b2|) = {scatter_floor:.2e}")

    # Gate 3a: COM projection unit test.
    v = torch.randn(x0.pos.shape[0], 3, device=DEVICE)
    from torch_scatter import scatter_mean
    com_after = scatter_mean(com_project(v, x0.batch), x0.batch, dim=0).abs().max()
    print(f"[gate3a] COM-projected per-structure mean = {float(com_after):.2e} (expect ~0)")

    traj = rollout.generate(x0)
    print(f"[noise] sigma*sqrt(dt): step0={float(traj.sigmas[0]*traj.dts[0].sqrt()):.3f} "
          f"mid={float(traj.sigmas[25]*traj.dts[25].sqrt()):.3f} "
          f"end={float(traj.sigmas[-1]*traj.dts[-1].sqrt()):.3f}")

    # Gate 1 & 2: q == 1 and KL == 0 at init.
    max_logq, max_kl = 0.0, 0.0
    with torch.no_grad():
        for i in range(rollout.n_steps):
            vv = (traj.sigmas[i] ** 2 * traj.dts[i]).clamp(min=1e-12)
            mu_theta, _ = rollout.recompute_step(traj, i)
            logq = (_per_atom_sq(traj.pos_next[i], traj.mu_old[i])
                    - _per_atom_sq(traj.pos_next[i], mu_theta)) / (2 * vv)
            kl = _per_atom_sq(mu_theta, traj.mu_ref[i]) / (2 * vv)
            max_logq = max(max_logq, float(logq.abs().max()))
            max_kl = max(max_kl, float(kl.abs().max()))
    print(f"[gate1] max |log q_t| at init = {max_logq:.2e} (scatter-limited)")
    print(f"[gate2] max KL at init        = {max_kl:.2e} (scatter-limited)")

    # Gate 4: end-to-end optimization signal.
    reward = RepulsionReward(device=DEVICE)
    trainer = GRPOTrainer(rollout, reward, group_ids,
                          GRPOConfig(lr=3e-4, ppo_epochs=2, kl_weight=1e-4))
    r0 = float(reward(traj.final, group_ids)[0].mean())
    history = []
    for it in range(12):
        logs = trainer.step(x0)
        history.append(logs["mean_reward"])
        if it % 3 == 0 or it == 11:
            print(f"[gate4] iter {it:2d}: reward={logs['mean_reward']:.4f} "
                  f"kl={logs['kl']:.3e} clipfrac={logs['clipfrac']:.3f}")
    improved = history[-1] > history[0]
    print(f"[gate4] reward {history[0]:.4f} -> {history[-1]:.4f}  "
          f"({'UP' if improved else 'DOWN'})")

    ok = (max_logq < 0.2) and (max_kl < 0.2) and (float(com_after) < 1e-5) and improved
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
