"""Training driver for OMatG-IRL Section 4.2 (energy reinforcement).

Runs GRPO on a fixed set of groups, logs per-iteration training diagnostics and
periodic held-out evaluation (match rate / RMSE / relative energy / invalid rate)
to a CSV, and saves the final reinforced policy. Designed to run one variant per
GPU in the background.

Usage:
  python omatg_irl/train.py --mode velocity --out experiments/velocity \
      --n-iters 300 --group-size 32 [--noise-scale 0.1 ...]
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from huggingface_hub import snapshot_download
from mace.calculators import mace_mp

from omatg_irl.checkpoint import get_pos_interpolant, load_lightning_module, split_policy_reference
from omatg_irl.data import build_groups
from omatg_irl.eval import EvalHarness
from omatg_irl.grpo import GRPOConfig, GRPOTrainer
from omatg_irl.policy import FullModelPolicy
from omatg_irl.reward import EnergyReward
from omatg_irl.rollout import Rollout
from omatg_irl.schedules import SqrtNoiseSchedule

TEST_LMDB = "data/mp_20/test.lmdb"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["velocity", "score"], default="velocity")
    p.add_argument("--out", required=True)
    p.add_argument("--n-iters", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--n-groups", type=int, default=16)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--noise-scale", type=float, default=0.1)
    p.add_argument("--sigma-max", type=float, default=1.0)
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-weight", type=float, default=1e-4)
    p.add_argument("--distill-weight", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    base = os.path.join(snapshot_download("OMatG/MP-20-CSP", allow_patterns=["Trig-SDE-Gamma/*"]),
                        "Trig-SDE-Gamma")
    lm = load_lightning_module(os.path.join(base, "train.yaml"),
                               os.path.join(base, "checkpoint.ckpt"), device)
    policy_model, reference_model = split_policy_reference(lm)
    gamma = get_pos_interpolant(lm)._gamma
    calc = mace_mp(model="medium-mpa-0", device=device, default_dtype="float32")

    policy = FullModelPolicy(policy_model)
    reference = FullModelPolicy(reference_model)
    noise = SqrtNoiseSchedule(a=args.noise_scale, sigma_max=args.sigma_max, sigma_min=args.sigma_min)
    rollout = Rollout(policy, reference, noise, gamma, n_steps=args.n_steps, mode=args.mode,
                      device=device)
    reward = EnergyReward(calc, device=device)

    x0, group_ids, _ = build_groups(TEST_LMDB, lm.sampler, n_groups=args.n_groups,
                                    group_size=args.group_size, seed=args.seed, device=device)
    cfg = GRPOConfig(lr=args.lr, ppo_epochs=args.ppo_epochs, clip_eps=args.clip_eps,
                     kl_weight=args.kl_weight, distill_weight=args.distill_weight)
    trainer = GRPOTrainer(rollout, reward, group_ids, cfg)
    evalharness = EvalHarness(TEST_LMDB, lm.sampler, gamma, calc, device)
    eval_indices = list(range(args.eval_n))

    csv_path = os.path.join(args.out, "metrics.csv")
    fields = ["iter", "mean_reward", "mean_energy_per_atom", "invalid_rate", "kl", "clipfrac",
              "adv_abs_mean", "match_rate", "mean_rmsd", "corr_rmsd", "relative_energy_per_atom",
              "gen_energy_per_atom", "ref_energy_per_atom", "invalid_energy_rate", "sec"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"[train] mode={args.mode} groups={args.n_groups}x{args.group_size} "
          f"Nt={args.n_steps} a={args.noise_scale} iters={args.n_iters} -> {args.out}", flush=True)

    best_relE = float("inf")
    for it in range(args.n_iters):
        t0 = time.time()
        logs = trainer.step(x0)
        row = {k: logs.get(k, "") for k in fields}
        row["iter"] = it
        row["sec"] = round(time.time() - t0, 2)

        if it % args.eval_every == 0 or it == args.n_iters - 1:
            m = evalharness.evaluate(policy, reference, eval_indices, n_steps=args.n_steps,
                                     seed=args.seed)
            row.update({k: m[k] for k in ["match_rate", "mean_rmsd", "corr_rmsd",
                                          "relative_energy_per_atom", "gen_energy_per_atom",
                                          "ref_energy_per_atom", "invalid_energy_rate"]})
            # Save the validation-optimal checkpoint (lowest relative energy/atom).
            if m["relative_energy_per_atom"] == m["relative_energy_per_atom"] and \
                    m["relative_energy_per_atom"] < best_relE:
                best_relE = m["relative_energy_per_atom"]
                torch.save(policy.state_dict(), os.path.join(args.out, "policy_best.pt"))
                with open(os.path.join(args.out, "best.json"), "w") as bf:
                    json.dump({"iter": it, **m}, bf, indent=2)
            print(f"[iter {it:3d}] reward={logs['mean_reward']:.3f} "
                  f"E/atom={logs['mean_energy_per_atom']:.3f} match={m['match_rate']:.3f} "
                  f"relE={m['relative_energy_per_atom']:.3f} kl={logs['kl']:.2e} "
                  f"inval={logs['invalid_rate']:.3f} {row['sec']}s", flush=True)

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        if it % 50 == 0 or it == args.n_iters - 1:
            torch.save(policy.state_dict(), os.path.join(args.out, "policy.pt"))

    torch.save(policy.state_dict(), os.path.join(args.out, "policy.pt"))
    print(f"[train] done -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
