"""Evaluate the S.U.N. metrics of an OMatG DNG model (pretrained baseline or an IRL-tuned checkpoint).

Generates a set of structures with the inference-time RL rollout procedure (the same one optimized during
training), scores them as a single group with the full CRYSTAL S.U.N. reward, and reports:

  * validity / stability / novelty / uniqueness rates,
  * the S.U.N. rate (fraction that are simultaneously stable, unique and novel),
  * the mean multiplicative reward.

Usage:
    python -m omg.irl.evaluate --ckpt ckpts/dng_linear_sde_gamma --n-samples 64 --out runs/eval_base.json
    python -m omg.irl.evaluate --ckpt runs/irl_g1/irl_checkpoint.ckpt --base ckpts/dng_linear_sde_gamma ...
"""
import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch_geometric.data import Batch

from .load import load_pretrained
from .irl_sampler import IRLSampler
from .structures import omgdata_to_ase
from .reward import SUNReward
from .stability import ElementalReferenceStability
from .novelty import NoveltyScorer
from omg.datamodule import StructureDataset, OMGDataset


def evaluate_module(module, reward: SUNReward, template_ds, n_samples: int = 64, n_steps: int = 50,
                    noise_scale: float = 0.1, device: str = "cuda", batch_size: int = 16) -> dict:
    """Generate ``n_samples`` structures and compute S.U.N. metrics over them as a single group."""
    module = module.to(device).eval()
    sampler = IRLSampler(module, n_steps=n_steps, noise_scale=noise_scale)
    n_templates = len(template_ds)
    atoms_all = []
    for start in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - start)
        idx = np.random.randint(0, n_templates, size=bs)
        tb = Batch.from_data_list([template_ds[int(i)] for i in idx]).to(device)
        x0 = module.sampler.sample_p_0(tb).to(device)
        gen, _ = sampler.rollout(x0)
        atoms_all.extend(omgdata_to_ase(gen))
    res = reward.compute_group(atoms_all)
    n = len(res)
    valid = np.array([r.valid for r in res])
    stable = np.array([r.r_stab > 0 for r in res])
    novel = np.array([r.r_nov > 0 for r in res])
    unique = np.array([r.r_match > 0 for r in res])  # set only for stable structures
    sun = valid & stable & novel & unique
    rewards = np.array([r.reward for r in res])
    return {
        "n_samples": int(n),
        "validity_rate": float(valid.mean()),
        "stability_rate": float(stable.mean()),
        "novelty_rate": float(novel.mean()),
        "uniqueness_rate": float(unique.mean()),
        "sun_rate": float(sun.mean()),
        "mean_reward": float(rewards.mean()),
        "max_reward": float(rewards.max()) if n else 0.0,
    }


def _load_any(ckpt: str, base_dir: Optional[str], device: str):
    """Load a module either from a HF-style checkpoint directory or from an IRL ``irl_checkpoint.ckpt`` file."""
    p = Path(ckpt)
    if p.is_dir():
        return load_pretrained(ckpt, map_location=device)
    # IRL checkpoint file: load base architecture, then overwrite weights.
    if base_dir is None:
        raise ValueError("--base is required when --ckpt is an IRL checkpoint file.")
    module = load_pretrained(base_dir, map_location=device)
    sd = torch.load(p, map_location=device, weights_only=False)["state_dict"]
    module.load_state_dict(sd, strict=False)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate S.U.N. metrics of an OMatG DNG model.")
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir, or an IRL irl_checkpoint.ckpt file.")
    ap.add_argument("--base", default="ckpts/dng_linear_sde_gamma", help="Base ckpt dir (for IRL ckpt files).")
    ap.add_argument("--train-lmdb", default="data/mp_20/train.lmdb")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--noise-scale", type=float, default=0.1)
    ap.add_argument("--relax-steps", type=int, default=30)
    ap.add_argument("--n-reference", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    module = _load_any(args.ckpt, args.base, args.device)
    template_ds = OMGDataset(StructureDataset(args.train_lmdb, lazy_storage=True, floating_point_precision="32-true"))
    reward = SUNReward(stability=ElementalReferenceStability(relax_steps=args.relax_steps),
                       novelty=NoveltyScorer(reference_lmdb=args.train_lmdb, n_reference=args.n_reference),
                       relax_steps=args.relax_steps)
    metrics = evaluate_module(module, reward, template_ds, n_samples=args.n_samples, n_steps=args.n_steps,
                              noise_scale=args.noise_scale, device=args.device)
    metrics["ckpt"] = args.ckpt
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
