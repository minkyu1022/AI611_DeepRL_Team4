"""Entry point for OMatG-DNG-IRL: GRPO fine-tuning of a pretrained OMatG DNG model with the CRYSTAL S.U.N. reward.

Example:
    python -m omg.irl.run --ckpt ckpts/dng_linear_sde_gamma --device cuda \
        --group-size 16 --n-groups 2 --n-steps 50 --ppo-epochs 2 --iterations 100 \
        --out runs/irl_dng
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .load import load_pretrained
from .reward import SUNReward
from .stability import ElementalReferenceStability
from .novelty import NoveltyScorer
from .grpo import GRPOTrainer, GRPOConfig
from omg.datamodule import StructureDataset, OMGDataset


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OMatG-DNG-IRL GRPO fine-tuning with the CRYSTAL S.U.N. reward.")
    p.add_argument("--ckpt", default="ckpts/dng_linear_sde_gamma", help="Pretrained DNG checkpoint directory.")
    p.add_argument("--train-lmdb", default="data/mp_20/train.lmdb", help="Dataset providing n_atoms templates.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/irl_dng", help="Output directory for checkpoints and metrics.")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--n-groups", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--noise-scale", type=float, default=0.1)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--relax-steps", type=int, default=30)
    p.add_argument("--n-reference", type=int, default=2000, help="MP-20 structures fingerprinted for novelty.")
    p.add_argument("--ablate", default=None, choices=[None, "stability", "novelty", "uniqueness"],
                   help="Drop one reward factor for the CRYSTAL-style ablation study.")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    module = load_pretrained(args.ckpt)
    template_ds = OMGDataset(StructureDataset(args.train_lmdb, lazy_storage=True, floating_point_precision="32-true"))
    reward = SUNReward(
        stability=ElementalReferenceStability(relax_steps=args.relax_steps),
        novelty=NoveltyScorer(reference_lmdb=args.train_lmdb, n_reference=args.n_reference),
        relax_steps=args.relax_steps, ablate=args.ablate)
    cfg = GRPOConfig(group_size=args.group_size, n_groups=args.n_groups, n_steps=args.n_steps,
                     noise_scale=args.noise_scale, ppo_epochs=args.ppo_epochs, clip_eps=args.clip_eps,
                     kl_coef=args.kl_coef, lr=args.lr)
    trainer = GRPOTrainer(module, reward, template_ds, cfg, device=args.device)

    print(f"OMatG-DNG-IRL | ckpt={args.ckpt} | G={cfg.group_size}x{cfg.n_groups} groups | "
          f"Nt={cfg.n_steps} | ppo={cfg.ppo_epochs} | kl={cfg.kl_coef} | lr={cfg.lr}")
    for it in range(args.iterations):
        m = trainer.train_iteration()
        print(f"[iter {it+1:4d}] R_mean={m['mean_reward']:.3f} R_max={m['max_reward']:.2f} "
              f"valid={m['valid_rate']:.2f} stable={m['stable_rate']:.2f} novel={m['novel_rate']:.2f} "
              f"loss={m['loss']:.4f} kl={m['kl']:.4f} ({m['time']:.0f}s)", flush=True)
        (out / "metrics.json").write_text(json.dumps(trainer.history, indent=2))
        if (it + 1) % args.save_every == 0 or it + 1 == args.iterations:
            torch.save({"state_dict": module.state_dict(), "iteration": it + 1, "config": vars(args)},
                       out / "irl_checkpoint.ckpt")
    print(f"Done. Checkpoint + metrics in {out}")


if __name__ == "__main__":
    main()
