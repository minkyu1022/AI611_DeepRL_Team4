"""Full MP-20 test-set evaluation for OMatG-IRL (less noisy, paper-comparable).

Generates one structure for every test composition (deterministic Nt=50) with the
pretrained baseline and each reinforced policy, then computes index-aligned match
rate / RMSE / cRMSE (PyMatgen StructureMatcher, ltol=0.3 stol=0.5 angle_tol=10,
parallel over CPUs) plus relative MACE-MPA-0 energy/atom and invalid rate over the
whole test set. Run from the OMatG package dir.
"""
import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from ase import Atoms
from huggingface_hub import snapshot_download
from mace.calculators import mace_mp
from torch_geometric.data import Batch

from omg.analysis import ValidAtoms, match_rmsds
from omg.datamodule.omg_data import OMGData
from omg.datamodule.structure_dataset import StructureDataset
from omatg_irl.checkpoint import get_pos_interpolant, load_lightning_module, split_policy_reference
from omatg_irl.eval import deterministic_generate
from omatg_irl.paths import EXPERIMENTS, TEST_LMDB
from omatg_irl.policy import FullModelPolicy
from omatg_irl.reward import is_invalid, omgdata_to_atoms


def gen_all(policy, reference, gamma, sampler, ds, indices, n_steps, batch, device, seed=0):
    atoms = []
    for s in range(0, len(indices), batch):
        idx = indices[s:s + batch]
        template = Batch.from_data_list([OMGData(ds[i]) for i in idx])
        torch.manual_seed(seed + s)
        x0 = sampler.sample_p_0(template).to(device)
        gen = deterministic_generate(policy, reference, gamma, x0, n_steps=n_steps, device=device)
        atoms.extend(omgdata_to_atoms(gen))
    return atoms


def energies(calc, atoms_list):
    epa, invalid = [], []
    for a in atoms_list:
        if is_invalid(a):
            invalid.append(True); continue
        invalid.append(False)
        try:
            b = a.copy(); b.calc = calc
            epa.append(b.get_potential_energy() / len(b))
        except Exception:
            epa.append(np.nan)
    return np.array(epa, dtype=float), np.array(invalid)


def metrics(gen_atoms, ref_atoms, gen_epa, ref_epa_mean, ncpu):
    gv = ValidAtoms.get_valid_atoms(gen_atoms, skip_validation=True, number_cpus=1, enable_progress_bar=False)
    rv = ValidAtoms.get_valid_atoms(ref_atoms, skip_validation=True, number_cpus=1, enable_progress_bar=False)
    match_rate, mean_rmsd, _, _, _, _, corr_rmsd, _ = match_rmsds(
        gv, rv, ltol=0.3, stol=0.5, angle_tol=10.0, number_cpus=ncpu, enable_progress_bar=False)
    gv_epa = gen_epa[~np.isnan(gen_epa)]
    return {
        "match_rate": float(match_rate), "mean_rmsd": float(mean_rmsd), "corr_rmsd": float(corr_rmsd),
        "gen_energy_per_atom": float(gv_epa.mean()) if len(gv_epa) else float("nan"),
        "relative_energy_per_atom": float(gv_epa.mean() - ref_epa_mean) if len(gv_epa) else float("nan"),
        "invalid_energy_rate": float(np.mean([is_invalid(a) for a in gen_atoms])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=str(EXPERIMENTS / "full_test_eval.json"))
    args = ap.parse_args()
    device = "cuda"
    ncpu = min(64, os.cpu_count())

    base = os.path.join(snapshot_download("OMatG/MP-20-CSP", allow_patterns=["Trig-SDE-Gamma/*"]),
                        "Trig-SDE-Gamma")
    lm = load_lightning_module(os.path.join(base, "train.yaml"),
                               os.path.join(base, "checkpoint.ckpt"), device)
    policy_model, reference_model = split_policy_reference(lm)
    gamma = get_pos_interpolant(lm)._gamma
    policy, reference = FullModelPolicy(policy_model), FullModelPolicy(reference_model)
    calc = mace_mp(model="medium-mpa-0", device=device, default_dtype="float32")

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        ds = StructureDataset(file_path=TEST_LMDB, lazy_storage=True, niggli_reduce=True,
                              floating_point_precision="32-true")
    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    indices = list(range(n))
    print(f"[eval_full] test structures: {n}  Nt={args.n_steps}  cpus={ncpu}", flush=True)

    ref_atoms = [Atoms(numbers=(o := OMGData(ds[i])).species.numpy(),
                       scaled_positions=o.pos.numpy(), cell=o.cell[0].numpy(), pbc=(1, 1, 1))
                 for i in indices]
    t = time.time()
    ref_epa, _ = energies(calc, ref_atoms)
    ref_epa_mean = float(np.nanmean(ref_epa[~np.isnan(ref_epa)]))
    print(f"[eval_full] reference MACE energies done ({time.time()-t:.0f}s), "
          f"mean ref E/atom={ref_epa_mean:.3f}", flush=True)

    models = [("baseline (pretrained, Nt=50)", reference_model.state_dict()),
              ("velocity-based OMatG-IRL", str(EXPERIMENTS / "full_velocity" / "policy_best.pt")),
              ("score-based OMatG-IRL", str(EXPERIMENTS / "full_score" / "policy_best.pt"))]

    results = {}
    for name, sd in models:
        if isinstance(sd, str):
            if not os.path.exists(sd):
                continue
            sd = torch.load(sd, map_location=device)
        policy_model.load_state_dict(sd)
        t = time.time()
        gen_atoms = gen_all(policy, reference, gamma, lm.sampler, ds, indices, args.n_steps,
                            args.batch, device)
        gen_epa, _ = energies(calc, gen_atoms)
        m = metrics(gen_atoms, ref_atoms, gen_epa, ref_epa_mean, ncpu)
        results[name] = m
        print(f"[{name}] ({time.time()-t:.0f}s) match={m['match_rate']:.4f} "
              f"RMSE={m['mean_rmsd']:.4f} cRMSE={m['corr_rmsd']:.4f} "
              f"relE={m['relative_energy_per_atom']:.4f} invalid={m['invalid_energy_rate']:.4f}",
              flush=True)
    policy_model.load_state_dict(reference_model.state_dict())

    with open(args.out, "w") as f:
        json.dump({"n_structures": n, "ref_energy_per_atom": ref_epa_mean, "results": results}, f, indent=2)
    cols = ["match_rate", "mean_rmsd", "corr_rmsd", "relative_energy_per_atom", "invalid_energy_rate"]
    print("\n" + f"{'model':32s} " + " ".join(f"{c:>16s}" for c in cols))
    for name, m in results.items():
        print(f"{name:32s} " + " ".join(f"{m[c]:16.4f}" for c in cols))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
