"""Build the final-assignment DNG extension tutorial notebook.

The notebook is intentionally report-first: it can be opened and graded without
CUDA or the full OMatG stack, while optional live cells show how to run the DNG
extension when an OMatG checkout with ``omg.irl`` is available.
"""
from __future__ import annotations

import json
from pathlib import Path


EXTENSION_DIR = Path(__file__).resolve().parent
WORKSPACE = EXTENSION_DIR.parent
OUT = EXTENSION_DIR / "omatg-irl-extension.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown(
        """# OMatG-DNG-IRL: Extending Inference-Time RL to De Novo Crystal Generation

This notebook is the final-assignment report notebook. It follows the class guideline:

1. Reproduce the paper baseline with a GitHub-style supplementary implementation.
2. Explain the method in a tutorial/report format.
3. Instantiate the method through a new research question.

**Research question.** Can the OMatG-IRL inference-time reinforcement idea, originally evaluated for
crystal structure prediction, be reused for de novo crystal generation by replacing the paper's
energy-only reward with a multi-objective S.U.N. reward for stable, unique, and novel crystals?

**Supplementary code.** The reproduction implementation is in `omatg-irl-reproduce/`. Its own
notebook, `omatg_irl/omatg-irl-reproduce.ipynb`, walks through the Section 4.2 baseline and
paper-comparable results. This notebook is the extension report that builds on that foundation.

**Data and checkpoint paths.**

- Reproduction checkpoint: Hugging Face `OMatG/MP-20-CSP`, subfolder `Trig-SDE-Gamma/`.
- Extension checkpoint: Hugging Face `OMatG/MP-20-DNG`, subfolder `Linear-SDE-Gamma/`.
- MP-20 LMDB data: `omg/data/mp_20/train.lmdb` or `omg/data/mp_20/test.lmdb`, resolved by OMatG's
  `StructureDataset` relative to the OMatG checkout root. The data ships with the OMatG repository
  under `omg/data/mp_20/`, so there is no separate dataset download step here.

**Running the live extension.** The S.U.N.-reward RL code lives in `omg.irl`, which upstream OMatG
does not ship. This folder vendors those modules under `omg_irl/` plus the reward caches under
`cache/`. The live cell calls `setup_live_demo.setup_omatg()`, which clones upstream OMatG, overlays
the `omg.irl` modules and caches into the clone, and installs it — so the demo runs from a fresh
checkout of this repository with no manual wiring.
"""
    ),
    markdown(
        """## 1. Reproduction Anchor: What Was Reproduced First

The paper baseline is reproduced in the supplementary repository. The implementation uses the public
`OMatG/MP-20-CSP/Trig-SDE-Gamma` checkpoint and evaluates both Section 4.2 variants:

- **Velocity-based OMatG-IRL:** reinforce only the velocity field.
- **Score-based OMatG-IRL:** reinforce the velocity field plus denoiser, with distillation.

The key result is that inference-time RL at `Nt=50` closes most of the gap to the much more expensive
pretrained `Nt=740` sampler while lowering MACE energy. The next cell loads the committed full-test
metrics from the supplementary repository without importing the heavy ML stack.
"""
    ),
    code(
        """import json
from pathlib import Path

HERE = Path.cwd()
candidates = [HERE, *HERE.parents, HERE / "omatg-irl-reproduce"]
SUPP = next((p for p in candidates
             if (p / "experiments" / "full_test_eval.json").exists()), None)
if SUPP is None:
    SUPP = next((p / "omatg-irl-reproduce" for p in candidates
                 if (p / "omatg-irl-reproduce" / "experiments" / "full_test_eval.json").exists()),
                None)
if SUPP is None:
    raise FileNotFoundError("Could not find omatg-irl-reproduce/experiments/full_test_eval.json")

result_path = SUPP / "experiments" / "full_test_eval.json"
data = json.loads(result_path.read_text())
cols = ["match_rate", "mean_rmsd", "corr_rmsd", "relative_energy_per_atom", "invalid_energy_rate"]

print(f"Loaded: {result_path}")
print(f"Full MP-20 test set: {data['n_structures']} structures\\n")
print(f"{'model':32s} " + " ".join(f"{c:>16s}" for c in cols))
for name, metrics in data["results"].items():
    print(f"{name:32s} " + " ".join(f"{metrics[c]:16.4f}" for c in cols))
"""
    ),
    markdown(
        """## 2. Method Summary: Generation as an RL Policy

OMatG-IRL treats the finite-step generative integration as an MDP:

- **State:** the partially generated crystal at time `t`.
- **Action:** the next integration step for fractional coordinates, and in DNG also the species update.
- **Policy:** the pretrained OMatG velocity/denoising network.
- **Reward:** a black-box terminal score assigned after a complete crystal is generated.

For continuous positions, the Euler or Euler-Maruyama step gives a Gaussian transition. Because both the
old policy and updated policy share the same variance, the PPO ratio and KL term have closed forms. GRPO
then compares samples within a group, standardizes rewards into group-relative advantages, and updates
the policy with a clipped PPO objective plus a KL pull back toward the frozen pretrained model.
"""
    ),
    markdown(
        """## 3. Extension: From CSP Energy Reward to DNG S.U.N. Reward

The reproduction reward is energy-only: valid structures are scored by negative MACE-MPA-0 energy per
atom. The DNG extension changes the task and reward while keeping the same inference-time RL template.

| Component | Paper reproduction | DNG extension |
|---|---|---|
| Task | Crystal structure prediction conditioned on composition | De novo generation |
| Position dynamics | Continuous ODE/SDE policy | Same |
| Species dynamics | Fixed by CSP composition | Discrete unmasking policy is part of generation |
| Lattice | Frozen reference path | Frozen reference path |
| Reward | Negative energy per atom | S.U.N.: stable, unique, novel |
| Research aim | Lower energy and preserve match quality | Increase useful de novo hit rate |

The S.U.N. reward is multiplicative. A generated structure must pass validity, stability, novelty, and
uniqueness pressure together; if one term collapses, the total reward collapses. This is stricter than
optimizing a single energy proxy and is closer to the actual goal of de novo discovery.
"""
    ),
    code(
        """# Optional execution preflight.
# The report can be read without these modules. Live DNG cells require an OMatG checkout that
# includes the extension modules under omg.irl plus CHGNet and the usual OMatG dependencies.
import importlib.util

required = [
    "torch",
    "omg",
    "omg.irl.load",
    "omg.irl.irl_sampler",
    "omg.irl.reward",
    "chgnet",
]

def available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False

missing = [name for name in required if not available(name)]
if missing:
    print("omg.irl is not importable in the current process yet:", ", ".join(missing))
    print("That is expected before setup -- the next cell calls setup_live_demo.setup_omatg(),")
    print("which clones OMatG, overlays the vendored omg.irl modules + caches, and installs them.")
    print("Heavy prerequisites (torch, chgnet, pymatgen, matminer) must already be in the environment.")
else:
    print("Live DNG execution prerequisites are available.")
"""
    ),
    markdown(
        """## 4. Optional Live Demo Path

The cell below is gated by `RUN_LIVE_DNG_DEMO = False` so the notebook stays safe to open in a clean
grading environment. Set it to `True` in an environment that has the heavy scientific stack
(torch, torch_scatter, pymatgen, matminer, chgnet) and the cell becomes self-contained: it calls
`setup_live_demo.setup_omatg()` to clone upstream OMatG, overlay the vendored `omg.irl` extension
modules and reward caches from this folder, install the package, then runs a small DNG rollout,
scores it with the S.U.N. reward, and takes a few GRPO steps. No separate OMatG checkout or manual
`omg.irl` wiring is required.
"""
    ),
    code(
        """RUN_LIVE_DNG_DEMO = False

if RUN_LIVE_DNG_DEMO:
    import os
    import sys
    from pathlib import Path

    # Locate this extension folder (it holds setup_live_demo.py + the vendored omg.irl modules
    # and reward caches) regardless of the kernel's starting directory.
    ext_dir = Path.cwd()
    if not (ext_dir / "setup_live_demo.py").exists():
        for cand in [*ext_dir.parents,
                     *ext_dir.glob("**/omatg-irl-extension"),
                     *(p / "omatg-irl-extension" for p in [ext_dir, *ext_dir.parents])]:
            if (cand / "setup_live_demo.py").exists():
                ext_dir = cand
                break
    sys.path.insert(0, str(ext_dir))

    # Clone upstream OMatG, overlay the vendored omg.irl modules + reward caches, install. Idempotent.
    from setup_live_demo import setup_omatg
    repo = setup_omatg()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    import numpy as np
    from huggingface_hub import hf_hub_download
    from torch_geometric.data import Batch
    from omg.datamodule import StructureDataset, OMGDataset
    from omg.irl.load import load_pretrained
    from omg.irl.irl_sampler import IRLSampler
    from omg.irl.reward import SUNReward
    from omg.irl.stability import ElementalReferenceStability
    from omg.irl.novelty import NoveltyScorer
    from omg.irl.grpo import GRPOTrainer, GRPOConfig
    from omg.irl.structures import omgdata_to_ase

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = repo / "ckpts" / "dng_linear_sde_gamma"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["Linear-SDE-Gamma/checkpoint.ckpt", "Linear-SDE-Gamma/train.yaml"]:
        target = ckpt_dir / Path(filename).name
        if not target.exists():
            target.write_bytes(Path(hf_hub_download("OMatG/MP-20-DNG", filename)).read_bytes())

    module = load_pretrained(str(ckpt_dir), map_location=device).to(device).eval()
    templates = OMGDataset(StructureDataset("omg/data/mp_20/train.lmdb", lazy_storage=True,
                                            floating_point_precision="32-true"))

    def sample_base(n):
        idx = np.random.randint(0, len(templates), size=n)
        batch = Batch.from_data_list([templates[int(i)] for i in idx]).to(device)
        return module.sampler.sample_p_0(batch).to(device)

    sampler = IRLSampler(module, n_steps=50, noise_scale=0.1)
    gen, traj = sampler.rollout(sample_base(8))
    atoms = omgdata_to_ase(gen)

    reward = SUNReward(
        stability=ElementalReferenceStability(relax_steps=30),
        novelty=NoveltyScorer(n_reference=1500),
        relax_steps=30,
    )
    scores = reward.compute_group(atoms)
    print("Initial group mean reward:", np.mean([s.reward for s in scores]))

    cfg = GRPOConfig(group_size=8, n_groups=1, n_steps=40, ppo_epochs=2,
                     noise_scale=0.1, clip_eps=0.2, kl_coef=0.001, lr=3e-5)
    trainer = GRPOTrainer(module, reward, templates, cfg, device=device)
    trainer.train(n_iterations=4)
else:
    print("Live DNG demo skipped. Set RUN_LIVE_DNG_DEMO = True in an environment with the heavy stack.")
"""
    ),
    markdown(
        """## 5. Extension Results

The extension sweep used the same GRPO idea with DNG-specific rewards. The most useful comparison is the
relative improvement over the pretrained DNG baseline, because the stability term here uses an offline
CHGNet elemental-reference proxy rather than a full Materials Project convex hull.

| config | SUN | valid | stable | novel | uniq | mean reward | delta SUN |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline, no RL | 0.552 | 0.78 | 0.77 | 0.55 | 0.77 | 2.17 | - |
| g1 full, kl=0.001, lr=1e-5 | 0.656 | 0.79 | 0.78 | 0.66 | 0.78 | 2.57 | +0.104 |
| **g2 full, lr=3e-5** | **0.802** | **0.86** | **0.84** | **0.80** | **0.84** | **3.17** | **+0.250** |
| g3 full, noise=0.05 | 0.688 | 0.80 | 0.79 | 0.69 | 0.79 | 2.74 | +0.135 |
| g4 full, kl=0 | 0.656 | 0.80 | 0.78 | 0.66 | 0.78 | 2.62 | +0.104 |
| g5 ablate uniqueness | 0.635 | 0.78 | 0.77 | 0.64 | 0.77 | 2.51 | +0.083 |
| g6 ablate novelty | 0.688 | 0.82 | 0.80 | 0.69 | 0.80 | 2.74 | +0.135 |
| g7 ablate stability | 0.708 | 0.82 | 0.80 | 0.71 | 0.80 | 2.78 | +0.156 |
"""
    ),
    code(
        """baseline_sun = 0.552
runs = {
    "g1 full, lr=1e-5": 0.656,
    "g2 full, lr=3e-5": 0.802,
    "g3 full, noise=0.05": 0.688,
    "g4 full, kl=0": 0.656,
    "g5 no uniqueness": 0.635,
    "g6 no novelty": 0.688,
    "g7 no stability": 0.708,
}

best = max(runs.items(), key=lambda item: item[1])
print(f"Best run: {best[0]}  SUN={best[1]:.3f}")
print(f"Absolute improvement: {best[1] - baseline_sun:+.3f}")
print(f"Relative improvement: {(best[1] / baseline_sun - 1.0) * 100:.1f}%")
"""
    ),
    markdown(
        """## 6. Interpretation and Limitations

The result supports the research question at the level of a course project: the same inference-time
reinforcement template can be instantiated for DNG by changing the rollout policy surface and terminal
reward. The best DNG run improves SUN from 0.552 to 0.802 under the offline proxy.

Important limitations:

- The absolute SUN value is not directly comparable to literature values that use a true DFT or Materials
  Project convex hull. The trustworthy signal here is the relative improvement under a fixed evaluator.
- Novelty thresholds and uniqueness checks should be calibrated against a held-out reference set before
  using the model for claims about new materials.
- The DNG extension is more expensive than CSP energy reinforcement because reward evaluation includes
  relaxation and batch-level diversity checks.

Most impactful next step: replace the elemental-reference stability proxy with a cached convex-hull
stability scorer, then rerun the same GRPO sweep.
"""
    ),
    markdown(
        """## 7. Reproduction Commands

Reproduce the paper baseline from the supplementary repository:

```bash
cd omatg-irl-reproduce
python omatg_irl/verify_gates.py velocity
python omatg_irl/verify_gates.py score
python omatg_irl/train.py --mode velocity --out experiments/full_velocity --n-iters 200 --group-size 32
python omatg_irl/train.py --mode score --out experiments/full_score --n-iters 200 --group-size 32 --distill-weight 1e-4
python omatg_irl/eval_full.py --limit 0 --out experiments/full_test_eval.json
```

Run the DNG extension when the external `omg.irl` implementation is available:

```bash
python -m omg.irl.run --ckpt ckpts/dng_linear_sde_gamma --device cuda \\
    --group-size 20 --n-groups 2 --n-steps 50 --ppo-epochs 2 --iterations 300 \\
    --noise-scale 0.1 --kl-coef 0.001 --lr 3e-5 --out runs/my_run
python -m omg.irl.evaluate --ckpt runs/my_run/irl_checkpoint.ckpt \\
    --base ckpts/dng_linear_sde_gamma --n-samples 96 --out runs/my_run/eval.json
```
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=2) + "\n")
print(f"wrote {OUT} with {len(cells)} cells")
