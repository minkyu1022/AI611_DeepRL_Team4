# OMatG-IRL - Reproduction Code for the Deep RL Final Assignment

This repository is the supplementary implementation for the Deep RL final assignment. It provides
the "solid starting point" requested by the guideline: a reproduction of **Section 4.2** of
Höllmer & Martiniani (2026), *Open Materials Generation with Inference-Time Reinforcement Learning*
(OMatG-IRL), built on top of the public
[OMatG](https://github.com/FERMat-ML/OMatG) framework and the public `OMatG/MP-20-CSP/Trig-SDE-Gamma`
checkpoint.

OMatG-IRL is a GRPO/PPO policy-gradient method that reinforces a **pretrained** continuous-time
crystal generator **at inference time**: it perturbs the ODE/SDE integration with noise, scores the
terminal structures with a black-box reward (the **negative MACE-MPA-0 energy per atom**), and
updates the velocity field — without retraining from scratch. We reproduce both the **velocity-based**
(Eq. 13, velocity only) and **score-based** (Eq. 7, velocity + denoiser) variants on MP-20.

## Results (full MP-20 test set, 9046 structures, Nt=50)

| model | match rate ↑ | RMSE ↓ | cRMSE ↓ | rel. energy/atom (eV) ↓ | invalid ↓ |
|---|---|---|---|---|---|
| baseline (pretrained, Nt=50) | 0.596 | 0.203 | 0.323 | 1.192 | 0.0064 |
| **velocity-based OMatG-IRL** | **0.675** | **0.086** | 0.221 | 0.300 | 0.0073 |
| **score-based OMatG-IRL** | **0.674** | **0.085** | 0.220 | 0.240 | 0.0042 |

**Paper anchors:** baseline Nt=50 (no annealing) = 0.594 / 0.202 (Table 1) — reproduced almost
exactly; pretrained Nt=740 (SDE + velocity annealing) = 0.686 / 0.125; RL @ Nt=50 ≈ 0.65–0.68 / ~0.09
(Fig. 3). The reinforced models match the Nt=740 annealed reference's match rate with **lower RMSE at
14× fewer integration steps**, reduce the relative energy per atom by ~0.9 eV, and **velocity-based ≈
score-based** — confirming the paper's central claim that effective RL is possible without an explicit
score.

## Final Assignment Deliverables

The assignment package has two notebook/report layers:

| file | purpose |
|---|---|
| `omatg_irl/omatg-irl-reproduce.ipynb` | Reproduction notebook for the OMatG-IRL paper baseline and Section 4.2 method. |
| `../omatg-irl-extension/omatg-irl-extension.ipynb` | Report-style extension notebook: "Can OMatG-IRL be instantiated for de novo generation with a S.U.N. reward?" |

This repository contains the full runnable reproduction code. The top-level DNG notebook is the
course-report extension: it references this repo for the reproduced baseline and presents the
research question, method adaptation, results, and limitations for de novo generation.

The single self-contained reproduction notebook **`omatg_irl/omatg-irl-reproduce.ipynb`** walks
through the whole pipeline (checkpoint -> baseline -> groups -> RL training -> Fig. 3 curves ->
final tables).

## Setup

Requires a CUDA GPU (CUDA 12.8 toolchain) and Python 3.11–3.13.

```bash
# 1. Clone the upstream OMatG framework alongside this repo (pinned commit)
git clone https://github.com/FERMat-ML/OMatG.git
git -C OMatG checkout fcb9ba2

# 2. Create the environment (uv recommended; or use venv/conda)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install torch_geometric==2.7.0
uv pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install -e ./OMatG
uv pip install mace-torch nbformat nbconvert ipykernel
```

The MP-20 dataset ships inside `OMatG/omg/data/`; the Trig-SDE-Gamma checkpoint and MACE-MPA-0 weights
download automatically on first use.

## Running

All commands run from the `OMatG/` directory (so the configs' relative data paths resolve):

```bash
cd OMatG && source ../.venv/bin/activate

# Correctness gates (no MACE) — fast sanity check of the RL loop
python ../omatg_irl/verify_gates.py velocity      # or: score

# Train (one variant per GPU, ~1.8 h each at G=32, 200 iters)
CUDA_VISIBLE_DEVICES=0 python ../omatg_irl/train.py --mode velocity --out ../experiments/full_velocity --n-iters 200 --group-size 32
CUDA_VISIBLE_DEVICES=1 python ../omatg_irl/train.py --mode score    --out ../experiments/full_score    --n-iters 200 --group-size 32 --distill-weight 1e-4

# Full test-set evaluation (paper-comparable)
python ../omatg_irl/eval_full.py --limit 0 --out ../experiments/full_test_eval.json

# Rebuild + execute the notebook
python ../omatg_irl/build_omatg_irl_reproduce_notebook.py
jupyter nbconvert --to notebook --execute --inplace ../omatg_irl/omatg-irl-reproduce.ipynb
```

## Code layout (`omatg_irl/`)

The Nt-step Euler integration is treated as an MDP; the per-step transition is a Gaussian over the
atomic positions, giving a closed-form per-atom PPO ratio and KL. Two extension seams keep §4.3
(velocity-annealing) a later add-on.

| file | role |
|---|---|
| `rollout.py` | differentiable Euler / Euler–Maruyama rollout (no torchsde/torchdiffeq); shared `step_mean` for rollout + grad recompute |
| `policy.py` | **SEAM 1** — `Policy`; `FullModelPolicy` (trainable CSPNet velocity) |
| `reward.py` | **SEAM 2** — `Reward`; `EnergyReward` (MACE-MPA-0 + validity penalties + in-group clipping) |
| `grpo.py` | group advantages, clipped PPO, KL, denoiser distillation; per-atom & per-step |
| `checkpoint.py` | build module from shipped YAML + strict load; policy + frozen reference |
| `schedules.py` | square-root noise schedule `σ(t)=a√((1−t)/t)` with floor/clamp |
| `data.py` / `eval.py` / `eval_full.py` | group building / held-out eval / full-test-set eval |
| `train.py` | GRPO driver with periodic eval, CSV logging, best-checkpoint saving |
| `omatg-irl-reproduce.ipynb` | the single self-contained reproduction notebook |

See `CLAUDE.md` for architecture details and environment invariants.

## Notes
- Trained checkpoints (`*.pt`, ~50 MB each) and the upstream `OMatG/` clone are git-ignored; the
  lightweight results (`experiments/**/metrics*.csv`, `best.json`, `full_test_eval.json`,
  `fig3_curves.png`) are committed.
- Hyperparameters are mid-range picks within the paper's published search ranges (the exact
  Optuna-selected values are not published). The noise scale `a_m ≈ 0.1` is read off the paper's Fig. 2.
- **Out of scope:** §4.3 (velocity-annealing) requires a non-public MP-20-polymorph-split checkpoint
  (multi-day pretraining); the two seams above make it a small add-on otherwise.
