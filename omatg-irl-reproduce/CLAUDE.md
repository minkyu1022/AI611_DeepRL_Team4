# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

A reproduction of **§4.2 ("Inference-Time Energy Reinforcement")** of Höllmer & Martiniani
(2026), *Open Materials Generation with Inference-Time Reinforcement Learning* (`papers/`).
OMatG-IRL is a GRPO/PPO policy-gradient method that reinforces a **pretrained** continuous-time
crystal generator *at inference time*: it perturbs the ODE/SDE integration with noise, scores the
terminal structures with the **negative MACE-MPA-0 energy/atom**, and updates the velocity field —
no retraining from scratch.

Layout:
- `OMatG/` — the upstream OMatG framework (cloned, `pip install -e`). **Do not modify it**; the
  reproduction imports it as the `omg` package.
- `omatg_irl/` — the reproduction code (the RL layer). All new work goes here.
- `papers/` — the two source PDFs (base OMatG paper + the IRL paper being reproduced).
- `experiments/` — run outputs (`metrics.csv`, `run.log`, `policy.pt`, figures).
- `.venv/` — the environment (created with `uv`).

## Environment & invariants

- **Activate:** `source .venv/bin/activate` from the repo root (built with `uv`, not
  `python -m venv`; the original host had no `venv`/`conda`/sudo). torch 2.8.0+cu128,
  `torch_scatter` from the PyG wheel index, `omg` editable, `mace-torch`.
- **Paths are portable.** The `omatg_irl` code derives all locations from `omatg_irl/paths.py`
  (relative to the package), and dataset paths like `data/mp_20/...` resolve against the
  installed `omg` package (`files("omg")`), so scripts run from any cwd. The upstream `omg`
  CLI (`omg predict`, etc.) still expects `cd OMatG` for its YAML-relative data paths.
- **Multi-GPU:** run one variant per GPU by setting `CUDA_VISIBLE_DEVICES` (see commands below);
  a single GPU works too (run the variants sequentially).
- **Positions are fractional** on the torus `[0,1)^3`; the model predicts the velocity in
  fractional space. All position math uses minimum-image displacements (`utils_pbc`).
- **Datasets must be loaded with `floating_point_precision="32-true"`** — the default is
  `64-true`, which gives float64 tensors that mismatch the float32 model weights.

## Common commands

```bash
# From the repo root (paths are portable; data resolves via the installed omg package):
source .venv/bin/activate

# Correctness gates (no MACE) — fast sanity check of the whole RL loop, per variant:
python omatg_irl/verify_gates.py velocity     # or: score

# Full training (one variant per GPU, ~1.8-3 h each at G=32):
CUDA_VISIBLE_DEVICES=0 python omatg_irl/train.py --mode velocity --out experiments/full_velocity --n-iters 200 --group-size 32
CUDA_VISIBLE_DEVICES=1 python omatg_irl/train.py --mode score    --out experiments/full_score    --n-iters 200 --group-size 32 --distill-weight 1e-4

# Full MP-20 test-set evaluation (paper-comparable):
python omatg_irl/eval_full.py --limit 0

# Rebuild the reproduction notebook from its builder:
python omatg_irl/build_omatg_irl_reproduce_notebook.py  # -> omatg_irl/omatg-irl-reproduce.ipynb

# Baseline using the upstream CLI (heavier weight-load gate / paper-parity check), run from OMatG/:
cd OMatG
CKPT=$(python -c "from huggingface_hub import snapshot_download as s; print(s('OMatG/MP-20-CSP', allow_patterns=['Trig-SDE-Gamma/*']))")/Trig-SDE-Gamma
omg predict --config "$CKPT/train.yaml" --ckpt_path "$CKPT/checkpoint.ckpt" --model.generation_xyz_filename gen.xyz --trainer.limit_predict_batches 2
omg csp_metrics --config "$CKPT/train.yaml" --xyz_file gen.xyz
```

## omatg_irl architecture (the RL layer)

The Nt-step Euler integration is treated as an MDP: state `(t, x_t)`, policy
`π^θ(x_{t+Δt}|x_t) = N(μ^θ, σ²(t)Δt I)` over the **atomic positions only** (the lattice is always
integrated with the frozen reference ODE). Same per-step variance for policy/old/reference, so the
PPO ratio and KL are closed-form (norm constants cancel):
- `log q_t = (‖x'−μ_old‖² − ‖x'−μ_θ‖²) / (2σ²Δt)`, `KL = ‖μ_θ−μ_ref‖² / (2σ²Δt)`, per atom.

Module map and the two **extension seams** (so §4.3 velocity-annealing is a later add-on):
- `rollout.py` — differentiable forward-Euler / Euler–Maruyama. **Does not use torchsde/torchdiffeq**
  (those run under `no_grad`). `step_mean()` is shared between the no-grad rollout and the grad
  recomputation, guaranteeing identical math. Stores detached `(x_t, x', μ_old, μ_ref, σ, z_ref)`.
- `policy.py` — **SEAM 1.** `Policy` produces the velocity fields; `FullModelPolicy` = the
  trainable CSPNet (§4.2). A learned `s^θ(t)` annealing policy would slot in here for §4.3.
- `reward.py` — **SEAM 2.** `EnergyReward` = MACE-MPA-0 energy/atom with validity penalties
  (vol<0.1, PBC dist<0.5, polar-sine<1e-3 → +3 eV/atom) and in-group ±3σ clipping. A `CRMSEReward`
  would slot in here for §4.3.
- `grpo.py` — group-relative advantages, clipped PPO surrogate, KL, denoiser-distillation (score
  mode). Ratios/KL are **per-atom and averaged** (size-invariant; realizes the `1/(G·Nt·S)`
  prefactor). Loss is accumulated and back-propagated **per step** to bound memory.
- `checkpoint.py` — the checkpoint stores only `model.*` weights and no hyper-parameters, so it
  builds the full module from the shipped `train.yaml` via `OMGCLI(..., run=False)` then loads the
  state dict (strict). Returns a trainable policy + a frozen deepcopy reference.
- `data.py` — builds 16 fixed compositions × G base samples as one batch + `group_ids`.
- `eval.py` — deterministic Nt=50 generation + `match_rmsds`/`metre_rmsds` + relative energy +
  invalid-energy rate.
- `train.py` — driver: GRPO loop with periodic held-out eval, CSV logging, policy checkpoint.

### Upstream `omg` touch points
- `omg.model.model.Model.forward(x, t)` → Data with `pos_b`/`pos_eta` `[N,3]`, `cell_b` `[B,3,3]`;
  `t` must be length B (repeated per structure).
- `omg.si.corrector.PeriodicBoundaryConditionsCorrector` — `correct` (wrap mod 1) / `unwrap`
  (minimum image).
- `omg.sampler.IndependentSampler.sample_p_0(template)` — resamples pos/cell, keeps species (CSP).
- `omg.analysis.match_rmsds` / `metre_rmsds`, `omg.analysis.ValidAtoms` — metrics; the paper's
  match rate uses `ltol=0.3, stol=0.5, angle_tol=10`.

## Gotchas / known risks
- **Noise schedule** `σ(t)=a√((1−t)/t)` diverges at `t→0` and vanishes at `t→1` (ill-conditions the
  ratio/KL). `SqrtNoiseSchedule` has `sigma_max` and a `sigma_min` floor; `a_m` (≈0.1) is read off
  the paper's Fig. 2 and is the main knob to ablate.
- The tiny non-zero `q` at the first PPO epoch is **GPU scatter nondeterminism** (~1e-4 floor), not a
  bug — `verify_gates.py` measures and reports the floor explicitly.
- Exact best hyper-parameters were chosen by an unpublished Optuna sweep; `config.py`/`train.py`
  defaults are mid-range picks inside the published search ranges.
