# OMatG-IRL Reproduction — §4.2 Inference-Time Energy Reinforcement

## Context
We reproduce the §4.2 results of *"Open Materials Generation with Inference-Time
Reinforcement Learning"* (Höllmer & Martiniani, 2026) on top of the existing OMatG
framework (`OMatG`). OMatG-IRL is a policy-gradient (GRPO/PPO)
method that reinforces a pretrained continuous-time generative model **at inference time**
by adding stochastic perturbations to the integration and optimizing a black-box reward —
here the **negative MACE-MPA-0 energy/atom** — without retraining from scratch.

**Decisions (confirmed with user):**
- **Scope:** §4.2 only — both *score-based* and *velocity-based* OMatG-IRL on MP-20 with the
  public `Trig-SDE-Gamma` checkpoint. §4.3 (velocity-annealing) is **deferred** because it
  needs a non-public MP-20-polymorph-split `Linear-ODE` checkpoint (multi-day pretraining).
- **Hyperparameters:** use paper/checkpoint reported values (no Optuna sweep — infeasible in a day).
- **Goal:** approach the paper's numbers. Validate the pipeline at tiny scale first, then run full scale.
- **Extensibility (per user):** §4.2 and §4.3 share the entire RL core. We design two seams so
  §4.3 later is a small add-on, **not** a rewrite:
  1. **Pluggable policy / velocity-modulation** — the rollout's drift comes from a `Policy`
     object: §4.2 = full fine-tuned CSPNet velocity; §4.3 = frozen base velocity × `(1+sᶿ(t))` MLP.
  2. **Pluggable reward** — a `Reward` interface: §4.2 = energy; §4.3 = cRMSE.
  The only real §4.3 blocker remains the checkpoint pretraining, not the code.

**Compute:** a CUDA GPU. Full runs can spread the two variants across two GPUs (one each) via
`CUDA_VISIBLE_DEVICES`, or run sequentially on a single GPU.

## Environment setup
Fresh system Python 3.12, CUDA 12.8, no torch/conda. Create a project venv:
`python3 -m venv .venv && pip install` → install PyTorch 2.8 (cu128 wheels) + PyG 2.7 +
torch_scatter, then `pip install -e OMatG` (pulls the rest of `pyproject.toml`), then
`pip install mace-torch` (new dependency, used only by the reward). Verify CUDA + a forward pass.

## Validated math (per-step Gaussian MDP, position coords only)
Per step the policy is `πᶿ(x'|x_t)=N(x'; μᶿ, vI)`, `v=σ²(t)Δt`, with `μᶿ=corr_COM(x_t+driftᶿ·Δt)`.
Same `v` for policy/old/ref — only the mean differs. Therefore (summed over all 3N position dims,
using **minimum-image** displacements on the torus):
- **log-ratio** `log q_t = (‖x'−μ_old‖² − ‖x'−μᶿ‖²)/(2σ²(t)Δt)` (norm consts cancel ⇒ clean).
- **KL** `D_KL(πᶿ‖π_ref) = ‖μᶿ−μ_ref‖²/(2σ²(t)Δt)` (analytic, exact, differentiable).
- Gradients flow **only through μᶿ**; `x', μ_old, μ_ref, σ_t, z_ref` are stored detached.
- Invariants used as correctness gates: `q_t=1` at the first PPO epoch; `KL=0` at init.

**Drifts** (lattice always frozen reference ODE, no noise; `velocity_annealing=0` everywhere):
- *Velocity-based* (Eq.13): `μ=corr_COM(x_t+bᶿ·Δt)`, noise `σ(t)=a·√((1−t)/t)`.
- *Score-based* (Eq.7): `μ=corr_COM(x_t+[bᶿ−(σ²(t)/(2γ(t)))zᶿ]·Δt)`, same `σ(t)` in drift+diffusion.

## New package: `omatg_irl/` (sibling of `omg/`, imports it; existing `omg/` is NOT modified)
```
omatg_irl/
  config.py        # dataclasses: RolloutConfig, GRPOConfig, RewardConfig, ExperimentConfig
  checkpoint.py    # load via shipped YAML + OMGLightning.load_from_checkpoint(...).model;
                   #   deepcopy → frozen reference. Gate on baseline match-rate.
  policy.py        # SEAM 1. Policy ABC -> drift(x,t)->(b[,z]).
                   #   FullModelPolicy (§4.2, trainable Model). [§4.3: AnnealingSchedulePolicy(MLP sᶿ)]
  schedules.py     # SqrtNoiseSchedule σ(t)=a√((1−t)/t) with clamp; wraps gamma/epsilon.
  rollout.py       # Rollout: differentiable Nt-step Euler / Euler-Maruyama (NO torchsde/torchdiffeq).
                   #   step_mean() shared by no-grad rollout and grad recompute (identical math).
                   #   stores (x_t, x', μ_old, μ_ref, σ_t, z_ref); COM-projects displacement (+noise).
  reward.py        # SEAM 2. Reward ABC. EnergyReward(MACE-MPA-0): validity(vol<0.1, dist<0.5 PBC,
                   #   polar-sine<1e-3)→+3 eV/atom penalty; valid→MACE e/atom clipped ±3σ in-group;
                   #   reward=−energy/atom. [§4.3: CRMSEReward reusing match_rmsds.]
  grpo.py          # GRPOTrainer: group advantages (r−mean)/std; PPO clip ε; KL β; denoiser-distill δ
                   #   (score only); Adam; PPO epochs K; per-atom S normalization. train(n_iters).
  eval.py          # match_rmsds/metre_rmsds + relative-energy/atom + invalid-energy-rate.
  data.py          # build_groups(): 16 fixed compositions from test set; G copies each; sampler.sample_p_0.
  utils_pbc.py     # minimum_image, com_project, sq_dist_torus.
```
**Reuse (no reinvention):** `omg.model.model.Model`, `omg.si.corrector.PeriodicBoundaryConditionsCorrector`
(`correct`/`unwrap`), `omg.si.gamma.LatentGammaSqrt`, `omg.si.epsilon.VanishingEpsilon`,
`omg.sampler.IndependentSampler` (as in the checkpoint YAML), `omg.analysis.match_rmsds`/`metre_rmsds`,
`omg.analysis.ValidAtoms`, `omg.datamodule.StructureDataset`. Baseline generation reuses the original
`StochasticInterpolants.integrate` (740 steps) for the weight-load gate.

## Reported configuration to use
From `Trig-SDE-Gamma/train.yaml`: pos = PeriodicTrigonometricInterpolant/SDE, γ=√(a t(1−t)) a=0.0492,
ε=VanishingEpsilon(c=9.42,μ=0.197,σ=0.040); cell = TrigonometricInterpolant/ODE.
RL settings (paper §4.2 / App. E–F): `Nt=50`, 16 groups, `G=32`, square-root noise medium scale
`a_m` (read off Fig.2 log-axis — start `a_m≈0.1`, ablate {0.03,0.1,0.3}), PPO `ε≈0.2`, `K=2` epochs,
KL `β`, denoiser `δ` from the App.F choice sets (start `β=1e-4`, `δ=1e-4`), Adam `lr≈1e-4`, per-atom
S-normalization, 300 iterations. (`velocity_annealing=0`.) Exact best HP values aren't published —
these are mid-range picks inside the published search ranges.

## Verification (cheap gates before any 300-iter run; single GPU, G=8, 2 groups, 5 iters)
1. `q_t=1` (|log q|<1e-5) at first PPO epoch. 2. `KL=0` at init. 3. KL grows / q deviates after an update.
4. Finite-difference check of `∂(−‖x'−μᶿ‖²/2v)` vs autograd. 5. Reward trends up / MACE e/atom down.
6. COM invariance (per-structure mean displacement ≈0; cell path bitwise-equal to frozen ODE).
7. **Weight-load gate:** 740-step baseline generation reproduces the pretrained match-rate.
8. Noise-blowup guard: print/clamp `σ(t_min)√Δt`. Wire all as assert cells so a full run fails fast.

## Single reproduction notebook: `omatg_irl/omatg-irl-reproduce.ipynb`
Env check → download `OMatG/MP-20-CSP/Trig-SDE-Gamma` (huggingface_hub) → load policy+reference →
**baseline gen + metrics (correctness gate)** → build 16 groups → **velocity-based** RL training →
**score-based** RL training → Fig.3-style 4-panel curves (rel-energy/atom, match rate, RMSE,
invalid-energy-rate vs iteration; velocity vs score) → **final test-set eval** table vs baseline →
save checkpoints/metrics/figures.

## Key risks (carried from design review)
- **`a_m` + σ blow-up at t→0** (highest): √((1−t)/t) diverges; clamp σ or use `t_min≈0.02`; ablate `a_m`.
- **MACE-MPA-0 version + reference-energy convention** for "relative energy/atom" — pin version, state
  the reference (MACE-on-ref-structures), keep consistent so the curve is comparable to Fig.3.
- **Polar-sine validity** not in `ValidAtoms` — implement explicitly (near-degenerate cell), document formula.
- **S normalization / KL direction** — use per-atom mean and the analytic same-variance KL; ablate if needed.
- Enforce `velocity_annealing=0` on **both** policy drift and the frozen cell ODE.

## Out of scope (this task)
§4.3 velocity-annealing, Optuna/Ray-Tune sweeps, MP-20-polymorph-split pretraining, DNG task.
The two seams above make §4.3 a later add-on.

## Final-assignment extension boundary

For the course submission, this repository is the paper-reproduction and baseline-validation
layer. The extension is presented in `../omatg-irl-extension/omatg-irl-extension.ipynb`:
it asks whether the same inference-time RL template can be instantiated for de novo generation
with a S.U.N. reward. That notebook is report-first and gates its live DNG cells behind an
external `omg.irl` implementation check, so the reproduction code here remains focused on
the public Section 4.2 CSP checkpoint and metrics.
