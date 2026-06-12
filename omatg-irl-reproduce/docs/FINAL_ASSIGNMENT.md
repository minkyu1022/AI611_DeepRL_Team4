# Final Assignment Package

This package follows the assignment guideline in two stages.

## 1. Paper Reproduction

The repository reproduces Section 4.2 of OMatG-IRL on MP-20 CSP.

- Main code: `omatg_irl/`
- Reproduction notebook: `omatg_irl/omatg-irl-reproduce.ipynb`
- Full-test metrics: `experiments/full_test_eval.json`
- Figure artifacts: `experiments/fig3_curves.png`, `experiments/fig3_curves.pdf`

The reproduced full-test result is:

| model | match rate | RMSE | cRMSE | rel. energy/atom | invalid |
|---|---:|---:|---:|---:|---:|
| baseline, Nt=50 | 0.596 | 0.203 | 0.323 | 1.192 | 0.0064 |
| velocity-based OMatG-IRL | 0.675 | 0.086 | 0.221 | 0.300 | 0.0073 |
| score-based OMatG-IRL | 0.674 | 0.085 | 0.220 | 0.240 | 0.0042 |

This anchors the implementation against the paper before introducing the extension.

## 2. Extension Research Question

The extension notebook `../omatg-irl-extension/omatg-irl-extension.ipynb` is the report-style
extension notebook.

Research question:

> Can the OMatG-IRL inference-time reinforcement idea, originally evaluated for crystal
> structure prediction, be reused for de novo crystal generation by replacing the paper's
> energy-only reward with a multi-objective S.U.N. reward for stable, unique, and novel crystals?

The DNG notebook is report-first and safe to open without CUDA. Optional live cells are gated and
only run when an OMatG checkout with the external `omg.irl` extension modules is available.

## Validation

Run the lightweight artifact checks from this repository:

```bash
python tests/test_assignment_package.py
```

These tests do not import torch, OMatG, MACE, ASE, or torch-geometric. They validate the
assignment artifacts and committed reproduction metrics in a clean environment.

Full scientific verification still requires the GPU environment described in `README.md`.
