# omatg-irl-extension

Report-style extension notebook for the Deep RL final assignment.

Research question:

> Can the OMatG-IRL inference-time reinforcement idea, originally evaluated for crystal
> structure prediction, be reused for de novo crystal generation by replacing the paper's
> energy-only reward with a multi-objective S.U.N. reward for stable, unique, and novel crystals?

Files:

- `omatg-irl-extension.ipynb` - Colab-style report notebook for the DNG extension.
- `build_omatg_irl_extension_notebook.py` - regenerates the notebook from source.

Data and checkpoint paths:

- Reproduction checkpoint: Hugging Face `OMatG/MP-20-CSP`, subfolder `Trig-SDE-Gamma/`.
- Extension checkpoint: Hugging Face `OMatG/MP-20-DNG`, subfolder `Linear-SDE-Gamma/`.
- MP-20 LMDB data: `data/mp_20/train.lmdb` and `data/mp_20/test.lmdb`, resolved by OMatG's
  `StructureDataset` relative to the installed `omg` package. The data ships with the OMatG
  repository under `OMatG/omg/data/mp_20/`.

The notebook is report-first and safe to open without CUDA or OMatG installed. Optional live
demo cells are gated behind `RUN_LIVE_DNG_DEMO = False` and require an external OMatG checkout
with the `omg.irl` extension modules.
