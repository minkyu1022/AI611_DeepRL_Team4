# omatg-irl-extension

Report-style extension notebook for the Deep RL final assignment.

Research question:

> Can the OMatG-IRL inference-time reinforcement idea, originally evaluated for crystal
> structure prediction, be reused for de novo crystal generation by replacing the paper's
> energy-only reward with a multi-objective S.U.N. reward for stable, unique, and novel crystals?

Files:

- `omatg-irl-extension.ipynb` - Colab-style report notebook for the DNG extension.
- `build_omatg_irl_extension_notebook.py` - regenerates the notebook from source.
- `setup_live_demo.py` - prepares a runnable OMatG checkout for the live demo (see below).
- `omg_irl/` - the vendored `omg.irl` extension modules (S.U.N. reward, stability and novelty
  scorers, GRPO trainer, IRL sampler, loaders). Upstream OMatG does not ship these.
- `cache/` - precomputed reward caches: CHGNet elemental references
  (`chgnet_elem_refs.json`) and MP-20 novelty fingerprints (`mp20_novelty_ref.npz`).

## Running the live extension (S.U.N. reward + GRPO)

The S.U.N.-reward RL code lives in `omg.irl`, which upstream OMatG does not provide. Rather than
vendoring the whole OMatG framework, this folder vendors only the `omg.irl` modules (`omg_irl/`)
and overlays them onto an upstream clone at run time:

```python
from setup_live_demo import setup_omatg
repo = setup_omatg()   # clone OMatG, overlay omg.irl + caches, install; returns the checkout path
```

`setup_omatg()` is idempotent and does, in order:

1. clone upstream OMatG (brings the `omg` package + the MP-20 LMDB data under `omg/data/mp_20/`),
2. overlay `omg_irl/*.py` into `<clone>/omg/irl/`,
3. seed `cache/*` into `<clone>/ckpts/hull/` so the first reward call is instant and offline-safe,
4. editable-install the clone and CHGNet.

In the notebook this is wired into the live cell: set `RUN_LIVE_DNG_DEMO = True` in an environment
that has the heavy scientific stack (torch, torch_scatter, pymatgen, matminer, chgnet) and the cell
runs end to end from a fresh clone of this repository — no manual `omg.irl` wiring needed.

Data and checkpoint paths:

- Reproduction checkpoint: Hugging Face `OMatG/MP-20-CSP`, subfolder `Trig-SDE-Gamma/`.
- Extension checkpoint: Hugging Face `OMatG/MP-20-DNG`, subfolder `Linear-SDE-Gamma/`.
- MP-20 LMDB data: `omg/data/mp_20/train.lmdb` and `omg/data/mp_20/test.lmdb`, resolved by OMatG's
  `StructureDataset` relative to the OMatG checkout root (the data ships under `omg/data/mp_20/`).

The notebook remains report-first and safe to open without CUDA or OMatG installed; only the gated
live cell needs the heavy stack.
