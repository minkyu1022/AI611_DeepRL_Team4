"""Novelty scoring for OMatG-DNG-IRL (CRYSTAL Sec. 3.3).

A structure is novel iff it is sufficiently distinct from a reference dataset (default: MP-20 train) in BOTH
of two orthogonal descriptors:
  * structural:  mean CrystalNNFingerprint (local coordination / topology),
  * compositional: ElementProperty / Magpie fingerprint (elemental identity).

The AND of the two (rather than OR) prevents the model from claiming novelty via only-composition or
only-structure innovation (CRYSTAL Sec. 3.3 / App. B). Reference fingerprints are precomputed once over a
subset of the reference dataset and cached to disk.
"""
import warnings
from pathlib import Path
from typing import List, Optional
import numpy as np


class NoveltyScorer:
    """
    :param reference_lmdb:
        Path to the reference dataset (LMDB) whose fingerprints define "known" materials.
    :param n_reference:
        Number of reference structures to fingerprint (subset for speed). Use -1 for all.
    :param d_struct / d_comp:
        L2-distance thresholds above which a structure is considered novel in the structural / compositional
        descriptor respectively.
    :param cache:
        Path of the ``.npz`` cache of reference fingerprints.
    """

    def __init__(self, reference_lmdb: str = "omg/data/mp_20/train.lmdb", n_reference: int = 5000,
                 d_struct: float = 0.2, d_comp: float = 2.0,
                 cache: str = "ckpts/hull/mp20_novelty_ref.npz") -> None:
        self.reference_lmdb = reference_lmdb
        self.n_reference = n_reference
        self.d_struct = d_struct
        self.d_comp = d_comp
        self.cache = Path(cache)
        self._ref_struct: Optional[np.ndarray] = None
        self._ref_comp: Optional[np.ndarray] = None

    def _ensure_reference(self) -> None:
        if self._ref_struct is not None:
            return
        if self.cache.exists():
            data = np.load(self.cache)
            self._ref_struct, self._ref_comp = data["struct"], data["comp"]
            return
        # Build the reference fingerprint bank from the reference dataset.
        from omg.datamodule import StructureDataset, OMGDataset
        from omg.analysis.valid_atoms import ValidAtoms
        from .structures import omgdata_to_ase
        ds = OMGDataset(StructureDataset(self.reference_lmdb, lazy_storage=True, floating_point_precision="32-true"))
        n = len(ds) if self.n_reference < 0 else min(self.n_reference, len(ds))
        struct_fps, comp_fps = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(n):
                atoms = omgdata_to_ase(ds[i])[0]
                va = ValidAtoms(atoms, skip_validation=True)
                try:
                    comp_fp, struct_fp = ValidAtoms._get_fingerprints(va.structure.composition, va.structure)
                    struct_fps.append(struct_fp)
                    comp_fps.append(comp_fp)
                except Exception:
                    continue
        self._ref_struct = np.asarray(struct_fps, dtype=np.float64)
        self._ref_comp = np.asarray(comp_fps, dtype=np.float64)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache, struct=self._ref_struct, comp=self._ref_comp)

    def is_novel(self, structures: List[object]) -> List[bool]:
        """Return a boolean per structure: novel (AND of structural and compositional novelty)."""
        from omg.analysis.valid_atoms import ValidAtoms
        self._ensure_reference()
        flags = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in structures:
                try:
                    comp_fp, struct_fp = ValidAtoms._get_fingerprints(s.composition, s)
                    ds_struct = np.linalg.norm(self._ref_struct - np.asarray(struct_fp), axis=1).min()
                    ds_comp = np.linalg.norm(self._ref_comp - np.asarray(comp_fp), axis=1).min()
                    flags.append(bool(ds_struct > self.d_struct and ds_comp > self.d_comp))
                except Exception:
                    flags.append(False)
        return flags
