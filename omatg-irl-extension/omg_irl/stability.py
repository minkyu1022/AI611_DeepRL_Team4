"""Stability scoring for OMatG-DNG-IRL via a machine-learned interatomic potential (CHGNet).

A ``StabilityScorer`` relaxes generated structures and returns, per structure, a triple
``(energy_above_hull, relaxed_structure, converged)`` where ``energy_above_hull`` is in eV/atom.

The default ``ElementalReferenceStability`` uses a self-consistent CHGNet formation energy with respect to
elemental references (CRYSTAL Eq. 5). This is fully offline. It is a *lenient* proxy for the true
Materials-Project convex hull (it ignores competing binary/ternary phases); a cached-hull backend with the
same interface can be substituted without touching the reward or trainer code.
"""
from abc import ABC, abstractmethod
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

_CHGNET = None
_RELAXER = None


def _get_chgnet():
    """Lazily construct and cache the CHGNet model and ASE relaxer."""
    global _CHGNET, _RELAXER
    if _CHGNET is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from chgnet.model import CHGNet
            from chgnet.model import StructOptimizer
            _CHGNET = CHGNet.load(verbose=False)
            _RELAXER = StructOptimizer(model=_CHGNET, use_device=None)
    return _CHGNET, _RELAXER


class StabilityScorer(ABC):
    """Maps relaxed structures to an energy above hull (eV/atom)."""

    @abstractmethod
    def score(self, structures: List["object"]) -> List[Tuple[float, object, bool]]:
        """Return ``(energy_above_hull, relaxed_structure, converged)`` for each input pymatgen ``Structure``."""
        raise NotImplementedError


class ElementalReferenceStability(StabilityScorer):
    """
    CHGNet formation energy per atom relative to elemental references (self-consistent, offline).

    :param relax_steps:
        Maximum number of FIRE relaxation steps per structure.
    :param fmax:
        Force convergence threshold (eV/A) for the relaxation.
    :param ref_cache:
        Path of the JSON cache of per-element reference energies (eV/atom).
    """

    def __init__(self, relax_steps: int = 50, fmax: float = 0.1,
                 ref_cache: str = "ckpts/hull/chgnet_elem_refs.json") -> None:
        self.relax_steps = relax_steps
        self.fmax = fmax
        self.ref_cache = Path(ref_cache)
        self._refs: Dict[str, float] = {}
        if self.ref_cache.exists():
            self._refs = {k: float(v) for k, v in json.loads(self.ref_cache.read_text()).items()}

    # ---- elemental references ----------------------------------------------------------------
    def _elemental_reference(self, symbol: str) -> float:
        """CHGNet energy per atom of the (relaxed) elemental reference structure for ``symbol``."""
        if symbol in self._refs:
            return self._refs[symbol]
        from ase.build import bulk
        from pymatgen.io.ase import AseAtomsAdaptor
        _, relaxer = _get_chgnet()
        atoms = None
        for crystal in ("fcc", "bcc", "hcp", "sc", "diamond"):
            try:
                atoms = bulk(symbol, crystalstructure=crystal, a=3.5, cubic=False)
                break
            except Exception:
                continue
        if atoms is None:
            # Fallback: a single atom in a large box (very rough reference).
            from ase import Atoms as _A
            atoms = _A(symbol, positions=[[0, 0, 0]], cell=[8, 8, 8], pbc=True)
        structure = AseAtomsAdaptor.get_structure(atoms)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = relaxer.relax(structure, steps=self.relax_steps, fmax=self.fmax, verbose=False)
        e_per_atom = float(res["trajectory"].energies[-1]) / len(structure)
        self._refs[symbol] = e_per_atom
        self.ref_cache.parent.mkdir(parents=True, exist_ok=True)
        self.ref_cache.write_text(json.dumps(self._refs))
        return e_per_atom

    def score(self, structures: List[object]) -> List[Tuple[float, object, bool]]:
        _, relaxer = _get_chgnet()
        out: List[Tuple[float, object, bool]] = []
        for structure in structures:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = relaxer.relax(structure, steps=self.relax_steps, fmax=self.fmax, verbose=False)
                relaxed = res["final_structure"]
                e_total = float(res["trajectory"].energies[-1])
                n = len(relaxed)
                e_per_atom = e_total / n
                comp = relaxed.composition
                # Formation energy per atom relative to elemental references.
                ref = 0.0
                for el, amt in comp.get_el_amt_dict().items():
                    ref += (amt / n) * self._elemental_reference(el)
                e_form = e_per_atom - ref
                converged = bool(np.isfinite(e_form))
                out.append((e_form, relaxed, converged))
            except Exception:
                out.append((float("nan"), structure, False))
        return out
