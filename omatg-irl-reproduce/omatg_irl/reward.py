"""Reward abstraction (extension SEAM 2) and the energy-based reward of Section 4.2.

The energy reward (Appendix E):
  * Invalid structures (Appendix B) are assigned a penalty energy of +3 eV/atom.
    Invalid := unit-cell volume < 0.1 A^3, OR any PBC pairwise distance < 0.5 A,
    OR polar-sine of the lattice vectors < 1e-3 (near-degenerate cell).
  * Valid structures get the MACE-MPA-0 energy per atom.
  * Within each GRPO group the per-atom energies are clipped to mean +/- 3 std
    so outliers do not dominate the group-relative advantage.
  * Reward = -(clipped energy per atom). Any constant offset cancels in the
    group-relative advantage, so no explicit reference energy is needed.

Section 4.3 would instead provide a ``CRMSEReward`` reusing ``match_rmsds``; the
GRPO trainer is agnostic to which ``Reward`` is plugged in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data


def omgdata_to_atoms(data: Data) -> list[Atoms]:
    """Convert a batched OMGData (fractional positions) to per-structure ase.Atoms."""
    data = data.to("cpu")
    atoms_list = []
    for i in range(len(data.n_atoms)):
        lo, hi = int(data.ptr[i]), int(data.ptr[i + 1])
        atoms_list.append(Atoms(
            numbers=data.species[lo:hi].numpy(),
            scaled_positions=data.pos[lo:hi].numpy(),
            cell=data.cell[i].numpy(),
            pbc=(1, 1, 1),
        ))
    return atoms_list


def polar_sine(cell: np.ndarray) -> float:
    """|det(L)| / (||a|| ||b|| ||c||); -> 0 for a near-degenerate cell."""
    norms = np.linalg.norm(cell, axis=1)
    denom = float(np.prod(norms))
    if denom < 1e-12:
        return 0.0
    return abs(float(np.linalg.det(cell))) / denom


def is_invalid(atoms: Atoms, vol_cutoff: float = 0.1, dist_cutoff: float = 0.5,
               polar_sine_cutoff: float = 1e-3) -> bool:
    """Return True if the structure fails any of the three validity criteria."""
    try:
        vol = abs(atoms.get_volume())
    except Exception:
        return True
    if vol < vol_cutoff:
        return True
    if polar_sine(np.asarray(atoms.cell)) < polar_sine_cutoff:
        return True
    if len(atoms) > 1:
        d = atoms.get_all_distances(mic=True)
        n = len(atoms)
        iu = np.triu_indices(n, k=1)
        if d[iu].min() < dist_cutoff:
            return True
    return False


class Reward(ABC):
    """Maps a batch of generated structures + group ids to per-structure rewards."""

    @abstractmethod
    def __call__(self, final: Data, group_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Return (rewards [B], diagnostics dict)."""


class EnergyReward(Reward):
    def __init__(self, mace_calc, device: str = "cuda", invalid_penalty: float = 3.0,
                 clip_std: float = 3.0, vol_cutoff: float = 0.1, dist_cutoff: float = 0.5,
                 polar_sine_cutoff: float = 1e-3):
        self.calc = mace_calc
        self.device = device
        self.invalid_penalty = invalid_penalty
        self.clip_std = clip_std
        self.vol_cutoff = vol_cutoff
        self.dist_cutoff = dist_cutoff
        self.polar_sine_cutoff = polar_sine_cutoff

    def _energy_per_atom(self, atoms: Atoms) -> float:
        a = atoms.copy()
        a.calc = self.calc
        return float(a.get_potential_energy()) / len(a)

    def __call__(self, final: Data, group_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        atoms_list = omgdata_to_atoms(final)
        gids = group_ids.detach().cpu().numpy()
        n = len(atoms_list)

        epa = np.full(n, np.nan, dtype=np.float64)
        invalid = np.zeros(n, dtype=bool)
        for i, atoms in enumerate(atoms_list):
            if is_invalid(atoms, self.vol_cutoff, self.dist_cutoff, self.polar_sine_cutoff):
                invalid[i] = True
                continue
            try:
                epa[i] = self._energy_per_atom(atoms)
            except Exception:
                invalid[i] = True

        # Penalty for invalid structures.
        epa_pen = np.where(invalid, self.invalid_penalty, epa)

        # Per-group clipping of the valid per-atom energies to mean +/- clip_std * std.
        clipped = epa_pen.copy()
        for g in np.unique(gids):
            mask = (gids == g) & (~invalid)
            if mask.sum() >= 2:
                vals = epa_pen[mask]
                mu, sd = vals.mean(), vals.std()
                if sd > 0:
                    clipped[mask] = np.clip(vals, mu - self.clip_std * sd, mu + self.clip_std * sd)

        rewards = -clipped
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        diagnostics = {
            "mean_energy_per_atom": float(np.nanmean(epa[~invalid])) if (~invalid).any() else float("nan"),
            "invalid_rate": float(invalid.mean()),
            "mean_reward": float(rewards.mean()),
        }
        return rewards_t, diagnostics
