"""Evaluation harness: match rate / RMSE / cRMSE + relative energy + invalid rate.

Generation for evaluation is deterministic (forward-Euler ODE with the reinforced
velocity field, no exploration noise, Nt steps), matching how the paper evaluates
the reduced-step policy. Structure metrics reuse OMatG's ``match_rmsds`` /
``metre_rmsds`` and ``ValidAtoms``; energies reuse the same MACE calculator as
the reward.
"""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Batch, Data

from omg.analysis import ValidAtoms, match_rmsds, metre_rmsds
from omg.datamodule.omg_data import OMGData
from omg.datamodule.structure_dataset import StructureDataset
from omg.si.corrector import PeriodicBoundaryConditionsCorrector

from .reward import is_invalid, omgdata_to_atoms
from .rollout import _set_state, step_mean
from .utils_pbc import com_project


@torch.no_grad()
def deterministic_generate(policy, reference, gamma, x0: Data, n_steps: int = 50,
                           t_min: float = 1e-3, t_max: float = 1.0 - 1e-3,
                           com_proj: bool = True, device: str = "cuda") -> Data:
    """Forward-Euler ODE with the policy velocity (positions) + frozen cell ODE; no noise."""
    corrector = PeriodicBoundaryConditionsCorrector(0.0, 1.0)
    times = torch.linspace(t_min, t_max, n_steps + 1, device=device)
    x0 = x0.to(device)
    template = x0.clone()
    batch = x0.batch
    B = int(x0.n_atoms.shape[0])
    pos = x0.pos.clone()
    cell = x0.cell.clone()
    zero = torch.zeros((), device=device)
    for i in range(n_steps):
        t = times[i]
        dt = times[i + 1] - times[i]
        x_int = _set_state(template, pos, cell)
        pol = policy.velocity_fields(x_int, t.repeat(B))
        ref = reference.velocity_fields(x_int, t.repeat(B))
        pos = step_mean(pol, pos, t, dt, zero, "velocity", gamma, batch, corrector, com_proj)
        cell = cell + ref["cell_b"] * dt
    return _set_state(template, pos, cell)


class EvalHarness:
    def __init__(self, test_lmdb: str, sampler, gamma, mace_calc, device: str = "cuda"):
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            self.ds = StructureDataset(file_path=test_lmdb, lazy_storage=True, niggli_reduce=True,
                                       floating_point_precision="32-true")
        self.sampler = sampler
        self.gamma = gamma
        self.calc = mace_calc
        self.device = device

    def _energy_per_atom(self, atoms: Atoms):
        try:
            if is_invalid(atoms):
                return None
            a = atoms.copy()
            a.calc = self.calc
            return float(a.get_potential_energy()) / len(a)
        except Exception:
            return None

    def evaluate(self, policy, reference, indices, n_steps: int = 50, seed: int = 0,
                 metre: bool = False) -> dict:
        """Generate one structure per test index and compute metrics vs the references."""
        ref_items = [OMGData(self.ds[i]) for i in indices]
        template = Batch.from_data_list([OMGData(self.ds[i]) for i in indices])
        torch.manual_seed(seed)
        x0 = self.sampler.sample_p_0(template).to(self.device)
        gen = deterministic_generate(policy, reference, self.gamma, x0, n_steps=n_steps,
                                     device=self.device)

        gen_atoms = omgdata_to_atoms(gen)
        ref_atoms = [Atoms(numbers=it.species.numpy(), scaled_positions=it.pos.numpy(),
                           cell=it.cell[0].numpy(), pbc=(1, 1, 1)) for it in ref_items]

        gen_valid = ValidAtoms.get_valid_atoms(gen_atoms, skip_validation=True, number_cpus=1,
                                               enable_progress_bar=False)
        ref_valid = ValidAtoms.get_valid_atoms(ref_atoms, skip_validation=True, number_cpus=1,
                                               enable_progress_bar=False)
        fn = metre_rmsds if metre else match_rmsds
        match_rate, mean_rmsd, _, _, _, _, corr_rmsd, _ = fn(
            gen_valid, ref_valid, ltol=0.3, stol=0.5, angle_tol=10.0, enable_progress_bar=False)

        gen_epa = np.array([e for e in (self._energy_per_atom(a) for a in gen_atoms) if e is not None])
        ref_epa = np.array([e for e in (self._energy_per_atom(a) for a in ref_atoms) if e is not None])
        invalid_rate = float(np.mean([is_invalid(a) for a in gen_atoms]))

        return {
            "match_rate": float(match_rate),
            "mean_rmsd": float(mean_rmsd),
            "corr_rmsd": float(corr_rmsd),
            "gen_energy_per_atom": float(gen_epa.mean()) if len(gen_epa) else float("nan"),
            "ref_energy_per_atom": float(ref_epa.mean()) if len(ref_epa) else float("nan"),
            "relative_energy_per_atom": (float(gen_epa.mean() - ref_epa.mean())
                                         if len(gen_epa) and len(ref_epa) else float("nan")),
            "invalid_energy_rate": invalid_rate,
        }
