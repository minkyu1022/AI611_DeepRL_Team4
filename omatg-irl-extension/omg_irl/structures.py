"""Convert generated OMatG ``OMGData`` batches into ASE ``Atoms`` / pymatgen ``Structure`` objects.

The generative process in OMatG produces a ``torch_geometric``-style batch whose atoms are concatenated and
delimited by ``ptr``. Fractional vs. Cartesian coordinates are flagged per structure by ``pos_is_fractional``.
"""
from typing import List
import numpy as np
import torch
from ase import Atoms


def omgdata_to_ase(data) -> List[Atoms]:
    """
    Convert an ``OMGData`` batch to a list of ASE ``Atoms`` (one per structure in the batch).

    :param data:
        A (batched) ``OMGData`` / ``torch_geometric.data.Data`` object with ``species``, ``pos``, ``cell``,
        ``n_atoms``, ``ptr`` and ``pos_is_fractional`` attributes.

    :return:
        List of ASE ``Atoms`` with periodic boundary conditions.
    """
    data = data.to("cpu")
    species = data.species.numpy()
    pos = data.pos.numpy().astype(np.float64)
    cell = data.cell.numpy().astype(np.float64)
    n_atoms = np.atleast_1d(data.n_atoms.numpy())
    n_structures = len(n_atoms)
    # ``ptr`` is only present on batched Data; reconstruct it from ``n_atoms`` for single items.
    if hasattr(data, "ptr") and data.ptr is not None:
        ptr = data.ptr.numpy()
    else:
        ptr = np.concatenate([[0], np.cumsum(n_atoms)])
    # ``cell`` is (3, 3) for a single structure and (n, 3, 3) when batched.
    if cell.ndim == 2:
        cell = cell[None, :, :]
    frac_flags = np.atleast_1d(np.asarray(data.pos_is_fractional))
    atoms_list = []
    for i in range(n_structures):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        if bool(frac_flags[i]):
            atoms = Atoms(numbers=species[lo:hi], scaled_positions=pos[lo:hi], cell=cell[i], pbc=True)
        else:
            atoms = Atoms(numbers=species[lo:hi], positions=pos[lo:hi], cell=cell[i], pbc=True)
        atoms_list.append(atoms)
    return atoms_list
