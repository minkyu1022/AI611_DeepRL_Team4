"""Periodic-boundary helpers for the fractional-coordinate position MDP.

All atomic-position quantities live on the torus [0, 1)^3 (the OMatG model uses
``frac_coords = x.pos`` and predicts the velocity in fractional space). Squared
distances and means must therefore respect the minimum-image convention and the
per-structure centre-of-mass (COM) constraint.
"""
from __future__ import annotations

import torch
from torch_scatter import scatter_mean


def minimum_image(delta: torch.Tensor) -> torch.Tensor:
    """Wrap fractional displacements into [-0.5, 0.5).

    This is the minimum-image difference on the unit torus and is invariant to
    wrapping either endpoint by an integer lattice vector, so storing the wrapped
    next-state is sufficient for later log-prob recomputation.

    :param delta: Raw fractional differences, any shape.
    :return: Minimum-image differences in [-0.5, 0.5).
    """
    return delta - torch.round(delta)


def com_project(vec: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Remove the per-structure mean (centre-of-mass component) of a per-atom vector.

    The CSPNet model is translation invariant in fractional space and cannot
    predict COM motion (``correct_center_of_mass_motion=True`` in the pretrained
    model). Projecting both the drift displacement and the injected noise keeps
    every structure's COM fixed, which is required for the per-step transition
    density to be consistent between rollout and log-prob recomputation.

    :param vec: Per-atom tensor of shape [N_total, 3].
    :param batch: Per-atom structure index of shape [N_total].
    :return: ``vec`` with the per-structure mean subtracted.
    """
    means = scatter_mean(vec, batch, dim=0)  # [B, 3]
    return vec - means[batch]


def per_structure_sq_dist(a: torch.Tensor, b: torch.Tensor, batch: torch.Tensor,
                          num_structures: int) -> torch.Tensor:
    """Minimum-image squared distance ||a - b||^2 summed within each structure.

    :param a: Per-atom fractional coordinates [N_total, 3].
    :param b: Per-atom fractional coordinates [N_total, 3].
    :param batch: Per-atom structure index [N_total].
    :param num_structures: Number of structures B.
    :return: Per-structure summed squared minimum-image distance, shape [B].
    """
    d = minimum_image(a - b)  # [N_total, 3]
    per_atom = (d * d).sum(dim=-1)  # [N_total]
    out = torch.zeros(num_structures, device=a.device, dtype=a.dtype)
    out.scatter_add_(0, batch, per_atom)
    return out
