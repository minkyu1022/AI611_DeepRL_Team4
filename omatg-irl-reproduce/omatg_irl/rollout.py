"""Differentiable Euler / Euler-Maruyama rollout for OMatG-IRL.

The pretrained OMatG integrators run under ``torch.no_grad()`` via torchsde /
torchdiffeq and expose no per-step transition densities, so we implement a fresh
forward-Euler loop that reuses only the ``Model``, the correctors, and the
gamma schedule. The same ``step_mean`` is used for the (no-grad) rollout and the
(grad) log-prob recomputation, guaranteeing identical math.

Position MDP (fractional coords, torus), velocity-annealing removed (factor 0):
  * velocity-based (Eq. 13): mu = corr_COM(x_t + b * dt),               noise sigma(t)
  * score-based   (Eq. 7) : mu = corr_COM(x_t + [b - (sigma^2/2gamma) z] * dt), noise sigma(t)
Lattice is always integrated with the frozen reference ODE (Eq. 4), no noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from omg.si.corrector import PeriodicBoundaryConditionsCorrector

from .policy import Policy
from .schedules import SqrtNoiseSchedule
from .utils_pbc import com_project


@dataclass
class Trajectory:
    """Per-step tensors of one rollout batch (all detached; constants for the update)."""
    template: Data                 # holds species, batch, ptr, n_atoms, pos_is_fractional
    batch: torch.Tensor            # [N_total] per-atom structure index
    num_structures: int
    pos_states: torch.Tensor       # [Nt, N_total, 3] fractional state x_t at each step
    cell_states: torch.Tensor      # [Nt, B, 3, 3] cell at each step
    pos_next: torch.Tensor         # [Nt, N_total, 3] sampled x_{t+dt} (wrapped)
    mu_old: torch.Tensor           # [Nt, N_total, 3] policy-snapshot mean
    mu_ref: torch.Tensor           # [Nt, N_total, 3] frozen-reference mean
    z_ref: torch.Tensor | None     # [Nt, N_total, 3] reference denoiser (score mode only)
    sigmas: torch.Tensor           # [Nt] sigma(t)
    dts: torch.Tensor              # [Nt]
    ts: torch.Tensor               # [Nt]
    final: Data                    # final OMGData (x_1)


def _set_state(template: Data, pos: torch.Tensor, cell: torch.Tensor) -> Data:
    """Clone the template and overwrite pos/cell (species/topology unchanged)."""
    x = template.clone()
    x.pos = pos
    x.cell = cell
    return x


def step_mean(fields: dict[str, torch.Tensor], pos: torch.Tensor, t: torch.Tensor,
              dt: torch.Tensor, sigma_t: torch.Tensor, mode: str, gamma, batch: torch.Tensor,
              corrector: PeriodicBoundaryConditionsCorrector, com_project_drift: bool) -> torch.Tensor:
    """Compute the position transition mean mu = corr_COM(pos + drift * dt).

    :param fields: dict with ``pos_b`` (and ``pos_eta`` for score mode).
    :param pos: current fractional positions [N,3].
    :param t: scalar time.
    :param dt: scalar step.
    :param sigma_t: scalar sigma(t) (for the score drift correction).
    :param mode: "velocity" or "score".
    :param gamma: LatentGammaSqrt instance (pretrained gamma(t)).
    :param batch: per-atom structure index.
    :param corrector: periodic corrector (wrap mod 1).
    :param com_project_drift: subtract per-structure mean displacement.
    :return: transition mean mu [N,3].
    """
    b = fields["pos_b"]
    if mode == "score":
        g = gamma.gamma(t).clamp(min=1e-4)
        drift = b - (sigma_t * sigma_t / (2.0 * g)) * fields["pos_eta"]
    elif mode == "velocity":
        drift = b
    else:
        raise ValueError(mode)
    disp = drift * dt
    if com_project_drift:
        disp = com_project(disp, batch)
    return corrector.correct(pos + disp)


class Rollout:
    """Generates stochastic trajectories and recomputes policy means with gradients."""

    def __init__(self, policy: Policy, reference: Policy, noise: SqrtNoiseSchedule, gamma,
                 n_steps: int = 50, mode: str = "velocity", t_min: float = 1e-3,
                 t_max: float = 1.0 - 1e-3, com_project: bool = True, device: str = "cuda"):
        assert mode in ("velocity", "score")
        self.policy = policy
        self.reference = reference
        self.noise = noise
        self.gamma = gamma
        self.n_steps = n_steps
        self.mode = mode
        self.com_project = com_project
        self.device = device
        self.corrector = PeriodicBoundaryConditionsCorrector(0.0, 1.0)
        # Nt forward-Euler steps integrate over [t_min, t_max].
        self.times = torch.linspace(t_min, t_max, n_steps + 1, device=device)

    def _t_vec(self, t: torch.Tensor, b: int) -> torch.Tensor:
        return t.repeat(b)

    @torch.no_grad()
    def generate(self, x0: Data) -> Trajectory:
        """Roll out the perturbed dynamics from a base sample ``x0`` (one big batch)."""
        x0 = x0.to(self.device)
        template = x0.clone()
        batch = x0.batch
        B = int(x0.n_atoms.shape[0])
        pos = x0.pos.clone()
        cell = x0.cell.clone()

        pos_states, cell_states, pos_next_l, mu_old_l, mu_ref_l, z_ref_l = [], [], [], [], [], []
        sigmas, dts, ts = [], [], []

        for i in range(self.n_steps):
            t = self.times[i]
            dt = self.times[i + 1] - self.times[i]
            sigma_t = self.noise(t).to(self.device)
            t_vec = self._t_vec(t, B)

            x_int = _set_state(template, pos, cell)
            pol = self.policy.velocity_fields(x_int, t_vec)
            ref = self.reference.velocity_fields(x_int, t_vec)

            mu_old = step_mean(pol, pos, t, dt, sigma_t, self.mode, self.gamma, batch,
                               self.corrector, self.com_project)
            mu_ref = step_mean(ref, pos, t, dt, sigma_t, self.mode, self.gamma, batch,
                               self.corrector, self.com_project)

            xi = torch.randn_like(pos)
            if self.com_project:
                xi = com_project(xi, batch)
            pos_new = self.corrector.correct(mu_old + sigma_t * torch.sqrt(dt) * xi)

            # Lattice: frozen reference ODE, no noise, no COM, velocity-annealing 0.
            cell_new = cell + ref["cell_b"] * dt

            pos_states.append(pos)
            cell_states.append(cell)
            pos_next_l.append(pos_new)
            mu_old_l.append(mu_old)
            mu_ref_l.append(mu_ref)
            if self.mode == "score":
                z_ref_l.append(ref["pos_eta"])
            sigmas.append(sigma_t)
            dts.append(dt)
            ts.append(t)

            pos = pos_new
            cell = cell_new

        final = _set_state(template, pos, cell)
        return Trajectory(
            template=template, batch=batch, num_structures=B,
            pos_states=torch.stack(pos_states), cell_states=torch.stack(cell_states),
            pos_next=torch.stack(pos_next_l), mu_old=torch.stack(mu_old_l),
            mu_ref=torch.stack(mu_ref_l),
            z_ref=torch.stack(z_ref_l) if self.mode == "score" else None,
            sigmas=torch.stack(sigmas), dts=torch.stack(dts), ts=torch.stack(ts),
            final=final,
        )

    def recompute_step(self, traj: Trajectory, i: int):
        """Recompute the live-policy mean (and denoiser) for step ``i`` WITH gradients.

        :return: (mu_theta [N,3], z_theta [N,3] or None).
        """
        t = traj.ts[i]
        dt = traj.dts[i]
        sigma_t = traj.sigmas[i]
        B = traj.num_structures
        x_int = _set_state(traj.template, traj.pos_states[i], traj.cell_states[i])
        pol = self.policy.velocity_fields(x_int, self._t_vec(t, B))
        mu_theta = step_mean(pol, traj.pos_states[i], t, dt, sigma_t, self.mode, self.gamma,
                             traj.batch, self.corrector, self.com_project)
        z_theta = pol["pos_eta"] if self.mode == "score" else None
        return mu_theta, z_theta
