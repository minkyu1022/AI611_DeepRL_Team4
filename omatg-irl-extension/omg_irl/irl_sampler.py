"""Inference-time RL rollout for OMatG-DNG-IRL (OMatG-IRL, arXiv:2602.00424, extended to DNG).

We re-implement the OMatG integration loop as an explicit Markov decision process so that the per-step
transition log-probabilities are available for policy-gradient RL. Three modalities are integrated jointly:

  * fractional positions -- score-based stochastic policy (Eq. 7): an Euler-Maruyama step whose Gaussian
    transition  N(x + f dt, sigma^2 dt)  uses the score-corrected drift  f = b - sigma^2/(2 gamma) z  and a
    tunable square-root noise schedule  sigma(t) = a sqrt((1-t)/t)  (Flow-GRPO-style). Reinforced.
  * atomic species -- discrete flow matching (Campbell et al. 2024): per-atom categorical unmasking. The
    log-probability that contributes policy gradient is  log softmax(logits)[chosen]  for atoms that unmask
    on that step (the unmask/remask Bernoulli factors are policy-independent and cancel in the ratio).
    Reinforced.
  * lattice vectors -- frozen deterministic ODE (Euler). Not reinforced (no denoiser available).

A ``Trajectory`` stores, per step, the full input state and the realized action, so that log-probabilities and
KL terms can be recomputed under an updated policy during PPO epochs.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import math
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch_scatter import scatter_add

from omg.globals import SMALL_TIME, BIG_TIME, MAX_ATOM_NUM

_LOG2PI = math.log(2.0 * math.pi)


def sqrt_noise(t: float, scale: float) -> float:
    """Square-root noise schedule sigma(t) = scale * sqrt((1 - t) / t), clamped away from the singular t->0."""
    t = min(max(t, 1e-4), 1.0 - 1e-6)
    return scale * math.sqrt((1.0 - t) / t)


@dataclass
class StepRecord:
    """Everything needed to recompute the per-step transition log-prob under a (possibly updated) policy."""
    species_in: torch.Tensor      # (N,) long, species state before the step (0 = masked)
    pos_in: torch.Tensor          # (N, 3) fractional positions before the step
    cell_in: torch.Tensor         # (B, 3, 3) lattice before the step
    t: float
    dt: float
    sigma: float                  # noise scale sigma(t) for the position policy
    pos_out: torch.Tensor         # (N, 3) realized next fractional positions (the position action)
    unmask_mask: torch.Tensor     # (N,) bool, atoms that unmasked on this step
    unmask_target: torch.Tensor   # (N,) long, species the unmasked atoms moved to (shifted species, 1..100)


@dataclass
class Trajectory:
    batch: torch.Tensor           # (N,) structure index per atom
    n_atoms: torch.Tensor         # (B,)
    ptr: torch.Tensor             # (B + 1,)
    steps: List[StepRecord] = field(default_factory=list)
    old_logp: Optional[torch.Tensor] = None   # (B,) summed log-prob under the sampling policy
    old_logp_steps: List[torch.Tensor] = field(default_factory=list)  # per-step (B,) log-prob under sampling policy

    @property
    def n_structures(self) -> int:
        return int(self.n_atoms.shape[0])


def _gaussian_logp_per_structure(disp: torch.Tensor, sigma: float, dt: float, batch: torch.Tensor,
                                 n_structures: int) -> torch.Tensor:
    """Sum the isotropic-Gaussian log density of displacement ``disp`` (N,3) over atoms within each structure."""
    var = sigma * sigma * dt
    # log N(disp; 0, var I_3) per atom = -0.5 * |disp|^2/var - 3/2 log(2 pi var)
    per_atom = -0.5 * (disp * disp).sum(dim=-1) / var - 1.5 * (_LOG2PI + math.log(var))
    return scatter_add(per_atom, batch, dim=0, dim_size=n_structures)


def _minimal_image(disp: torch.Tensor) -> torch.Tensor:
    """Wrap fractional displacements to (-0.5, 0.5] (tangent space of the torus)."""
    return disp - torch.round(disp)


class IRLSampler:
    """
    Roll out the OMatG DNG generative process as an MDP and expose per-step log-probabilities.

    :param module:
        The ``OMGLightning`` module providing ``model`` (policy network) and ``si`` (interpolant params).
    :param n_steps:
        Number of integration steps N_t (reduced for RL, e.g. 50).
    :param noise_scale:
        Scale ``a`` of the square-root position-noise schedule used during exploration rollouts.
    """

    def __init__(self, module, n_steps: int = 50, noise_scale: float = 0.1) -> None:
        self.module = module
        self.n_steps = n_steps
        self.noise_scale = noise_scale
        pos_si = module.si.get_stochastic_interpolant("pos")
        self._pos_corrector = pos_si.get_corrector()
        self._gamma = pos_si._gamma
        species_si = module.si.get_stochastic_interpolant("species")
        self._species_noise = species_si._noise
        self._mask_index = 0

    def _model_forward(self, model, species, pos, cell, n_atoms, batch, ptr, t_scalar):
        """Run the policy network at state (species, pos, cell, t); return (pos_b, pos_eta, species_logits, cell_b)."""
        from omg.datamodule import OMGData
        from torch_geometric.data import Data
        x = Data(species=species, pos=pos, cell=cell, n_atoms=n_atoms, batch=batch, ptr=ptr)
        x.pos_is_fractional = torch.ones(self_n_structures(n_atoms), dtype=torch.bool, device=pos.device)
        t = torch.full((self_n_structures(n_atoms),), t_scalar, device=pos.device)
        out = model(x, t)
        return out["pos_b"], out["pos_eta"], out["species_b"], out["cell_b"]

    @torch.no_grad()
    def rollout(self, x0) -> "tuple":
        """
        Integrate a batch of base samples to generated structures, recording the trajectory and sampling log-probs.

        :param x0:
            Batched ``OMGData`` from the base distribution (masked species, uniform positions, informed cell).

        :return:
            (generated OMGData, Trajectory).
        """
        module = self.module
        model = module.model
        device = x0.pos.device
        n_atoms = x0.n_atoms
        batch = x0.batch
        ptr = x0.ptr
        B = int(n_atoms.shape[0])

        species = x0.species.clone().long()
        pos = self._pos_corrector.correct(x0.pos.clone())
        cell = x0.cell.clone()

        times = torch.linspace(SMALL_TIME, BIG_TIME, self.n_steps, device=device)
        traj = Trajectory(batch=batch, n_atoms=n_atoms, ptr=ptr)
        old_logp = torch.zeros(B, device=device)

        for k in range(1, len(times)):
            t = float(times[k - 1])
            dt = float(times[k] - times[k - 1])
            sigma = sqrt_noise(t, self.noise_scale)

            pos_b, pos_eta, species_logits, cell_b = self._model_forward(
                model, species, pos, cell, n_atoms, batch, ptr, t)

            # ---- position score-based SDE step (Eq. 7) -------------------------------------------
            gamma_t = float(self._gamma.gamma(torch.tensor(t)))
            drift = pos_b - (sigma * sigma / (2.0 * gamma_t)) * pos_eta
            mean = pos + drift * dt
            std = sigma * math.sqrt(dt)
            noise = torch.randn_like(pos)
            pos_new = self._pos_corrector.correct(mean + std * noise)
            disp = _minimal_image(pos_new - mean)
            pos_logp = _gaussian_logp_per_structure(disp, sigma, dt, batch, B)

            # ---- species discrete-flow-matching step --------------------------------------------
            species_in = species.clone()
            probs = F.softmax(species_logits, dim=-1)
            x1 = Categorical(probs).sample() + 1  # predicted clean species (shifted to 1..100)
            is_masked = species_in == self._mask_index
            final = abs(t + dt - BIG_TIME) < 5e-3
            if final:
                will_unmask = is_masked
            else:
                p_unmask = dt * (1.0 + self._species_noise * t) / (1.0 - t)
                will_unmask = (torch.rand_like(pos[:, 0]) < p_unmask) & is_masked
            will_mask = (torch.rand_like(pos[:, 0]) < dt * self._species_noise) & (~is_masked)
            species_new = species_in.clone()
            species_new[will_unmask] = x1[will_unmask]
            if not final:
                species_new[will_mask] = self._mask_index
            # log-prob of the chosen species for atoms that unmasked this step.
            log_probs = F.log_softmax(species_logits, dim=-1)
            chosen_logp = log_probs.gather(1, (x1 - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            sp_logp_atom = torch.where(will_unmask, chosen_logp, torch.zeros_like(chosen_logp))
            species_logp = scatter_add(sp_logp_atom, batch, dim=0, dim_size=B)

            # ---- lattice frozen ODE Euler step (not reinforced) ---------------------------------
            cell_new = cell + cell_b * dt

            traj.steps.append(StepRecord(
                species_in=species_in, pos_in=pos.clone(), cell_in=cell.clone(), t=t, dt=dt, sigma=sigma,
                pos_out=pos_new.clone(), unmask_mask=will_unmask.clone(),
                unmask_target=x1.clone()))
            step_logp = pos_logp + species_logp
            traj.old_logp_steps.append(step_logp)
            old_logp = old_logp + step_logp

            species, pos, cell = species_new, pos_new, cell_new

        traj.old_logp = old_logp
        gen = x0.clone()
        gen.species = species
        gen.pos = pos
        gen.cell = cell
        return gen, traj

    def logp_and_kl(self, model, traj: Trajectory, ref_model=None):
        """
        Recompute the summed per-structure log-probability under ``model`` (with gradient) and, if a reference
        model is given, the summed per-structure KL divergence to the reference policy.

        :return:
            (logp[B], kl[B] or None, n_pos_terms[B], n_species_terms[B]) where the term counts enable per-atom
            normalization of the policy loss and KL.
        """
        B = traj.n_structures
        batch = traj.batch
        device = traj.n_atoms.device
        logp = torch.zeros(B, device=device)
        kl = torch.zeros(B, device=device) if ref_model is not None else None
        n_pos = torch.zeros(B, device=device)
        n_sp = torch.zeros(B, device=device)

        for rec in traj.steps:
            pos_b, pos_eta, species_logits, _ = self._model_forward(
                model, rec.species_in, rec.pos_in, rec.cell_in, traj.n_atoms, batch, traj.ptr, rec.t)
            gamma_t = float(self._gamma.gamma(torch.tensor(rec.t)))
            sigma, dt = rec.sigma, rec.dt
            mean = rec.pos_in + (pos_b - (sigma * sigma / (2.0 * gamma_t)) * pos_eta) * dt
            disp = _minimal_image(rec.pos_out - mean)
            pos_logp = _gaussian_logp_per_structure(disp, sigma, dt, batch, B)

            log_probs = F.log_softmax(species_logits, dim=-1)
            chosen_logp = log_probs.gather(1, (rec.unmask_target - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            sp_logp_atom = torch.where(rec.unmask_mask, chosen_logp, torch.zeros_like(chosen_logp))
            species_logp = scatter_add(sp_logp_atom, batch, dim=0, dim_size=B)

            logp = logp + pos_logp + species_logp
            n_pos = n_pos + scatter_add(torch.ones_like(disp[:, 0]), batch, dim=0, dim_size=B)
            n_sp = n_sp + scatter_add(rec.unmask_mask.float(), batch, dim=0, dim_size=B)

            if ref_model is not None:
                with torch.no_grad():
                    rpos_b, rpos_eta, rspecies_logits, _ = self._model_forward(
                        ref_model, rec.species_in, rec.pos_in, rec.cell_in, traj.n_atoms, batch, traj.ptr, rec.t)
                    rmean = rec.pos_in + (rpos_b - (sigma * sigma / (2.0 * gamma_t)) * rpos_eta) * dt
                # KL between two isotropic Gaussians with equal variance sigma^2 dt: |mean - rmean|^2 / (2 var).
                var = sigma * sigma * dt
                pos_kl_atom = 0.5 * ((mean - rmean) * (mean - rmean)).sum(dim=-1) / var
                # Categorical KL over the per-atom species distribution, for atoms eligible to unmask (masked).
                is_masked = (rec.species_in == self._mask_index).float()
                rlog_probs = F.log_softmax(rspecies_logits, dim=-1)
                sp_kl_atom = (log_probs.exp() * (log_probs - rlog_probs)).sum(dim=-1) * is_masked
                kl = kl + scatter_add(pos_kl_atom + sp_kl_atom, batch, dim=0, dim_size=B)

        return logp, kl, n_pos, n_sp

    def grpo_loss(self, model, traj: Trajectory, advantages: torch.Tensor, ref_model=None,
                  clip_eps: float = 0.2, kl_coef: float = 0.0):
        """
        GRPO/PPO-clipped policy-gradient loss (OMatG-IRL Eqs. 10-11) for one group of trajectories.

        A per-step probability ratio is formed from the geometric-mean per-atom log-probability (size
        normalization, Appendix A), clipped against the group advantage, and averaged over steps and the group.
        An optional KL term regularizes toward the frozen reference policy.

        :param advantages:
            (B,) group-relative advantages (one per trajectory).

        :return:
            (loss scalar tensor, metrics dict).
        """
        B = traj.n_structures
        batch = traj.batch
        n_atoms = traj.n_atoms.float()
        device = traj.n_atoms.device
        n_steps = max(len(traj.steps), 1)
        policy_terms = torch.zeros(B, device=device)
        kl_terms = torch.zeros(B, device=device)

        for rec, old_step_logp in zip(traj.steps, traj.old_logp_steps):
            pos_b, pos_eta, species_logits, _ = self._model_forward(
                model, rec.species_in, rec.pos_in, rec.cell_in, traj.n_atoms, batch, traj.ptr, rec.t)
            gamma_t = float(self._gamma.gamma(torch.tensor(rec.t)))
            sigma, dt = rec.sigma, rec.dt
            mean = rec.pos_in + (pos_b - (sigma * sigma / (2.0 * gamma_t)) * pos_eta) * dt
            disp = _minimal_image(rec.pos_out - mean)
            pos_logp = _gaussian_logp_per_structure(disp, sigma, dt, batch, B)
            log_probs = F.log_softmax(species_logits, dim=-1)
            chosen = log_probs.gather(1, (rec.unmask_target - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            sp_logp = scatter_add(torch.where(rec.unmask_mask, chosen, torch.zeros_like(chosen)),
                                  batch, dim=0, dim_size=B)
            new_step_logp = pos_logp + sp_logp

            # Geometric-mean per-atom probability ratio keeps the ratio near 1 and removes size bias.
            log_ratio = (new_step_logp - old_step_logp) / n_atoms
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_terms = policy_terms + torch.min(unclipped, clipped)

            if ref_model is not None and kl_coef > 0.0:
                with torch.no_grad():
                    rpos_b, rpos_eta, rspecies_logits, _ = self._model_forward(
                        ref_model, rec.species_in, rec.pos_in, rec.cell_in, traj.n_atoms, batch, traj.ptr, rec.t)
                    rmean = rec.pos_in + (rpos_b - (sigma * sigma / (2.0 * gamma_t)) * rpos_eta) * dt
                var = sigma * sigma * dt
                pos_kl_atom = 0.5 * ((mean - rmean) * (mean - rmean)).sum(dim=-1) / var
                is_masked = (rec.species_in == self._mask_index).float()
                rlog_probs = F.log_softmax(rspecies_logits, dim=-1)
                sp_kl_atom = (log_probs.exp() * (log_probs - rlog_probs)).sum(dim=-1) * is_masked
                kl_terms = kl_terms + scatter_add(pos_kl_atom + sp_kl_atom, batch, dim=0, dim_size=B) / n_atoms

        policy_obj = (policy_terms / n_steps).mean()
        kl_obj = (kl_terms / n_steps).mean()
        loss = -(policy_obj) + kl_coef * kl_obj
        return loss, {"policy_obj": float(policy_obj.detach()), "kl": float(kl_obj.detach())}


def self_n_structures(n_atoms: torch.Tensor) -> int:
    return int(n_atoms.shape[0])
