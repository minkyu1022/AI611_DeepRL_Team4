"""Load the pretrained OMatG model and expose policy + frozen reference.

The public Trig-SDE-Gamma checkpoint stores only ``model.*`` weights and no
hyper-parameters, so we instantiate the full ``OMGLightning`` module from the
shipped ``train.yaml`` (via the LightningCLI in ``run=False`` mode) and load the
state dict strictly. This reuses the exact architecture/sampler/interpolant
configuration that produced the checkpoint and avoids silent weight corruption
from hand-built kwargs.
"""
from __future__ import annotations

import copy
import io
import os
from contextlib import redirect_stderr, redirect_stdout

import torch

from omg.datamodule import OMGDataModule
from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer


def load_lightning_module(config_yaml: str, ckpt_path: str, device: str = "cpu") -> OMGLightning:
    """Instantiate ``OMGLightning`` from the YAML and load the checkpoint weights.

    :param config_yaml: Path to the shipped ``train.yaml``.
    :param ckpt_path: Path to ``checkpoint.ckpt``.
    :param device: Device to move the module to.
    :return: The loaded ``OMGLightning`` (with ``.model``, ``.si``, ``.sampler``).
    """
    # The CLI eagerly loads the LMDB datasets and prints large progress bars;
    # silence stdout/stderr during construction. Must run from a directory where
    # the YAML's relative data paths resolve (the OMatG package dir).
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        cli = OMGCLI(OMGLightning, OMGDataModule, trainer_class=OMGTrainer,
                     args=["--config", config_yaml], run=False)
        lm = cli.model
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = lm.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    lm.to(device)
    lm.eval()
    return lm


def split_policy_reference(lm: OMGLightning):
    """Return (policy_model, reference_model) where the reference is a frozen deepcopy.

    Both are ``omg.model.model.Model`` instances. The policy is trainable; the
    reference holds the frozen pretrained weights used for the KL-regularization
    target, the denoiser-distillation target, and the lattice ODE.
    """
    policy_model = lm.model
    reference_model = copy.deepcopy(lm.model)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad_(False)
    return policy_model, reference_model


def get_pos_interpolant(lm: OMGLightning):
    """Return the SingleStochasticInterpolant for atomic positions (index of 'pos')."""
    names = [d.name for d in lm.si._data_fields]
    return lm.si._stochastic_interpolants[names.index("pos")]
