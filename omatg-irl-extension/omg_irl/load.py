"""Utilities to load a pretrained OMatG DNG model from a HuggingFace-style checkpoint directory.

A checkpoint directory contains ``checkpoint.ckpt`` (a PyTorch-Lightning checkpoint) and ``train.yaml``
(the LightningCLI configuration used for training). We instantiate the ``OMGLightning`` module from the
``model`` section of ``train.yaml`` using jsonargparse (the same machinery the OMatG CLI relies on), then
load the weights from the checkpoint.
"""
from pathlib import Path
from typing import Union
import yaml
import torch
from jsonargparse import ArgumentParser
from omg.omg_lightning import OMGLightning


def load_pretrained(ckpt_dir: Union[str, Path], map_location: str = "cpu") -> OMGLightning:
    """
    Load a pretrained ``OMGLightning`` module from a checkpoint directory.

    :param ckpt_dir:
        Directory containing ``checkpoint.ckpt`` and ``train.yaml``.
    :param map_location:
        Device onto which the checkpoint weights are loaded.

    :return:
        The pretrained ``OMGLightning`` module in eval mode.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "checkpoint.ckpt"
    yaml_path = ckpt_dir / "train.yaml"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    cfg = yaml.safe_load(yaml_path.read_text())
    model_cfg = cfg["model"]

    # Instantiate the OMGLightning module (with nested si/sampler/model objects) via jsonargparse.
    parser = ArgumentParser()
    parser.add_class_arguments(OMGLightning, "model", fail_untyped=False)
    parsed = parser.parse_object({"model": model_cfg})
    init = parser.instantiate_classes(parsed)
    module: OMGLightning = init.model

    # Load the weights. Lightning checkpoints store parameters under "state_dict".
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    # The only acceptable mismatches are buffers that are re-created on init; warn loudly otherwise.
    real_missing = [k for k in missing if not k.endswith("_corrector")]
    if real_missing:
        print(f"[load_pretrained] WARNING missing keys: {real_missing[:10]}"
              f"{' ...' if len(real_missing) > 10 else ''} ({len(real_missing)} total)")
    if unexpected:
        print(f"[load_pretrained] WARNING unexpected keys: {unexpected[:10]}"
              f"{' ...' if len(unexpected) > 10 else ''} ({len(unexpected)} total)")

    module.eval()
    return module
