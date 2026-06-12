"""Portable path resolution for the OMatG-IRL reproduction.

All locations are derived from this file's position (``<repo>/omatg_irl/paths.py``),
so the repository works regardless of where it is checked out. The MP-20 LMDB path
is left relative because ``omg.datamodule.StructureDataset`` resolves it against the
installed ``omg`` package (``files("omg").joinpath(...)``), not the cwd.
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # <repo>/omatg_irl
REPO_ROOT = PKG_DIR.parent                          # <repo>
OMATG_DIR = REPO_ROOT / "OMatG"                     # upstream framework (cloned alongside)
EXPERIMENTS = REPO_ROOT / "experiments"             # run outputs

# Resolved within the installed omg package, independent of cwd.
TEST_LMDB = "data/mp_20/test.lmdb"
