"""Self-contained setup for the OMatG-DNG-IRL live demo.

The extension reuses the base OMatG framework (sampler, datamodule, checkpoint loading, the
bundled MP-20 LMDB data) and layers the S.U.N.-reward RL code on top of it. The base framework is
NOT vendored here (it is large and has its own license); instead it is cloned from upstream and the
extension modules under ``omg/irl/`` — which upstream does not ship — are overlaid into the clone.

``setup_omatg()`` performs the whole preparation and is idempotent:

  1. clone upstream OMatG (brings the ``omg`` package + the MP-20 LMDB data under ``omg/data/mp_20/``),
  2. overlay the vendored ``omg_irl/*.py`` extension modules into ``<clone>/omg/irl/``,
  3. seed the reward caches (elemental references + MP-20 novelty fingerprints) into
     ``<clone>/ckpts/hull/`` so the first reward call is instant and offline-safe,
  4. editable-install the clone and CHGNet (the reward's machine-learned potential).

It returns the path to the prepared OMatG checkout. The heavy scientific dependencies
(torch, torch_scatter, pymatgen, matminer, chgnet) are assumed to be the documented prerequisite
environment; only the omg package itself and chgnet are installed here.
"""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

EXT_DIR = Path(__file__).resolve().parent
OMATG_URL = "https://github.com/FERMat-ML/OMatG.git"


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


def setup_omatg(workdir=None, install: bool = True, verbose: bool = True):
    """Clone upstream OMatG, overlay the extension modules + caches, and (optionally) install.

    :param workdir: where to place the ``OMatG`` checkout (default: ``<this folder>/_omatg_checkout``).
    :param install: editable-install the checkout and CHGNet if not already importable.
    :returns: ``pathlib.Path`` to the prepared OMatG checkout.
    """
    def log(*a):
        if verbose:
            print(*a)

    workdir = Path(workdir) if workdir is not None else EXT_DIR / "_omatg_checkout"
    repo = workdir / "OMatG"

    # 1. clone upstream OMatG (skip if already present)
    if not (repo / "omg").is_dir():
        workdir.mkdir(parents=True, exist_ok=True)
        log("cloning upstream OMatG ...")
        subprocess.run(["git", "clone", "--depth", "1", OMATG_URL, str(repo)], check=True)
    else:
        log("OMatG checkout already present:", repo)

    # 2. overlay the vendored extension modules into omg/irl/
    src = EXT_DIR / "omg_irl"
    dst = repo / "omg" / "irl"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.py")):
        shutil.copy2(f, dst / f.name)
    log(f"overlaid {len(list(src.glob('*.py')))} extension modules into {dst}")

    # 3. seed the reward caches so the first reward call is instant and offline-safe
    cache_dst = repo / "ckpts" / "hull"
    cache_dst.mkdir(parents=True, exist_ok=True)
    for f in sorted((EXT_DIR / "cache").glob("*")):
        if f.is_file():
            shutil.copy2(f, cache_dst / f.name)
    log("seeded reward caches into", cache_dst)

    # 4. register the package + install CHGNet if needed.
    # omg is installed with --no-deps on purpose: the heavy scientific stack (torch, torch_scatter,
    # pymatgen, matminer) is a documented prerequisite and is often version-pinned (e.g. a CUDA-matched
    # torch + torch_scatter wheel pair). Letting pip resolve omg's dependencies here could upgrade torch
    # and break torch_scatter, so we only register the package and leave the prerequisites untouched.
    if install:
        if not _have("omg"):
            log("registering omg (editable, --no-deps) ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo), "--no-deps"],
                           check=True)
        if not _have("chgnet"):
            log("installing chgnet ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chgnet"], check=True)

    log("setup OK -- prepared OMatG checkout at", repo)
    return repo


if __name__ == "__main__":
    setup_omatg()
