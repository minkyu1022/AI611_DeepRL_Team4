"""Build GRPO groups from MP-20 test compositions.

A GRPO "group" is ``group_size`` independent base samples that share one fixed
composition (CSP: species and atom count are conditioning, only pos/cell are
generated). We build one large batch of ``n_groups * group_size`` structures so
the whole rollout runs in a single batched trajectory; ``group_ids`` records the
group membership used for the group-relative advantage.
"""
from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout
import io

import torch
from torch_geometric.data import Batch

from omg.datamodule.omg_data import OMGData
from omg.datamodule.structure_dataset import StructureDataset


def _load_test_dataset(test_lmdb: str) -> StructureDataset:
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        ds = StructureDataset(file_path=test_lmdb, lazy_storage=True, niggli_reduce=True,
                              floating_point_precision="32-true")
    return ds


def build_groups(test_lmdb: str, sampler, n_groups: int = 16, group_size: int = 32,
                 seed: int = 0, device: str = "cuda"):
    """Pick ``n_groups`` fixed test compositions and build a base-distribution batch.

    :param test_lmdb: Path to MP-20 ``test.lmdb``.
    :param sampler: An ``omg.sampler.IndependentSampler`` (from the loaded module).
    :param n_groups: Number of distinct compositions (groups).
    :param group_size: Number of rollouts per group (G).
    :param seed: Seed for composition selection and base sampling.
    :param device: Target device.
    :return: (x0 OMGData of n_groups*group_size structures, group_ids [n_groups*group_size],
              ref_indices list[int]).
    """
    ds = _load_test_dataset(test_lmdb)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    ref_indices = perm[:n_groups]

    data_list = []
    group_ids = []
    for gi, idx in enumerate(ref_indices):
        item = ds[idx]
        for _ in range(group_size):
            data_list.append(OMGData(item))
            group_ids.append(gi)
    template = Batch.from_data_list(data_list)

    torch.manual_seed(seed)
    x0 = sampler.sample_p_0(template).to(device)
    group_ids = torch.tensor(group_ids, dtype=torch.long, device=device)
    return x0, group_ids, ref_indices
