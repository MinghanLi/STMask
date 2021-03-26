from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader
import numpy as np
from .sampler import GroupSampler, DistributedGroupSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     batch_size,
                     num_workers,
                     dist=True,
                     shuffle_sampler=True,
                     **kwargs):
    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedGroupSampler(dataset, batch_size, world_size,
                                          rank)

    else:
        if not kwargs.get('shuffle', True):
            sampler = None
        else:
            sampler = GroupSampler(dataset, batch_size, shuffle=shuffle_sampler)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
        **kwargs)

    return data_loader
