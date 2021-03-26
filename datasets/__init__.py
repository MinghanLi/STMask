from .config import *
from .utils import to_tensor, random_scale, show_ann, get_dataset

from .custom import CustomDataset
from .ytvos import YTVOSDataset
from .loader import GroupSampler, DistributedGroupSampler, build_dataloader
from .utils import to_tensor, random_scale, show_ann, get_dataset, prepare_data
from .concat_dataset import ConcatDataset
from .repeat_dataset import RepeatDataset
from .extra_aug import ExtraAugmentation

__all__ = [
    'cfg', 'MEANS', 'STD', 'set_cfg', 'set_dataset', 'detection_collate',
    'CustomDataset', 'YTVOSDataset',
    'GroupSampler', 'DistributedGroupSampler', 'build_dataloader',
    'to_tensor', 'random_scale', 'show_ann', 'get_dataset', 'prepare_data',
    'ConcatDataset', 'RepeatDataset', 'ExtraAugmentation'
]

import torch
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    batch_out = {}
    # batch_out['img'] = torch.cat([batch[i]['img'].data for i in range(batch_size)])
    # if 'ref_imgs' in batch[0].keys():
    #     batch_out['ref_imgs'] = torch.cat([batch[i]['ref_imgs'].data for i in range(batch_size)])

    for k in batch[0].keys():
        batch_out[k] = []

    for i in range(len(batch)):
        for k in batch_out.keys():
            if isinstance(batch[i][k], list):
                batch_out[k].append(batch[i][k])
            else:
                batch_out[k].append(batch[i][k].data)

    return batch_out

