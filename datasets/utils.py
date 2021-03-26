import copy
from collections import Sequence
import os

import mmcv
from mmcv.runner import obj_from_dict
import torch

import matplotlib.pyplot as plt
import numpy as np
from .concat_dataset import ConcatDataset
from .repeat_dataset import RepeatDataset
import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import random


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.
    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".
    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'range_keep_ratio':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            scale = np.random.rand(1) * (max(img_scale_long) / min(img_scale_long)-1) + 1
            img_scale = (int(min(img_scale_long) * scale), int(min(img_scale_short) * scale))
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


def get_dataset(data_cfg):
    data_cfg = vars(data_cfg)
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset']), data_cfg['times'])

    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset


def prepare_data(data_batch, devices: list = None, allocation: list = None, batch_size=None, is_cuda=False,
                 train_mode=True):
    if train_mode:
        with torch.no_grad():
            if batch_size is None:
                batch_size = 1
            if devices is None:
                devices = ['cuda:0'] if is_cuda else ['cpu']
            if allocation is None:
                allocation = [batch_size // len(devices)] * (len(devices) - 1)
                allocation.append(batch_size - sum(allocation))  # The rest might need more/less

            images_list = data_batch['img']
            bboxes_list = data_batch['bboxes']
            labels_list = data_batch['labels']
            masks_list = data_batch['masks']
            ids_list = data_batch['ids']
            images_meta_list = data_batch['img_meta']
            n_clip = images_list[0].size(0)

            split_images, split_bboxes, split_labels, split_masks, split_ids, split_images_meta = \
                [[None for alloc in allocation] for _ in range(6)]
            for idx, device, alloc in zip(range(len(devices)), devices, allocation):
                split_images[idx] = gradinator(torch.stack(images_list[alloc * idx:alloc * (idx + 1)], dim=0).to(device))
                for cur_idx in range(alloc):
                    bboxes_list[alloc * idx + cur_idx] = [gradinator(
                        bboxes_list[alloc * idx + cur_idx][i].to(device)) for i in range(n_clip)]
                    labels_list[alloc * idx + cur_idx] = [gradinator(
                        labels_list[alloc * idx + cur_idx][i].to(device)) for i in range(n_clip)]
                    masks_list[alloc * idx + cur_idx] = [gradinator(masks_list[alloc * idx + cur_idx][i].to(device))
                                                         for i in range(n_clip)]
                    ids_list[alloc * idx + cur_idx] = [gradinator(ids_list[alloc * idx + cur_idx][i].to(device))
                                                       for i in range(n_clip)]

                split_bboxes[idx] = bboxes_list[alloc * idx:alloc * (idx + 1)]
                split_labels[idx] = labels_list[alloc * idx:alloc * (idx + 1)]
                split_masks[idx] = masks_list[alloc * idx:alloc * (idx + 1)]
                split_ids[idx] = ids_list[alloc * idx:alloc * (idx + 1)]
                split_images_meta[idx] = images_meta_list[alloc * idx:alloc * (idx + 1)]

            return split_images, split_bboxes, split_labels, split_masks, split_ids, split_images_meta
    else:
        # [0] is downsample image [1, 3, 384, 640], [1] is original image [1, 3, 736, 1280]
        images = torch.stack([img[0].data for img in data_batch['img']], dim=0)
        images_meta = [img_meta[0].data for img_meta in data_batch['img_meta']]
        if 'ref_imgs' in data_batch.keys():
            ref_images = torch.stack([ref_img[0].data for ref_img in data_batch['ref_imgs']], dim=0)
            ref_images_meta = [ref_img_meta[0].data for ref_img_meta in data_batch['ref_img_metas']]
        else:
            ref_images = None
            ref_images_meta = None

        if is_cuda:
            images = gradinator(images.cuda())
            images_meta = images_meta
            if ref_images is not None:
                ref_images = gradinator(ref_images.cuda())
                ref_images_meta = ref_images_meta
        else:
            images = gradinator(images)
            images_meta = images_meta
            if ref_images is not None:
                ref_images = gradinator(ref_images)
                ref_images_meta = ref_images_meta

        return images, images_meta, ref_images, ref_images_meta


def gradinator(x):
    x.requires_grad = False
    return x


def enforce_size(img, targets, masks, num_crowds, new_w, new_h):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()

        if h == new_h and w == new_w:
            return img, targets, masks, num_crowds

        # Resize the image so that it fits within new_w, new_h
        w_prime = new_w
        h_prime = h * new_w / w

        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h

        w_prime = int(w_prime)
        h_prime = int(h_prime)

        # Do all the resizing
        img = F.interpolate(img.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        img.squeeze_(0)

        # Act like each object is a color channel
        masks = F.interpolate(masks.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        masks.squeeze_(0)

        # Scale bounding boxes (this will put them in the top left corner in the case of padding)
        targets[:, [0, 2]] *= (w_prime / new_w)
        targets[:, [1, 3]] *= (h_prime / new_h)

        # Finally, pad everything to be the new_w, new_h
        pad_dims = (0, new_w - w_prime, 0, new_h - h_prime)
        img = F.pad(img, pad_dims, mode='constant', value=0)
        masks = F.pad(masks, pad_dims, mode='constant', value=0)

        return img, targets, masks, num_crowds

