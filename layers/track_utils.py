# -*- coding: utf-8 -*-
import torch


def split_bbox(bbox, idx_bbox_ori, nums_bbox_per_layer):
    num_layers = len(nums_bbox_per_layer)
    num_bboxes = len(idx_bbox_ori)
    for i in range(1, num_layers):
        nums_bbox_per_layer[i] = nums_bbox_per_layer[i-1] + nums_bbox_per_layer[i]

    split_bboxes = [[] for _ in range(num_layers)]
    split_bboxes_idx = [[] for _ in range(num_layers)]
    for i in range(num_bboxes):
        for j in range(num_layers):
            if idx_bbox_ori[i] < nums_bbox_per_layer[j]:
                split_bboxes[j].append(bbox[i].unsqueeze(0))
                split_bboxes_idx[j].append(i)
                break

    for j in range(num_layers):
        if len(split_bboxes[j]) > 0:
            split_bboxes[j] = torch.cat(split_bboxes[j])
        if j > 0:
            split_bboxes_idx[0] += split_bboxes_idx[j]

    return split_bboxes, split_bboxes_idx[0]


def mask_iou(mask1, mask2):
    intersection = torch.sum(mask1 * mask2, dim=(0, 1))
    area1 = torch.sum(mask1, dim=(0, 1))
    area2 = torch.sum(mask2, dim=(0, 1))
    union = (area1 + area2) - intersection
    ret = intersection / union
    return ret












