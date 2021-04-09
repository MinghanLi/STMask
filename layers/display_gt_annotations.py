# -*- coding: utf-8 -*-
import torch
import numpy as np
import mmcv
import os.path as osp
import pycocotools.mask as mask_util
from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS
import matplotlib.pyplot as plt
from datasets import cfg, MEANS, STD
import cv2


def display_gt_ann(anno_file, img_prefix, save_path, mask_alpha=0.45):
    ytvosGt = YTVOS(anno_file)
    anns = ytvosGt.anns
    videos_info = ytvosGt.dataset['videos']
    video_id = anns[3394]['video_id']
    cat_id, bboxes, segm = [], [], []
    n_vid = 0
    for idx, ann_id in enumerate(anns):
        video_id_cur = anns[ann_id]['video_id']
        cat_id_cur = anns[ann_id]['category_id']
        bboxes_cur = anns[ann_id]['bboxes']
        segm_cur = anns[ann_id]['segmentations']
        if video_id_cur == video_id:
            cat_id.append(cat_id_cur)
            bboxes.append(bboxes_cur)
            segm.append(segm_cur)
        else:
            vid_info = videos_info[n_vid]
            h, w = vid_info['height'], vid_info['width']
            display_masks(n_vid, h, w, bboxes, segm, cat_id, vid_info, img_prefix, save_path, mask_alpha)
            n_vid += 1
            video_id = video_id_cur
            cat_id = [cat_id_cur]
            bboxes = [bboxes_cur]
            segm = [segm_cur]


def display_masks(n_vid, h, w, bboxes, segm, cat_id, vid_info, img_prefix, save_path, mask_alpha=0.45):
    for frame_id in range(len(bboxes[0])):
        print(n_vid, frame_id)
        img_numpy = mmcv.imread(osp.join(img_prefix, vid_info['file_names'][frame_id]))
        img_numpy = img_numpy[:, :, (2, 1, 0)] / 255.
        img_numpy = np.clip(img_numpy, 0, 1)
        img_gpu = torch.Tensor(img_numpy).cuda()
        img_numpy = img_gpu.cpu().numpy()

        # plot masks
        masks, colors = [], []
        for j in range(len(bboxes)):
            if segm[j][frame_id] is not None:
                # polygons to rle, rle to binary mask
                mask_rle = mask_util.frPyObjects(segm[j][frame_id], h, w)
                masks.append(mask_util.decode(mask_rle))
                colors.append(np.array(get_color(j)).reshape([1, 1, 3]))

        if len(masks) == 0:
            img_numpy = np.clip(img_numpy * 255, 0, 255).astype(np.int32)
        else:
            masks = np.stack(masks, axis=0)[:, :, :, None]
            colors = np.stack(colors, axis=0)
            masks_color = np.repeat(masks, 3, axis=3) * colors * mask_alpha
            inv_alph_masks = masks * (-mask_alpha) + 1

            masks_color_summand = masks_color[0]
            if len(colors) > 1:
                inv_alph_cumul = inv_alph_masks[:(len(colors) - 1)].cumprod(0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(0)
            img_numpy = img_numpy * inv_alph_masks.prod(axis=0) + masks_color_summand
            img_numpy = np.clip(img_numpy*255, 0, 255).astype(np.int32)
            # img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
            # img_numpy = cv2.cvtColor(np.float32(img_numpy), cv2.COLOR_RGB2GRAY)

            # plot bboxes and text
            for j in range(len(bboxes)):
                if bboxes[j][frame_id] is not None:
                    color = get_color(j)
                    x1, y1, w, h = bboxes[j][frame_id]
                    # x1, x2 = cx - w / 2, cx + w / 2
                    x2, y2 = x1 + w, y1 + h
                    y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                    _class = cfg.classes[cat_id[j] - 1]
                    text_str = '%s' % _class
                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 1
                    font_thickness = 1
                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_pt = (max(x1, 50), max(y1 - 3, 50))
                    text_color = [255, 255, 255]
                    cv2.rectangle(img_numpy, (max(int(x1), 5), max(int(y1), 5)), (int(x1 + text_w), int(y1 - text_h - 4)), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                cv2.LINE_AA)
        plt.imshow(img_numpy)
        plt.axis('off')
        plt.title(str([n_vid, frame_id]))
        plt.savefig(''.join([save_path, str([n_vid, frame_id]), '.png']))
        plt.clf()


# Quick and dirty lambda for selecting the color for a particular index
# Also keeps track of a per-gpu color cache for maximum speed
def get_color(j, norm=True):
    global color_cache
    color_idx = (j * 5) % len(cfg.COLORS)

    color = cfg.COLORS[color_idx]
    if norm:
        color = [color[0] / 255., color[1] / 255., color[2] / 255.]

    return color


if __name__ == '__main__':
    anno_file = ''.join(['/home/lmh/Downloads/VIS/code/', cfg.valid_sub_dataset.ann_file[3:]])
    img_prefix = cfg.valid_sub_dataset.img_prefix
    save_path = '/home/lmh/Downloads/VIS/code/yolact_JDT_VIS/results/gt_anno/'
    display_gt_ann(anno_file, img_prefix, save_path)