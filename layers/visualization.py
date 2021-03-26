import torch
import numpy as np
import cv2
from datasets import cfg, mask_type, MEANS, STD
import random
from math import sqrt
import matplotlib.pyplot as plt
import mmcv
import torch.nn.functional as F


def display_box_shift(box, box_shift, conf=None, img_gpu=None, img_meta=None, idx=0):
    if img_meta is not None:
        path = ''.join(['results/results_1024_2/box_shift1/', str(img_meta[0]['video_id']), '_',
                        str(img_meta[0]['frame_id']), '_', str(idx), '.png'])
        path_ori = ''.join(['results/results_1024_2/box_shift1/', str(img_meta[0]['video_id']), '_',
                           str(img_meta[0]['frame_id']), '_', str(idx), '_ori.png'])
    else:
        path = 'results/results_1024_2/box_shift/0.png'
        path_ori = 'results/results_1024_2/box_shift/0_ori.png'

    h, w = 384, 640
    # Make empty black image
    if img_gpu is None:
        image = np.ones((h, w, 3), np.uint8) * 255
    else:
        img_numpy = img_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_numpy = img_numpy[:, :, (2, 1, 0)]  # To BRG
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
        # img_numpy = img_numpy[:, :, (2, 1, 0)]  # To RGB
        img_numpy = np.clip(img_numpy, 0, 1)
        img_gpu = torch.Tensor(img_numpy).cuda()
        image = (img_gpu * 255).byte().cpu().numpy()

    if conf is not None:
        scores, classes = conf[:, 1:].max(dim=1)
    display_labels = False
    # cv2.imwrite(path_ori, image)

    # Create a named colour
    red = [0, 0, 255]
    black = [0, 0, 0]
    # plot pred bbox
    for i in range(box.size(0)):
        cv2.rectangle(image, (box[i, 0]*w, box[i, 1]*h), (box[i, 2]*w, box[i, 3]*h), black, 4)

    for i in range(box.size(0)):
        cv2.rectangle(image, (box_shift[i, 0] * w, box_shift[i, 1] * h),
                             (box_shift[i, 2] * w, box_shift[i, 3] * h), red, 4)

        if conf is not None and display_labels:
            text_str = '%s: %.2f' % (classes[i].item()+1, scores[i])

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            font_thickness = 1
            text_pt = (box_shift[i, 0]*w, box_shift[i, 1]*h - 3)
            text_color = [255, 255, 255]
            cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.imwrite(path, image)


def display_feature_align_dcn(detection, offset, loc_data, img_gpu=None, img_meta=None, use_yolo_regressors=False):
    h, w = 384, 640
    # Make empty black image
    if img_gpu is None:
        image = np.ones((h, w, 3), np.uint8) * 255
    else:
        img_numpy = img_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_numpy = img_numpy[:, :, (2, 1, 0)]  # To BRG
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
        # img_numpy = img_numpy[:, :, (2, 1, 0)]  # To RGB
        img_numpy = np.clip(img_numpy, 0, 1)
        img_gpu = torch.Tensor(img_numpy).cuda()
        image = (img_gpu * 255).byte().cpu().numpy()

    n_dets = detection['box'].size(0)
    n = 0
    p = detection['priors'][n]
    decoded_loc = detection['box'][n]
    id = detection['bbox_idx'][n]
    loc = loc_data[0, id, :]
    pixel_id = id // 3
    prior_id = id % 3
    if prior_id == 0:
        o = offset[0, :18, pixel_id]
        ks_h, ks_w = 3, 3
        grid_w = torch.tensor([-1, 0, 1] * ks_h)
        grid_h = torch.tensor([[-1], [0], [1]]).repeat(1, ks_w).view(-1)
    elif prior_id == 1:
        o = offset[0, 18:48, pixel_id]
        ks_h, ks_w = 3, 5
        grid_w = torch.tensor([-2, -1, 0, 1, 2] * ks_h)
        grid_h = torch.tensor([[-1], [0], [1]]).repeat(1, ks_w).view(-1)
    else:
        o = offset[0, 48:, pixel_id]
        ks_h, ks_w = 5, 3
        grid_w = torch.tensor([-1, 0, 1] * ks_h)
        grid_h = torch.tensor([[-2], [-1], [0], [1], [2]]).repeat(1, ks_w).view(-1)

    # thransfer the rectange to 9 points
    cx1, cy1, w1, h1 = p[0], p[1], p[2], p[3]
    dw1 = grid_w * w1 / (ks_w-1) + cx1
    dh1 = grid_h * h1 / (ks_h-1) + cy1

    dwh = p[2:] * ((loc.detach()[2:] * 0.2).exp() - 1)
    # regressed bounding boxes
    new_dh1 = dh1 + loc[1] * p[3] * 0.1 + dwh[1] / ks_h * grid_h
    new_dw1 = dw1 + loc[0] * p[2] * 0.1 + dwh[0] / ks_w * grid_w
    # points after the offsets of dcn
    new_dh2 = dh1 + o[::2].view(-1) * 0.5 * p[3]
    new_dw2 = dw1 + o[1::2].view(-1) * 0.5 * p[2]

    # Create a named colour
    blue = [255, 0, 0]  # bgr
    purple = [128, 0, 128]
    red = [0, 0, 255]

    # plot pred bbox
    cv2.rectangle(image, (decoded_loc[0] * w, decoded_loc[1] * h), (decoded_loc[2] * w, decoded_loc[3] * h),
                  blue, 2, lineType=8)

    # plot priors
    pxy1 = p[:2] - p[2:] / 2
    pxy2 = p[:2] + p[2:] / 2
    cv2.rectangle(image, (pxy1[0] * w, pxy1[1] * h), (pxy2[0] * w, pxy2[1] * h),
                  purple, 2, lineType=8)
    for i in range(len(dw1)):
        cv2.circle(image, (new_dw2[i] * w, new_dh2[i] * h), radius=0, color=blue, thickness=10)
        cv2.circle(image, (new_dw1[i]*w, new_dh1[i]*h), radius=0, color=blue, thickness=6)
        cv2.circle(image, (dw1[i] * w, dh1[i] * h), radius=0, color=purple, thickness=6)

    if img_meta is not None:
        path = ''.join(['results/results_1024_2/FCB/', str(img_meta[0]['video_id']), '_',
                        str(img_meta[0]['frame_id']), '.png'])
    else:
        path = 'results/results_1024_2/FCB/0.png'
    cv2.imwrite(path, image)


def display_correlation_map(x_corr, img_meta=None, idx=0):
    x_corr = x_corr[:, :36]
    bs, ch, h, w = x_corr.size()
    r = int(sqrt(ch))
    x_show = x_corr.view(r, r, h, w).permute(0, 2, 1, 3).contiguous()
    x_show = x_show.view(h*r, r*w)
    x_numpy = (x_show).cpu().numpy()

    if img_meta is not None:
        path = ''.join(['results/results_1024_2/fea_ref/', str(img_meta[0]['video_id']), '_',
                        str(img_meta[0]['frame_id']), '_', str(idx), '.png'])
    else:
        path = 'results/results_1024_2/fea_ref/0.png'

    plt.axis('off')
    plt.pcolormesh(x_numpy)
    plt.savefig(path)
    plt.clf()


def display_embedding_map(matching_map_all, idx, img_meta=None):
    if img_meta is not None:
        path = ''.join(['results/results_1227_1/embedding_map/', str(img_meta['video_id']), '_',
                        str(img_meta['frame_id']), '_', str(idx), '.png'])
        path2 = ''.join(['results/results_1227_1/embedding_map/', str(img_meta['video_id']), '_',
                        str(img_meta['frame_id']), '_', str(idx), '_m.png'])

    else:
        path = 'results/results_1227_1/embedding_map/0.png'
        path2 = 'results/results_1227_1/embedding_map/0_m.png'

    matching_map_all = matching_map_all.squeeze(0)
    r, r, h, w = matching_map_all.size()
    # matching_map_mean = matching_map_all.view(r**2, h, w).mean(0)  # / (r**2)
    matching_map, _ = matching_map_all.view(r ** 2, h, w).max(0)  # / (r**2)
    x_show = matching_map_all.permute(0, 2, 1, 3).contiguous()
    x_show = x_show.view(h * r, r * w)
    x_numpy = (x_show[h*2:h*10, w*2:w*10]).cpu().numpy()

    plt.axis('off')
    plt.pcolormesh(mmcv.imflip(x_numpy, direction='vertical'))
    plt.savefig(path)
    plt.clf()

    matching_map_numpy = matching_map.squeeze(0).cpu().numpy()
    plt.axis('off')
    plt.imshow(matching_map_numpy)
    plt.savefig(path2)
    plt.clf()


def display_shifted_masks(shifted_masks, img_meta=None):
    n, h, w = shifted_masks.size()

    for i in range(n):
        if img_meta is not None:
            path = ''.join(['results/results_1227_1/embedding_map/', str(img_meta['video_id']), '_',
                            str(img_meta['frame_id']), '_', str(i), '_shifted_masks.png'])

        else:
            path = 'results/results_1227_1/fea_ref/0_shifted_mask.png'
        shifted_masks = shifted_masks.gt(0.3).float()
        shifted_masks_numpy = shifted_masks[i].cpu().numpy()
        plt.axis('off')
        plt.pcolormesh(mmcv.imflip(shifted_masks_numpy*10, direction='vertical'))
        plt.savefig(path)
        plt.clf()



