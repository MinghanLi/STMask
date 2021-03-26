""" Contains functions used to sanitize and prepare the output of Yolact. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import cfg, mask_type, MEANS, STD, activation_func
from utils.augmentations import Resize
from utils import timer
from .box_utils import crop, sanitize_coordinates, center_size, decode
import eval as eval_script
import matplotlib.pyplot as plt


def display_train_output(images, predictions, conf_t, pids_t, gt_bboxes, gt_labels, gt_masks, ref_images, ref_bboxes,
                         gt_pids, img_meta, epoch, iteration, path=None):
    setup_eval()
    loc_data = predictions['loc']
    conf_data = predictions['conf']
    mask_data = predictions['mask']
    priors = predictions['priors']
    priors = priors[0, :, :]
    match_score = predictions['track']
    ref_boxes_n = predictions['ref_boxes_n']
    if cfg.mask_type == mask_type.lincomb:
        proto_data = predictions['proto']

    batch_size, _, h, w = images.size()

    if cfg.use_sigmoid_focal_loss:
        # Note: even though conf[0] exists, this mode doesn't train it so don't use it
        conf_data = torch.sigmoid(conf_data)
    elif cfg.use_objectness_score:
        # See focal_loss_sigmoid in multibox_loss.py for details
        objectness = torch.sigmoid(conf_data[:, :, 0])
        conf_data[:, :, 1:] = objectness[:, :, None] * F.softmax(conf_data[:, :, 1:], -1)
        conf_data[:, :, 0] = 1 - objectness
    else:
        conf_data = F.softmax(conf_data, -1)

    # visualization
    pos = conf_t > 0
    for batch_idx in range(batch_size):
        # detection results
        dets_out = {}
        idx_pos = pos[batch_idx, :] == 1

        dets_out['score'], class_pred = conf_data[batch_idx, idx_pos, 1:].max(dim=1)
        dets_out['class_pred'] = class_pred + 1  # classes begins from 1
        dets_out['class'] = conf_t[batch_idx, idx_pos]
        dets_out['pids'] = pids_t[batch_idx, idx_pos]
        dets_out['box'] = decode(loc_data[batch_idx, idx_pos, :], priors[idx_pos])
        dets_out['mask'] = mask_data[batch_idx, idx_pos, :]
        if cfg.mask_type == mask_type.lincomb:
            dets_out['proto'] = proto_data[batch_idx]

        img_numpy = eval_script.prep_display(dets_out, images[batch_idx], h, w, img_meta[batch_idx], display_mode='train')

        # gt results
        dets_out = {}
        dets_out['class'] = gt_labels[batch_idx]
        dets_out['box'] = gt_bboxes[batch_idx]
        dets_out['segm'] = gt_masks[batch_idx].type(torch.cuda.FloatTensor)
        dets_out['pids'] = gt_pids[batch_idx].type(torch.cuda.LongTensor)

        img_numpy_gt = eval_script.prep_display(dets_out, images[batch_idx], h, w, img_meta[batch_idx])

        # gt results of the last frame
        dets_out = {}
        gt_class_last = []
        for i in range(1, len(ref_bboxes[batch_idx])+1):
            if i in gt_pids[batch_idx].tolist():
                gt_class_last.append(gt_labels[batch_idx][gt_pids[batch_idx].tolist().index(i)])
            else:
                gt_class_last.append(-1)
        dets_out['class'] = torch.tensor(gt_class_last)

        dets_out['box'] = ref_bboxes[batch_idx]
        dets_out['pids'] = torch.arange(1, len(ref_bboxes[batch_idx])+1)

        img_numpy_gt_last = eval_script.prep_display(dets_out, ref_images[batch_idx], h, w, img_meta[batch_idx])

        # show results and save figs
        plt.imshow(img_numpy)
        plt.title('train')
        plt.savefig(''.join([path, 'out/', str(epoch), '_', str(iteration), '_', str(batch_idx), '_train', '.png']))
        plt.show()

        plt.imshow(img_numpy_gt)
        plt.title('gt')
        plt.savefig(''.join([path, 'out/', str(epoch), '_', str(iteration), '_', str(batch_idx), '_gt', '.png']))
        plt.show()

        plt.imshow(img_numpy_gt_last)
        plt.title('gt_last')
        plt.savefig(''.join([path, 'out/', str(epoch), '_', str(iteration), '_', str(batch_idx), '_gt_last', '.png']))
        plt.show()


def setup_eval():
    eval_script.parse_args(['--no_bar',
                            '--output_json',
                            ])

