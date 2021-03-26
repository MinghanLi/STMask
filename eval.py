from datasets import *
from STMask import STMask
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess_ytbvis, undo_image_transformation, display_fpn_outs

from datasets import get_dataset, prepare_data
import mmcv
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.autograd import Variable
import argparse
import random
import os
from collections import defaultdict
from layers.eval_utils import bbox2result_with_id, results2json_videoseg, ytvos_eval, calc_metrics

import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--clip_eval_mode', default=False, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--top_k', default=100, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display_fpn_outs', default=True, type=str2bool,
                        help='Whether or not to display outputs after fpn')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_json', default=True, dest='output_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/eval_bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/eval_mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=True, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--eval_dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')
    parser.add_argument('--eval_types', type=str, nargs='+', choices=['bbox', 'segm'], help='eval types')
    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False, emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


def prep_display(dets_out, img, pad_h, pad_w, img_ids=None, img_meta=None, undo_transform=True, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    -- display_model: 'train', 'test', 'None' means groundtruth results
    """

    if undo_transform:
        img_numpy = undo_image_transformation(img, img_meta, pad_h, pad_w)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        pad_h, pad_w, _ = img.shape

    # if img_gpu is None:
    # img_gpu = torch.tensor(np.zeros(img_gpu.size(), np.uint8)).cuda()

    with timer.env('Postprocess'):
        cfg.mask_proto_debug = args.mask_proto_debug
        cfg.preserve_aspect_ratio = False
        dets_out = postprocess_ytbvis(dets_out, pad_h, pad_w, img_meta, display_mask=True,
                                      visualize_lincomb = args.display_lincomb,
                                      crop_masks        = args.crop,
                                      score_threshold   = cfg.eval_conf_thresh,
                                      img_ids           = img_ids,
                                      mask_det_file     = args.mask_det_file)
        torch.cuda.synchronize()
        scores = dets_out['score'][:args.top_k].detach().cpu().numpy()
        boxes = dets_out['box'][:args.top_k].detach().cpu().numpy()

    if 'segm' in dets_out:
        masks = dets_out['segm'][:args.top_k]
        args.display_masks = True
    else:
        args.display_masks = False

    classes = dets_out['class'][:args.top_k].detach().cpu().numpy()
    num_dets_to_consider = min(args.top_k, classes.shape[0])
    color_type = dets_out['box_ids']
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, color_type, on_gpu=img_gpu.device.index, undo_transform=undo_transform).view(1, 1, 1, 3)
                            for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)
        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
    if args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            # get the bbox_idx to know box's layers (after FPN): p3-p7
            # box_idx = dets_out['bbox_idx'][j]
            # p_nums = [34560, 43200, 45360, 45900, 46035]
            # p_nums = [11520, 14400, 15120, 15300, 15345]
            # p = 0
            # for i in range(len(p_nums)):
            #     if box_idx < p_nums[i]:
            #         p = i + 3
            #         break

            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j, color_type)
            # plot priors
            h, w, _ = img_meta['img_shape']
            if 'priors' in dets_out.keys():
                priors = dets_out['priors'].view(-1, 4).detach().cpu().numpy()
                if j < dets_out['priors'].size(0):
                    cpx, cpy, pw, ph = priors[j, :] * [w, h, w, h]
                    px1, py1 = cpx - pw/2.0, cpy - ph/2.0
                    px2, py2 = cpx + pw / 2.0, cpy + ph / 2.0
                    px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                    pcolor = [255, 0, 255]

            # plot the range of features for classification and regression
            pred_scales = [24, 48, 96, 192, 384]
            # cpx, cpy = (px1+px2)/2, (py1+py2)/2
            # fx1, fy1 = cpx - pred_scales[p - 3] / 2, cpy - pred_scales[p - 3] / 2
            # fx2, fy2 = cpx + pred_scales[p - 3] / 2, cpy + pred_scales[p - 3] / 2
            # fx1, fy1, fx2, fy2 = int(fx1), int(fy1), int(fx2), int(fy2)
            # fcolor = [255, 128, 0]
            x = torch.clamp(torch.tensor([x1, x2]), min=2, max=pad_w).tolist(),
            y = torch.clamp(torch.tensor([y1, y2]), min=2, max=pad_h).tolist(),
            x, y = x[0], y[0]

            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x[0], y[0]), (x[1], y[1]), color, 2)
                # if 'priors' in dets_out.keys():
                #     if j < dets_out['priors'].size(0):
                #         cv2.rectangle(img_numpy, (px1, py1), (px2, py2), pcolor, 2, lineType=8)
                # cv2.rectangle(img_numpy, (x[4], y[4]), (x[5], y[5]), fcolor, 2)

            if args.display_text:
                if classes[j]-1 < 0:
                    _class = 'None'
                else:
                    _class = cfg.classes[classes[j]-1]

                if score is not None:
                    # if cfg.use_maskiou and not cfg.rescore_bbox:
                    train_DIoU = False
                    if train_DIoU:
                        rescore = dets_out['DIoU_score'][j] * score
                        text_str = '%s: %.2f: %.2f: %s' % (_class, score, rescore, str(color_type[j].cpu().numpy())) \
                            if args.display_scores else _class
                    else:

                        text_str = '%s: %.2f: %s' % (
                        _class, score, str(color_type[j].cpu().numpy())) if args.display_scores else _class
                else:
                    text_str = '%s' % _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.5
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (max(x1, 10), max(y1 - 3, 10))
                text_color = [255, 255, 255]
                cv2.rectangle(img_numpy, (max(x1, 10), max(y1, 10)), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy


# Quick and dirty lambda for selecting the color for a particular index
# Also keeps track of a per-gpu color cache for maximum speed
def get_color(j, color_type, on_gpu=None, undo_transform=True):
    global color_cache
    color_idx = ( color_type[j] * 5 ) % len(cfg.COLORS)

    if on_gpu is not None and color_idx in color_cache[on_gpu]:
        return color_cache[on_gpu][color_idx]
    else:
        color = cfg.COLORS[color_idx]
        if not undo_transform:
            # The image might come in as RGB or BRG, depending
            color = (color[2], color[1], color[0])
        if on_gpu is not None:
            color = torch.Tensor(color).to(on_gpu).float() / 255.
            color_cache[on_gpu][color_idx] = color
        return color


def prep_display_single(dets_out, img, pad_h, pad_w, img_ids=None, img_meta=None, undo_transform=True, mask_alpha=0.45,
                        fps_str='', display_mode=None):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    -- display_model: 'train', 'test', 'None' means groundtruth results
    """

    if undo_transform:
        img_numpy = undo_image_transformation(img, img_meta, pad_h, pad_w)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        pad_h, pad_w, _ = img.shape

    with timer.env('Postprocess'):
        cfg.mask_proto_debug = args.mask_proto_debug
        cfg.preserve_aspect_ratio = False
        dets_out = postprocess_ytbvis(dets_out, pad_h, pad_w, img_meta, display_mask=True,
                                      visualize_lincomb=args.display_lincomb,
                                      crop_masks=args.crop,
                                      score_threshold=cfg.eval_conf_thresh,
                                      img_ids=img_ids,
                                      mask_det_file=args.mask_det_file)
        torch.cuda.synchronize()
        scores = dets_out['score'][:args.top_k].detach().cpu().numpy()
        boxes = dets_out['box'][:args.top_k].detach().cpu().numpy()

    if 'segm' in dets_out:
        masks = dets_out['segm'][:args.top_k]
        args.display_masks = True
    else:
        args.display_masks = False

    classes = dets_out['class'][:args.top_k].detach().cpu().numpy()

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    color_type = dets_out['box_ids']
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, color_type, on_gpu=img_gpu.device.index, undo_transform=undo_transform).view(1, 1, 1, 3)
             for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)
        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j, color_type)
            # plot priors
            h, w, _ = img_meta['img_shape']
            priors = dets_out['priors'].detach().cpu().numpy()
            if j < dets_out['priors'].size(0):
                cpx, cpy, pw, ph = priors[j, :] * [w, h, w, h]
                px1, py1 = cpx - pw / 2.0, cpy - ph / 2.0
                px2, py2 = cpx + pw / 2.0, cpy + ph / 2.0
                px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                pcolor = [255, 0, 255]

            # plot the range of features for classification and regression
            pred_scales = [24, 48, 96, 192, 384]
            x = torch.clamp(torch.tensor([x1, x2]), min=2, max=638).tolist(),
            y = torch.clamp(torch.tensor([y1, y2]), min=2, max=358).tolist(),
            x, y = x[0], y[0]

            if display_mode is not None:
                score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x[0], y[0]), (x[1], y[1]), color, 1)
                if j < dets_out['priors'].size(0):
                    cv2.rectangle(img_numpy, (px1, py1), (px2, py2), pcolor, 2, lineType=8)
                # cv2.rectangle(img_numpy, (x[4], y[4]), (x[5], y[5]), fcolor, 2)

            if args.display_text:
                if classes[j] - 1 < 0:
                    _class = 'None'
                else:
                    _class = cfg.classes[classes[j] - 1]

                if display_mode == 'test':
                    # if cfg.use_maskiou and not cfg.rescore_bbox:
                    train_centerness = False
                    if train_centerness:
                        rescore = dets_out['DIoU_score'][j] * score
                        text_str = '%s: %.2f: %.2f: %s' % (_class, score, rescore, str(color_type[j].cpu().numpy())) \
                            if args.display_scores else _class
                    else:

                        text_str = '%s: %.2f: %s' % (
                            _class, score, str(color_type[j].cpu().numpy())) if args.display_scores else _class
                else:
                    text_str = '%s' % _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.5
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]
                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

    return img_numpy


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def validation(net:STMask, valid_data=False, output_metrics_file=None):
    cfg.mask_proto_debug = args.mask_proto_debug
    if not valid_data:
        cfg.valid_sub_dataset.test_mode = True
        dataset = get_dataset(cfg.valid_sub_dataset)
    else:
        cfg.valid_dataset.test_mode = True
        dataset = get_dataset(cfg.valid_dataset)

    frame_times = MovingAverage()
    dataset_size = math.ceil(len(dataset)/args.batch_size) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    results = []

    try:
        # Main eval loop
        for it, data_batch in enumerate(data_loader):
            timer.reset()
            with timer.env('Load Data'):
                images, images_meta, ref_images, ref_images_meta = prepare_data(data_batch, is_cuda=True, train_mode=False)
                pad_h, pad_w = images.size()[2:4]

            with timer.env('Network Extra'):
                preds = net(images, img_meta=images_meta, ref_x=ref_images, ref_imgs_meta=ref_images_meta)

                if it == dataset_size - 1:
                    batch_size = len(dataset) % args.batch_size
                else:
                    batch_size = images.size(0)

                for batch_id in range(batch_size):
                    cfg.preserve_aspect_ratio = True
                    preds_cur = postprocess_ytbvis(preds[batch_id], pad_h, pad_w, images_meta[batch_id],
                                                   score_threshold=cfg.eval_conf_thresh)
                    segm_results = bbox2result_with_id(preds_cur, cfg.classes)
                    results.append(segm_results)

            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                if batch_size == 0:
                    batch_size = 1
                frame_times.add(timer.total_time()/batch_size)

            if it > 1 and frame_times.get_avg() > 0:
                fps = 1 / frame_times.get_avg()
            else:
                fps = 0
            progress = (it + 1) / dataset_size * 100
            progress_bar.set_val(it + 1)
            print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                  % (repr(progress_bar), it + 1, dataset_size, progress, fps), end='')

        print()
        print('Dumping detections...')

        if not valid_data:
            results2json_videoseg(dataset, results, args.mask_det_file)
            print('calculate evaluation metrics ...')
            ann_file = cfg.valid_sub_dataset.ann_file
            dt_file = args.mask_det_file
            calc_metrics(ann_file, dt_file, output_file=output_metrics_file)
        else:
            results2json_videoseg(dataset, results, output_metrics_file.replace('.txt', '.json'))

    except KeyboardInterrupt:
        print('Stopping...')


def evaluate(net:STMask, dataset):
    net.detect.use_fast_nms = args.fast_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    frame_times = MovingAverage()
    dataset_size = math.ceil(len(dataset) / args.batch_size) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    results = []

    try:
        # Main eval loop
        for it, data_batch in enumerate(data_loader):
            timer.reset()

            with timer.env('Load Data'):
                images, images_meta, ref_images, ref_images_meta = prepare_data(data_batch, is_cuda=True, train_mode=False)
            pad_h, pad_w = images.size()[2:4]

            with timer.env('Network Extra'):
                preds = net(images, img_meta=images_meta, ref_x=ref_images, ref_imgs_meta=ref_images_meta)

            # Perform the meat of the operation here depending on our mode.
            if it == dataset_size-1:
                batch_size = len(dataset) % args.batch_size
            else:
                batch_size = images.size(0)

            for batch_id in range(batch_size):
                if args.display:
                    img_id = (images_meta[batch_id]['video_id'], images_meta[batch_id]['frame_id'])
                    if not cfg.display_mask_single:
                        img_numpy = prep_display(preds[batch_id], images[batch_id], pad_h, pad_w,
                                                 img_meta=images_meta[batch_id], img_ids=img_id)
                    else:
                        for p in range(preds[batch_id]['detection']['box'].size(0)):
                            preds_single = {'detection': {}}
                            for k in preds[batch_id]['detection']:
                                if preds[batch_id]['detection'][k] is not None and k not in {'proto'}:
                                    preds_single['detection'][k] = preds[batch_id]['detection'][k][p]
                                else:
                                    preds_single['detection'][k] = None
                            preds_single['net'] = preds[batch_id]['net']
                            preds_single['detection']['box_ids'] = torch.tensor(-1)

                            img_numpy = prep_display(preds_single, images[batch_id], pad_h, pad_w,
                                                     img_meta=images_meta[batch_id], img_ids=img_id)
                            plt.imshow(img_numpy)
                            plt.axis('off')
                            plt.savefig(''.join([args.mask_det_file[:-12], 'out_single/', str(img_id), '_', str(p),
                                                 '.png']))
                            plt.clf()

                else:
                    cfg.preserve_aspect_ratio = True
                    preds_cur = postprocess_ytbvis(preds[batch_id], pad_h, pad_w, images_meta[batch_id],
                                                   score_threshold=cfg.eval_conf_thresh)
                    segm_results = bbox2result_with_id(preds_cur, cfg.classes)
                    results.append(segm_results)

                # First couple of images take longer because we're constructing the graph.
                # Since that's technically initialization, don't include those in the FPS calculations.
                if it > 1:
                    frame_times.add(timer.total_time() / batch_size)
            
                if args.display and not cfg.display_mask_single:
                    if it > 1:
                        print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                    plt.imshow(img_numpy)
                    plt.axis('off')
                    plt.title(str(img_id))

                    root_dir = ''.join([args.mask_det_file[:-12], 'out/', str(images_meta[batch_id]['video_id']), '/'])
                    if not os.path.exists(root_dir):
                        os.makedirs(root_dir)
                    plt.savefig(''.join([root_dir, str(images_meta[batch_id]['frame_id']), '.png']))
                    plt.clf()
                    # plt.show()
                elif not args.no_bar:
                    if it > 1: fps = 1 / frame_times.get_avg()
                    else: fps = 0
                    progress = (it+1) / dataset_size * 100
                    progress_bar.set_val(it+1)
                    print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                        % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')

        if not args.display and not args.benchmark:
            print()
            if args.output_json:
                print('Dumping detections...')
                results2json_videoseg(dataset, results, args.mask_det_file)

                if cfg.use_valid_sub or cfg.use_train_sub:
                    if cfg.use_valid_sub:
                        print('calculate evaluation metrics ...')
                        ann_file = cfg.valid_sub_dataset.ann_file
                    else:
                        print('calculate train_sub metrics ...')
                        ann_file = cfg.train_dataset.ann_file
                    dt_file = args.mask_det_file
                    metrics = calc_metrics(ann_file, dt_file)

                    return metrics

        elif args.benchmark:
            print()
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000*avg_seconds))

    except KeyboardInterrupt:
        print('Stopping...')


def evaluate_single(net:STMask, im_path=None, save_path=None):
        im = mmcv.imread(im_path)
        ori_shape = im.shape
        im, w_scale, h_scale = mmcv.imresize(im, (640, 360), return_scale=True)
        img_shape = im.shape
        im = mmcv.imnormalize(im, np.array([123.675, 116.28, 103.53], dtype=np.float32),
                              np.array([58.395, 57.12, 57.375], dtype=np.float32), to_rgb=True)
        im = mmcv.impad_to_multiple(im, 32)
        pad_shape = im.shape
        im = torch.tensor(im).permute(2, 0, 1).contiguous().unsqueeze(0).cuda()
        pad_h, pad_w = im.size()[2:4]
        preds = net(im)
        preds[0]['detection']['box_ids'] = torch.arange(preds[0]['detection']['box'].size(0))
        img_meta = {'ori_shape': ori_shape, 'img_shape': img_shape, 'pad_shape': pad_shape}
        cfg.preserve_aspect_ratio = True
        img_numpy = prep_display(preds[0], im[0], pad_h, pad_w, img_meta=img_meta,
                                 img_ids=(0, 0))
        if save_path is None:
            plt.imshow(img_numpy)
            plt.axis('off')
            plt.show()
        else:
            cv2.imwrite(save_path, img_numpy)


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.image is None:
        if args.eval_dataset is not None:
            set_dataset(args.eval_dataset, 'eval')

        if cfg.use_train_sub:
            print('load train_sub dataset')
            cfg.train_dataset.test_mode = True
            val_dataset = get_dataset(cfg.train_dataset)
        elif cfg.use_valid_sub:
            print('load valid_sub dataset')
            cfg.valid_sub_dataset.test_mode = True
            val_dataset = get_dataset(cfg.valid_sub_dataset)
        else:
            print('load valid dataset')
            val_dataset = get_dataset(cfg.valid_dataset)
    else:
        val_dataset = None

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        print('Loading model...', end='')
        net = STMask()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        if args.image is None:
            if cfg.only_calc_metrics:
                print('calculate evaluation metrics ...')
                ann_file = cfg.valid_sub_dataset.ann_file
                dt_file = args.mask_det_file
                print('det_file:', dt_file)
                metrics = calc_metrics(ann_file, dt_file)
                metrics_name = ['mAP', 'AP50', 'AP75', 'small', 'medium', 'large',
                                'AR1', 'AR10', 'AR100', 'AR100_small', 'AR100_medium', 'AR100_large']
                log_dir = 'weights/temp/train_log'
                writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix='VIS')
                for i_m in range(len(metrics_name)):
                    writer.add_scalar('valid_metrics/' + metrics_name[i_m], metrics[i_m], 1)
            else:
                evaluate(net, val_dataset)
        else:
            im_path = 'results/Jie/53_2.tif'
            save_path = 'results/Jie/53_2_mask.png'
            evaluate_single(net, im_path, save_path)


