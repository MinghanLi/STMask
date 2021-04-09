# -*- coding: utf-8 -*-
import torch
import numpy as np
import mmcv
import os
import pycocotools.mask as mask_util
from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS
from cocoapi.PythonAPI.pycocotools.ytvoseval import YTVOSeval
import matplotlib as plt
from datasets import cfg
import cv2
from utils.functions import SavePath


def bbox2result_with_id(preds, classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        classes (int): class category, including background class

    Returns:
        list(ndarray): bbox results of each class
    """

    if preds['box'].shape[0] == 0:
        return dict()
    else:
        bboxes = preds['box'].cpu().numpy()
        if preds['class'] is not None:
            labels = preds['class'].cpu().numpy()
            # labels_all = preds['class_all'].cpu().numpy()
        else:
            labels = None
        scores = preds['score'].cpu().numpy()
        segms = preds['segm']
        obj_ids = preds['box_ids'].cpu().numpy()
        results = {}
        if labels is not None:
            for bbox, label, score, segm, obj_id in zip(bboxes, labels, scores, segms, obj_ids):
                if obj_id >= 0:
                    results[obj_id] = {'bbox': bbox, 'label': label, 'score': score, 'segm': segm,
                                       'category': classes[label-1]}
        else:
            for bbox, score, segm, obj_id in zip(bboxes, scores, segms, obj_ids):
                if obj_id >= 0:
                    results[obj_id] = {'bbox': bbox, 'score': score, 'segm': segm}

        return results


def results2json_videoseg(dataset, results, out_file, sampler_img_ids=None):
    json_results = []
    vid_objs = {}
    size = len(results)
    # if current sub_dataset is only a part of dataset, you should give image index in Sampler to sampler_img_ids
    if sampler_img_ids is None:
        sampler_img_ids = range(size)

    for idx in range(size):
        # assume results is ordered

        vid_id, frame_id = dataset.img_ids[sampler_img_ids[idx]]
        vid_id = dataset.vid_ids[vid_id]
        if idx == size - 1:
            is_last = True
        else:
            _, frame_id_next = dataset.img_ids[sampler_img_ids[idx] + 1]
            is_last = frame_id_next == 0

        det = results[idx]
        for obj_id in det:
            if det[obj_id]['segm']['counts'] != b'PPTl0':
                bbox = det[obj_id]['bbox']
                score = det[obj_id]['score']
                segm = det[obj_id]['segm']
                label = det[obj_id]['label']
                # label_all = det[obj_id]['label_all']
                if obj_id not in vid_objs:
                    vid_objs[obj_id] = {'scores': [], 'cats': [], 'segms': {}}
                vid_objs[obj_id]['scores'].append(score)
                vid_objs[obj_id]['cats'].append(label)
                segm['counts'] = segm['counts'].decode()
                vid_objs[obj_id]['segms'][frame_id] = segm
        if is_last:
            # store results of  the current video
            for obj_id, obj in vid_objs.items():
                # majority voting for sequence category
                all_cats = np.bincount(np.array(obj['cats']))
                data_score = np.array(obj['scores']).mean().item()
                vid_seg = []
                for fid in range(frame_id + 1):
                    if fid in obj['segms']:
                        vid_seg.append(obj['segms'][fid])
                    else:
                        vid_seg.append(None)

                for cat_id in range(len(all_cats)):
                    if all_cats[cat_id] > 0.1 * len(obj['cats']) or cat_id == all_cats.argmax().item():
                        data = dict()

                        data['video_id'] = vid_id
                        data['score'] = data_score
                        # majority voting for sequence category
                        data['category_id'] = cat_id

                        data['segmentations'] = vid_seg
                        json_results.append(data)

            vid_objs = {}
    if not os.path.exists(out_file[:-13]):
        os.makedirs(out_file[:-13])

    mmcv.dump(json_results, out_file)
    print('Done')


def calc_metrics(anno_file, dt_file, output_file=None):
    ytvosGt = YTVOS(anno_file)
    ytvosDt = ytvosGt.loadRes(dt_file)

    E = YTVOSeval(ytvosGt, ytvosDt, iouType='segm', output_file=output_file)
    E.evaluate()
    E.accumulate()
    E.summarize()
    print('finish validation')

    return E.stats


def ytvos_eval(result_file, result_types, ytvos, max_dets=(100, 300, 1000), save_path_valid_metrics=None):
    if mmcv.is_str(ytvos):
        ytvos = YTVOS(ytvos)
    assert isinstance(ytvos, YTVOS)

    if len(ytvos.anns) == 0:
        print("Annotations does not exist")
        return
    assert result_file.endswith('.json')
    ytvos_dets = ytvos.loadRes(result_file)

    vid_ids = ytvos.getVidIds()
    for res_type in result_types:
        iou_type = res_type
        ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type, output_file=save_path_valid_metrics)
        ytvosEval.params.vidIds = vid_ids
        if res_type == 'proposal':
            ytvosEval.params.useCats = 0
            ytvosEval.params.maxDets = list(max_dets)
        ytvosEval.evaluate()
        ytvosEval.accumulate()
        ytvosEval.summarize()
