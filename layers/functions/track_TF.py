import torch
import torch.nn.functional as F
from ..box_utils import jaccard,  mask_iou
from ..mask_utils import generate_mask
from .TF_utils import CandidateShift, compute_comp_scores
from utils import timer

from datasets import cfg

import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)


class Track_TF(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self):
        self.prev_candidate = None

    def __call__(self, net, candidates, imgs_meta, imgs=None):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]

        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        with timer.env('Track'):
            results = []

            # only support batch_size = 1 for video test
            for batch_idx, candidate in enumerate(candidates):
                result = self.track(net, candidate, imgs_meta[batch_idx], img=imgs[batch_idx])
                results.append({'detection': result, 'net': net})

        return results

    def track(self, net, candidate, img_meta, img=None):
        # only support batch_size = 1 for video test
        is_first = img_meta['is_first']
        if is_first:
            self.prev_candidate = None

        if candidate['box'].nelement() == 0 and self.prev_candidate is None:
            return {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                    'score': torch.Tensor(), 'box_ids': torch.Tensor()}

        else:
            if candidate['box'].nelement() == 0 and self.prev_candidate is not None:
                prev_candidate_shift = CandidateShift(net, self.prev_candidate, candidate,
                                                      img=img, img_meta=img_meta)
                for k, v in prev_candidate_shift.items():
                    self.prev_candidate[k] = v.clone()
                self.prev_candidate['tracked_mask'] = self.prev_candidate['tracked_mask'] + 1
            else:

                # get bbox and class after NMS
                det_bbox = candidate['box']
                det_score = candidate['score']
                det_labels = candidate['class']
                det_masks_coeff = candidate['mask_coeff']
                if cfg.train_track:
                    det_track_embed = candidate['track']
                else:
                    det_track_embed = F.normalize(det_masks_coeff, dim=1)

                n_dets = det_bbox.size(0)
                # get masks
                det_masks_soft = generate_mask(candidate['proto'], det_masks_coeff, det_bbox)
                candidate['mask'] = det_masks_soft
                det_masks = det_masks_soft.gt(0.5).float()

                # compared bboxes in current frame with bboxes in previous frame to achieve tracking
                if is_first or (not is_first and self.prev_candidate is None):
                    # save bbox and features for later matching
                    self.prev_candidate = dict()
                    for k, v in candidate.items():
                        self.prev_candidate[k] = v
                    self.prev_candidate['tracked_mask'] = torch.zeros(n_dets)

                else:

                    assert self.prev_candidate is not None
                    prev_candidate_shift = CandidateShift(net, self.prev_candidate, candidate,
                                                          img=img, img_meta=img_meta)
                    for k, v in prev_candidate_shift.items():
                        self.prev_candidate[k] = v.clone()
                    self.prev_candidate['tracked_mask'] = self.prev_candidate['tracked_mask'] + 1

                    n_prev = self.prev_candidate['box'].size(0)
                    # only support one image at a time
                    cos_sim = det_track_embed @ self.prev_candidate['track'].t()
                    cos_sim = torch.cat([torch.zeros(n_dets, 1), cos_sim], dim=1)
                    cos_sim = (cos_sim + 1) / 2  # [0, 1]

                    bbox_ious = jaccard(det_bbox, self.prev_candidate['box'])
                    prev_masks_shift = self.prev_candidate['mask'].gt(0.5).float()

                    mask_ious = mask_iou(det_masks, prev_masks_shift)  # [n_dets, n_prev]
                    # print(img_meta['video_id'], img_meta['frame_id'], cos_sim[:, 1:], mask_ious)

                    # compute comprehensive score
                    prev_det_labels = self.prev_candidate['class']
                    label_delta = (prev_det_labels == det_labels.view(-1, 1)).float()
                    comp_scores = compute_comp_scores(cos_sim,
                                                      det_score.view(-1, 1),
                                                      bbox_ious,
                                                      mask_ious,
                                                      label_delta,
                                                      add_bbox_dummy=True,
                                                      bbox_dummy_iou=0.3,
                                                      match_coeff=cfg.match_coeff)
                    match_likelihood, match_ids = torch.max(comp_scores, dim=1)
                    # translate match_ids to det_obj_ids, assign new id to new objects
                    # update tracking features/bboxes of exisiting object,
                    # add tracking features/bboxes of new object
                    det_obj_ids = torch.ones(n_dets, dtype=torch.int32) * (-1)
                    best_match_scores = torch.ones(n_prev) * (-1)
                    best_match_idx = torch.ones(n_prev) * (-1)
                    for idx, match_id in enumerate(match_ids):
                        if match_id == 0:
                            det_obj_ids[idx] = self.prev_candidate['box'].size(0)
                            for k, v in self.prev_candidate.items():
                                if k not in {'proto', 'T2S_feat', 'fpn_feat', 'tracked_mask'}:
                                    self.prev_candidate[k] = torch.cat([v, candidate[k][idx][None]], dim=0)
                            self.prev_candidate['tracked_mask'] = torch.cat([self.prev_candidate['tracked_mask'],
                                                                             torch.zeros(1)], dim=0)

                        else:
                            # multiple candidate might match with previous object, here we choose the one with
                            # largest comprehensive score
                            obj_id = match_id - 1
                            match_score = det_score[idx]  # match_likelihood[idx]
                            if match_score > best_match_scores[obj_id]:
                                if best_match_idx[obj_id] != -1:
                                    det_obj_ids[int(best_match_idx[obj_id])] = -1
                                det_obj_ids[idx] = obj_id
                                best_match_scores[obj_id] = match_score
                                best_match_idx[obj_id] = idx
                                # udpate feature
                                for k, v in self.prev_candidate.items():
                                    if k not in {'proto', 'T2S_feat', 'fpn_feat', 'tracked_mask'}:
                                        self.prev_candidate[k][obj_id] = candidate[k][idx]
                                self.prev_candidate['tracked_mask'][obj_id] = 0

            det_obj_ids = torch.arange(self.prev_candidate['box'].size(0))
            # whether add some tracked masks
            cond1 = self.prev_candidate['tracked_mask'] <= 10
            # whether tracked masks are greater than a small threshold, which removes some false positives
            cond2 = self.prev_candidate['mask'].gt(0.5).sum([1, 2]) > 1
            # a declining weights (0.8) to remove some false positives that cased by consecutively track to segment
            cond3 = self.prev_candidate['score'].clone().detach() > cfg.eval_conf_thresh
            keep = cond1 & cond2 & cond3

            if keep.sum() == 0:
                detection = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor(), 'box_ids': torch.Tensor()}
            else:

                detection = {'box': self.prev_candidate['box'][keep],
                             'mask_coeff': self.prev_candidate['mask_coeff'][keep],
                             'track': self.prev_candidate['track'][keep],
                             'class': self.prev_candidate['class'][keep],
                             'score': self.prev_candidate['score'][keep],
                             'centerness': self.prev_candidate['centerness'][keep],
                             'proto': candidate['proto'], 'mask': self.prev_candidate['mask'][keep],
                             'box_ids': det_obj_ids[keep]}

            return detection
