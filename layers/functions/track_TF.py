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
        self.CandidateShift = CandidateShift()

    def __call__(self, net, candidate, img_meta, img=None):
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
            # only support batch_size = 1 for video test
            result = self.track(net, candidate, img_meta, img=img)
            out = [{'detection': result, 'net': net}]

        return out

    def track(self, net, candidate, img_meta, img=None):
        # only support batch_size = 1 for video test
        is_first = img_meta['is_first']
        if is_first:
            self.prev_candidate = None

        if candidate['box'].nelement() == 0:
            detection = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                         'score': torch.Tensor(), 'box_ids': torch.Tensor()}

        else:
            # get bbox and class after NMS
            det_bbox = candidate['box']
            det_score, det_labels = candidate['conf'][:, 1:].max(1)  # class
            det_masks_coeff = candidate['mask_coeff']
            if cfg.train_track:
                det_track_embed = F.normalize(candidate['track'], dim=1)
            else:
                det_track_embed = F.normalize(det_masks_coeff, dim=1)
            proto_data = candidate['proto']

            n_dets = det_bbox.size(0)
            # get masks
            det_masks = generate_mask(proto_data,
                                      cfg.mask_proto_coeff_activation(det_masks_coeff),
                                      det_bbox,
                                      cfg.mask_proto_mask_activation,
                                      cfg.use_sipmask)
            det_masks = det_masks.gt(0.5).float()

            # compared bboxes in current frame with bboxes in previous frame to achieve tracking
            if is_first or (not is_first and self.prev_candidate is None):
                # save bbox and features for later matching
                self.prev_candidate = candidate
                self.prev_candidate['tracked_mask'] = torch.zeros(n_dets)
            else:

                assert self.prev_candidate is not None
                T2S_feat_next = candidate['T2S_feat']
                fpn_feat_next = candidate['fpn_feat']
                prev_candidate_shift = self.CandidateShift(net, self.prev_candidate, T2S_feat_next, fpn_feat_next,
                                                           img=img, img_meta=[img_meta])
                self.prev_candidate['box'] = prev_candidate_shift['box'].clone()
                self.prev_candidate['conf'] = prev_candidate_shift['conf'].clone()
                self.prev_candidate['mask_coeff'] = prev_candidate_shift['mask_coeff'].clone()
                self.prev_candidate['tracked_mask'] = self.prev_candidate['tracked_mask'] + 1

                n_prev = self.prev_candidate['conf'].size(0)
                # only support one image at a time
                cos_sim = det_track_embed @ F.normalize(self.prev_candidate['track'], dim=1).t()
                cos_sim = torch.cat([torch.zeros(n_dets, 1), cos_sim], dim=1)
                cos_sim = (cos_sim + 1) / 2  # [0, 1]

                bbox_ious = jaccard(det_bbox, prev_candidate_shift['box'])
                prev_masks_shift = generate_mask(proto_data,
                                                 cfg.mask_proto_coeff_activation(prev_candidate_shift['mask_coeff']),
                                                 prev_candidate_shift['box'],
                                                 cfg.mask_proto_mask_activation,
                                                 cfg.use_sipmask)

                mask_ious = mask_iou(det_masks, prev_masks_shift)  # [n_dets, n_prev]
                # print(img_meta['video_id'], img_meta['frame_id'], cos_sim[:, 1:], mask_ious)

                # compute comprehensive score
                prev_det_score, prev_det_labels = self.prev_candidate['conf'][:, 1:].max(1)  # class
                label_delta = (prev_det_labels == det_labels.view(-1, 1)).float()
                comp_scores = compute_comp_scores(cos_sim,
                                                  det_score.view(-1, 1),
                                                  bbox_ious,
                                                  mask_ious,
                                                  label_delta,
                                                  add_bbox_dummy=True,
                                                  bbox_dummy_iou=0.3,
                                                  match_coeff=cfg.match_coeff)
                comp_scores[:, 1:] = comp_scores[:, 1:] * 0.95 ** (self.prev_candidate['tracked_mask'] - 1).view(1, -1)
                match_likelihood, match_ids = torch.max(comp_scores, dim=1)
                # translate match_ids to det_obj_ids, assign new id to new objects
                # update tracking features/bboxes of exisiting object,
                # add tracking features/bboxes of new object
                det_obj_ids = torch.ones(n_dets, dtype=torch.int32) * (-1)
                best_match_scores = torch.ones(n_prev) * (-1)
                best_match_idx = torch.ones(n_prev) * (-1)
                for idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        det_obj_ids[idx] = self.prev_candidate['conf'].size(0)
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

                for k, v in self.prev_candidate.items():
                    if k in {'proto', 'T2S_feat', 'fpn_feat'}:
                        self.prev_candidate[k] = candidate[k]

            det_score, _ = self.prev_candidate['conf'][:, 1:].max(1)
            det_obj_ids = torch.arange(self.prev_candidate['box'].size(0))
            det_masks_out = generate_mask(proto_data,
                                          cfg.mask_proto_coeff_activation(self.prev_candidate['mask_coeff']),
                                          self.prev_candidate['box'],
                                          cfg.mask_proto_mask_activation,
                                          cfg.use_sipmask)

            # whether add some tracked masks
            cond1 = self.prev_candidate['tracked_mask'] <= 7
            # whether tracked masks are greater than a small threshold, which removes some false positives
            cond2 = det_masks_out.gt_(0.5).sum([1, 2]) > 2
            # a declining weights (0.8) to remove some false positives that cased by consecutively track to segment
            cond3 = det_score.clone().detach() > cfg.eval_conf_thresh
            keep = cond1 & cond2 & cond3

            if keep.sum() == 0:
                detection = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor(), 'box_ids': torch.Tensor()}
            else:
                det_score, det_labels = self.prev_candidate['conf'][keep, 1:].max(1)
                detection = {'box': self.prev_candidate['box'][keep],
                             'mask_coeff': self.prev_candidate['mask_coeff'][keep],
                             'track': self.prev_candidate['track'][keep],
                             'class': det_labels+1, 'score': det_score,
                             'centerness': self.prev_candidate['centerness'][keep], 'proto': proto_data,
                             'box_ids': det_obj_ids[keep], 'mask': det_masks_out[keep]}

        return detection
