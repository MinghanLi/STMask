import torch
import torch.nn.functional as F
from ..box_utils import jaccard, mask_iou,  DIoU
from ..mask_utils import generate_mask
from utils import timer

from datasets import cfg, mask_type

import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)


class Track(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self):
        self.prev_det_bbox = None
        self.prev_track_embed = None
        self.prev_det_labels = None
        self.prev_det_masks = None

    def __call__(self, pred_outs_after_NMS, img_meta):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]

        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        with timer.env('Track_local_matching'):
            batch_size = len(pred_outs_after_NMS)

            for batch_idx in range(batch_size):
                detection = pred_outs_after_NMS[batch_idx]['detection']
                pred_outs_after_NMS[batch_idx]['detection'] = self.track(detection, img_meta[batch_idx])

        return pred_outs_after_NMS

    def track(self, detection, img_meta):

        # only support batch_size = 1 for video test
        is_first = img_meta['is_first']
        if is_first:
            self.prev_det_bbox = None
            self.prev_track_embed = None
            self.prev_det_labels = None
            self.prev_det_masks = None

        if detection['class'].nelement() == 0:
            detection['box_ids'] = np.array([], dtype=np.int64)
            return detection

        # get bbox and class after NMS
        det_bbox = detection['box']
        det_labels = detection['class']  # class
        det_score = detection['score']
        det_masks_coff = detection['mask_coeff']
        if cfg.train_track:
            det_track_embed = F.normalize(detection['track'], dim=1)
        else:
            det_track_embed = F.normalize(det_masks_coff, dim=1)
        proto_data = detection['proto']

        n_dets = det_bbox.size(0)

        # get masks
        det_masks = generate_mask(proto_data, det_masks_coff, det_bbox, cfg.mask_proto_mask_activation)
        det_masks_out = det_masks
        det_masks = det_masks.gt(0.5).float()

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_det_bbox is None):
            det_obj_ids = torch.arange(det_bbox.size(0))
            # save bbox and features for later matching
            self.prev_det_bbox = det_bbox
            self.prev_track_embed = det_track_embed
            self.prev_det_labels = det_labels.view(-1)
            self.prev_det_masks = det_masks

        else:

            assert self.prev_track_embed is not None
            n_prev = self.prev_det_bbox.size(0)
            # only support one image at a time
            cos_sim = det_track_embed @ self.prev_track_embed.t()  # [n_dets, n_prev], val in [-1, 1]
            cos_sim = torch.cat([torch.zeros(n_dets, 1), cos_sim], dim=1)
            cos_sim = (cos_sim + 1) / 2  # [0, 1]

            bbox_ious = jaccard(det_bbox, self.prev_det_bbox)
            mask_ious = mask_iou(det_masks, self.prev_det_masks)

            # print(img_meta['video_id'], img_meta['frame_id'], mask_ious)
            if cfg.use_DIoU_in_comp_scores:
                term_DIoU = DIoU(det_bbox, self.prev_det_bbox)
                bbox_ious = bbox_ious - term_DIoU

            # compute comprehensive score
            comp_scores = self.compute_comp_scores(cos_sim,
                                                   det_score.view(-1, 1),
                                                   bbox_ious,
                                                   mask_ious,
                                                   add_bbox_dummy=True,
                                                   bbox_dummy_iou=0.3,
                                                   match_coeff=cfg.match_coeff)

            match_likelihood, match_ids = torch.max(comp_scores, dim=1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            match_ids = match_ids
            det_obj_ids = torch.ones(n_dets, dtype=torch.int32) * (-1)
            best_match_scores = torch.ones(n_prev) * (-1)
            best_match_idx = torch.ones(n_prev) * (-1)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    det_obj_ids[idx] = self.prev_det_masks.size(0)
                    self.prev_track_embed = torch.cat([self.prev_track_embed, det_track_embed[idx][None]], dim=0)
                    self.prev_det_bbox = torch.cat((self.prev_det_bbox, det_bbox[idx][None]), dim=0)
                    if det_labels is not None:
                        self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                    self.prev_det_masks = torch.cat((self.prev_det_masks, det_masks[idx][None]), dim=0)
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
                        if (mask_ious[idx] > 0.3).sum() < 2:
                            if det_labels is not None:
                                self.prev_det_labels[obj_id] = det_labels[idx]
                            self.prev_track_embed[obj_id] = det_track_embed[idx]
                            self.prev_det_bbox[obj_id] = det_bbox[idx]
                            self.prev_det_masks[obj_id] = det_masks[idx]

        detection['box_ids'] = det_obj_ids
        detection['mask'] = det_masks_out
        if cfg.remove_false_inst:
            keep = det_obj_ids >= 0
            for k in detection:
                if k not in {'proto', 'bbox_idx', 'priors', 'loc_t'} and detection[k] is not None:
                    detection[k] = detection[k][keep]

        return detection

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, mask_ious, add_bbox_dummy=False, bbox_dummy_iou=0,
                            match_coeff=None):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                        device=torch.cuda.current_device()) * bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            mask_ious = torch.cat((bbox_iou_dummy, mask_ious), dim=1)

        if match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 4
            assert (len(match_coeff) == 4)
            return match_ll + match_coeff[0] * \
                   bbox_scores + match_coeff[1] * bbox_ious \
                   + match_coeff[2] * mask_ious
