import torch
import torch.nn.functional as F
from ..box_utils import decode, jaccard, index2d, mask_iou, crop, center_size, DIoU
from ..mask_utils import generate_mask
from layers.mask_utils import generate_rel_coord
from utils import timer
from .TF_utils import compute_comp_scores

from datasets import cfg

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
        self.prev_det_masks_coeff = None
        self.prev_protos = None
        self.det_scores = None

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

        with timer.env('Track'):
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
            self.prev_det_masks_coeff = None
            self.prev_protos = None
            self.det_scores = None
            # self.prev_track = {}

        if detection['class'].nelement() == 0:
            detection['box_ids'] = torch.tensor([], dtype=torch.int64)
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
        mask_h, mask_w = proto_data.size()[:2]

        # get masks
        det_masks = generate_mask(proto_data, det_masks_coff, det_bbox, cfg.mask_proto_mask_activation)
        soft_crop = generate_rel_coord(det_bbox, mask_h, mask_w, sigma_scale=1.8)
        det_masks_out = soft_crop * det_masks
        det_masks = det_masks.gt(0.5).float()

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_det_bbox is None):
            det_obj_ids = torch.arange(det_bbox.size(0))
            # save bbox and features for later matching
            self.prev_det_bbox = det_bbox
            self.prev_track_embed = det_track_embed
            self.prev_det_labels = det_labels.view(-1)
            self.prev_det_masks = det_masks
            self.prev_det_masks_coeff = det_masks_coff
            self.prev_protos = proto_data.unsqueeze(0).repeat(n_dets, 1, 1, 1)
            self.prev_scores = det_score
            # self.prev_track = {i: det_track_embed[i].view(1, -1) for i in range(det_bbox.size(0))}

        else:

            assert self.prev_track_embed is not None
            n_prev = self.prev_det_bbox.size(0)
            # only support one image at a time
            cos_sim = det_track_embed @ self.prev_track_embed.t()  # [n_dets, n_prev], val in [-1, 1]
            cos_sim = torch.cat([torch.zeros(n_dets, 1), cos_sim], dim=1)
            cos_sim = (cos_sim + 1) / 2  # [0, 1]

            bbox_ious = jaccard(det_bbox, self.prev_det_bbox)
            # mask_ious = mask_iou(det_masks, self.prev_det_masks)
            mask_ious = []
            det_bbox = torch.clamp(det_bbox, min=0, max=1)

            for i in range(n_prev):
                det_masks_shift = self.prev_protos[i] @ det_masks_coff.t()
                det_masks_shift = cfg.mask_proto_mask_activation(det_masks_shift).permute(2, 0, 1).contiguous()
                det_masks_shift = soft_crop * det_masks_shift
                det_masks_shift = det_masks_shift.gt(0.5).float()  # [n_dets, h, w]
                mask_ious.append(mask_iou(det_masks_shift, self.prev_det_masks[i].unsqueeze(0)))  # [n_dets, 1]

            mask_ious = torch.cat(mask_ious, dim=1)

            if cfg.use_DIoU_in_comp_scores:
                term_DIoU = DIoU(det_bbox, self.prev_det_bbox)
                bbox_ious = bbox_ious - term_DIoU

            # compute comprehensive score
            comp_scores = compute_comp_scores(cos_sim,
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
                    self.prev_det_masks_coeff = torch.cat((self.prev_det_masks_coeff, det_masks_coff[idx][None]), dim=0)
                    self.prev_protos = torch.cat((self.prev_protos, proto_data[None]), dim=0)
                    self.prev_scores = torch.cat((self.prev_scores, det_score[idx][None]), dim=0)
                    # self.prev_track[self.prev_det_masks.size(0)-1] = det_track_embed[idx].view(1, -1)

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
                            self.prev_det_masks_coeff[obj_id] = det_masks_coff[idx]
                            self.prev_protos[obj_id] = proto_data
                            self.prev_scores[obj_id] = det_score[idx]
                            # self.prev_track[int(obj_id)] = torch.cat([self.prev_track[int(obj_id)], det_track_embed[idx][None]], dim=0)

            missed_inst_id = (best_match_idx == -1) * (self.prev_scores[:n_prev] > 0.75)
            if cfg.add_missed_masks and missed_inst_id.sum() > 0:
                missed_masks_coeff = self.prev_det_masks_coeff[:n_prev][missed_inst_id].t()
                missed_masks = proto_data @ missed_masks_coeff
                missed_masks = missed_masks.permute(2, 0, 1).contiguous()
                missed_masks = cfg.mask_proto_mask_activation(missed_masks)

                # use relative coord to remove pixels that are far away bbox
                bbox_missed = torch.clamp(self.prev_det_bbox[:n_prev][missed_inst_id], min=0, max=1)
                soft_crop = generate_rel_coord(bbox_missed, mask_h, mask_w, sigma_scale=2)
                missed_masks = soft_crop * missed_masks

                used_idx = missed_masks.gt(0.5).sum([1, 2]) > 5
                if used_idx.sum() > 0:
                    add_inst_idx = torch.arange(n_prev)[missed_inst_id][used_idx]
                    det_masks_out = torch.cat([det_masks_out, missed_masks[used_idx]], dim=0)
                    det_obj_ids = torch.cat([det_obj_ids, add_inst_idx.type(torch.int32)])
                    det_bbox = torch.cat([det_bbox, self.prev_det_bbox[:n_prev][add_inst_idx]], dim=0)
                    det_labels = torch.cat([det_labels, self.prev_det_labels[:n_prev][add_inst_idx]], dim=0)
                    det_score = torch.cat([det_score, self.prev_scores[add_inst_idx]], dim=0)
                    det_track_embed = torch.cat([det_track_embed, self.prev_track_embed[:n_prev][add_inst_idx]], dim=0)
                    det_masks_coff = torch.cat([det_masks_coff, self.prev_det_masks_coeff[:n_prev][add_inst_idx]],
                                               dim=0)

                    detection['box'] = det_bbox
                    detection['class'] = det_labels
                    detection['score'] = det_score
                    detection['track'] = det_track_embed
                    detection['mask_coeff'] = det_masks_coff
                    if cfg.train_centerness:
                        detection['centerness'] = torch.cat([detection['centerness'],
                                                             torch.ones(add_inst_idx.size())], dim=0)

        detection['box_ids'] = det_obj_ids
        detection['mask'] = det_masks_out
        if cfg.remove_false_inst:
            keep = det_obj_ids >= 0
            for k in detection:
                if k not in {'proto', 'bbox_idx', 'priors', 'loc_t'} and detection[k] is not None:
                    detection[k] = detection[k][keep]

        return detection