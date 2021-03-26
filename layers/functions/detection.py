import torch
import torch.nn.functional as F
from ..box_utils import decode, jaccard, mask_iou, crop
from utils import timer

from datasets import cfg

import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)

from utils.cython_nms import nms as cnms


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        
        self.use_cross_class_nms = True
        self.use_fast_nms = True

    def __call__(self, predictions, net):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        if cfg.train_boxes:
            loc_data   = predictions['loc']
        if cfg.train_centerness:
            centerness_data = predictions['centerness']
        else:
            centerness_data = None

        conf_data  = predictions['conf'] if cfg.train_class else None
        mask_data  = predictions['mask_coeff']
        track_data = predictions['track'] if cfg.train_track else None
        prior_data = predictions['priors'].squeeze(0)
        proto_data = predictions['proto']

        inst_data  = predictions['inst'] if 'inst' in predictions else None

        out = []

        with timer.env('Detect'):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)

            if cfg.train_class:
                conf_data = conf_data.view(batch_size, num_priors, -1).transpose(2, 1).contiguous()

            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)

                result = self.detect(batch_idx, conf_data, decoded_boxes, centerness_data,
                                     mask_data, track_data, proto_data, inst_data)

                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]
                    if len(result['bbox_idx']) != 0:
                        result['priors'] = prior_data[result['bbox_idx']]
                
                out.append({'detection': result, 'net': net})
        
        return out

    def detect(self, batch_idx, conf_preds, decoded_boxes, centerness_data,
               mask_data, track_data, proto_data, inst_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        assert cfg.train_class or cfg.train_clip_class, \
            'The training process should include train_class or train_clip_class.'
        if cfg.train_class:
            cur_scores = conf_preds[batch_idx, 1:, :]
            conf_scores, _ = torch.max(cur_scores, dim=0)

        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]

        if cfg.train_centerness:
            centerness_scores = centerness_data[batch_idx, keep].view(-1)
        else:
            centerness_scores = None

        boxes = decoded_boxes[keep, :]
        masks_coeff = mask_data[batch_idx, keep, :]
        track = track_data[batch_idx, keep, :] if cfg.train_track else None
        proto_data = proto_data[batch_idx]

        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]
    
        if boxes.size(0) == 0:
            return {'box': boxes, 'mask_coeff': masks_coeff, 'class': torch.Tensor(), 'score': torch.Tensor(),
                    'bbox_idx': torch.Tensor()}
        
        if self.use_fast_nms:
            if self.use_cross_class_nms:
                boxes_aft_nms, masks_aft_nms, track_aft_nms, classes_aft_nms, scores_aft_nms, \
                centerness_score_aft_nms, idx_out = self.cc_fast_nms(boxes, masks_coeff, proto_data,
                                                                     track, scores, centerness_scores,
                                                                     self.nms_thresh, self.top_k)
            else:
                boxes_aft_nms, masks_aft_nms, track_aft_nms, classes_aft_nms, scores_aft_nms, idx_out = \
                    self.fast_nms(boxes, masks_coeff, track, scores, self.nms_thresh, self.top_k)
        else:
            boxes_aft_nms, masks_aft_nms, track_aft_nms, classes_aft_nms, scores_aft_nms = \
                self.traditional_nms(boxes, masks_coeff, track, scores, self.nms_thresh, self.conf_thresh)

        idx = torch.arange(len(keep))[keep][idx_out]

        return {'box': boxes_aft_nms, 'mask_coeff': masks_aft_nms, 'track': track_aft_nms, 'class': classes_aft_nms,
                'score': scores_aft_nms, 'centerness': centerness_score_aft_nms, 'bbox_idx': idx}

    def cc_fast_nms(self, boxes, masks_coeff, proto_data, track, conf, centerness_scores,
                    iou_threshold: float = 0.5, top_k: int = 200):
        # Collapse all the classes into 1
        scores, classes = conf.max(dim=0)

        if centerness_scores is not None:
            scores = scores * centerness_scores

        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]

        if cfg.nms_as_miou:
            det_masks = proto_data @ masks_coeff[idx].t()
            det_masks = cfg.mask_proto_mask_activation(det_masks)
            _, det_masks = crop(det_masks.squeeze(0), boxes[idx])
            det_masks = det_masks.permute(2, 0, 1).contiguous()  # [n_masks, h, w]
            det_masks = det_masks.gt(0.5).float()
            iou = mask_iou(det_masks, det_masks)
        else:
            # Compute the pairwise IoU between the boxes
            boxes_idx = boxes[idx]
            iou = jaccard(boxes_idx, boxes_idx)

        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        iou = torch.triu(iou, diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        iou_max, _ = torch.max(iou, dim=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        idx_out = idx[iou_max <= iou_threshold]

        boxes = boxes[idx_out]
        masks_coeff = masks_coeff[idx_out]
        if track is not None:
            track = track[idx_out]
        if classes is not None:
            classes = classes[idx_out] + 1
        scores = scores[idx_out]  # conf.max(dim=0)[0][idx_out]  # scores[idx_out]
        if centerness_scores is not None:
            centerness_scores = centerness_scores[idx_out]

        return boxes, masks_coeff, track, classes, scores, centerness_scores, idx_out

    def coefficient_nms(self, coeffs, scores, cos_threshold=0.9, top_k=400):
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        coeffs_norm = F.normalize(coeffs[idx], dim=1)

        # Compute the pairwise cosine similarity between the coefficients
        cos_similarity = coeffs_norm @ coeffs_norm.t()
        
        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        cos_similarity.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the cos similarity matrix along the columns, each column will represent the
        # maximum cosine similarity between this element and every element with a higher
        # score than this element.
        cos_max, _ = torch.max(cos_similarity, dim=0)

        # Now just filter out the ones higher than the threshold
        idx_out = idx[cos_max <= cos_threshold]
        
        return idx_out, idx_out.size(0)

    def fast_nms(self, boxes, masks, track, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=True):
        scores, idx = scores.sort(1, descending=True)  # [num_classes, num_dets]

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)
        if cfg.train_track:
            track = track[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)  # [num_classes, num_dets, num_dets]
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)  # [num_classes, num_dets]

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)  # [num_classes, num_dets]

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        if cfg.train_track:
            track = track[keep]
        scores = scores[keep]
        idx_out = idx[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
        if cfg.train_track:
            track = track[idx]
        idx_out = idx_out[idx]

        return boxes, masks, track, classes+1, scores, idx_out

    def traditional_nms(self, boxes, masks, track, scores, iou_threshold=0.5, conf_thresh=0.05):
        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.size(0) == 0:
                continue
            
            preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], track[idx], classes, scores
