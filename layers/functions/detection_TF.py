import torch
from ..box_utils import jaccard, mask_iou, crop
from utils import timer
from datasets import cfg


class Detect_TF(object):
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

        self.use_cross_class_nms = False
        self.use_fast_nms = True

    def __call__(self, net, candidate, is_output_candidate=False):
        """
        Args:
             net: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            candidate: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        with timer.env('Detect'):

            result = self.detect(candidate, is_output_candidate)
            if is_output_candidate:
                return result
            else:
                return [{'detection': result, 'net': net}]

    def detect(self, candidate, is_output_candidate=False):
        """ Perform nms for only the max scoring class that isn't background (class 0) """

        scores = candidate['conf'].t()[1:]
        boxes = candidate['box']
        centerness_scores = candidate['centerness']
        mask_coeff = candidate['mask_coeff']
        track = candidate['track']
        proto_data = candidate['proto']

        if boxes.size(0) == 0:
            if is_output_candidate:
                return candidate
            else:
                return {'box': boxes, 'mask_coeff': mask_coeff, 'class': torch.Tensor(), 'score': torch.Tensor()}

        if self.use_fast_nms:
            if self.use_cross_class_nms:
                idx_out, out_aft_nms = self.cc_fast_nms(boxes, mask_coeff, proto_data, track, scores, centerness_scores,
                                                        self.nms_thresh, self.top_k)
            else:
                idx_out, out_aft_nms = self.fast_nms(boxes, mask_coeff, proto_data, track, scores, centerness_scores,
                                                     self.nms_thresh, self.top_k)

        if is_output_candidate:
            candidate['conf'] = candidate['conf'][idx_out]
            candidate['box'] = boxes[idx_out]
            candidate['mask_coeff'] = mask_coeff[idx_out]
            candidate['track'] = candidate['track'][idx_out]
            if cfg.train_centerness:
                candidate['centerness'] = centerness_scores[idx_out]
                candidate['conf'] *= candidate['centerness'][:, None]
            return candidate
        else:
            return out_aft_nms

    def cc_fast_nms(self, boxes, masks_coeff, proto_data, track, conf, centerness_scores,
                    iou_threshold: float = 0.5, top_k: int = 200):
        # Collapse all the classes into 1
        scores, classes = conf.max(dim=0)

        if centerness_scores is not None:
            scores = scores * centerness_scores

        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        idx_out = None

        if len(idx) == 0:
            out_after_NMS = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor()}

        else:
            if cfg.nms_as_miou:
                det_masks = proto_data @ cfg.mask_proto_coeff_activation(masks_coeff.t())
                det_masks = cfg.mask_proto_mask_activation(det_masks)
                _, det_masks = crop(det_masks, boxes)
                det_masks = det_masks.permute(2, 0, 1).contiguous()  # [n_masks, h, w]
                det_masks = det_masks.gt(0.5).float()
                iou = mask_iou(det_masks[idx], det_masks[idx])
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
            scores = scores[idx_out]
            if centerness_scores is not None:
                centerness_scores = centerness_scores[idx_out]

            out_after_NMS = {'box': boxes, 'mask_coeff': masks_coeff, 'track': track, 'class': classes,
                             'score': scores, 'centerness': centerness_scores, 'proto': proto_data}
        return idx_out, out_after_NMS

    def fast_nms(self, boxes, masks_coeff, proto_data, track, conf, centerness_scores,
                 iou_threshold: float = 0.5, top_k: int = 200,
                 second_threshold: bool = True):

        if centerness_scores is not None:
            centerness_scores = centerness_scores.view(-1, 1)
            conf = conf * centerness_scores.t()

        scores, idx = conf.sort(1, descending=True)  # [num_classes, num_dets]
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        if len(idx) == 0:
            out_after_NMS = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor()}

        else:
            num_classes, num_dets = idx.size()
            boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
            masks_coeff = masks_coeff[idx.view(-1), :].view(num_classes, num_dets, -1)
            if cfg.train_track:
                track = track[idx.view(-1), :].view(num_classes, num_dets, -1)
            if centerness_scores is not None:
                centerness_scores = centerness_scores[idx.view(-1), :].view(num_classes, num_dets, -1)

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
            masks_coeff = masks_coeff[keep]
            if cfg.train_track:
                track = track[keep]
            if centerness_scores is not None:
                centerness_scores = centerness_scores[keep]
            scores = scores[keep]
            idx_out = idx[keep]

            # Only keep the top cfg.max_num_detections highest scores across all classes
            scores, idx = scores.sort(0, descending=True)
            idx = idx[:cfg.max_num_detections]
            scores = scores[:cfg.max_num_detections]

            classes = classes[idx]
            boxes = boxes[idx]
            masks_coeff = masks_coeff[idx]
            if cfg.train_track:
                track = track[idx]
            if centerness_scores is not None:
                centerness_scores = centerness_scores[idx]
            idx_out = idx_out[idx]

            out_after_NMS = {'box': boxes, 'mask_coeff': masks_coeff, 'track': track, 'class': classes,
                             'score': scores, 'centerness': centerness_scores, 'proto': proto_data}

        return idx_out, out_after_NMS

