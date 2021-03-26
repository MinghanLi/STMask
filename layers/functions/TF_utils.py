import torch
import torch.nn.functional as F
from layers.box_utils import jaccard, center_size, point_form, decode, crop, mask_iou
from layers.modules import correlate, bbox_feat_extractor
from layers.visualization import display_box_shift, display_correlation_map
from datasets import cfg
from utils import timer


class CandidateShift(object):
    def __init__(self):
        self.correlation_patch_size = cfg.correlation_patch_size
        self.correlation_selected_layer = cfg.correlation_selected_layer

    def __call__(self, net, ref_candidate, box_feat_next, img=None, img_meta=None):
        # extend the frames from time t to t+1
        ref_candidate_shift = self.candidate_shift(net, box_feat_next, ref_candidate, img,
                                                   img_meta=img_meta,
                                                   correlation_patch_size=self.correlation_patch_size)

        return ref_candidate_shift

    def candidate_shift(self, net, T2S_feat_next, ref_candidate,
                        img=None, img_meta=None, correlation_patch_size=11):
        """
        The function try to shift the candidates of reference frame to that of target frame.
        The most important step is to shift the bounding box of reference frame to that of target frame
        :param net: network
        :param T2S_feat_next: features of the last layer to predict bounding box on target frame
        :param ref_candidate: the candidate dictionary that includes 'box', 'conf', 'mask_coeff', 'track' items.
        :param correlation_patch_size: the output size of roialign
        :return: candidates on the target frame
        """
        ref_candidate_shift = {'T2S_feat': ref_candidate['T2S_feat'].clone(),
                               'proto': ref_candidate['proto'].clone()}

        if ref_candidate['box'].size(0) == 0:
            ref_candidate_shift['box'] = torch.tensor([])
            for k, v in ref_candidate.items():
                if k not in {'box', 'T2S_feat'}:
                    ref_candidate_shift[k] = v.clone()
        else:
            # we only use the features in the P3 layer
            T2S_feat_ref = ref_candidate['T2S_feat']
            x_corr = correlate(T2S_feat_ref, T2S_feat_next, patch_size=correlation_patch_size)
            # display_correlation_map(fpn_ref, img_meta, idx)
            concatenated_features = F.relu(torch.cat([x_corr, T2S_feat_ref, T2S_feat_next], dim=1))

            # extract features on the predicted bbox
            box_ref_c = center_size(ref_candidate['box'])
            # we use 1.2 box to crop features
            box_ref_crop = point_form(torch.cat([box_ref_c[:, :2],
                                                 torch.clamp(box_ref_c[:, 2:] * 1.2, min=0, max=1)], dim=1))
            bbox_feat_input = bbox_feat_extractor(concatenated_features, box_ref_crop, 7)
            loc_ref_shift, mask_coeff_shift = net.TemporalNet(bbox_feat_input)
            box_ref_shift = torch.cat([(loc_ref_shift[:, :2] * box_ref_c[:, 2:] + box_ref_c[:, :2]),
                                        torch.exp(loc_ref_shift[:, 2:]) * box_ref_c[:, 2:]], dim=1)
            box_ref_shift = point_form(box_ref_shift)

            ref_candidate_shift['box'] = box_ref_shift.clone()
            ref_candidate_shift['conf'] = ref_candidate['conf'].clone() * 0.8
            ref_candidate_shift['mask_coeff'] = ref_candidate['mask_coeff'].clone() + mask_coeff_shift

            for k, v in ref_candidate.items():
                if k not in {'box', 'conf', 'T2S_feat', 'proto', 'mask_coeff'}:
                    ref_candidate_shift[k] = v.clone()

        return ref_candidate_shift


def generate_candidate(predictions):
    # only support batch_size=1
    batch_Size = predictions['loc'].size(0)
    candidate = []
    prior_data = predictions['priors'].squeeze(0)
    for i in range(batch_Size):
        if cfg.train_boxes:
            loc_data = predictions['loc'][i]
        if cfg.train_centerness:
            centerness_data = predictions['centerness'][i]

        conf_data = predictions['conf'][i]
        mask_data = predictions['mask_coeff'][i]
        track_data = predictions['track'][i] if cfg.train_track else None

        candidate_cur = {'T2S_feat': predictions['T2S_feat'][i].unsqueeze(0)}

        with timer.env('Detect'):
            decoded_boxes = decode(loc_data, prior_data)

            conf_data = conf_data.t().contiguous()
            cur_scores = conf_data[1:, :]
            if cfg.train_centerness:
                cur_scores = cur_scores * centerness_data.t()
            conf_scores, _ = torch.max(cur_scores, dim=0)

            keep = (conf_scores > cfg.eval_conf_thresh)
            conf = conf_data[:, keep].t()
            boxes = decoded_boxes[keep, :]
            mask_coeff = mask_data[keep, :]
            track = track_data[keep, :] if cfg.train_track else None

            if cfg.train_centerness:
                centerness_scores = centerness_data[keep].view(-1)
                candidate_cur['centerness'] = centerness_scores

            candidate_cur['proto'] = predictions['proto'][i]
            candidate_cur['conf'] = conf
            candidate_cur['box'] = boxes
            candidate_cur['mask_coeff'] = mask_coeff
            candidate_cur['track'] = track

        candidate.append(candidate_cur)

    return candidate


def merge_candidates(candidate, ref_candidate_clip_shift):
    merged_candidate = {}
    for k, v in candidate.items():
        merged_candidate[k] = v.clone()

    for ref_candidate in ref_candidate_clip_shift:
        if ref_candidate['box'].nelement() > 0:
            for k, v in merged_candidate.items():
                if k not in {'proto', 'T2S_feat'}:
                    merged_candidate[k] = torch.cat([v.clone(), ref_candidate[k].clone()], dim=0)

    return merged_candidate


def cc_fast_nms(boxes, conf, DIoU_scores, proto_data, mask_coeff, iou_threshold: float = 0.3, top_k: int = 200):
    # Collapse all the classes into 1
    if boxes.size(0) > 0:
        scores, classes = conf[:, 1:].max(dim=1)
        if DIoU_scores is not None:
            scores = scores * DIoU_scores

        sorted_scores, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]

        det_masks = proto_data @ mask_coeff.t()
        det_masks = cfg.mask_proto_mask_activation(det_masks)
        _, det_masks = crop(det_masks, boxes)
        det_masks = det_masks.permute(2, 0, 1).contiguous()  # [n_masks, h, w]
        det_masks = det_masks.gt(0.5).float()

        if cfg.nms_as_miou:
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

    return idx_out


def compute_comp_scores(match_ll, bbox_scores, bbox_ious, mask_ious, add_bbox_dummy=False, bbox_dummy_iou=0,
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
        assert (len(match_coeff) == 3)
        return match_ll + match_coeff[0] * bbox_scores \
               + match_coeff[1] * mask_ious \
               + match_coeff[2] * bbox_ious
