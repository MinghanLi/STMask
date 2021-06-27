import torch
import torch.nn.functional as F
from layers.box_utils import jaccard, center_size, point_form, decode, crop, mask_iou
from layers.mask_utils import generate_mask
from layers.modules import correlate, bbox_feat_extractor
from layers.visualization import display_box_shift, display_correlation_map_patch

from datasets import cfg
from utils import timer


def CandidateShift(net, ref_candidate, next_candidate, img=None, img_meta=None, display=False):
    """
    The function try to shift the candidates of reference frame to that of target frame.
    The most important step is to shift the bounding box of reference frame to that of target frame
    :param net: network
    :param next_candidate: features of the last layer to predict bounding box on target frame
    :param ref_candidate: the candidate dictionary that includes 'box', 'conf', 'mask_coeff', 'track' items.
    :return: candidates on the target frame
     """

    ref_candidate_shift = {}
    for k, v in next_candidate.items():
        if k in {'proto', 'fpn_feat', 'T2S_feat'}:
            ref_candidate_shift[k] = v.clone()

    # we only use the features in the P3 layer to perform correlation operation
    T2S_feat_ref, T2S_feat_next = ref_candidate['T2S_feat'], next_candidate['T2S_feat']
    fpn_feat_ref, fpn_feat_next = ref_candidate['fpn_feat'], next_candidate['fpn_feat']
    x_corr = correlate(fpn_feat_ref, fpn_feat_next, patch_size=cfg.correlation_patch_size)
    concatenated_features = F.relu(torch.cat([x_corr, T2S_feat_ref, T2S_feat_next], dim=1))

    box_ref = ref_candidate['box'].clone()
    feat_h, feat_w = fpn_feat_ref.size()[2:]
    bbox_feat_input = bbox_feat_extractor(concatenated_features, box_ref, feat_h, feat_w, 7)
    loc_ref_shift, mask_coeff_shift = net.TemporalNet(bbox_feat_input)
    box_ref_shift = decode(loc_ref_shift, center_size(box_ref))
    mask_coeff_ref_shift = ref_candidate['mask_coeff'].clone() + mask_coeff_shift
    masks_ref_shift = generate_mask(next_candidate['proto'], mask_coeff_ref_shift, box_ref_shift)

    # display = 1
    if display:
        # display_correlation_map_patch(bbox_feat_input[:, :121], img_meta)
        display_box_shift(box_ref, box_ref_shift, mask_shift=masks_ref_shift, img_meta=img_meta, img_gpu=img)

    ref_candidate_shift['box'] = box_ref_shift.clone()
    ref_candidate_shift['score'] = ref_candidate['score'].clone() * 0.95
    ref_candidate_shift['mask_coeff'] = mask_coeff_ref_shift.clone()
    ref_candidate_shift['mask'] = masks_ref_shift.clone()

    return ref_candidate_shift


def generate_candidate(predictions):
    batch_Size = predictions['loc'].size(0)
    candidate = []
    prior_data = predictions['priors'].squeeze(0)
    for i in range(batch_Size):
        loc_data = predictions['loc'][i]
        conf_data = predictions['conf'][i]

        candidate_cur = {'T2S_feat': predictions['T2S_feat'][i].unsqueeze(0),
                         'fpn_feat': predictions['fpn_feat'][i].unsqueeze(0)}

        with timer.env('Detect'):
            decoded_boxes = decode(loc_data, prior_data)

            conf_data = conf_data.t().contiguous()
            conf_scores, _ = torch.max(conf_data[1:, :], dim=0)

            keep = (conf_scores > cfg.eval_conf_thresh)
            candidate_cur['proto'] = predictions['proto'][i]
            candidate_cur['conf'] = conf_data[:, keep].t()
            candidate_cur['box'] = decoded_boxes[keep, :]
            candidate_cur['mask_coeff'] = predictions['mask_coeff'][i][keep, :]
            candidate_cur['track'] = predictions['track'][i][keep, :] if cfg.train_track else None
            if cfg.train_centerness:
                candidate_cur['centerness'] = predictions['centerness'][i][keep].view(-1)

        candidate.append(candidate_cur)

    return candidate


def merge_candidates(candidate, ref_candidate_clip_shift):
    merged_candidate = {}
    for k, v in candidate.items():
        merged_candidate[k] = v.clone()

    for ref_candidate in ref_candidate_clip_shift:
        if ref_candidate['box'].nelement() > 0:
            for k, v in merged_candidate.items():
                if k not in {'proto', 'T2S_feat', 'fpn_feat'}:
                    merged_candidate[k] = torch.cat([v.clone(), ref_candidate[k].clone()], dim=0)

    return merged_candidate


def compute_comp_scores(match_ll, bbox_scores, bbox_ious, mask_ious, label_delta, add_bbox_dummy=False, bbox_dummy_iou=0,
                        match_coeff=None):
    # compute comprehensive matching score based on matchig likelihood,
    # bbox confidence, and ious
    if add_bbox_dummy:
        bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                    device=torch.cuda.current_device()) * bbox_dummy_iou
        bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
        mask_ious = torch.cat((bbox_iou_dummy, mask_ious), dim=1)
        label_dummy = torch.ones(bbox_ious.size(0), 1,
                                 device=torch.cuda.current_device())
        label_delta = torch.cat((label_dummy, label_delta), dim=1)

    if match_coeff is None:
        return match_ll
    else:
        # match coeff needs to be length of 4
        assert (len(match_coeff) == 4)
        return match_ll + match_coeff[0] * bbox_scores \
               + match_coeff[1] * mask_ious \
               + match_coeff[2] * bbox_ious \
               + match_coeff[3] * label_delta


