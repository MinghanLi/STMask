# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, encode, decode, center_size, crop, crop_sipmask, jaccard, point_form
from .track_to_segment_head import bbox_feat_extractor
from ..track_utils import mask_iou
from ..visualization import display_correlation_map_patch
from ..mask_utils import generate_mask

from datasets import cfg, mask_type, activation_func


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes

        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

        if cfg.use_class_balanced_conf:
            self.class_instances = None
            self.total_instances = 0

    def forward(self, net, predictions, gt_bboxes, gt_labels, gt_masks, gt_ids):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.

            * Only if mask_type == lincomb
        """
        gt_bboxes_fold = sum(gt_bboxes, [])
        gt_labels_fold = sum(gt_labels, [])
        gt_masks_fold = sum(gt_masks, [])
        gt_ids_fold = sum(gt_ids, [])

        loc_data = predictions['loc']
        centerness_data = predictions['centerness'] if cfg.train_centerness else None
        score_data = predictions['score'] if cfg.use_mask_scoring else None

        conf_data = predictions['conf'] if cfg.train_class else None
        mask_data = predictions['mask_coeff']
        track_data = predictions['track'] if cfg.train_track else None

        # This is necessary for training on multiple GPUs because
        # DataParallel will cat the priors from each GPU together
        priors = predictions['priors']
        proto_data = predictions['proto']
        segm_data = predictions['segm'] if cfg.use_semantic_segmentation_loss else None

        losses, conf_t, ids_t, idx_t, pos_weights_per_img = self.multibox_loss(net, loc_data, conf_data,
                                                                               mask_data, centerness_data, score_data,
                                                                               proto_data, priors, segm_data,
                                                                               gt_bboxes_fold, gt_labels_fold,
                                                                               gt_masks_fold, gt_ids_fold)

        if cfg.temporal_fusion_module:
            loc_data_ref = loc_data[::2].detach()
            ids_t_ref = ids_t[::2]
            mask_coeff_ref = mask_data[::2].detach()
            proto_data_next = proto_data[1::2].detach()
            losses_shift = self.track_to_segment_loss(net, predictions['T2S_concat_feat'],
                                                      loc_data_ref, ids_t_ref, mask_coeff_ref, proto_data_next,
                                                      priors[0], gt_bboxes, gt_ids, gt_masks)
            losses.update(losses_shift)

        if cfg.train_track:
            losses['T'] = self.track_loss(pos_weights_per_img, track_data, conf_t, ids_t)

        for k in losses:
            if torch.isinf(losses[k]) or torch.isnan(losses[k]):
                print(k)

        return losses

    def split(self, x):
        x1, x2 = torch.split(x, [1, 1], dim=-1)
        return x1.squeeze(-1), x2.squeeze(-1)

    def multibox_loss(self, net, loc_data, conf_data, mask_data, centerness_data, score_data,
                      proto_data, priors, segm_data, gt_bboxes, gt_labels, gt_masks, gt_ids):
        batch_size, num_priors = loc_data.size()[0:2]

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_boxes_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()
        ids_t = loc_data.new(batch_size, num_priors).long()  # pids for tracking

        # assign positive samples
        for idx in range(batch_size):
            match(self.pos_threshold, self.neg_threshold, gt_bboxes[idx], gt_labels[idx], gt_ids[idx], priors[idx],
                  loc_data[idx], conf_data[idx], loc_t, conf_t, idx_t, ids_t, idx)

            gt_boxes_t[idx] = gt_bboxes[idx][idx_t[idx]]

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)
        ids_t = Variable(ids_t, requires_grad=False)

        pos = conf_t > 0
        total_num_pos = pos.data.sum()
        num_pos_per_img = []
        for idx in range(batch_size):
            num_pos_per_img.append(pos[idx].sum().long())

        losses = {}

        # design weights for all pos samples in per img
        IoU_split = torch.ones(int(total_num_pos.tolist()), device=loc_data.device).split(num_pos_per_img)
        pos_weights_per_img = [IoU_cur / torch.clamp(IoU_cur.sum(), min=1) for IoU_cur in IoU_split]
        pos_weights = torch.cat(pos_weights_per_img, dim=0)

        # Localization Loss (Smooth L1)
        if cfg.train_boxes:
            loc_p = loc_data[pos].view(-1, 4)
            loc_t = loc_t[pos].view(-1, 4)
            pos_priors = priors[pos].view(-1, 4)

            if cfg.use_boxiou_loss:
                decoded_loc_p = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
                DIoU = self.get_DIoU(decoded_loc_p, gt_boxes_t[pos])
                losses['BIoU'] = (pos_weights * (1-DIoU)).sum() * cfg.bboxiou_alpha
            else:
                losses['B'] = (pos_weights.view(-1, 1) * F.smooth_l1_loss(loc_p, loc_t,
                                                                          reduction='none')).sum() * cfg.bbox_alpha

        # Mask Loss
        if cfg.train_masks:
            if cfg.mask_type == mask_type.lincomb:
                ret = self.lincomb_mask_loss(pos_weights_per_img, pos, idx_t, ids_t, loc_data, mask_data, score_data, priors,
                                             proto_data, gt_masks, gt_boxes_t, gt_labels)
                if cfg.use_maskiou:
                    loss, maskiou_targets = ret
                else:
                    loss = ret
                losses.update(loss)

                if cfg.mask_proto_loss is not None:
                    if cfg.mask_proto_loss == 'l1':
                        losses['P'] = torch.mean(torch.abs(proto_data)) / self.l1_expected_area * self.l1_alpha
                    elif cfg.mask_proto_loss == 'disj':
                        losses['P'] = -torch.mean(torch.max(F.log_softmax(proto_data, dim=-1), dim=-1)[0])

        # Confidence loss
        if cfg.train_class:
            if cfg.use_sigmoid_focal_loss:
                loss_c = self.focal_conf_sigmoid_loss(conf_data, conf_t)
            else:
                loss_c = self.ohem_conf_loss(pos_weights, conf_data, conf_t, centerness_data, loc_data, priors, gt_boxes_t)
            losses.update(loss_c)

        # Mask IoU Loss
        if cfg.use_maskiou and maskiou_targets is not None:
            losses['I'] = self.mask_iou_loss(net, maskiou_targets)

        # These losses also don't depend on anchors
        if cfg.use_semantic_segmentation_loss:
            losses['S'] = self.semantic_segmentation_loss(segm_data, gt_masks, gt_labels)

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        # loss[P] and loss [IC] have been divided the number of positives
        for k in losses:
            losses[k] /= batch_size
        # Loss Key:
        #  - B: Box Localization Loss
        #  - C: Class Confidence Loss
        #  - M: Mask Loss
        #  - T: Tracking Loss
        #  - T_acc: Tracking accuracy
        #  - P: Prototype Loss
        #  - D: Coefficient Diversity Loss
        #  - E: Class Existence Loss
        #  - S: Semantic Segmentation Loss
        return losses, conf_t, ids_t, idx_t, pos_weights_per_img

    def get_DIoU(self, pos_pred_boxes, pos_gt_boxes):
        # calculate bbox IoUs
        IoU = jaccard(pos_gt_boxes, pos_pred_boxes).diag().view(-1)

        # calculate the diagonal length of the smallest enclosing box
        x_label = torch.cat([pos_pred_boxes[:, ::2], pos_gt_boxes[:, ::2]], dim=1)
        y_label = torch.cat([pos_pred_boxes[:, 1::2], pos_gt_boxes[:, 1::2]], dim=1)
        c2 = (x_label.max(dim=1)[0] - x_label.min(dim=1)[0])**2 + (y_label.max(dim=1)[0] - y_label.min(dim=1)[0])**2
        c2 = torch.clamp(c2, min=1e-10)

        # get the distance of centres between pred_bbox and gt_bbox
        pos_pred_boxes_c = pos_pred_boxes[:, :2] / 2 + pos_pred_boxes[:, 2:] / 2
        pos_gt_boxes_c = pos_gt_boxes[:, :2] / 2 + pos_gt_boxes[:, 2:] / 2
        d2 = ((pos_pred_boxes_c - pos_gt_boxes_c)**2).sum(dim=1)

        # DIoU
        DIoU = IoU - d2/c2

        return DIoU

    def track_to_segment_loss(self, net, concat_feat, loc_ref, ids_t_ref, mask_data_ref, proto_data_next,
                              priors, gt_bboxes, gt_ids, gt_masks, interpolation_mode='bilinear'):
        feat_h, feat_w = concat_feat.size()[2:]
        loss_B_shift, loss_mask_shift = torch.zeros(1), torch.zeros(1)
        bs = loc_ref.size(0)
        n_pos = 0
        for i in range(bs):
            ids_t_ref_cur = ids_t_ref[i].clone()
            # select instances that exists in both two frames
            gt_ids_ref = gt_ids[i][0]
            gt_ids_next = gt_ids[i][1]
            gt_bboxes_reg = torch.zeros_like(loc_ref[i])
            for j, id in enumerate(gt_ids_ref):
                if id in gt_ids_next:
                    keep_inst = ids_t_ref_cur == id
                    # calculate tracking regression values between two frames of bounding boxes
                    gt_bbox_ref = gt_bboxes[i][0][j].view(1, 4)
                    gt_bbox_next = gt_bboxes[i][1][gt_ids_next == id].view(1, 4)
                    gt_bboxes_reg_cur = encode(gt_bbox_next, center_size(gt_bbox_ref))
                    gt_bboxes_reg[keep_inst] = gt_bboxes_reg_cur.repeat(keep_inst.sum(), 1)
                else:
                    ids_t_ref_cur[ids_t_ref[i] == id] = 0

            gt_bboxes_reg = Variable(gt_bboxes_reg, requires_grad=False)
            pos = ids_t_ref_cur > 0

            if pos.sum() == 0:
                continue

            n_pos += pos.sum()
            loc_p = loc_ref[i][pos].view(-1, 4).detach()
            priors_p = priors[pos].view(-1, 4)
            bbox_p = decode(loc_p, priors_p, cfg.use_yolo_regressors)

            # extract features on the predicted bbox
            bbox_feats = bbox_feat_extractor(concat_feat[i].unsqueeze(0), bbox_p, feat_h, feat_w, 7)
            # display_correlation_map_patch(bbox_feats[:, :121])
            bbox_reg, shift_mask_coeff = net.TemporalNet(bbox_feats)
            pre_loss_B = F.smooth_l1_loss(bbox_reg, gt_bboxes_reg[pos], reduction='none').sum(1)
            loss_B_shift += pre_loss_B.mean()

            if cfg.maskshift_loss:
                # create mask_gt and bbox_gt in the reference frame
                cur_pos_ids_t = ids_t_ref_cur[pos]
                pos_idx_t = [gt_ids[i][1].tolist().index(id) for id in cur_pos_ids_t]
                bbox_t_next = gt_bboxes[i][1][pos_idx_t]
                mask_t_next = gt_masks[i][1][pos_idx_t].float()

                tar_mask_coeff = mask_data_ref[i, pos] + shift_mask_coeff

                if cfg.mask_proto_crop:
                    pred_masks_soft = generate_mask(proto_data_next[i], tar_mask_coeff, bbox_t_next)  # [n, h, w]
                else:
                    pred_masks_soft = generate_mask(proto_data_next[i], tar_mask_coeff)

                mask_gt_h, mask_gt_w = mask_t_next.size()[1:]
                upsampled_pred_masks = F.interpolate(pred_masks_soft.unsqueeze(0),
                                                     (mask_gt_h, mask_gt_w),
                                                     mode=interpolation_mode, align_corners=False).squeeze(0)

                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    pre_loss = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks, 0, 1), mask_t_next,
                                                      reduction='none')
                else:
                    pre_loss = F.smooth_l1_loss(upsampled_pred_masks, mask_t_next, reduction='none')

                if cfg.mask_proto_crop:
                    pos_get_csize = center_size(bbox_t_next)
                    gt_box_width = pos_get_csize[:, 2] * mask_gt_w
                    gt_box_height = pos_get_csize[:, 3] * mask_gt_h
                    pre_loss = pre_loss.sum(dim=(1, 2)) / gt_box_width / gt_box_height
                    loss_mask_shift += torch.mean(pre_loss)
                else:
                    loss_mask_shift += torch.mean(pre_loss) / mask_gt_h / mask_gt_w

        losses = {'B_shift': loss_B_shift / bs * cfg.boxshift_alpha}
        if cfg.maskshift_loss:
            losses['M_shift'] = loss_mask_shift / bs * cfg.maskshift_alpha

        return losses

    def track_loss(self, pos_weights_per_img, track_data, conf_t, ids_t=None):
        pos = conf_t > 0  # [2, n]

        pos_track_data = track_data[pos]
        pos_ids_t = ids_t[pos]
        cos_sim = pos_track_data @ pos_track_data.t()  # [n_pos, n_ref_pos]
        inst_eq = (pos_ids_t.view(-1, 1) == pos_ids_t.view(1, -1)).float()
        cur_weights = torch.cat(pos_weights_per_img)
        loss_weights = cur_weights.view(-1, 1) @ cur_weights.view(1, -1)
        loss_weights.triu_(diagonal=1)

        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2
        cos_sim.triu_(diagonal=1)

        # If they are the same instance, use cosine distance, else use consine similarity
        # loss += ((1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)).sum() / (cos_sim.size(0) * cos_sim.size(1))
        # pos: -log(cos_sim), neg: -log(1-cos_sim)
        cos_sim_diff = torch.clamp(1 - cos_sim, min=1e-10)
        loss_m = -1*(inst_eq * torch.clamp(cos_sim, min=1e-10).log() + (1 - inst_eq) * cos_sim_diff.log())
        loss_m.triu_(diagonal=1)
        loss = (loss_m * loss_weights).sum()

        return loss * cfg.track_alpha / loss_weights.sum()

    def instance_conf_loss(self, pos_weights_per_img, ref_pos_weights_per_img, conf_data, ref_conf_data,
                           conf_t, ref_conf_t, ids_t, ref_ids_t):
        batch_size, num_priors, n_classes = conf_data.size()
        loss = 0

        for idx in range(batch_size):
            # calculate the instance confident in a single frame: the mean of the bboxes with same instance id
            inst_pt, inst_conf_t, inst_id = self.get_inst_conf(pos_weights_per_img[idx], conf_data[idx],  conf_t[idx], ids_t[idx])
            ref_inst_pt, ref_inst_conf_t, ref_inst_id = self.get_inst_conf(ref_pos_weights_per_img[idx], ref_conf_data[idx],
                                                                           ref_conf_t[idx], ref_ids_t[idx])

            # get the instance confident cross multi frames by mean them in different frames
            all_inst_id = torch.unique(torch.cat([inst_id, ref_inst_id]))
            all_inst_pt, all_inst_conf_t = [], []
            for i, id in enumerate(all_inst_id):
                used_idx = inst_id == id
                used_ref_idx = ref_inst_id == id
                if id in inst_id and id in ref_inst_id:
                    inst_pt_id = (inst_pt[used_idx] + ref_inst_pt[used_ref_idx]) / 2.0
                    inst_conf_t_id = inst_conf_t[used_idx]
                elif id in inst_id:
                    inst_pt_id = inst_pt[used_idx]
                    inst_conf_t_id = inst_conf_t[used_idx]
                else:
                    inst_pt_id = ref_inst_pt[used_ref_idx]
                    inst_conf_t_id = ref_inst_conf_t[used_ref_idx]
                all_inst_pt.append(inst_pt_id)
                all_inst_conf_t.append(inst_conf_t_id)

            # get the confident loss of instances in the a short subset (includes 2 frames here)
            loss += F.cross_entropy(torch.cat(all_inst_pt, dim=0), torch.cat(all_inst_conf_t), reduction='sum') / all_inst_id.size(0)

        return loss / batch_size

    def get_inst_conf(self, pos_weights, conf_data,  conf_t, ids_t):
        # put bboxes with same instances ids into a set, called instance confidence
        pos = (conf_t > 0)
        pos_pt = F.softmax(conf_data[pos], -1)  # [n_pos, 41]
        pos_ids_t = ids_t[pos]
        pos_conf_t = conf_t[pos]

        # merge those bboxes with same instances into a subset
        # use the conf data's mean of all bboxes in the subset as the conf data of the instance in the subset
        inst_id, inv_id = torch.unique(pos_ids_t, sorted=True, return_inverse=True)
        inst_pt = [pos_pt[inv_id == i].sum(dim=0) / (inv_id == i).sum() for i in range(len(inst_id))]
        inst_conf_t = [pos_conf_t[inv_id == i][0].view(1) for i in range(len(inst_id))]

        return torch.stack(inst_pt, dim=0), torch.cat(inst_conf_t), inst_id

    def select_neg_bboxes(self, conf_data, conf_t):
        if len(conf_t.size()) == 2 or len(conf_data.size()) == 3:
            conf_t = conf_t.view(-1)                          # [batch_size*num_priors]
            conf_data = conf_data.view(-1, self.num_classes)  # [batch_size*num_priors, num_classes]

        # Compute max conf across batch for hard negative mining
        if cfg.ohem_use_most_confident:
            conf_data = F.softmax(conf_data, dim=1)
            loss_c, _ = conf_data[:, 1:].max(dim=1)
        else:
            loss_c = log_sum_exp(conf_data) - conf_data[:, 0]

        # Hard Negative Mining
        num_pos = (conf_t > 0).sum()
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=conf_t.size()[0] - 1)
        loss_c[conf_t > 0] = 0  # filter out pos samples and neutrals
        loss_c[conf_t < 0] = 0  # filter out pos samples and neutrals
        _, loss_idx = loss_c.sort(descending=True)
        neg = torch.zeros(conf_t.size(), device=conf_t.get_device())
        neg[loss_idx[:num_neg]] = 1
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[conf_t > 0] = 0  # filter out pos samplers and neutrals
        neg[conf_t < 0] = 0  # filter out pos samplers and neutrals

        return neg

    def ohem_conf_loss(self, pos_weights, conf_data, conf_t, centerness_data, loc, priors, gt_boxes_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        batch_size = conf_t.size(0)
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1,  self.num_classes)

        pos = (conf_t > 0).float()
        neg = self.select_neg_bboxes(conf_data, conf_t)
        keep = (pos + neg).gt(0)
        use_conf_t = conf_t[keep]
        use_conf_data = conf_data[keep]
        num_neg = (neg > 0).sum()
        neg_weights = torch.ones(num_neg, device=pos_weights.device) / num_neg * self.negpos_ratio * batch_size
        loss_weights = torch.cat([pos_weights, neg_weights])

        loss = F.cross_entropy(use_conf_data, use_conf_t, reduction='none')
        losses = {'C': cfg.conf_alpha * (loss_weights * loss).sum() / (self.negpos_ratio + 1)}

        if centerness_data is not None:
            pos = pos.gt(0)
            decoded_loc = decode(loc.view(-1, 4)[pos], priors.view(-1, 4)[pos], cfg.use_yolo_regressors)
            DIoU = self.get_DIoU(decoded_loc, gt_boxes_t.view(-1, 4)[pos])
            loss_cn = F.smooth_l1_loss(centerness_data.view(-1)[pos], DIoU, reduction='none')
            losses['center'] = cfg.center_alpha * (pos_weights * loss_cn).sum()

        return losses

    def focal_conf_loss(self, conf_data, conf_t):
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, self.num_classes)
        pos = (conf_t > 0).float()
        neg = self.select_neg_bboxes(conf_data, conf_t)
        keep = (pos + neg).gt(0)

        # Confidence Loss Including Positive and Negative Examples
        # adapted from focal loss to reduce the effect of class unbalance
        conf_p = conf_data[keep].view(-1, self.num_classes)
        logpt = F.log_softmax(conf_p, dim=-1)
        use_conf_t = conf_t[keep].view(-1)
        logpt = logpt.gather(1, use_conf_t.unsqueeze(-1)).view(-1)
        pt = logpt.exp()
        at = (1 - cfg.focal_loss_alpha) * pos.float() + cfg.focal_loss_alpha * neg.float()
        at = at[keep].view(-1)
        loss_c = -1 * at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        return cfg.conf_alpha * loss_c.sum()

    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """

        bs, _, num_classes = conf_data.size()

        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(-1, num_classes)  # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0  # can't mask with -1, so filter that out

        # Compute a one-hot embedding of conf_t
        # From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
        conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
        conf_pm_t = conf_one_t * 2 - 1  # -1 if background, +1 if forground for specific class

        logpt = F.logsigmoid(conf_data * conf_pm_t)  # note: 1 - sigmoid(x) = sigmoid(-x)
        pt = logpt.exp()

        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (1 - conf_one_t)
        at[..., 0] = 0  # Set alpha for the background class to 0 because sigmoid focal loss doesn't use it

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(dim=-1)

        losses = {'C': cfg.conf_alpha * loss.sum() / keep.sum() * bs}

        return losses

    def coeff_sparse_loss(self, coeffs):
        """
        coeffs:  should be size [num_pos, num_coeffs]
        """
        torch.abs(coeffs).sum(1)

    def coeff_diversity_loss(self, weights, coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        """
        num_pos = coeffs.size(0)
        instance_t = instance_t.view(-1)  # juuuust to make sure

        coeffs_norm = F.normalize(coeffs, dim=1)
        cos_sim = torch.mm(coeffs_norm, coeffs_norm.t())

        inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim)).float()

        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2

        # If they're the same instance, use cosine distance, else use cosine similarity
        cos_sim_diff = torch.clamp(1 - cos_sim, min=1e-10)
        loss = -1 * (torch.clamp(cos_sim, min=1e-10).log() * inst_eq + cos_sim_diff.log() * (1 - inst_eq))
        weights = weights.view(-1, 1) * weights.view(1, -1)

        # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
        # and all the losses will be divided by num_pos at the end, so just one extra time.
        return cfg.mask_proto_coeff_diversity_alpha * (weights * loss).sum()

    def lincomb_mask_loss(self, pos_weights_per_img, pos, idx_t, ids_t, loc_data, mask_data, score_data,
                          priors, proto_data, masks_gt, gt_box_t, labels_gt, interpolation_mode='bilinear'):
        process_gt_bboxes = cfg.mask_proto_normalize_emulate_roi_pooling or cfg.mask_proto_crop

        loss_m, loss_miou = 0, 0
        loss_d = 0  # Coefficient diversity loss
        proto_coef_clip, pos_ids_clip, weights_clip = [], [], []
        maskiou_t_list = []
        maskiou_net_input_list = []
        label_t_list = []

        for idx in range(mask_data.size(0)):
            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            pos_ids_t = ids_t[idx, cur_pos]
            pos_pred_box = decode(loc_data[idx, cur_pos], priors[idx, cur_pos], cfg.use_yolo_regressors).detach()
            pos_pred_box = center_size(pos_pred_box)
            pos_pred_box[:, 2:] *= 1.2
            pos_pred_box = point_form(pos_pred_box)
            pos_pred_box = torch.clamp(pos_pred_box, min=1e-5, max=1)

            if process_gt_bboxes:
                # Note: this is in point-form
                if cfg.mask_proto_crop_with_pred_box:
                    pos_gt_box_t = pos_pred_box
                else:
                    pos_gt_box_t = gt_box_t[idx, cur_pos]

            if pos_idx_t.size(0) == 0:
                continue

            mask_t = masks_gt[idx][pos_idx_t].float()       # [n, h, w]
            label_t = labels_gt[idx][pos_idx_t]

            proto_masks = proto_data[idx]                   # [mask_h, mask_w, 32]
            proto_coef = mask_data[idx, cur_pos, :]         # [num_pos, 32]
            if cfg.use_mask_scoring:
                mask_scores = score_data[idx, cur_pos, :]
            if cfg.mask_proto_coeff_diversity_loss:
                proto_coef_clip.append(proto_coef)
                pos_ids_clip.append(pos_ids_t)
                weights_clip.append(pos_weights_per_img[idx])
                if (idx + 1) % 2 == 0:
                    proto_coef_clip = torch.cat(proto_coef_clip, dim=0)
                    pos_ids_clip = torch.cat(pos_ids_clip, dim=0)
                    weights_clip = torch.cat(weights_clip, dim=0)
                    loss_d += self.coeff_diversity_loss(weights_clip, proto_coef_clip, pos_ids_clip)
                    proto_coef_clip, pos_ids_clip, weights_clip = [], [], []

            if cfg.mask_proto_crop:
                pred_masks_soft = generate_mask(proto_masks, proto_coef, pos_gt_box_t)  # [n_pos, mask_h, mask_w]
            else:
                pred_masks_soft = generate_mask(proto_masks, proto_coef)
            mask_gt_h, mask_gt_w = mask_t.size()[1:]
            upsampled_pred_masks = F.interpolate(pred_masks_soft.unsqueeze(0),
                                                 (mask_gt_h, mask_gt_w),
                                                 mode=interpolation_mode, align_corners=False).squeeze(0)

            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks, 0, 1), mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(upsampled_pred_masks, mask_t, reduction='none')

            if cfg.mask_proto_crop:
                pos_get_csize = center_size(pos_gt_box_t)
                gt_box_width = pos_get_csize[:, 2] * mask_gt_w
                gt_box_width = torch.clamp(gt_box_width, min=1)
                gt_box_height = pos_get_csize[:, 3] * mask_gt_h
                gt_box_height = torch.clamp(gt_box_height, min=1)
                pre_loss = pre_loss.sum(dim=(1, 2)) / gt_box_width / gt_box_height
                loss_m += torch.sum(pos_weights_per_img[idx] * pre_loss)
            else:
                loss_m += torch.sum(pos_weights_per_img[idx] * pre_loss) / mask_gt_h / mask_gt_w

            if cfg.use_maskiou_loss:
                # calculate
                tmp_pred_masks = upsampled_pred_masks.view(-1, mask_gt_h * mask_gt_w).gt(0.5).float()
                tmp_mask_t = mask_t.view(-1, mask_gt_h * mask_gt_w)
                intersection = (tmp_pred_masks * tmp_mask_t).sum(dim=-1)
                union = (tmp_pred_masks.sum(dim=-1) + tmp_mask_t.sum(dim=-1)) - intersection
                union = torch.clamp(union, min=1e-10)
                loss_miou += (1 - intersection / union).sum()    # [n_pos]

            if cfg.use_maskiou:
                maskiou_net_input = upsampled_pred_masks.unsqueeze(1)        # [n, 1, h, w]
                upsampled_pred_masks = upsampled_pred_masks.gt(0.5).float()
                maskiou_t = self._mask_iou(upsampled_pred_masks, mask_t)

                maskiou_net_input_list.append(maskiou_net_input)
                maskiou_t_list.append(maskiou_t)
                label_t_list.append(label_t)

        losses = {'M': loss_m * cfg.mask_alpha}

        if cfg.use_maskiou_loss:
            losses['MIoU'] = loss_miou * cfg.maskiou_alpha

        if cfg.mask_proto_coeff_diversity_loss:
            losses['D'] = loss_d

        if cfg.use_maskiou:
            # discard_mask_area discarded every mask in the batch, so nothing to do here
            if len(maskiou_t_list) == 0:
                return losses, None

            maskiou_t = torch.cat(maskiou_t_list)
            label_t = torch.cat(label_t_list)
            maskiou_net_input = torch.cat(maskiou_net_input_list)

            return losses, [maskiou_net_input, maskiou_t, label_t]

        return losses

    def _mask_iou(self, mask1, mask2):
        intersection = torch.sum(mask1*mask2, dim=(0, 1))
        area1 = torch.sum(mask1, dim=(0, 1))
        area2 = torch.sum(mask2, dim=(0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def mask_iou_loss(self, net, maskiou_targets):
        maskiou_net_input, maskiou_t, label_t = maskiou_targets

        maskiou_p = net.maskiou_net(maskiou_net_input)

        label_t = label_t[:, None]
        maskiou_p = torch.gather(maskiou_p, dim=1, index=label_t).view(-1)

        loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='sum')

        return loss_i * cfg.maskiou_alpha

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, interpolation_mode='bilinear'):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        # mask_h, mask_w = 2*mask_h, 2*mask_w
        loss_s = 0

        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]-1

            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_t[idx].float().unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                # cur_segment = F.interpolate(cur_segment.unsqueeze(0), (mask_h, mask_w),
                #                             mode=interpolation_mode, align_corners=False).squeeze(0)

                # Construct Semantic Segmentation
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(mask_t[idx].size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(segment_t[cur_class_t[obj_idx]],
                                                                downsampled_masks[obj_idx])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')

        return loss_s / mask_h / mask_w * cfg.semantic_segmentation_alpha
