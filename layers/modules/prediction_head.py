
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from datasets.config import cfg, mask_type

from .make_net import make_net
from .Featurealign import FeatureAlign
from utils import timer
from itertools import product
from math import sqrt
from mmcv.ops import DeformConv2d


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.
    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """

    def __init__(self, in_channels, out_channels=1024,
                 pred_aspect_ratios=None, pred_scales=None, parent=None, deform_groups=1):
        super().__init__()

        self.out_channels = out_channels
        self.num_classes = cfg.num_classes
        self.mask_dim = cfg.mask_dim
        self.num_priors = len(pred_aspect_ratios[0]) * len(pred_scales)
        self.embed_dim = cfg.embed_dim
        self.pred_aspect_ratios = pred_aspect_ratios
        self.pred_scales = pred_scales
        self.deform_groups = deform_groups
        self.parent = [parent]  # Don't include this in the state dict
        self.num_heads = cfg.num_heads
        if cfg.use_sipmask:
            self.mask_dim = self.mask_dim * cfg.sipmask_head

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim

        if parent is None:
            if cfg.extra_head_net is None:
                self.out_channels = in_channels
            else:
                self.upfeature, self.out_channels = make_net(in_channels, cfg.extra_head_net)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **cfg.head_layer_params)

            kernel_size = cfg.head_layer_params['kernel_size']
            if cfg.train_class:
                if cfg.use_cascade_pred and cfg.use_dcn_class:
                    self.conf_layer = FeatureAlign(self.out_channels,
                                                   self.num_priors * self.num_classes,
                                                   kernel_size=kernel_size,
                                                   deformable_groups=self.deform_groups,
                                                   use_pred_offset=cfg.use_pred_offset)
                else:
                    self.conf_layer = nn.Conv2d(self.out_channels, self.num_priors * self.num_classes,
                                                **cfg.head_layer_params)

            if cfg.train_track:
                if cfg.use_cascade_pred and cfg.use_dcn_track:
                    self.track_layer = FeatureAlign(self.out_channels,
                                                    self.num_priors * self.embed_dim,
                                                    kernel_size=kernel_size,
                                                    deformable_groups=self.deform_groups,
                                                    use_pred_offset=cfg.use_pred_offset)
                else:
                    self.track_layer = nn.Conv2d(out_channels, self.num_priors * self.embed_dim, **cfg.head_layer_params)

            if cfg.use_cascade_pred and cfg.use_dcn_mask:
                self.mask_layer = FeatureAlign(self.out_channels,
                                               self.num_priors * self.mask_dim,
                                               kernel_size=kernel_size,
                                               deformable_groups=self.deform_groups,
                                               use_pred_offset=cfg.use_pred_offset)
            else:
                self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **cfg.head_layer_params)

            if cfg.train_centerness:
                self.centerness_layer = nn.Conv2d(out_channels, self.num_priors, **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs,
                                            **cfg.head_layer_params)

            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra, self.track_extra = [make_extra(x) for x in cfg.extra_layers]

            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])
        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]

        bs, _, conv_h, conv_w = x.size()

        if cfg.extra_head_net is not None:
            x = src.upfeature(x)

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)
        track_x = src.track_extra(x)

        bbox = src.bbox_layer(bbox_x)
        if cfg.use_cascade_pred:
            offset = src.conv_offset(bbox.detach())
            # o1, o2, offset_mask = torch.chunk(offset_all, 3, dim=1)
            # offset = torch.cat((o1, o2), dim=1)
            # offset_mask = offset.new_ones(bs, int(offset.size(1)/2), conv_h, conv_w)
        bbox = bbox.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        if cfg.train_class:
            if cfg.use_cascade_pred and cfg.use_dcn_class:
                conf = src.conf_layer(conf_x, offset)
            else:
                conf = src.conf_layer(conf_x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        if cfg.train_track:
            if cfg.use_cascade_pred and cfg.use_dcn_track:
                track = src.track_layer(track_x, offset)
            else:
                track = src.track_layer(track_x)
            track = track.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.embed_dim)

        if cfg.use_cascade_pred and cfg.use_dcn_mask:
            mask = src.mask_layer(mask_x, offset)
        else:
            mask = src.mask_layer(mask_x)
        mask = mask.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)

        if cfg.train_centerness:
            centerness = src.centerness_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)
            centerness = torch.tanh(centerness)

        if cfg.use_mask_scoring:
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim),
                         mode='constant', value=0)

        priors = self.make_priors(conv_h, conv_w, x.device)
        preds = {'loc': bbox, 'conf': conf, 'mask_coeff': mask, 'priors': priors}

        if cfg.train_centerness:
            preds['centerness'] = centerness

        if cfg.train_track:
            preds['track'] = F.normalize(track, dim=-1)

        if cfg.use_mask_scoring:
            preds['score'] = score

        if cfg.use_instance_coeff:
            preds['inst'] = inst

        if cfg.temporal_fusion_module:
            preds['T2S_feat'] = x

        return preds

    def make_priors(self, conv_h, conv_w, device):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        with timer.env('makepriors'):
            prior_data = []
            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                for ars in self.pred_aspect_ratios:
                    for scale in self.pred_scales:
                        for ar in ars:
                            # [1, 1/2, 2]
                            ar = sqrt(ar)
                            r = scale / self.pred_scales[0] * 3
                            w = r * ar / conv_w
                            h = r / ar / conv_h

                            prior_data += [x, y, w, h]

            priors = torch.Tensor(prior_data, device=device).view(1, -1, 4).detach()
            priors.requires_grad = False

        return priors

