import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.config import cfg, mask_type
from .make_net import make_net
from .Featurealign import FeatureAlign
from utils import timer
from itertools import product
from mmcv.ops import DeformConv2d


class PredictionModule_FC(nn.Module):
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

    def __init__(self, in_channels, out_channels=1024, deform_groups=1,
                 pred_aspect_ratios=None, pred_scales=None, parent=None):
        super().__init__()

        self.out_channels = out_channels
        self.num_classes = cfg.num_classes
        self.mask_dim = cfg.mask_dim
        self.num_priors = len(pred_scales)
        self.deform_groups = deform_groups
        self.embed_dim = cfg.embed_dim
        self.pred_aspect_ratios = pred_aspect_ratios
        self.pred_scales = pred_scales
        self.parent = [parent]  # Don't include this in the state dict
        self.num_heads = cfg.num_heads

        if cfg.use_sipmask:
            self.mask_dim = self.mask_dim * self.sipmask_head

        if parent is None:
            if cfg.extra_head_net is None:
                self.out_channels = in_channels
            else:
                self.upfeature, self.out_channels = make_net(in_channels, cfg.extra_head_net)

            # init single or multi kernel prediction modules
            self.bbox_layer, self.track_layer, self.mask_layer = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])

            if cfg.train_centerness:
                self.centerness_layer = nn.ModuleList([])

            if cfg.train_class:
                self.conf_layer = nn.ModuleList([])

            for k in range(len(cfg.head_layer_params)):
                kernel_size = cfg.head_layer_params[k]['kernel_size']

                if cfg.train_centerness:
                    # self.DIoU_layer.append(nn.Conv2d(self.out_channels, self.num_priors, **cfg.head_layer_params[k]))
                    self.centerness_layer.append(nn.Conv2d(self.out_channels, self.num_priors, **cfg.head_layer_params[k]))

                if cfg.train_boxes:
                    self.bbox_layer.append(nn.Conv2d(self.out_channels, self.num_priors*4, **cfg.head_layer_params[k]))

                if cfg.train_class:
                    if cfg.use_dcn_class:
                        self.conf_layer.append(FeatureAlign(self.out_channels,
                                                            self.num_priors * self.num_classes,
                                                            kernel_size=kernel_size,
                                                            deformable_groups=self.deform_groups,
                                                            use_pred_offset=cfg.use_pred_offset))
                    else:
                        self.conf_layer.append(nn.Conv2d(self.out_channels, self.num_priors * self.num_classes,
                                                         **cfg.head_layer_params[k]))

                if cfg.train_track:
                    if cfg.use_dcn_track:
                        self.track_layer.append(FeatureAlign(self.out_channels,
                                                             self.num_priors * self.embed_dim,
                                                             kernel_size=kernel_size,
                                                             deformable_groups=self.deform_groups,
                                                             use_pred_offset=cfg.use_pred_offset))
                    else:
                        self.track_layer.append(nn.Conv2d(self.out_channels, self.num_priors*self.embed_dim,
                                                          **cfg.head_layer_params[k]))
                if cfg.use_dcn_mask:
                    self.mask_layer.append(FeatureAlign(self.out_channels,
                                                        self.num_priors * self.mask_dim,
                                                        kernel_size=kernel_size,
                                                        deformable_groups=self.deform_groups,
                                                        use_pred_offset=cfg.use_pred_offset))
                else:
                    self.mask_layer.append(nn.Conv2d(self.out_channels, self.num_priors*self.mask_dim,
                                                     **cfg.head_layer_params[k]))

            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers, out_channels):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            if cfg.train_track:
                self.track_extra = make_extra(cfg.extra_layers[2], self.out_channels)
            if cfg.train_class:
                self.conf_extra = make_extra(cfg.extra_layers[0], self.out_channels)
            self.bbox_extra, self.mask_extra = [make_extra(x, self.out_channels) for x in cfg.extra_layers[:2]]

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

        batch_size, _, conv_h, conv_w = x.size()

        if cfg.extra_head_net is not None:
            x = src.upfeature(x)

        if cfg.train_class:
            conf_x = src.conf_extra(x)
        bbox_x = src.bbox_extra(x)
        mask_x = src.mask_extra(x)
        if cfg.train_track:
            track_x = src.track_extra(x)

        conf, bbox, centerness_data, mask, track = [], [], [], [], []
        for k in range(len(cfg.head_layer_params)):
            if cfg.train_centerness:
                centerness_cur = src.centerness_layer[k](bbox_x)
                centerness_data.append(centerness_cur.permute(0, 2, 3, 1).contiguous())

            bbox_cur = src.bbox_layer[k](bbox_x)
            bbox.append(bbox_cur.permute(0, 2, 3, 1).contiguous())

            if cfg.train_class:
                if cfg.use_dcn_class:
                    conf_cur = src.conf_layer[k](conf_x, bbox_cur.detach())
                else:
                    conf_cur = src.conf_layer[k](conf_x)
                conf.append(conf_cur.permute(0, 2, 3, 1).contiguous())

            if cfg.train_track:
                if cfg.use_dcn_track:
                    track_emb = src.track_layer[k](track_x, bbox_cur.detach())
                else:
                    track_emb = src.track_layer[k](track_x)
                track.append(track_emb.permute(0, 2, 3, 1).contiguous())

            if cfg.use_dcn_mask:
                mask_cur = src.mask_layer[k](mask_x, bbox_cur.detach())
            else:
                mask_cur = src.mask_layer[k](mask_x)
            mask.append(mask_cur.permute(0, 2, 3, 1).contiguous())

        # cat for all anchors
        if cfg.train_boxes:
            bbox = torch.cat(bbox, dim=-1).view(x.size(0), -1, 4)
        if cfg.train_centerness:
            centerness_data = torch.cat(centerness_data, dim=1).view(x.size(0), -1, 1)
            centerness_data = torch.tanh(centerness_data)
        if cfg.train_class:
            conf = torch.cat(conf, dim=-1).view(x.size(0), -1, src.num_classes)
        if cfg.train_track:
            track = torch.cat(track, dim=-1).view(x.size(0), -1, src.embed_dim)
        mask = torch.cat(mask, dim=-1).view(x.size(0), -1, src.mask_dim)

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        priors = self.make_priors(x.size(2), x.size(3), x.device)  # [1, h*w*num_priors*num_ratios, 4]

        preds = {'mask_coeff': mask, 'priors': priors}

        if cfg.train_boxes:
            preds['loc'] = bbox

        if cfg.train_centerness:
            preds['centerness'] = centerness_data

        if cfg.temporal_fusion_module:
            preds['T2S_feat'] = x

        if cfg.train_class:
            preds['conf'] = conf

        if cfg.train_track:
            preds['track'] = F.normalize(track, dim=-1)

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
                    for ar in ars:
                        for scale in self.pred_scales:
                            # [h, w]: [3, 3], [3, 5], [5, 3]
                            arh, arw = ar
                            ratio = scale / self.pred_scales[0]
                            w = ratio * arw / conv_w
                            h = ratio * arh / conv_h
                            prior_data += [x, y, w, h]

            priors = torch.Tensor(prior_data, device=device).view(1, -1, 4).detach()
            priors.requires_grad = False

        return priors
