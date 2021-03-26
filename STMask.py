import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from datasets.config import cfg, mask_type
from layers import Detect, Detect_TF, Track, generate_candidate, \
    Track_TF, PredictionModule, PredictionModule_FC, make_net, FPN, FastMaskIoUNet, TemporalNet, correlate
from backbone import construct_backbone
from utils import timer

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()
prior_cache = defaultdict(lambda: None)


class STMask(nn.Module):
    """
    The code comes from Yolact.
    You can set the arguments by chainging them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone)

        if cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size ** 2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            self.proto_src = cfg.mask_proto_src
            self.interpolation_mode = cfg.fpn.interpolation_mode

            if self.proto_src is None:
                in_channels = 3
            elif cfg.fpn is not None:
                in_channels = cfg.fpn.num_features
            else:
                in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.mask_proto_bias:
                cfg.mask_dim += 1

        self.selected_layers = cfg.backbone.selected_layers
        self.pred_scales = cfg.backbone.pred_scales
        self.pred_aspect_ratios = cfg.backbone.pred_aspect_ratios
        self.num_priors = len(self.pred_scales[0])
        src_channels = self.backbone.channels

        if cfg.use_maskiou:
            self.maskiou_net = FastMaskIoUNet()

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])

            if cfg.backbone_C2_as_features:
                self.selected_layers = list(range(1, len(self.selected_layers) + cfg.fpn.num_downsample))
                src_channels = [cfg.fpn.num_features] * (len(self.selected_layers) + 1)
            else:
                self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
                src_channels = [cfg.fpn.num_features] * len(self.selected_layers)

        # prediction layers for loc, conf, mask
        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)  # yolact++
        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent, parent_t = None, None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule_FC(src_channels[layer_idx], src_channels[layer_idx],
                                       deform_groups=1,
                                       pred_aspect_ratios=self.pred_aspect_ratios[idx],
                                       pred_scales=self.pred_scales[idx],
                                       parent=parent)

            self.prediction_layers.append(pred)

        # parameters in temporal correlation net
        if cfg.temporal_fusion_module:
            corr_channels = 2*in_channels + cfg.correlation_patch_size**2
            self.TemporalNet = TemporalNet(corr_channels)
            self.correlation_selected_layer = cfg.correlation_selected_layer

            # evaluation for frame-level tracking
            self.Detect_TF = Detect_TF(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
                                       conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)
            self.Track_TF = Track_TF()

        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)

        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes - 1, kernel_size=1)

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k, conf_thresh=cfg.nms_conf_thresh,
                             nms_thresh=cfg.nms_thresh)
        self.Track = Track()

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        model_dict = self.state_dict()

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]

        diff_layers1 = [k for k, v in state_dict.items() if k not in model_dict.keys()]
        print()
        print('layers in pre-trained model but not in current model:', diff_layers1)

        diff_layers2 = [k for k, v in model_dict.items() if k not in state_dict.keys()]
        print('layers in current model but not in pre-trained model:', diff_layers2)

        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def init_weights_coco(self, backbone_path):
        """ Initialize weights for training. """
        state_dict = torch.load(backbone_path)
        model_dict = self.state_dict()

        # only remain same modules and layers between pre-trained model and our model
        for key in list(state_dict.keys()):
            if key not in model_dict.keys():
                del state_dict[key]
            elif model_dict[key].shape != state_dict[key].shape:
                del state_dict[key]

        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        print('parameters without load weights from pre-trained models')
        print([k for k, v in model_dict.items() if k not in state_dict])
        model_dict.update(state_dict)

        # Initialize the rest of the conv layers with xavier
        for k, v in model_dict.items():
            if k not in state_dict:
                print('init weights by Xavier:', k)
                if 'bn3d' not in k:
                    if 'weight' in k:
                        nn.init.xavier_uniform_(model_dict[k])
                    elif 'bias' in k:
                        if cfg.use_sigmoid_focal_loss and 'conf_layer' in k:
                            data0 = -torch.tensor(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi)).log()
                            data1 = -torch.tensor((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi).log()
                            model_dict[k] = torch.cat([data0.repeat(self.num_priors), data1.repeat((cfg.num_classes-1)*self.num_priors)])
                        else:
                            model_dict[k].zero_()

        self.load_state_dict(model_dict)

    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def forward_single(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """

        with timer.env('backbone'):
            bb_outs = self.backbone(x)

        if cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [bb_outs[i] for i in cfg.backbone.selected_layers]
                fpn_outs = self.fpn(outs)

        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):
                if self.proto_src is None:
                    proto_x = x
                else:
                    # h, w = bb_outs[self.proto_src].size()[2:]
                    # p3_upsample = F.interpolate(fpn_outs[self.proto_src], size=(h, w), mode=self.interpolation_mode,
                    #                             align_corners=False)
                    # proto_x = p3_upsample # + bb_outs[self.proto_src]
                    proto_x = fpn_outs[self.proto_src]

                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                proto_out = cfg.mask_proto_prototype_activation(proto_out)

                # Move the features last so the multiplication is easy
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], 1)

        with timer.env('pred_heads'):
            pred_outs = {'mask_coeff': [], 'priors': []}

            if cfg.train_boxes:
                pred_outs['loc'] = []
            if cfg.temporal_fusion_module:
                pred_outs['T2S_feat'] = []
            if cfg.train_centerness:
                pred_outs['centerness'] = []

            if cfg.train_class:
                pred_outs['conf'] = []

            if cfg.train_track:
                pred_outs['track'] = []

            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = fpn_outs[idx]

                # A hack for the way dataparallel works
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]

                p = pred_layer(pred_x)

                for k, v in p.items():
                    pred_outs[k].append(v)  # [batch_size, h*w*anchors, dim]

                if cfg.backbone_C2_as_features:
                    idx -= 1

            for k, v in pred_outs.items():
                if k is not 'T2S_feat':
                    pred_outs[k] = torch.cat(v, 1)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        return fpn_outs, pred_outs

    def forward(self, x, img_meta=None, ref_x=None, ref_imgs_meta=None):
        if self.training:
            batch_size, nf, c, h, w = x.size()
            fpn_outs, pred_outs = self.forward_single(x.view(-1, c, h, w))

            if cfg.temporal_fusion_module:
                # calculate correlation map
                x_ref = pred_outs['T2S_feat'][self.correlation_selected_layer][::2].contiguous()
                x_next = pred_outs['T2S_feat'][self.correlation_selected_layer][1::2].contiguous()
                x_corr = correlate(x_ref, x_next, patch_size=cfg.correlation_patch_size)
                pred_outs['T2S_concat_feat'] = F.relu(torch.cat([x_corr, x_ref, x_next], dim=1))
                del pred_outs['T2S_feat']

            # For the extra loss functions
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(fpn_outs[-1].mean(dim=(2, 3)))

            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(fpn_outs[0])

            # for nn.DataParallel
            pred_outs['priors'] = pred_outs['priors'].repeat(batch_size*nf, 1, 1)

            return pred_outs
        else:
            # track instances frame-by-frame
            fpn_outs, pred_outs = self.forward_single(x)
            pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

            if cfg.temporal_fusion_module:
                # we only use the bbox features in the P3 layer
                pred_outs['T2S_feat'] = pred_outs['T2S_feat'][self.correlation_selected_layer]
                candidate = generate_candidate(pred_outs)
                candidate_after_NMS = self.Detect_TF(self, candidate[0], is_output_candidate=True)
                pred_outs_after_NMS = self.Track_TF(self, candidate_after_NMS, img_meta[0], img=x)

            else:
                pred_outs['mask_coeff'] = cfg.mask_proto_coeff_activation(pred_outs['mask_coeff'])
                # detect instances by NMS for each single frame
                pred_outs_after_NMS = self.detect(pred_outs, self)
                pred_outs_after_NMS = self.Track(pred_outs_after_NMS, img_meta)

            return pred_outs_after_NMS

