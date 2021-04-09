from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone
from math import sqrt
import torch

# for making bounding boxes pretty
COLORS = ((244, 67, 54),
          (233, 30, 99),
          (156, 39, 176),
          (103, 58, 183),
          (63, 81, 181),
          (33, 150, 243),
          (3, 169, 244),
          (0, 188, 212),
          (0, 150, 136),
          (76, 175, 80),
          (139, 195, 74),
          (205, 220, 57),
          (255, 235, 59),
          (255, 193, 7),
          (255, 152, 0),
          (255, 87, 34),
          (121, 85, 72),
          (158, 158, 158),
          (96, 125, 139))

# These are in BGR and are for YouTubeVOS
MEANS = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)

YouTube_VOS_CLASSES = ('person', 'giant_panda', 'lizard', 'parrot', 'skateboard',
                       'sedan', 'ape', 'dog', 'snake', 'monkey',
                       'hand', 'rabbit', 'duck', 'cat', 'cow',
                       'fish', 'train', 'horse', 'turtle', 'bear',
                       'motorbike', 'giraffe', 'leopard', 'fox', 'deer',
                       'owl', 'surfboard', 'airplane', 'truck', 'zebra',
                       'tiger', 'elephant', 'snowboard', 'boat', 'shark',
                       'mouse', 'frog', 'eagle', 'earless seal', 'tennis_racket')

YouTube_VOS_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                         9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                         17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24,
                         25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32,
                         33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40}

# ----------------------- CONFIG CLASS ----------------------- #


class Config(object):
    """
References[]    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """
        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


# ----------------------- DATASETS ----------------------- #

dataset_base = Config({
    'type': 'YTVOSDataset',

    # images and annotations path
    'ann_file': 'path_to_annotation_file',
    'img_prefix': 'path_to_images_file',
    # 'img_scale': [(640, 360), (1280, 720)],
    'img_scale': [(640, 360)],
    'img_norm_cfg': dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    'size_divisor': 32,
    'flip_ratio': 0.5,
    'resize_keep_ratio': False,
    'with_mask': True,
    'with_crowd': True,
    'with_label': True,
    'with_track': True,
    'proposal_file': None,
    'extra_aug': None,
    'clip_frames': 1,

    # A list of names for each of you classes.
    # 'class_names': YouTube_VOS_CLASSES,

})

train_YouTube_VOS_dataset = dataset_base.copy({
    'img_prefix': '../datasets/YouTube_VOS2019/train/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2019/annotations_instances/valid_sub.json',
    # 'extra_aug': dict(random_crop=dict(min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
    # 'extra_aug': dict(expand=dict(mean=(123.675, 116.28, 103.53), to_rgb=True, ratio_range=(1, 3))),
})

valid_sub_YouTube_VOS_dataset = dataset_base.copy({
    'img_prefix': '../datasets/YouTube_VOS2019/train/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2019/annotations_instances/valid_sub.json',
    'test_mode': False,
})

valid_YouTube_VOS_dataset = dataset_base.copy({

    'img_prefix': '../datasets/YouTube_VOS2019/valid/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2019/annotations_instances/valid.json',
    'test_mode': True,
})

test_YouTube_VOS_dataset = dataset_base.copy({

    'img_prefix': '../datasets/YouTube_VOS2019/test/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2019/annotations_instances/test.json',
    'has_gt': False,
})

# ----------------------- TRANSFORMS ----------------------- #

resnet_transform = Config({
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
})

vgg_transform = Config({
    # Note that though vgg is traditionally BGR,
    # the channel order of vgg_reducedfc.pth is RGB.
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': True,
    'to_float': False,
})

darknet_transform = Config({
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': False,
    'to_float': True,
})

# ----------------------- BACKBONES ----------------------- #

backbone_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/pretrained/weights',
    'type': object,
    'args': tuple(),
    'transform': resnet_transform,

    'selected_layers': list(),
    'pred_scales': list(),
    'pred_aspect_ratios': list(),

    'use_pixel_scales': False,
    'preapply_sqrt': True,
    'use_square_anchors': False,
})

resnet101_backbone = backbone_base.copy({
    'name': 'ResNet101',
    'path': 'yolact_base_54_800000.pth',
    # 'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]] * 6,
    'pred_aspect_ratios': [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]] * 6,
})

resnet101_gn_backbone = backbone_base.copy({
    'name': 'ResNet101_GN',
    'path': 'R-101-GN.pkl',
    'type': ResNetBackboneGN,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]] * 6,
    'pred_aspect_ratios': [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]] * 6,
})

resnet101_dcn_inter3_backbone = resnet101_backbone.copy({
    'name': 'ResNet101_DCN_Interval3',
    'path': 'yolact_plus_base_54_800000.pth',
    'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
})

resnet101_gn_dcn_inter3_backbone = resnet101_gn_backbone.copy({
    'name': 'ResNet101_GN_DCN_Interval3',
    'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'yolact_resnet50_54_800000.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
    'transform': resnet_transform,
})

resnet50_dcn_inter3_backbone = resnet50_backbone.copy({
    'name': 'ResNet50_DCN_Interval3',
    'path': 'yolact_plus_resnet50_54.pth',
    'args': ([3, 4, 6, 3], [0, 4, 6, 3], 3),
})

darknet53_backbone = backbone_base.copy({
    'name': 'DarkNet53',
    'path': 'darknet53.pth',
    'type': DarkNetBackbone,
    'args': ([1, 2, 8, 8, 4],),
    'transform': darknet_transform,

    'selected_layers': list(range(3, 9)),
    'pred_scales': [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [1.8, 1.92]],
    'pred_aspect_ratios': [[[1, sqrt(2), 1 / sqrt(2), sqrt(3), 1 / sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3]],
})

vgg16_arch = [[64, 64],
              ['M', 128, 128],
              ['M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              ['M', 512, 512, 512],
              [('M', {'kernel_size': 3, 'stride': 1, 'padding': 1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]]

vgg16_backbone = backbone_base.copy({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vgg16_arch, [(256, 2), (128, 2), (128, 1), (128, 1)], [3]),
    'transform': vgg_transform,

    'selected_layers': [3] + list(range(5, 10)),
    'pred_scales': [[5, 4]] * 6,
    'pred_aspect_ratios': [[[1], [1, sqrt(2), 1 / sqrt(2), sqrt(3), 1 / sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3]],
})

# ----------------------- MASK BRANCH TYPES ----------------------- #

mask_type = Config({
    'direct': 0,
    'lincomb': 1,
})

# ----------------------- ACTIVATION FUNCTIONS ----------------------- #

activation_func = Config({
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu': lambda x: torch.nn.functional.relu(x, inplace=True),
    'none': lambda x: x,
})

# ----------------------- FPN DEFAULTS ----------------------- #

fpn_base = Config({
    # The number of features to have in each FPN layer
    'num_features': 256,

    # The upsampling mode used
    'interpolation_mode': 'bilinear',

    # The number of extra layers to be produced by downsampling starting at P5
    'num_downsample': 1,

    # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
    'use_conv_downsample': False,

    # Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
    # This is just here for backwards compatibility
    'pad': True,

    # Whether to add relu to the downsampled layers.
    'relu_downsample_layers': False,

    # Whether to add relu to the regular layers
    'relu_pred_layers': True,
})

# ----------------------- CONFIG DEFAULTS ----------------------- #


YouTube_VOS_base_config = Config({
    'train_dataset': train_YouTube_VOS_dataset,
    'valid_dataset': valid_YouTube_VOS_dataset,
    'num_classes': 41,  # This should include the background class
    'num_classes_c4': 16,  # This should include the background class
    'classes': YouTube_VOS_CLASSES,
    'COLORS': COLORS,

    'max_iter': 600000,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 1e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (280000, 360000, 400000),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 500,

    # The terms to scale the respective loss by
    'conf_alpha': 1,
    'bbox_alpha': 1.5,
    'track_alpha': 10,
    'mask_alpha': 0.4 / 256 * 140 * 140,  # Some funky equation. Don't worry about it.

    # Eval.py sets this if you just want to run YOLACT as a detector
    'eval_mask_branch': True,

    # the dim of embedding vectors in track layers
    'embed_dim': 512,

    # Top_k examples to consider for NMS
    'nms_top_k': 200,
    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.3,
    # Boxes with IoU overlap greater than this threshold will be culled during NMS
    'nms_thresh': 0.5,
    # used in detection of eval, lower than conf_thresh will be ignored
    'eval_conf_thresh': 0.3,

    # See mask_type for details.
    'mask_type': mask_type.direct,
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],
    'mask_proto_bias': False,
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
    'mask_proto_coeff_activation': activation_func.tanh,
    'mask_proto_crop': False,
    'mask_proto_crop_expand': 0,
    'mask_proto_loss': None,
    'mask_proto_binarize_downsampled_gt': True,
    'mask_proto_grid_file': 'data/grid.npy',
    'mask_proto_use_grid': False,
    'mask_proto_remove_empty_masks': False,
    'mask_proto_reweight_coeff': 1,
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_coeff_diversity_alpha': 1,
    'mask_proto_normalize_emulate_roi_pooling': False,
    'mask_proto_double_loss': False,
    'mask_proto_double_loss_alpha': 1,
    'mask_proto_crop_with_pred_box': False,

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': True,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': True,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': False,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'augment_random_rot90': False,

    # Discard detections with width and height smaller than this (in absolute width and height)
    'discard_box_width': 4 / 550,
    'discard_box_height': 4 / 550,

    # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': None,

    # Use the same weights for each network head
    'share_prediction_module': False,

    # For hard negative mining, instead of using the negatives that are leastl confidently background,
    # use negatives that are most confidently not background.
    'ohem_use_most_confident': False,

    # Whether to use sigmoid focal loss instead of softmax, all else being the same.
    'use_sigmoid_focal_loss': False,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,

    # The initial bias toward forground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Keeps track of the average number of examples for each class, and weights the loss for that class accordingly.
    'use_class_balanced_conf': False,

    # Adds a global pool + fc layer to the smallest selected layer that predicts the existence of each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_class_existence_loss': False,
    'class_existence_alpha': 1,

    # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations for each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_semantic_segmentation_loss': False,
    'semantic_segmentation_alpha': 1,

    # Adds another branch to the netwok to predict Mask IoU.
    'use_mask_scoring': False,
    'mask_scoring_alpha': 1,

    # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
    # Note that the threshold you set for iou_threshold should be negative with this setting on.
    'use_change_matching': False,

    # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers before the final
    # prediction in prediction modules. If this is none, no extra layers will be added.
    'extra_head_net': None,

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'head_layer_params': {0: {'kernel_size': [3, 3], 'padding': 1}, 1: {'kernel_size': [3, 3], 'padding': 1},
                          2: {'kernel_size': [3, 3], 'padding': 1}},

    # Add extra layers between the backbone and the network heads
    # The order is (conf, bbox, track, mask)
    'extra_layers': (0, 0, 0, 0),

    # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
    # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
    # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
    # The rest are neutral and not used in calculating the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.5,

    # When using ohem, the ratio between positives and negatives (3 means 3 negatives to 1 positive)
    'ohem_negpos_ratio': 3,

    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 1,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Whether or not to do post processing on the cpu at test time
    'force_cpu_nms': True,

    # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
    'use_coeff_nms': False,

    # Whether or not to tie the mask loss / box loss to 0
    'train_masks': True,
    'train_boxes': True,
    'train_track': True,
    'train_class': True,
    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, uses the faster r-cnn resizing scheme.
    # If False, all images arte resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,

    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    # Use command-line arguments to set this.
    'no_jit': False,

    'backbone': None,
    'name': 'base_config',

    # Fast Mask Re-scoring Network
    # Inspried by Mask Scoring R-CNN (https://arxiv.org/abs/1903.00241)
    # Do not crop out the mask with bbox but slide a convnet on the image-size mask,
    # then use global pooling to get the final mask score
    'use_maskiou': False,

    # Archecture for the mask iou network. A (num_classes-1, 1, {}) layer is appended to the end.
    'maskiou_net': [],

    # Discard predicted masks whose area is less than this
    'discard_mask_area': -1,

    'rescore_mask': False,
    'rescore_bbox': False,
    'maskious_to_train': -1,

    # display output results in each epoch
    'train_output_visualization': True,

    # the format of mask in output .json file including 'polygon' and 'rle'.
    'mask_output_json': 'polygon',
})

# ----------------------- STMask CONFIGS ----------------------- #
STMask_base_config = YouTube_VOS_base_config.copy({
    'name': 'yolact_JDT_base',

    # Dataset stuff
    'train_dataset': train_YouTube_VOS_dataset,
    'valid_sub_dataset': valid_sub_YouTube_VOS_dataset,
    'valid_dataset': valid_YouTube_VOS_dataset,

    # Training params
    'lr_steps': (150000, 200000, 250000),
    'max_iter': 300000,

    # loss
    'conf_alpha': 6.125,
    'bbox_alpha': 1.5,
    'BIoU_alpha': 0.5,
    'bboxiou_alpha': 5,
    'maskiou_alpha': 5,
    'track_alpha': 5,
    'mask_proto_coeff_diversity_alpha': 5,
    'center_alpha': 20,

    # backbone and FCA settings
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(1, 4)),

        'pred_aspect_ratios': [[[ [3, 3], [3, 5],  [5, 3] ]]] * 5,
        'pred_scales': [[i*2**(j/1.0) for j in range(1)] for i in [24, 48, 96, 192, 384]],
    }),

    # FPN Settings
    'fpn': fpn_base.copy({
        'num_features': 256,
        'use_conv_downsample': True,
        'num_downsample': 2,
    }),

    # FCA and prediction module settings
    'share_prediction_module': True,
    'extra_head_net': [(256, 3, {'padding': 1})],
    'extra_layers': (2, 2, 2, 2),
    'head_layer_params': {0: {'kernel_size': [3, 3], 'padding': (1, 1)},
                          1: {'kernel_size': [3, 5], 'padding': (1, 2)},
                          2: {'kernel_size': [5, 3], 'padding': (2, 1)}},

    # Mask Settings
    'mask_type': mask_type.lincomb,
    'mask_alpha': 6.125,
    'mask_proto_src': 0,
    'mask_proto_crop': True,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(8, 1, {})],
    'use_rela_coord': False,
    'mask_proto_normalize_emulate_roi_pooling': False,
    'discard_mask_area': 5 * 5,
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_crop_with_pred_box': True,

    # Proto_net settings
    'backbone_C2_as_features': False,
    'display_protos': False,

    # train boxes
    'train_boxes': True,
    'train_class': True,
    'use_sigmoid_focal_loss': False,
    'train_centerness': True,

    # Track settings
    'train_track': True,
    'match_coeff': [0, 0.7, 0.3, 0],   # scores, mask_iou, box_iou, label
    'embed_dim': 128,

    # temporal fusion module settings
    'temporal_fusion_module': True,
    'correlation_patch_size': 11,
    'correlation_selected_layer': 1,
    'boxshift_alpha': 10,
    'maskshift_alpha': 6.125,
    'maskshift_loss': True,

    # FCB settings
    'use_pred_offset': False,
    'use_dcn_class': False,
    'use_dcn_track': False,
    'use_dcn_mask': False,

    # SipMask uses multi heads for obtaining better mask segmentation
    'use_sipmask': False,
    'sipmask_head': 4,

    # loss settings
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,
    'crowd_iou_threshold': 0.7,
    'use_conf_cross_frames': False,
    'use_boxiou_loss': True,
    'use_maskiou_loss': False,
    'use_semantic_segmentation_loss': False,

    # eval
    'nms_conf_thresh': 0.05,
    'nms_thresh': 0.5,
    'eval_conf_thresh': 0.05,
    'candidate_conf_thresh': 0.05,
    'nms_as_miou': False,
    'remove_false_inst': True,
    'add_missed_masks': False,
    'use_train_sub': False,
    'use_valid_sub': False,
    'only_calc_metrics': False,
    'only_count_classes': False,
    'use_DIoU_in_comp_scores': False,
    'display_corr': False,
    'display_mask_single': False,
    'eval_single_im': False,
})


# ----------------------- STMask-plus CONFIGS ----------------------- #
STMask_plus_base_config = STMask_base_config.copy({
    'name': 'STMask_plus_base',
    'backbone': resnet101_dcn_inter3_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'pred_scales': STMask_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
    }),

})

# ----------------------- STMask-resnet50 CONFIGS ----------------------- #
STMask_resnet50_config = STMask_base_config.copy({
    'name': 'STMask_resnet50',
    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'pred_scales': STMask_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
    }),
})

STMask_plus_resnet50_config = STMask_resnet50_config.copy({
    'name': 'STMask_plus_resnet50',
    'backbone': resnet50_dcn_inter3_backbone.copy({
        'selected_layers': list(range(1, 4)),

        'pred_scales': STMask_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
    }),

})


STMask_darknet53_config = STMask_base_config.copy({
    'name': 'STMask_darknet53',
    'backbone': darknet53_backbone.copy({
        'selected_layers': list(range(2, 5)),
        'pred_scales': STMask_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True,  # This is for backward compatability with a bug
    }),
})

# Default config
cfg = STMask_base_config.copy()


def set_cfg(config_name: str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]


def set_dataset(dataset_name: str, type: str):
    """ Sets the dataset of the current config. """
    if type == 'train':
        cfg.train_dataset = eval(dataset_name)
    elif type == 'eval':
        cfg.valid_dataset = eval(dataset_name)
