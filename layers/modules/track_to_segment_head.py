import torch
import torch.nn as nn
from datasets.config import cfg
from spatial_correlation_sampler import spatial_correlation_sample
import torch.nn.functional as F
from mmcv.ops import roi_align
from layers.box_utils import sanitize_coordinates_hw


class TemporalNet(nn.Module):
    def __init__(self, corr_channels, mask_proto_n=32):

        super().__init__()
        self.conv1 = nn.Conv2d(corr_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d((7, 7), stride=1)
        self.fc = nn.Linear(1024, 4)
        if cfg.use_sipmask:
            self.fc_coeff = nn.Linear(1024, mask_proto_n * cfg.sipmask_head)
        else:
            self.fc_coeff = nn.Linear(1024, mask_proto_n)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x_reg = self.fc(x)
        x_coeff = self.fc_coeff(x)

        return x_reg, x_coeff


def correlate(x1, x2, patch_size=11, dilation_patch=1):
    """
    :param x1: features 1
    :param x2: features 2
    :param patch_size: the size of whole patch is used to calculate the correlation
    :return:
    """

    # Output sizes oH and oW are no longer dependant of patch size, but only of kernel size and padding
    # patch_size is now the whole patch, and not only the radii.
    # stride1 is now stride and stride2 is dilation_patch, which behave like dilated convolutions
    # equivalent max_displacement is then dilation_patch * (patch_size - 1) / 2.
    # to get the right parameters for FlowNetC, you would have
    out_corr = spatial_correlation_sample(x1,
                                          x2,
                                          kernel_size=1,
                                          patch_size=patch_size,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=dilation_patch)
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph*pw, h, w) / x1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def bbox_feat_extractor(feature_maps, boxes_w_norm, h, w, pool_size):
    """
        feature_maps: size:1*C*h*w
        boxes: Mx5 float box with (x1, y1, x2, y2) **without normalization**
    """
    # Currently only supports batch_size 1
    boxes = sanitize_coordinates_hw(boxes_w_norm, h, w)
    # boxes = boxes_w_norm

    # Crop and Resize
    # Result: [num_boxes, pool_height, pool_width, channels]
    box_ind = torch.zeros(boxes.size(0))  # index of bbox in batch
    if boxes.is_cuda:
        box_ind = box_ind.cuda()

    # CropAndResizeFunction needs batch dimension
    if len(feature_maps.size()) == 3:
        feature_maps = feature_maps.unsqueeze(0)

    # make crops:
    rois = torch.cat([box_ind.unsqueeze(1), boxes], dim=1)
    pooled_features = roi_align(feature_maps, rois, pool_size)

    return pooled_features

