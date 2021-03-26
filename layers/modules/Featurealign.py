import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d


class FeatureAlign(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 deformable_groups=4,
                 use_pred_offset=True):
        super(FeatureAlign, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.use_pred_offset = use_pred_offset

        if self.use_pred_offset:
            offset_channels = kernel_size[0] * kernel_size[1] * 2
            self.conv_offset = nn.Conv2d(4,
                                         deformable_groups * offset_channels,
                                         1,
                                         bias=False)

        self.conv_adaption = DeformConv2d(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                          deform_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        # self.norm = nn.GroupNorm(32, in_channels)

    def init_weights(self, bias_value=0):
        torch.nn.init.normal_(self.conv_offset.weight, std=0.0)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)

    def forward(self, x, shape):
        if self.use_pred_offset:
            offset = self.conv_offset(shape.detach())
        else:
            ks_h, ks_w = self.kernel_size
            batch_size = x.size(0)

            variances = [0.1, 0.2]
            # dx = 2*\delta x , dy = 2*\delta y
            dxy = shape[:, :2].view(batch_size, 2, -1) * variances[0]  # [bs, 2, hw]
            dx = (dxy[:, 0] * ks_w).unsqueeze(1).repeat(1, ks_h * ks_w, 1)
            dy = (dxy[:, 1] * ks_h).unsqueeze(1).repeat(1, ks_h * ks_w, 1)

            # dw = exp(\delta w) - 1
            dwh = (shape[:, 2:].view(batch_size, 2, -1) * variances[1]).exp() - 1

            # build offset for h
            dh_R = torch.arange(-ks_h // 2 + 1, ks_h // 2 + 1).float()
            dh_R = dh_R.view(-1, 1).repeat(1, ks_w)
            dh = dwh[:, 1].unsqueeze(1) * dh_R.view(1, -1, 1)
            # build offset for w
            dw_R = torch.arange(-ks_w // 2 + 1, ks_w // 2 + 1).float()
            dw_R = dw_R.repeat(ks_h)
            dw = dwh[:, 0].unsqueeze(1) * dw_R.view(1, -1, 1)

            # [dy1, dx1, dy2, dx2, ..., dyn, dxn]
            offset = torch.stack([dy + dh, dx + dw], dim=1).permute(0, 2, 1, 3).contiguous()
            offset = offset.view(batch_size, -1, x.size(2), x.size(3))

        x = self.relu(self.conv_adaption(x, offset))
        # x = self.relu(self.norm(self.conv_adaption(x, offset)))
        return x


