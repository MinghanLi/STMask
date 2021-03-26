import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.config import cfg
from .make_net import make_net

try:
    from dcn_v2 import DCN, DCNv2
except ImportError:
    def DCN(*args, **kwdargs):
        raise Exception('DCN could not be imported. If you want to use YOLACT++ models, compile DCN. Check the README for instructions.')

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn


class FastMaskIoUNet(ScriptModuleWrapper):

    def __init__(self):
        super().__init__()
        input_channels = 1
        last_layer = [(cfg.num_classes-1, 1, {})]
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

        return maskiou_p