"""CAM Only ablation of SCANet: CAM ON, SAM OFF, mask supervision ON.
The `sa` head is kept ONLY to produce the 4 part maps for the mask loss (decision 2026-11-03, option a).
g / SAM / W2 are absent. Output format is identical to scanet.model.SCANet, so losses/train code are unchanged.
Not verified until tests/test_cam_only.py has run."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import model as M
from ..model import SCANet, SpatialAttn


class IATCamOnly(nn.Module):
    """z = W1(CAM(x)) + x ; a = sa(z). Same CAM code and init as scanet.model.IAT."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.sa = SpatialAttn(in_channels, number=4)
        self.W1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels))
        M._init_conv_bn(self)
        nn.init.constant_(self.W1[1].weight.data, 0.0)
        nn.init.constant_(self.W1[1].bias.data, 0.0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(g_x, g_x.permute(0, 2, 1))
        f = F.softmax(f, dim=-1)
        y = torch.matmul(f, g_x)
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        y = self.W1(y)
        z = y + x
        a = self.sa(z)
        return z, a


class SCANetCamOnly(SCANet):
    """SCANet with IAT2/IAT3 replaced by IATCamOnly; forward is inherited unchanged.
    Note: the parent builds full IATs first, so init RNG draws differ from other variants (report as confound)."""
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes, pretrained=pretrained)
        self.IAT2 = IATCamOnly(512)
        self.IAT3 = IATCamOnly(1024)
