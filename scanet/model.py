"""Image-based SCANet re-implementation following the public image notebook of
Zulfiqar et al. (github.com/FatimaZulfiqar/Person-ReID-STCANet, commit 1b1eb9dd).
Parameter names mirror the notebook for parity testing. Spatial attention only."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

RESNET50_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


def _normal_conv_init(m):
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2.0 / n))


def _init_conv_bn(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            _normal_conv_init(m)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class Gconv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fsm = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        _init_conv_bn(self)

    def forward(self, W, x):
        bs, n, c = x.size()
        x_neighbor = torch.bmm(W, x)
        x = torch.cat([x, x_neighbor], 2)
        x = x.view(-1, x.size(2), 1, 1)
        x = self.fsm(x)
        return x.view(bs, n, c)


class Wcompute(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.relation = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1), nn.BatchNorm2d(1))
        _init_conv_bn(self)

    def forward(self, x, W_id, y):
        bs, N, C = x.size()
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2)
        W_new = torch.abs(W1 - W2)
        W_new = torch.transpose(W_new, 1, 3)
        y = y.view(bs, C, 1, 1).expand_as(W_new)
        W_new = torch.cat((W_new, y), 1)
        W_new = self.relation(W_new)
        W_new = torch.transpose(W_new, 1, 3)
        W_new = W_new.squeeze(3)
        W_new = W_new - W_id.expand_as(W_new) * 1e8
        return F.softmax(W_new, dim=2)


class SAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.module_w = Wcompute(in_channels)
        self.module_l = Gconv(in_channels)

    def forward(self, x, y):
        bs, N, C = x.size()
        W_init = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(bs, 1, 1)
        W = self.module_w(x, W_init, y)
        return self.module_l(W, x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class SpatialAttn(nn.Module):
    def __init__(self, in_channels, number):
        super().__init__()
        self.conv = ConvBlock(in_channels, number, 1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class IAT(nn.Module):
    """CAM (channel self-attention) followed by SAM (4-part graph attention)."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.sa = SpatialAttn(in_channels, number=4)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.SAM = SAM(self.inter_channels)
        self.W1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels))
        self.W2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels))
        _init_conv_bn(self)
        nn.init.constant_(self.W1[1].weight.data, 0.0)
        nn.init.constant_(self.W1[1].bias.data, 0.0)
        nn.init.constant_(self.W2[1].weight.data, 0.0)
        nn.init.constant_(self.W2[1].bias.data, 0.0)

    def reduce_dimension(self, x, global_node):
        bs, c = global_node.size()
        x = x.transpose(1, 2).unsqueeze(3)
        x = torch.cat((x, global_node.view(bs, c, 1, 1)), 2)
        x = self.g(x).squeeze(3)
        global_node = x[:, :, -1]
        x = x[:, :, :-1].transpose(1, 2)
        return x, global_node

    def forward(self, x):
        # CAM
        batch_size = x.size(0)
        g_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(g_x, g_x.permute(0, 2, 1))
        f = F.softmax(f, dim=-1)
        y = torch.matmul(f, g_x)
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        y = self.W1(y)
        z = y + x
        # SAM
        x = z
        inputs = x
        b, c, h, w = x.size()
        u = x.view(b, c, -1).mean(2)
        a = self.sa(x)
        x = torch.bmm(a.view(b, -1, h * w), x.view(b, c, -1).transpose(1, 2))
        x, u = self.reduce_dimension(x, u)
        y = self.SAM(x, u)
        y = torch.mean(y, 1)
        u = torch.cat((y, u), 1)
        y = self.W2(u.view(u.size(0), u.size(1), 1, 1))
        return y + inputs, a


class SCANet(nn.Module):
    feat_dim = 2048

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        r.layer4[0].conv2.stride = (1, 1)            # last stride 1
        r.layer4[0].downsample[0].stride = (1, 1)
        if pretrained:
            sd = torch.hub.load_state_dict_from_url(
                RESNET50_URL, progress=False, check_hash=True, weights_only=False)
            r.load_state_dict(sd)
        self.conv1, self.bn1 = r.conv1, r.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.IAT2 = IAT(512)
        self.IAT3 = IAT(1024)
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.maxpool(self.relu(x))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x2, a2 = self.IAT2(x2)
        x3 = self.layer3(x2)
        x3, a3 = self.IAT3(x3)
        x4 = self.layer4(x3)
        f = F.avg_pool2d(x4, x4.size()[2:])
        f = f.view(f.size(0), -1)
        f = self.bn(f)                                # post-BN feature = test feature
        y = self.classifier(f)
        a_head = [a2[:, 0:1], a3[:, 0:1]]
        a_upper = [a2[:, 1:2], a3[:, 1:2]]
        a_lower = [a2[:, 2:3], a3[:, 2:3]]
        a_shoes = [a2[:, 3:4], a3[:, 3:4]]
        return y, f, a_head, a_upper, a_lower, a_shoes


def count_num_param(model):
    n = sum(p.numel() for p in model.parameters()) / 1e6
    return n - sum(p.numel() for p in model.classifier.parameters()) / 1e6
