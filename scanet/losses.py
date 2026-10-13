"""Losses of the authors' image notebook: CE label smoothing (eps 0.1),
cosine hard-mining triplet (margin 0.3, dist = -cos), BCE spatial-attention mask loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes, self.epsilon = num_classes, epsilon

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        t = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        t = (1 - self.epsilon) * t + self.epsilon / self.num_classes
        return (-t * log_probs).mean(0).sum()


class TripletLossCosine(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
        l2norm = inputs.div(fnorm.expand_as(inputs))
        dist = -torch.mm(l2norm, l2norm.t())
        same = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        dist_ap = dist.masked_fill(~same, float("-inf")).max(dim=1)[0]
        dist_an = dist.masked_fill(same, float("inf")).min(dim=1)[0]
        return F.relu(dist_ap - dist_an + self.margin).mean()


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        b, c, h, w = inputs.size()
        targets = F.interpolate(targets, (h, w), mode="bilinear", align_corners=True)
        return self.loss(inputs.reshape(b, -1), targets.reshape(b, -1))


def deep_supervision(criterion, xs, y):
    loss = 0.0
    for x in xs:
        loss = loss + criterion(x, y)
    return loss / len(xs)


class SCANetCriterion(nn.Module):
    """total = CE_smooth + triplet + alpha * mean_over_4_parts(mask loss over IAT2, IAT3)."""
    def __init__(self, num_classes, margin=0.3, alpha=0.5, epsilon=0.1):
        super().__init__()
        self.xent = CrossEntropyLabelSmooth(num_classes, epsilon)
        self.tri = TripletLossCosine(margin)
        self.mask = MaskLoss()
        self.alpha = alpha

    def forward(self, model_out, pids, masks):
        """masks: [B,4,H,W] in order head, upper, lower, shoes."""
        y, f, a_head, a_upper, a_lower, a_shoes = model_out
        xent = self.xent(y, pids)
        tri = self.tri(f, pids)
        parts = (a_head, a_upper, a_lower, a_shoes)
        mloss = sum(deep_supervision(self.mask, a, masks[:, i:i + 1]) for i, a in enumerate(parts)) / 4.0
        total = xent + tri + self.alpha * mloss
        return total, {"xent": xent.detach(), "tri": tri.detach(), "mask": mloss.detach()}
