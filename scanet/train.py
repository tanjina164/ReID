"""Training utilities of the SCANet reproduction.
Follows the authors' image notebook (cell 27 `train` / `main`) of
github.com/FatimaZulfiqar/Person-ReID-STCANet (commit 1b1eb9dd): Adam lr 3.5e-4 wd 5e-4 on all parameters,
MultiStepLR [20, 40] gamma 0.1 with scheduler.step() called at the START of every epoch (before training),
FP32, batch 64 (RandomIdentitySampler, K=4), total = CE_smooth + cosine triplet + 0.5 * mask loss.
Differences (documented): all RNGs seeded; DataLoader workers get their seeds from the DataLoader `generator`
(torch seeds python `random` and numpy inside workers from it); no DataParallel.
The full driver (checkpoints, resume, final-epoch evaluation) is NOT in this file yet.
Parity: tests/test_train.py."""
import random
import warnings

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from . import data as D


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def build_train_loader(items, batch_size=64, num_instances=4, num_workers=4, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(D.SCANetTrainDataset(items, D.JointTrainTransform()),
                      sampler=D.RandomIdentitySampler(items, num_instances),
                      batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      drop_last=True, generator=g)


def build_optimizer(model, lr=3.5e-4, weight_decay=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, milestones=(20, 40), gamma=0.1):
    return lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)


def lr_trace(max_epoch=60, base_lr=3.5e-4, milestones=(20, 40), gamma=0.1, step_first=True):
    """LR used in each epoch (index 0 = epoch 1). step_first=True is the authors' order."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=base_lr)
    sch = build_scheduler(opt, milestones, gamma)
    lrs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(max_epoch):
            if step_first:
                sch.step()
            lrs.append(opt.param_groups[0]["lr"])
            p.grad = torch.zeros_like(p)
            opt.step()
            if not step_first:
                sch.step()
    return lrs


def train_one_epoch(model, criterion, optimizer, loader, device="cuda"):
    """loader yields (img [B,3,H,W], masks [B,4,H,W], pid, camid). Returns epoch-averaged metrics."""
    model.train()
    n_tot, s_x, s_t, s_m, s_c = 0, 0.0, 0.0, 0.0, 0.0
    for x, m, pids, _ in loader:
        x = x.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        pids = pids.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        total, parts = criterion(out, pids, m)
        total.backward()
        optimizer.step()
        n = pids.size(0)
        s_x += parts["xent"].item() * n
        s_t += parts["tri"].item() * n
        s_m += parts["mask"].item() * n
        s_c += (out[0].detach().argmax(1) == pids).float().sum().item()
        n_tot += n
    return {"xent": s_x / n_tot, "tri": s_t / n_tot, "mask": s_m / n_tot, "acc": s_c / n_tot, "n": n_tot}
