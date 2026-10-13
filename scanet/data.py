"""Data layer of the SCANet reproduction (Market-1501 + SCHP part masks).
Re-implemented from the behaviour of the authors' image notebook (cells 1, 20, 24, 25) of
github.com/FatimaZulfiqar/Person-ReID-STCANet (commit 1b1eb9dd). Parity: tests/test_parity_data.py.
Deviations (documented): sorted glob, no infinite read-retry loop, foreground.png not loaded
(unused by the authors' loss, consumes no RNG)."""
import copy
import glob
import math
import os.path as osp
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ERASE_MEAN = (0.4914, 0.4822, 0.4465)
MASK_NAMES = ("head", "upper_clothes", "lower_clothes", "shoes")   # channel order: head, upper, lower, shoes
_PATTERN = re.compile(r"([-\d]+)_c(\d)")


def process_dir(dir_path, mask_root=None, relabel=False):
    """Returns (items, num_pids, num_imgs); item = (path, pid, camid).
    Train items: path = (img, head, upper, lower, shoes). Junk (pid -1) dropped, camid 0-based."""
    img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
    pid_container = set()
    for p in img_paths:
        pid, _ = map(int, _PATTERN.search(osp.basename(p)).groups())
        if pid == -1:
            continue
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
    items = []
    for p in img_paths:
        pid, camid = map(int, _PATTERN.search(osp.basename(p)).groups())
        if pid == -1:
            continue
        assert 0 <= pid <= 1501
        assert 1 <= camid <= 6
        camid -= 1
        if relabel:
            pid = pid2label[pid]
        if mask_root is not None:
            stem = osp.splitext(osp.basename(p))[0]
            mp = tuple(osp.join(mask_root, stem, n + ".png") for n in MASK_NAMES)
            for q in mp:
                assert osp.exists(q), q
            path = (p,) + mp
        else:
            path = p
        items.append((path, pid, camid))
    return items, len(pid_container), len(items)


def read_image(path, mode):
    with Image.open(path) as im:
        return im.convert(mode)


def _pil_to_tensor(pic):
    arr = np.array(pic)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div(255)


def _normalize_(t, mean, std):
    for c, m, s in zip(t, mean, std):
        c.sub_(m).div_(s)
    return t


def _random_erase(img, probability, sl, sh, r1, mean):
    """Same RNG consumption order as the authors' RandomErasing (python `random`)."""
    if random.uniform(0, 1) > probability:
        return img
    for _ in range(100):
        area = img.size()[1] * img.size()[2]
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < img.size()[2] and h < img.size()[1]:
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            img[0, x1:x1 + h, y1:y1 + w] = mean[0]
            img[1, x1:x1 + h, y1:y1 + w] = mean[1]
            img[2, x1:x1 + h, y1:y1 + w] = mean[2]
            return img
    return img


class JointTrainTransform:
    """One flip decision (random.random() < 0.5) shared by image and 4 masks; bicubic resize to
    (H, W) for all; image only: ImageNet normalize then RandomErasing; masks stay in [0, 1]."""
    def __init__(self, size=(256, 128), mean=IMAGENET_MEAN, std=IMAGENET_STD, erase_prob=0.5,
                 erase_mean=ERASE_MEAN, sl=0.02, sh=0.4, r1=0.3):
        self.size, self.mean, self.std = size, mean, std
        self.erase_prob, self.erase_mean, self.sl, self.sh, self.r1 = erase_prob, erase_mean, sl, sh, r1

    def _prep(self, im, flip):
        im = im.resize(self.size[::-1], Image.Resampling.BICUBIC)
        if flip:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return im

    def __call__(self, img, masks):
        flip = random.random() < 0.5
        x = _normalize_(_pil_to_tensor(self._prep(img, flip)), self.mean, self.std)
        x = _random_erase(x, self.erase_prob, self.sl, self.sh, self.r1, self.erase_mean)
        m = torch.cat([_pil_to_tensor(self._prep(mk, flip)) for mk in masks], 0)
        return x, m


def build_test_transform(size=(256, 128)):
    return T.Compose([
        T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


class SCANetTrainDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        paths, pid, camid = self.items[i]
        img = read_image(paths[0], "RGB")
        masks = [read_image(p, "L") for p in paths[1:5]]
        x, m = self.transform(img, masks)
        return x, m, pid, camid


class ReIDEvalDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, pid, camid = self.items[i]
        return self.transform(read_image(path, "RGB")), pid, camid


class RandomIdentitySampler(Sampler):
    """Per ID: shuffle indices, cut into groups of K (remainder dropped; IDs with < K images are
    resampled with replacement); then shuffle all groups globally. A batch of N*K therefore does
    NOT guarantee N distinct IDs (same as the authors')."""
    def __init__(self, data_source, num_instances=4):
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            num = max(len(self.index_dic[pid]), num_instances)
            self.length += num - num % num_instances

    def __iter__(self):
        groups = []
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch = []
            for idx in idxs:
                batch.append(idx)
                if len(batch) == self.num_instances:
                    groups.append(batch)
                    batch = []
        random.shuffle(groups)
        return iter([i for g in groups for i in g])

    def __len__(self):
        return self.length
