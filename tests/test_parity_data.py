"""Parity test of scanet.data against the authors' image notebook code (extracted and executed).
Env patches for the authors' code (documented): collections.Iterable shim; ToTensor fallback if it fails."""
import sys, os, ast, math, copy, glob, re, random, textwrap, time, types, json, collections, collections.abc
import os.path as osp
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, Sampler, DataLoader
sys.path.insert(0, "/kaggle/working/ReID")
from scanet import data as D

MARKET = "/kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15"
MASKS = "/kaggle/input/datasets/jiniyatanjina/mask-file"
NB = "/kaggle/working/ref/Person-ReID-STCANet/Jupyter Notebook/Image Based Person ReID STCANet.ipynb"
TRAIN_DIR, QUERY_DIR, GAL_DIR = f"{MARKET}/bounding_box_train", f"{MARKET}/query", f"{MARKET}/bounding_box_test"
FAILS = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
    if not ok: FAILS.append(name)
def neg(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'BAD '} | NEG-CONTROL {name}: {'DETECTED' if ok else 'NOT DETECTED'}" + (f" | {detail}" if detail else ""))
    if not ok: FAILS.append("neg-control " + name)

cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "authors_nb", "glob": glob, "re": re, "os": os, "osp": osp, "np": np, "random": random,
      "math": math, "torch": torch, "Image": Image, "Dataset": Dataset, "Sampler": Sampler, "copy": copy,
      "defaultdict": defaultdict, "T": T, "accimage": None,
      "collections": types.SimpleNamespace(Iterable=collections.abc.Iterable)}
def extract(i, names):
    src = "".join(cells[i]["source"])
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names:
            code = textwrap.dedent(ast.get_source_segment(src, n, padded=True))
            exec(compile(code, f"authors_cell{i}:{n.name}", "exec"), ns)
extract(1, {"_process_dir"})
extract(20, {"read_image", "ImageDataset", "ImageDataset_seg"})
extract(25, {"Compose", "ToTensor", "Normalize", "Scale", "RandomHorizontalFlip", "RandomErasing"})
exec(compile("".join(cells[24]["source"]), "authors_cell24", "exec"), ns)

# ---------- 0) authors' ToTensor on this env
try:
    ns["ToTensor"]()(Image.new("RGB", (4, 4)))
    print("authors' ToTensor runs unpatched on this env")
except Exception as e:
    print("authors' ToTensor FAILS here:", type(e).__name__, str(e)[:80], "-> PATCHED (same maths, numpy path)")
    def _patched(self, pic):
        arr = np.array(pic)
        if arr.ndim == 2: arr = arr[:, :, None]
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div(self.norm_value)
    ns["ToTensor"].__call__ = _patched
im_rgb = Image.fromarray(np.random.RandomState(0).randint(0, 256, (20, 10, 3), dtype=np.uint8))
im_l = Image.fromarray(np.random.RandomState(1).randint(0, 256, (20, 10), dtype=np.uint8))
check("ToTensor (authors') == torchvision to_tensor (RGB, L)",
      torch.equal(ns["ToTensor"]()(im_rgb), T.functional.to_tensor(im_rgb)) and
      torch.equal(ns["ToTensor"]()(im_l), T.functional.to_tensor(im_l)))
check("ToTensor (ours) == torchvision to_tensor (RGB, L)",
      torch.equal(D._pil_to_tensor(im_rgb), T.functional.to_tensor(im_rgb)) and
      torch.equal(D._pil_to_tensor(im_l), T.functional.to_tensor(im_l)))

# ---------- 1) loader parity + mask integrity
o_train, no_ids, no_n = D.process_dir(TRAIN_DIR, MASKS, relabel=True)
o_query, _, nq = D.process_dir(QUERY_DIR)
o_gal, _, ng = D.process_dir(GAL_DIR)
a_train, na_ids, na_n = ns["_process_dir"](None, TRAIN_DIR, MASKS, relabel=True)
a_query, _, _ = ns["_process_dir"](None, QUERY_DIR, relabel=False)
a_gal, _, _ = ns["_process_dir"](None, GAL_DIR, relabel=False)
print(f"counts ours: train {no_n} ({no_ids} ids) | query {nq} | gallery {ng}")
check("counts train/query/gallery == 12936/3368/15913, 751 ids",
      (no_n, nq, ng, no_ids) == (12936, 3368, 15913, 751) and (na_n, len(a_query), len(a_gal), na_ids) == (12936, 3368, 15913, 751))
a_by = {it[0][0]: it for it in a_train}
check("train image set identical", set(a_by) == {it[0][0] for it in o_train})
bad_meta, mp = 0, {}
for o in o_train:
    a = a_by[o[0][0]]
    mp.setdefault(a[1], o[1])
    if mp[a[1]] != o[1] or a[2] != o[2] or list(a[0][1:5]) != list(o[0][1:5]): bad_meta += 1
check("train camid + 4 mask paths same, relabel is a consistent bijection",
      bad_meta == 0 and len(set(mp.values())) == len(mp), f"bad={bad_meta}")
print("   relabel identical to authors' set-order labels:", all(k == v for k, v in mp.items()))
for nm, a_l, o_l in (("query", a_query, o_query), ("gallery", a_gal, o_gal)):
    check(f"{nm}: (path,pid,camid) identical",
          {it[0]: (it[1], it[2]) for it in a_l} == {it[0]: (it[1], it[2]) for it in o_l})

stems = {osp.splitext(osp.basename(it[0][0]))[0] for it in o_train}
check("mask folders == train image stems (12936)", set(os.listdir(MASKS)) == stems)
check("foreground.png exists for all train images", all(osp.exists(osp.join(MASKS, s, "foreground.png")) for s in stems))
bad_sz = bad_val = 0
for k in np.random.RandomState(0).choice(len(o_train), 300, replace=False):
    p = o_train[int(k)][0]
    isz = Image.open(p[0]).size
    for q in p[1:5]:
        m = Image.open(q)
        if m.size != isz: bad_sz += 1
        if not set(np.unique(np.array(m.convert("L"))).tolist()) <= {0, 255}: bad_val += 1
check("mask size == image size and values subset of {0,255} (300 sampled x 4)", bad_sz == 0 and bad_val == 0, f"size_bad={bad_sz} val_bad={bad_val}")

# ---------- 2) sampler parity
SA, SO = ns["RandomIdentitySampler"], D.RandomIdentitySampler
for s in range(3):
    random.seed(s); np.random.seed(s); ia = list(iter(SA(o_train, 4))); ra = (random.getstate(), np.random.get_state()[1].tolist(), np.random.get_state()[2])
    random.seed(s); np.random.seed(s); ib = list(iter(SO(o_train, 4))); rb = (random.getstate(), np.random.get_state()[1].tolist(), np.random.get_state()[2])
    check(f"sampler seed {s}: same index order + RNG state", [int(i) for i in ia] == [int(i) for i in ib] and ra == rb)
check("sampler len equal", len(SA(o_train, 4)) == len(SO(o_train, 4)), f"len={len(SO(o_train, 4))}")
random.seed(0); np.random.seed(0); ic = [int(i) for i in iter(SO(o_train, 4))]
random.seed(1); np.random.seed(1); idd = [int(i) for i in iter(SO(o_train, 4))]
neg("sampler different seed", ic != idd)
g = [o_train[i][1] for i in ic]
check("every group of 4 has one ID", all(len(set(g[j:j + 4])) == 1 for j in range(0, len(g), 4)))
dist_ids = [len(set(g[j:j + 64])) for j in range(0, len(g) - 63, 64)]
print(f"   distinct IDs per batch of 64: mean {np.mean(dist_ids):.2f} min {min(dist_ids)} max {max(dist_ids)} (16 would be guaranteed only with a PK sampler)")

# ---------- 3) joint transform parity
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tr_a = ns["Compose"]([ns["Scale"]((256, 128), interpolation=3), ns["RandomHorizontalFlip"](), ns["ToTensor"](),
                      ns["Normalize"](mean=MEAN, std=STD), ns["RandomErasing"](0.5)])
ds_a = ns["ImageDataset_seg"]([a_by[o[0][0]] for o in o_train], transform=tr_a)
N = 200
sel = [int(k) for k in np.random.RandomState(0).choice(len(o_train), N, replace=False)]
auth = []
for j, idx in enumerate(sel):
    random.seed(1000 + j)
    seq, pa, ca = ds_a[idx]
    auth.append((seq[:7].clone(), pa, ca, random.getstate()))
def ours_run(tf, off=0):
    ds = D.SCANetTrainDataset(o_train, tf); out = []
    for j, idx in enumerate(sel):
        random.seed(1000 + j + off)
        x, m, p, c = ds[idx]
        out.append((x, m, p, c, random.getstate()))
    return out
def n_diff(outs):
    return sum(not (torch.equal(a[0][:3], o[0]) and torch.equal(a[0][3:7], o[1])) for a, o in zip(auth, outs))
outs = ours_run(D.JointTrainTransform())
check("joint transform: image identical (200 samples)", all(torch.equal(a[0][:3], o[0]) for a, o in zip(auth, outs)))
check("joint transform: 4 masks identical (200 samples)", all(torch.equal(a[0][3:7], o[1]) for a, o in zip(auth, outs)))
check("pid/camid identical", all(a[1] == o[2] and a[2] == o[3] for a, o in zip(auth, outs)))
check("python-RNG state after each sample identical", all(a[3] == o[4] for a, o in zip(auth, outs)))
noerase = ours_run(D.JointTrainTransform(erase_prob=0.0))
def flipped(j):
    random.seed(1000 + j); return random.random() < 0.5
n_flip = sum(flipped(j) for j in range(N)); n_er = sum(not torch.equal(o[0], n_[0]) for o, n_ in zip(outs, noerase))
check("coverage: flip and erase both exercised", 20 < n_flip < 180 and n_er > 20, f"flipped={n_flip}/{N} erased={n_er}/{N}")
neg("different seed", n_diff(ours_run(D.JointTrainTransform(), off=1)) > 0)
neg("erase off", n_diff(noerase) > 0)
neg("erase value zeros", n_diff(ours_run(D.JointTrainTransform(erase_mean=(0., 0., 0.)))) > 0)
neg("std swapped", n_diff(ours_run(D.JointTrainTransform(std=(0.225, 0.229, 0.224)))) > 0)
neg("mask flipped vs image", any(not torch.equal(a[0][3:7], o[1].flip(-1)) for a, o in zip(auth, outs)))
neg("mask order swapped", any(not torch.equal(a[0][3:7], o[1][[1, 0, 2, 3]]) for a, o in zip(auth, outs)))

# ---------- 4) eval transform parity
try:
    tf_a = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    tf_a(Image.new("RGB", (64, 128))); print("authors' T.Resize(interpolation=3) works here")
except Exception as e:
    print("authors' T.Resize(interpolation=3) fails:", type(e).__name__, "-> using InterpolationMode.BICUBIC (== 3)")
    tf_a = T.Compose([T.Resize((256, 128), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
rs = np.random.RandomState(5)
sub_o = [o_query[int(k)] for k in rs.choice(len(o_query), 100, replace=False)] + [o_gal[int(k)] for k in rs.choice(len(o_gal), 100, replace=False)]
amap = {it[0]: it for it in a_query + a_gal}
da = ns["ImageDataset"]([amap[o[0]] for o in sub_o], transform=tf_a)
db = D.ReIDEvalDataset(sub_o, D.build_test_transform())
check("eval transform identical (200 query/gallery images) + pid/camid",
      all(torch.equal(da[i][0], db[i][0]) and da[i][1] == db[i][1] and da[i][2] == db[i][2] for i in range(len(sub_o))))
dbad = D.ReIDEvalDataset(sub_o, T.Compose([T.Resize((256, 128), interpolation=InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]))
neg("eval bilinear instead of bicubic", any(not torch.equal(da[i][0], dbad[i][0]) for i in range(40)))

# ---------- 5) DataLoader smoke
random.seed(0); np.random.seed(0); torch.manual_seed(0)
sampler = D.RandomIdentitySampler(o_train, 4)
loader = DataLoader(D.SCANetTrainDataset(o_train, D.JointTrainTransform()), batch_size=64, sampler=sampler,
                    num_workers=4, pin_memory=True, drop_last=True)
print("len(sampler):", len(sampler), "| len(loader):", len(loader))
check("len(loader) == len(sampler)//64", len(loader) == len(sampler) // 64)
t0 = time.time(); it = iter(loader); ok = True
for b in range(3):
    x, m, pid, cam = next(it)
    shape_ok = tuple(x.shape) == (64, 3, 256, 128) and tuple(m.shape) == (64, 4, 256, 128) and pid.shape == (64,) and x.dtype == torch.float32
    grp_ok = bool((pid.view(16, 4) == pid.view(16, 4)[:, :1]).all())
    rng_ok = float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    print(f"   batch {b}: shapes_ok={shape_ok} groups_ok={grp_ok} mask_range_ok={rng_ok} distinct_ids={len(set(pid.tolist()))} img_mean={x.mean():.3f} mask_mean={m.mean():.3f}")
    ok = ok and shape_ok and grp_ok and rng_ok
check("DataLoader batches (shape, 4-instance groups, mask in [0,1])", ok)
print(f"   3 batches in {time.time() - t0:.1f}s (incl. worker start-up)")

print("\nSUMMARY:", "ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
