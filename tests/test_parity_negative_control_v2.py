"""Negative control v2 (v1 was weak: thr 1e-4, near-uniform logits, iid 50% masks).
Fixes: thr 1e-6; logits x10 on BOTH sides; per-part mask densities; component-wise loss checks."""
import sys, json, copy
import torch, torch.nn as nn
sys.path.insert(0, "/kaggle/working/ReID")
from scanet.model import SCANet
from scanet.losses import SCANetCriterion

NB = "/kaggle/working/ref/Person-ReID-STCANet/Jupyter Notebook/Image Based Person ReID STCANet.ipynb"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "authors_nb"}
for i in (5, 7, 8, 9, 22):
    exec(compile("".join(cells[i]["source"]), f"authors_cell{i}", "exec"), ns)

THR, BAD = 1e-6, []
def rel(a, b):
    return ((a.double() - b.double()).abs().max() / (a.double().abs().max() + 1e-12)).item()
def report(name, val, expect_detect):
    detected = val > THR
    ok = detected if expect_detect else not detected
    tag = ("DETECTED" if detected else "NOT DETECTED") if expect_detect else ("clean (as expected)" if ok else "UNEXPECTED DIFF")
    print(f"{'OK  ' if ok else 'BAD '} | {name}: rel_err={val:.3e} -> {tag}")
    if not ok: BAD.append(name)

NC = 751
torch.manual_seed(0)
A = ns["STCANet"](NC).cuda()
g = torch.Generator().manual_seed(123)
with torch.no_grad():
    for n_, mod in A.named_modules():
        if (n_.startswith(("IAT2", "IAT3")) or n_ == "bn") and isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.copy_(0.5 + torch.rand(mod.weight.shape, generator=g))
            mod.bias.copy_(0.1 * torch.randn(mod.bias.shape, generator=g))
            mod.running_mean.copy_(0.1 * torch.randn(mod.running_mean.shape, generator=g))
            mod.running_var.copy_(0.5 + torch.rand(mod.running_var.shape, generator=g))
B = SCANet(NC, pretrained=False).cuda()
B.load_state_dict(A.state_dict(), strict=True)

torch.manual_seed(1)
x = torch.randn(32, 3, 256, 128).cuda()
pids = torch.randperm(NC)[:8].repeat_interleave(4).cuda()
dens = (0.1, 0.4, 0.7, 0.95)
masks = torch.stack([(torch.rand(32, 256, 128) < p).float() for p in dens], 1).cuda()

def maxrel(oa, ob):
    ka = [oa[0], oa[1]] + [t for grp in oa[2:] for t in grp]
    kb = [ob[0], ob[1]] + [t for grp in ob[2:] for t in grp]
    return max(rel(p, q) for p, q in zip(ka, kb))

# ---- model bugs
A.eval(); B.eval()
with torch.no_grad():
    oa = A(x)
    report("baseline model (unmutated)", maxrel(oa, B(x)), False)
def m_cam(m):    m.IAT2.W1[1].weight.data.mul_(1.001)
def m_sam(m):    m.IAT3.W2[1].weight.data.mul_(1.001)
def m_swap(m):
    w = m.IAT2.sa.conv.conv.weight.data
    w[[0, 1]] = w[[1, 0]].clone()
def m_stride(m):
    m.layer4[0].conv2.stride = (2, 2); m.layer4[0].downsample[0].stride = (2, 2)
for name, fn in [("CAM BN scale x1.001", m_cam), ("SAM BN scale x1.001", m_sam),
                 ("attention channel swap", m_swap), ("layer4 stride 2", m_stride)]:
    Bm = copy.deepcopy(B); fn(Bm); Bm.eval()
    with torch.no_grad():
        report("model bug: " + name, maxrel(oa, Bm(x)), True)

# ---- loss bugs (train mode, logits x10 on both sides)
A.train(); B.train()
cx = ns["CrossEntropyLabelSmooth"](num_classes=NC, use_gpu=True)
ct = ns["TripletLoss"](margin=0.3, distance="cosine")
cm = ns["MaskLoss"](mode="ce"); DS = ns["DeepSupervision"]
with torch.no_grad():
    o, f, ah, au, al, ash = A(x)
    o10 = o * 10
    xa = cx(o10, pids); ta = ct(f, pids)
    ma = (DS(cm, ah, masks[:, 0:1]) + DS(cm, au, masks[:, 1:2]) + DS(cm, al, masks[:, 2:3]) + DS(cm, ash, masks[:, 3:4])) / 4.0
    tot_a = xa + ta + 0.5 * ma
    ob = list(B(x)); ob[0] = ob[0] * 10; ob = tuple(ob)
    print(f"reference values: xent {xa.item():.4f} tri {ta.item():.4f} mask {ma.item():.4f} total {tot_a.item():.4f}")

    def run(alpha=0.5, margin=0.3, eps=0.1, perm=None):
        c = SCANetCriterion(NC, margin, alpha, eps).cuda()
        return c(ob, pids, masks if perm is None else masks[:, perm])

    t, p = run()
    report("baseline loss [total]", rel(tot_a, t), False)
    report("baseline loss [xent]", rel(xa, p["xent"]), False)
    report("baseline loss [tri]", rel(ta, p["tri"]), False)
    report("baseline loss [mask]", rel(ma, p["mask"]), False)
    t, p = run(alpha=0.4);   report("loss bug: alpha 0.4 [total]", rel(tot_a, t), True)
    t, p = run(margin=0.31); report("loss bug: margin 0.31 [tri]", rel(ta, p["tri"]), True)
    t, p = run(eps=0.11);    report("loss bug: label-smooth eps 0.11 [xent]", rel(xa, p["xent"]), True)
    t, p = run(eps=0.0);     report("loss bug: no label smoothing [xent]", rel(xa, p["xent"]), True)
    t, p = run(perm=[1, 0, 2, 3]); report("loss bug: mask head<->upper swap [mask]", rel(ma, p["mask"]), True)

print("\nSUMMARY:", "TEST IS SENSITIVE (thr 1e-6): baselines clean, all injected bugs detected" if not BAD else f"STILL WEAK/UNEXPECTED: {BAD}")
sys.exit(1 if BAD else 0)
