"""Negative control for test_parity_model_loss.py: injected bugs must be detected (rel_err > 1e-4)."""
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

THR, MISSED = 1e-4, []
def rel(a, b):
    return ((a.double() - b.double()).abs().max() / (a.double().abs().max() + 1e-12)).item()
def report(name, val, expect_detect):
    detected = val > THR
    ok = detected if expect_detect else not detected
    tag = ("DETECTED" if detected else "NOT DETECTED") if expect_detect else ("clean (as expected)" if ok else "UNEXPECTED DIFF")
    print(f"{'OK  ' if ok else 'BAD '} | {name}: rel_err={val:.3e} -> {tag}")
    if not ok: MISSED.append(name)

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
masks = (torch.rand(32, 4, 256, 128) > 0.5).float().cuda()

def maxrel(oa, ob):
    ka = [oa[0], oa[1]] + [t for grp in oa[2:] for t in grp]
    kb = [ob[0], ob[1]] + [t for grp in ob[2:] for t in grp]
    return max(rel(p, q) for p, q in zip(ka, kb))

A.eval()
with torch.no_grad():
    oa = A(x)
B.eval()
with torch.no_grad():
    report("baseline (unmutated)", maxrel(oa, B(x)), False)

def m_cam(m):    m.IAT2.W1[1].weight.data.mul_(1.1)
def m_sam(m):    m.IAT3.W2[1].weight.data.mul_(1.1)
def m_swap(m):
    w = m.IAT2.sa.conv.conv.weight.data
    w[[0, 1]] = w[[1, 0]].clone()
def m_stride(m):
    m.layer4[0].conv2.stride = (2, 2)
    m.layer4[0].downsample[0].stride = (2, 2)
for name, fn in [("CAM BN scale x1.1 (IAT2.W1)", m_cam), ("SAM BN scale x1.1 (IAT3.W2)", m_sam),
                 ("attention channel swap (head<->upper)", m_swap), ("layer4 stride 2 instead of 1", m_stride)]:
    Bm = copy.deepcopy(B); fn(Bm); Bm.eval()
    with torch.no_grad():
        report("model bug: " + name, maxrel(oa, Bm(x)), True)

# loss bugs (train mode, authors' loss formulas vs our criterion with a deliberate deviation)
A.train(); B.train()
crit_x = ns["CrossEntropyLabelSmooth"](num_classes=NC, use_gpu=True)
crit_t = ns["TripletLoss"](margin=0.3, distance="cosine")
crit_m = ns["MaskLoss"](mode="ce"); DS = ns["DeepSupervision"]
with torch.no_grad():
    o, f, ah, au, al, ash = A(x)
    xent = crit_x(o, pids); tri = crit_t(f, pids)
    mloss = (DS(crit_m, ah, masks[:, 0:1]) + DS(crit_m, au, masks[:, 1:2]) +
             DS(crit_m, al, masks[:, 2:3]) + DS(crit_m, ash, masks[:, 3:4])) / 4.0
    total_a = xent + tri + 0.5 * mloss
    ob = B(x)
    t0, _ = SCANetCriterion(NC, 0.3, 0.5, 0.1).cuda()(ob, pids, masks)
    report("baseline loss total", rel(total_a, t0), False)
    t1, _ = SCANetCriterion(NC, 0.3, 0.4, 0.1).cuda()(ob, pids, masks)
    report("loss bug: alpha 0.4", rel(total_a, t1), True)
    t2, _ = SCANetCriterion(NC, 0.31, 0.5, 0.1).cuda()(ob, pids, masks)
    report("loss bug: margin 0.31", rel(total_a, t2), True)
    t3, _ = SCANetCriterion(NC, 0.3, 0.5, 0.11).cuda()(ob, pids, masks)
    report("loss bug: label-smooth eps 0.11", rel(total_a, t3), True)
    t4, _ = SCANetCriterion(NC, 0.3, 0.5, 0.1).cuda()(ob, pids, masks[:, [1, 0, 2, 3]])
    report("loss bug: mask part order swapped", rel(total_a, t4), True)

print("\nSUMMARY:", "TEST IS SENSITIVE (all injected bugs detected, baseline clean)" if not MISSED else f"WEAK TEST, problems: {MISSED}")
sys.exit(1 if MISSED else 0)
