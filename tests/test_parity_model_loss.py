"""Parity test: scanet.model/losses vs the authors' image notebook (executed verbatim).
Authors: github.com/FatimaZulfiqar/Person-ReID-STCANet (commit 1b1eb9dd), notebook cells 5,7,8,9,22.
Thresholds (relative error) are the reproducer's choice: fwd/loss/buffers 1e-4, grads 1e-3."""
import sys, json
import torch
import torch.nn as nn
sys.path.insert(0, "/kaggle/working/ReID")
from scanet.model import SCANet, count_num_param
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

FAILS = []
def rel(a, b):
    return ((a.double() - b.double()).abs().max() / (a.double().abs().max() + 1e-12)).item()
def check(name, val, thr):
    ok = val < thr
    print(f"{'PASS' if ok else 'FAIL'} | {name}: rel_err={val:.3e} (thr {thr:.0e})")
    if not ok: FAILS.append(name)

NC = 751
torch.manual_seed(0)
A = ns["STCANet"](NC).cuda()                       # authors' model (pretrained, their code)
B = SCANet(NC, pretrained=False).cuda()            # ours

# randomize attention-module BN so zero-init does not hide bugs
g = torch.Generator().manual_seed(123)
with torch.no_grad():
    mods = [m for n_, mod in A.named_modules() if n_.startswith(("IAT2", "IAT3")) or n_ == "bn"
            for m in [mod] if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    for m in mods:
        m.weight.copy_(0.5 + torch.rand(m.weight.shape, generator=g))
        m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=g))
        m.running_mean.copy_(0.1 * torch.randn(m.running_mean.shape, generator=g))
        m.running_var.copy_(0.5 + torch.rand(m.running_var.shape, generator=g))
print("randomized BN modules:", len(mods))

# 1) structure
sa, sb = A.state_dict(), B.state_dict()
print("state_dict keys: authors", len(sa), "| ours", len(sb))
print("only in authors:", sorted(set(sa) - set(sb))[:10])
print("only in ours:", sorted(set(sb) - set(sa))[:10])
shape_bad = [k for k in sa if k in sb and sa[k].shape != sb[k].shape]
print("shape mismatches:", shape_bad[:10])
if set(sa) != set(sb) or shape_bad: FAILS.append("state_dict structure")
B.load_state_dict(sa, strict=True)
pa_, pb_ = count_num_param(A), count_num_param(B)
print(f"params excl classifier (M): authors {pa_:.6f} | ours {pb_:.6f}")
if abs(pa_ - pb_) > 1e-9: FAILS.append("param count")

# inputs
torch.manual_seed(1)
x = torch.randn(32, 3, 256, 128).cuda()
ids = torch.randperm(NC)[:8]
pids = ids.repeat_interleave(4).cuda()
masks = (torch.rand(32, 4, 256, 128) > 0.5).float().cuda()

def flat(out):
    y, f, ah, au, al, ash = out
    return {"y": y, "f": f, **{f"a_{n}{i}": t for n, a in zip(("head", "upper", "lower", "shoes"), (ah, au, al, ash))
                               for i, t in enumerate(a)}}

# 2) eval forward
A.eval(); B.eval()
with torch.no_grad():
    oa, ob = flat(A(x)), flat(B(x))
for k in oa: check(f"eval fwd {k}", rel(oa[k], ob[k]), 1e-4)

# 3) train forward + loss + backward
A.train(); B.train()
crit_x = ns["CrossEntropyLabelSmooth"](num_classes=NC, use_gpu=True)
crit_t = ns["TripletLoss"](margin=0.3, distance="cosine")
crit_m = ns["MaskLoss"](mode="ce")
DS = ns["DeepSupervision"]

A.zero_grad(); B.zero_grad()
outputs, features, a_head, a_upper, a_lower, a_shoes = A(x)
xent = crit_x(outputs, pids); tri = crit_t(features, pids)
hl = DS(crit_m, a_head, masks[:, 0:1]); ul = DS(crit_m, a_upper, masks[:, 1:2])
ll = DS(crit_m, a_lower, masks[:, 2:3]); sl = DS(crit_m, a_shoes, masks[:, 3:4])
mloss = (hl + ul + ll + sl) / 4.0
total_a = xent + tri + 0.5 * mloss
total_a.backward()

crit_ours = SCANetCriterion(NC, margin=0.3, alpha=0.5).cuda()
out_b = B(x)
total_b, parts = crit_ours(out_b, pids, masks)
total_b.backward()

check("train loss xent", rel(xent, parts["xent"]), 1e-4)
check("train loss triplet", rel(tri, parts["tri"]), 1e-4)
check("train loss mask", rel(mloss, parts["mask"]), 1e-4)
check("train loss total", rel(total_a, total_b), 1e-4)
print(f"values: xent {xent.item():.5f} tri {tri.item():.5f} mask {mloss.item():.5f} total {total_a.item():.5f}")
check("train fwd f", rel(features, out_b[1]), 1e-4)

pbn = dict(B.named_parameters())
worst, wname, none_mismatch = 0.0, "", 0
for n_, p in A.named_parameters():
    q = pbn[n_]
    if (p.grad is None) != (q.grad is None): none_mismatch += 1; continue
    if p.grad is None: continue
    r = rel(p.grad, q.grad)
    if r > worst: worst, wname = r, n_
print("grad params:", sum(1 for _ in A.parameters()), "| grad None mismatches:", none_mismatch, "| worst:", wname)
check("grad worst param", worst, 1e-3)
if none_mismatch: FAILS.append("grad None mismatch")

sa2, sb2 = A.state_dict(), B.state_dict()
wb = max(rel(sa2[k].float(), sb2[k].float()) for k in sa2 if sa2[k].is_floating_point())
check("BN buffers after train fwd", wb, 1e-4)

print("\nSUMMARY:", "ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
