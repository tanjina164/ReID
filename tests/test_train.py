"""Parity test of scanet.train against the authors' cell 27 (`train` executed verbatim).
Thresholds (relative error) are the reproducer's choice: parameters/buffers 1e-5."""
import sys, json, io, re, copy, time, contextlib
import numpy as np
import torch
sys.path.insert(0, "/kaggle/working/ReID")
from scanet import train as TR, data as D
from scanet.model import SCANet
from scanet.losses import SCANetCriterion

NB = "/kaggle/working/ref/Person-ReID-STCANet/Jupyter Notebook/Image Based Person ReID STCANet.ipynb"
MARKET = "/kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15"
MASKS = "/kaggle/input/datasets/jiniyatanjina/mask-file"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
TR.seed_everything(0, deterministic=True)

FAILS = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
    if not ok: FAILS.append(name)
def neg(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'BAD '} | NEG-CONTROL {name}: {'DETECTED' if ok else 'NOT DETECTED'}" + (f" | {detail}" if detail else ""))
    if not ok: FAILS.append("neg-control " + name)

cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
src27 = "".join(cells[27]["source"])
ns = {"__name__": "authors_nb"}
for i in (5, 7, 8, 9, 13, 22, 23, 27):
    exec(compile("".join(cells[i]["source"]), f"authors_cell{i}", "exec"), ns)

# ---------- 1) hyper-parameters from the authors' executed code + epoch/step order
keys = ["optim", "lr", "weight_decay", "stepsize", "gamma", "max_epoch", "train_batch", "num_instances",
        "margin", "alpha", "mode", "distance", "flip_cnt", "rerank", "height", "width", "seed"]
print("authors' cell 27 values:", {k: ns[k] for k in keys})
check("authors' hyper-parameters match the reproduction config",
      ns["optim"] == "adam" and ns["lr"] == 0.00035 and ns["weight_decay"] == 5e-04 and ns["stepsize"] == [20, 40]
      and ns["gamma"] == 0.1 and ns["max_epoch"] == 60 and ns["train_batch"] == 64 and ns["num_instances"] == 4
      and ns["margin"] == 0.3 and ns["alpha"] == 0.5 and ns["mode"] == "ce" and ns["distance"] == "cosine"
      and ns["flip_cnt"] == 1 and ns["rerank"] is False and (ns["height"], ns["width"]) == (256, 128))
i1, i2 = src27.find("scheduler.step()"), src27.find("train(epoch, model")
check("authors' main(): scheduler.step() comes before train(...) inside the epoch loop", 0 < i1 < i2, f"pos {i1} < {i2}")

# ---------- 2) LR trace
def transitions(lrs):
    out, prev = [], None
    for e, v in enumerate(lrs, 1):
        if v != prev: out.append((e, v)); prev = v
    return out
exp = [3.5e-4] * 19 + [3.5e-5] * 20 + [3.5e-6] * 21
lrs = TR.lr_trace(60, 3.5e-4, (20, 40), 0.1, step_first=True)
print("authors-order LR transitions (1-based epoch, lr):", [(e, f"{v:.2e}") for e, v in transitions(lrs)])
check("LR trace (authors' order) == derived 19/20/21 epochs of 3.5e-4/3.5e-5/3.5e-6",
      len(lrs) == 60 and np.allclose(lrs, exp, rtol=1e-9, atol=0))
lrs_modern = TR.lr_trace(60, 3.5e-4, (20, 40), 0.1, step_first=False)
print("modern-order LR transitions:", [(e, f"{v:.2e}") for e, v in transitions(lrs_modern)])
neg("scheduler.step() after the epoch (modern order)", not np.allclose(lrs_modern, exp))

# ---------- 3) training-loop parity
NC = 751
torch.manual_seed(0)
A = ns["STCANet"](NC).cuda()
init_state = copy.deepcopy(A.state_dict())
items, _, _ = D.process_dir(f"{MARKET}/bounding_box_train", MASKS, relabel=True)
loader = TR.build_train_loader(items, 64, 4, 4, seed=0)
batches = [b for _, b in zip(range(4), loader)]
del loader
print("batches:", len(batches), "| x", tuple(batches[0][0].shape), "| masks", tuple(batches[0][1].shape), "| pids", tuple(batches[0][2].shape))
auth_batches = [(torch.cat([x, m, torch.zeros_like(m[:, :1])], 1), pid, cam) for x, m, pid, cam in batches]

crit_x = ns["CrossEntropyLabelSmooth"](num_classes=NC, use_gpu=True)
crit_t = ns["TripletLoss"](margin=0.3, distance="cosine")
crit_m = ns["MaskLoss"](mode="ce")
opt_a = ns["init_optim"]("adam", A.parameters(), 0.00035, 5e-04)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns["train"](0, A, crit_x, crit_t, crit_m, opt_a, auth_batches)
line = buf.getvalue().strip()
print("authors' train() printed:", line)
mt = re.search(r"xentLoss:([\d.]+) triLoss:([\d.]+) MaskLoss:([\d.]+) Acc:([\d.]+)%", line)
auth_m = [float(g) for g in mt.groups()] if mt else None
sa = A.state_dict()

def worst_rel(s1, s2):
    w, wk, int_ok = 0.0, "", True
    for k in s1:
        if not s1[k].is_floating_point():
            int_ok = int_ok and torch.equal(s1[k], s2[k]); continue
        a, b = s1[k].double(), s2[k].double()
        r = ((a - b).abs().max() / (a.abs().max() + 1e-12)).item()
        if r > w: w, wk = r, k
    return w, wk, int_ok

moved, mk, _ = worst_rel(init_state, sa)
check("sanity: 4 Adam steps actually moved the weights (authors' run)", moved > 1e-4, f"max rel change {moved:.2e} at {mk}")

def run_ours(alpha=0.5, lr=3.5e-4, wd=5e-4, bl=None):
    B = SCANet(NC, pretrained=False).cuda()
    B.load_state_dict(init_state, strict=True)
    crit = SCANetCriterion(NC, 0.3, alpha).cuda()
    opt = TR.build_optimizer(B, lr, wd)
    met = TR.train_one_epoch(B, crit, opt, bl if bl is not None else batches)
    return B, met

torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
B, met = run_ours()
torch.cuda.synchronize(); dt = time.time() - t0
print(f"ours: {met}")
w, wk, int_ok = worst_rel(sa, B.state_dict())
check("params+buffers after 4 steps: authors' train() vs ours, rel<1e-5", w < 1e-5 and int_ok, f"worst rel {w:.3e} at '{wk}', int buffers equal={int_ok}")
if auth_m:
    ok = (abs(auth_m[0] - met["xent"]) < 6e-5 and abs(auth_m[1] - met["tri"]) < 6e-5 and
          abs(auth_m[2] - met["mask"]) < 6e-5 and abs(auth_m[3] - 100 * met["acc"]) < 6e-3)
    check("epoch metrics (xent/tri/mask/acc) match the authors' printed line", ok,
          f"authors {auth_m} | ours {[round(met['xent'], 4), round(met['tri'], 4), round(met['mask'], 4), round(100 * met['acc'], 2)]}")
else:
    check("could not parse authors' printed metrics", False, line[:120])
print(f"peak GPU memory (both models resident, batch 64): {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | "
      f"4 iters incl. warm-up: {dt:.1f}s (rough)")
del B; torch.cuda.empty_cache()

for name, kw in [("alpha 0.4", dict(alpha=0.4)), ("weight_decay 0", dict(wd=0.0)),
                 ("lr 3.5e-5", dict(lr=3.5e-5)), ("batch order reversed", dict(bl=batches[::-1]))]:
    Bn, _ = run_ours(**kw)
    wn, _, _ = worst_rel(sa, Bn.state_dict())
    neg(name, wn > 1e-5, f"worst rel {wn:.2e}")
    del Bn; torch.cuda.empty_cache()

print("\nSUMMARY:", "ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
