"""Tests of scanet.run: pretrained-weight loading, 2-epoch smoke + final-epoch eval, resume (with nondeterminism
floor and negative control), full-size rerank time/RAM. Thresholds are relative to the measured floor."""
import sys, os, json, math, time
import torch
sys.path.insert(0, "/kaggle/working/ReID")
from scanet import run as RUN, train as TR
from scanet.model import SCANet, RESNET50_URL

FAILS = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""), flush=True)
    if not ok: FAILS.append(name)

# ---------- 1) pretrained-weight loading
sd = torch.hub.load_state_dict_from_url(RESNET50_URL, progress=False, check_hash=True, weights_only=False)
m1 = SCANet(751, pretrained=True)
s1 = m1.state_dict()
keys = [k for k in sd if not k.startswith("fc.")]
bad = [k for k in keys if k not in s1 or not torch.equal(s1[k], sd[k])]
check("pretrained=True: all backbone tensors equal the resnet50-19c8e357 file", len(bad) == 0 and len(keys) > 300,
      f"{len(keys)} keys compared, mismatches {len(bad)} {bad[:3]}")
m0 = SCANet(751, pretrained=False)
check("negative control: pretrained=False differs from the file", not torch.equal(m0.state_dict()["conv1.weight"], sd["conv1.weight"]))
del m1, m0

# ---------- 2) smoke run A (2 epochs, ckpt every epoch, final eval)
n = 1
while os.path.exists(f"/kaggle/working/runs/smoke_run{n}"): n += 1
BASE = f"/kaggle/working/runs/smoke_run{n}"
print("run dir:", BASE, flush=True)
A = RUN.fit(f"{BASE}/A", seed=0, max_epoch=2, ckpt_every=1, do_eval=True, deterministic=True)
fin = all(math.isfinite(r[k]) for r in A["log"] for k in ("xent", "tri", "mask", "acc"))
check("smoke A: 2 epochs, all losses finite", fin and len(A["log"]) == 2, str([(r["epoch"], round(r["xent"], 3), round(r["tri"], 3), round(r["mask"], 3), r["n"]) for r in A["log"]]))
check("smoke A: LR in epoch 1,2 = 3.5e-4", all(abs(r["lr"] - 3.5e-4) < 1e-12 for r in A["log"]))
check("smoke A: samples per epoch = 185*64", all(r["n"] == 185 * 64 for r in A["log"]), str([r["n"] for r in A["log"]]))
check("smoke A: ckpt_epoch1/2 saved", all(os.path.exists(f"{BASE}/A/ckpt_epoch{e}.pt") for e in (1, 2)))
print("epoch time (s, deterministic=True):", [round(r["time_s"], 1) for r in A["log"]], flush=True)
E = A["final_epoch_eval"]
print("FINAL-EPOCH EVAL (2-epoch smoke model, NOT a result):", json.dumps(E["metrics"]), flush=True)
check("eval: sizes 3368 / 15913, feat 2048, plain+rerank both present",
      E["n_query"] == 3368 and E["n_gallery"] == 15913 and E["feat_dim"] == 2048 and set(E["metrics"]) == {"plain", "rerank"},
      f"{E['n_query']}/{E['n_gallery']}/{E['feat_dim']}/{sorted(E['metrics'])}")
print(f"full-size eval: extract {E['t_extract_s']:.1f}s | metrics incl. rerank {E['t_eval_s']:.1f}s | peak host RSS {E['peak_rss_gib']:.2f} GiB", flush=True)

# ---------- 3) nondeterminism floor, resume, negative control
A2 = RUN.fit(f"{BASE}/A2", seed=0, max_epoch=2, ckpt_every=1, do_eval=False, deterministic=True)
R = RUN.fit(f"{BASE}/R", seed=0, max_epoch=2, ckpt_every=1, do_eval=False, deterministic=True, resume=f"{BASE}/A/ckpt_epoch1.pt")
N = RUN.fit(f"{BASE}/N", seed=0, max_epoch=2, ckpt_every=1, do_eval=False, deterministic=True, resume=f"{BASE}/A/ckpt_epoch1.pt", restore_rng=False)

def rel(p1, p2):
    c1 = torch.load(p1, map_location="cpu", weights_only=False); c2 = torch.load(p2, map_location="cpu", weights_only=False)
    def cmp(d1, d2):
        w = 0.0
        for k in d1:
            a, b = d1[k], d2[k]
            if not torch.is_tensor(a): continue
            if not a.is_floating_point():
                if not torch.equal(a, b): return float("inf")
                continue
            a, b = a.double(), b.double()
            w = max(w, ((a - b).abs().max() / (a.abs().max() + 1e-12)).item())
        return w
    wm = cmp(c1["model"], c2["model"])
    wo = max(cmp(c1["opt"]["state"][i], c2["opt"]["state"][i]) for i in c1["opt"]["state"])
    return max(wm, wo), wm, wo

P = lambda d: f"{BASE}/{d}/ckpt_epoch2.pt"
floor, fm, fo = rel(P("A"), P("A2"))
res, rm, ro = rel(P("A"), P("R"))
neg, nm, no_ = rel(P("A"), P("N"))
print(f"epoch-2 worst rel (model, opt): floor A vs A2 = {floor:.3e} ({fm:.2e},{fo:.2e}) | resume A vs R = {res:.3e} ({rm:.2e},{ro:.2e}) | neg A vs N = {neg:.3e} ({nm:.2e},{no_:.2e})", flush=True)
check("resume (RNG restored) reproduces straight run within 3x the nondeterminism floor", res <= 3 * floor, f"{res:.3e} <= 3*{floor:.3e}")
ok_neg = neg > 3 * max(floor, res) and neg > 1e-5
print(f"{'OK  ' if ok_neg else 'BAD '} | NEG-CONTROL resume without RNG restore: {'DETECTED' if ok_neg else 'NOT DETECTED'} | {neg:.3e}")
if not ok_neg: FAILS.append("neg-control resume without RNG restore")
check("resume log: resumed run did only epoch 2, lr 3.5e-4", [r["epoch"] for r in R["log"]] == [2] and abs(R["log"][0]["lr"] - 3.5e-4) < 1e-12)

print("\nSUMMARY:", "ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
