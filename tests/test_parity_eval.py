"""Parity test of scanet.evaluation against the authors' cells 12, 19, 21 (executed) + a hand-computed example."""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/kaggle/working/ReID")
from scanet import evaluation as E, data as D
from scanet.model import SCANet
from torch.utils.data import DataLoader

NB = "/kaggle/working/ref/Person-ReID-STCANet/Jupyter Notebook/Image Based Person ReID STCANet.ipynb"
MARKET = "/kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FAILS = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
    if not ok: FAILS.append(name)
def neg(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'BAD '} | NEG-CONTROL {name}: {'DETECTED' if ok else 'NOT DETECTED'}" + (f" | {detail}" if detail else ""))
    if not ok: FAILS.append("neg-control " + name)

cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "authors_nb", "np": np, "torch": torch}
for i in (12, 19, 21):
    exec(compile("".join(cells[i]["source"]), f"authors_cell{i}", "exec"), ns)
print("authors' CYTHON_EVAL_AVAI:", ns.get("CYTHON_EVAL_AVAI"), "(False => pure-Python metric, same as here)")

# ---- 1) hand-computed examples (independent of authors' code)
dm = np.array([[0.0, 0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
qp, qc = np.array([1]), np.array([0])
gp, gc = np.array([1, 2, 1, 3, 1]), np.array([0, 1, 1, 2, 2])
cmc, mAP = E.eval_market1501(dm, qp, gp, qc, gc, 50)
check("hand example: same-pid+same-cam removed, AP=0.5, CMC=[0,1,1,1]",
      np.allclose(cmc, [0, 1, 1, 1]) and abs(mAP - 0.5) < 1e-9, f"cmc={cmc.tolist()} mAP={mAP:.6f}")
dm2 = np.array([[0.1, 0.2, 0.8, 0.9], [0.9, 0.8, 0.1, 0.2]], dtype=np.float32)
cmc2, mAP2 = E.eval_market1501(dm2, np.array([1, 2]), np.array([1, 1, 2, 2]), np.array([0, 0]), np.array([1, 2, 1, 2]), 50)
check("hand example: perfect retrieval -> mAP=1, R1=1", abs(mAP2 - 1.0) < 1e-9 and abs(cmc2[0] - 1.0) < 1e-9)

# ---- synthetic data with identity structure
def make(nq, ng, n_id, dim, seed, absent=10):
    rs = np.random.RandomState(seed)
    centers = rs.randn(n_id + absent, dim).astype(np.float32)
    g_pid, g_cam = rs.randint(0, n_id, ng), rs.randint(0, 6, ng)
    q_pid, q_cam = rs.randint(0, n_id + absent, nq), rs.randint(0, 6, nq)   # some query ids absent from gallery
    gf = centers[g_pid] + rs.randn(ng, dim).astype(np.float32)
    qf = centers[q_pid] + rs.randn(nq, dim).astype(np.float32)
    return torch.from_numpy(qf), torch.from_numpy(gf), q_pid, g_pid, q_cam, g_cam

CD = lambda a, b: ns["compute_distance_matrix"](a, b, "cosine")
def authors_pipeline(qf, gf, qp, gp, qc, gc, rerank):
    distmat = CD(qf, gf).numpy()
    if rerank:
        distmat = ns["re_ranking"](distmat, CD(qf, qf), CD(gf, gf))      # torch tensors, as in the notebook
    return ns["evaluate_model"](distmat, qp, gp, qc, gc, use_metrics_cuhk03=False)

# ---- 2) end-to-end metric parity (plain + rerank, same features)
qf, gf, qp, gp, qc, gc = make(300, 2000, 100, 128, 0)
ours = E.evaluate_features(qf, gf, qp, gp, qc, gc, rerank=True)
for key, rr in (("plain", False), ("rerank", True)):
    cmc_a, map_a = authors_pipeline(qf, gf, qp, gp, qc, gc, rr)
    o = ours[key]
    check(f"{key}: CMC array identical to authors' (50 values)", np.array_equal(cmc_a, o["cmc"]),
          f"maxdiff={np.abs(cmc_a - o['cmc']).max():.2e}")
    check(f"{key}: mAP identical to authors'", float(map_a) == o["mAP"], f"authors {float(map_a):.6f} ours {o['mAP']:.6f}")
    check(f"{key}: R1/R5/R10/R20 == cmc[0,4,9,19]", all(o[f"R{r}"] == float(cmc_a[r - 1]) for r in (1, 5, 10, 20)),
          f"R1={o['R1']:.4f} R5={o['R5']:.4f} mAP={o['mAP']:.4f}")
check("rerank run differs from plain run (rerank is actually applied)",
      ours["plain"]["mAP"] != ours["rerank"]["mAP"], f"plain {ours['plain']['mAP']:.4f} vs rerank {ours['rerank']['mAP']:.4f}")

# ---- 3) re_ranking array parity (small) + negative controls
qf2, gf2, *_ = make(150, 700, 40, 64, 3)
dqg, dqq, dgg = CD(qf2, gf2).numpy(), CD(qf2, qf2).numpy(), CD(gf2, gf2).numpy()
ra = ns["re_ranking"](dqg, dqq, dgg)
rb = E.re_ranking(dqg, dqq, dgg)
check("re_ranking array identical (150x700)", rb.shape == (150, 700) and np.array_equal(ra, rb), f"maxdiff={np.abs(ra - rb).max():.2e}")
neg("re_ranking k1=19", not np.array_equal(ra, E.re_ranking(dqg, dqq, dgg, k1=19)))
neg("re_ranking k2=7", not np.array_equal(ra, E.re_ranking(dqg, dqq, dgg, k2=7)))
neg("re_ranking lambda=0.31", not np.array_equal(ra, E.re_ranking(dqg, dqq, dgg, lambda_value=0.31)))

# ---- 4) metric negative controls
cmc_a, map_a = authors_pipeline(qf, gf, qp, gp, qc, gc, False)
_, m_nocam = E.eval_market1501(CD(qf, gf).numpy(), qp, gp, qc + 100, gc, 50)
neg("camera filter off", float(map_a) != float(m_nocam), f"mAP {float(map_a):.4f} vs {float(m_nocam):.4f}")
eu = torch.cdist(qf, gf).pow(2).numpy()
_, m_eu = E.eval_market1501(eu, qp, gp, qc, gc, 50)
neg("euclidean instead of cosine", float(map_a) != float(m_eu), f"mAP {float(map_a):.4f} vs {float(m_eu):.4f}")

# ---- 5) feature extraction parity (authors' test() loop: f = model(imgs)[1].data.cpu())
items, _, _ = D.process_dir(f"{MARKET}/query")
items = items[:96]
ds = D.ReIDEvalDataset(items, D.build_test_transform())
torch.manual_seed(0)
model = SCANet(751, pretrained=False).cuda()
model.eval()
with torch.no_grad():
    fa = torch.cat([model(imgs.cuda())[1].data.cpu() for imgs, _, _ in DataLoader(ds, batch_size=32, shuffle=False)], 0)
model.train()
fb, pb, cb = E.extract_features(model, DataLoader(ds, batch_size=32, shuffle=False), "cuda")
check("extract_features == authors' loop (96 query imgs, bs 32), bitwise", torch.equal(fa, fb), f"shape {tuple(fb.shape)}")
check("extract_features restores model.train() mode", model.training is True)
check("pids/camids from loader are numpy ints", pb.shape == (96,) and cb.shape == (96,) and pb.dtype.kind == "i")
fc, _, _ = E.extract_features(model, DataLoader(ds, batch_size=7, shuffle=False), "cuda")
rel = ((fa.double() - fc.double()).abs().max() / fa.double().abs().max()).item()
check("extract_features batch-size independent (bs 7 vs 32), rel<1e-4", rel < 1e-4, f"rel={rel:.2e}")
model.train()
with torch.no_grad():
    ft = torch.cat([model(imgs.cuda())[1].data.cpu() for imgs, _, _ in DataLoader(ds, batch_size=32, shuffle=False)], 0)
neg("model in train-mode (BN batch stats)", not torch.equal(fa, ft))

print("\nSUMMARY:", "ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
