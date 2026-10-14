"""Launcher for one seed of the SCANet reproduction (one process = one GPU, pinned by CUDA_VISIBLE_DEVICES).
Layout: <out_root>/seed<S>/attempt<N>/ (fit output), attempt<N>.log, attempt<N>_config.json (outside the attempt dir).
A relaunch resumes from the highest-epoch checkpoint of earlier attempts; finished seeds are refused."""
import argparse, glob, json, os, re, subprocess, sys, time
import torch
from . import run as R

def _git(root):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip() or "unknown"
    except Exception:
        return "unknown"

def plan(seed_dir):
    atts = []
    for d in glob.glob(os.path.join(seed_dir, "attempt*")):
        m = re.fullmatch(r"attempt(\d+)", os.path.basename(d))
        if m and os.path.isdir(d):
            atts.append((int(m.group(1)), d))
    atts.sort()
    for _, d in atts:
        if os.path.exists(os.path.join(d, "result.json")):
            return {"state": "finished", "dir": d}
    best = None
    for _, d in atts:
        for p in glob.glob(os.path.join(d, "ckpt_epoch*.pt")):
            e = int(re.search(r"ckpt_epoch(\d+)\.pt", p).group(1))
            if best is None or e > best[0]:
                best = (e, p)
    n = (atts[-1][0] + 1) if atts else 1
    return {"state": "resume" if best else "fresh", "attempt": n,
            "out_dir": os.path.join(seed_dir, f"attempt{n}"),
            "resume": best[1] if best else None, "resume_epoch": best[0] if best else 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_root", default="/kaggle/working/runs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    seed_dir = os.path.join(a.out_root, f"seed{a.seed}")
    p = plan(seed_dir)
    dev = {"visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "n": torch.cuda.device_count()}
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        dev.update(name=pr.name, uuid=str(getattr(pr, "uuid", "n/a")))
    print("PLAN", json.dumps(p), "| DEVICE", json.dumps(dev), flush=True)
    if p["state"] == "finished":
        print("seed already finished, refusing:", p["dir"]); sys.exit(2)
    if a.dry_run:
        print("DRY RUN: nothing created"); return
    os.makedirs(seed_dir, exist_ok=True)
    cfg = {"seed": a.seed, "epochs": a.epochs, "ckpt_every": a.ckpt_every, "deterministic": bool(a.deterministic),
           "num_workers": a.num_workers, "plan": p, "device": dev, "torch": torch.__version__,
           "git": _git(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "started": time.ctime()}
    with open(os.path.join(seed_dir, f"attempt{p['attempt']}_config.json"), "x") as fh:
        json.dump(cfg, fh, indent=1)
    R.fit(p["out_dir"], seed=a.seed, max_epoch=a.epochs, ckpt_every=a.ckpt_every, resume=p["resume"],
          do_eval=True, deterministic=bool(a.deterministic), num_workers=a.num_workers)

if __name__ == "__main__":
    main()
