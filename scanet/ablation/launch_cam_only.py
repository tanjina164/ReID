"""Launcher for one seed of the CAM Only ablation (CAM ON, SAM OFF, mask ON).
Same code path as scanet.launch (plan/resume/refuse-finished, authors' recipe, FP32); the ONLY change is that
scanet.run.SCANet is replaced by SCANetCamOnly at runtime. scanet/*.py files are untouched.
Layout: <out_root>/seed<S>/attempt<N>/ ; default out_root = /kaggle/working/runs_cam_only ; default ckpt_every = 20."""
import argparse, json, os, sys, time
import torch
from .. import run as R, launch as L
from .model_cam_only import SCANetCamOnly

VARIANT = "cam_only"

def apply_variant():
    R.SCANet = SCANetCamOnly
    assert R.SCANet is SCANetCamOnly, "variant swap failed"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_root", default="/kaggle/working/runs_cam_only")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ckpt_every", type=int, default=20)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    apply_variant()
    seed_dir = os.path.join(a.out_root, f"seed{a.seed}")
    p = L.plan(seed_dir)
    dev = {"visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "n": torch.cuda.device_count()}
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        dev.update(name=pr.name, uuid=str(getattr(pr, "uuid", "n/a")))
    nparam = None
    try:
        from ..model import count_num_param
        nparam = count_num_param(R.SCANet(751, pretrained=False))
    except Exception as e:
        print("param count failed:", repr(e)[:100])
    print("VARIANT", VARIANT, "| model class", R.SCANet.__name__, "| params excl. classifier (M)", nparam, flush=True)
    print("PLAN", json.dumps(p), "| DEVICE", json.dumps(dev), flush=True)
    if p["state"] == "finished":
        print("seed already finished, refusing:", p["dir"]); sys.exit(2)
    if a.dry_run:
        print("DRY RUN: nothing created"); return
    os.makedirs(seed_dir, exist_ok=True)
    cfg = {"variant": VARIANT, "model_class": R.SCANet.__name__, "params_M_excl_classifier": nparam,
           "seed": a.seed, "epochs": a.epochs, "ckpt_every": a.ckpt_every, "deterministic": bool(a.deterministic),
           "num_workers": a.num_workers, "plan": p, "device": dev, "torch": torch.__version__,
           "python": sys.version.split()[0],
           "git": L._git(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
           "started": time.ctime()}
    with open(os.path.join(seed_dir, f"attempt{p['attempt']}_config.json"), "x") as fh:
        json.dump(cfg, fh, indent=1)
    R.fit(p["out_dir"], seed=a.seed, max_epoch=a.epochs, ckpt_every=a.ckpt_every, resume=p["resume"],
          do_eval=True, deterministic=bool(a.deterministic), num_workers=a.num_workers)

if __name__ == "__main__":
    main()
