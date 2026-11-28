"""Launcher for one seed of CAM Only + Mask OFF (CAM ON, SAM OFF, alpha = 0).
Same as launch_cam_only.py except scanet.run.SCANetCriterion is replaced at runtime by a subclass with alpha=0.0
(fit() hard-codes alpha=0.5 at run.py:87; the subclass asserts that, so a changed run.py is detected).
scanet/*.py and existing ablation files are untouched. Default out_root = /kaggle/working/runs_cam_only_nomask.
GPU is NOT chosen here: caller must set CUDA_VISIBLE_DEVICES (refuses otherwise)."""
import argparse, json, os, sys, time
import torch
from .. import run as R, launch as L
from ..losses import SCANetCriterion
from .model_cam_only import SCANetCamOnly

VARIANT = "cam_only_nomask"

class CriterionAlpha0(SCANetCriterion):
    def __init__(self, num_classes, margin=0.3, alpha=0.5, epsilon=0.1):
        assert alpha == 0.5 and margin == 0.3, f"run.py changed: margin={margin} alpha={alpha}"
        super().__init__(num_classes, margin, 0.0, epsilon)

def apply_variant():
    R.SCANet = SCANetCamOnly
    R.SCANetCriterion = CriterionAlpha0
    assert R.SCANet is SCANetCamOnly and R.SCANetCriterion is CriterionAlpha0, "variant swap failed"
    assert R.SCANetCriterion(751, 0.3, 0.5).alpha == 0.0, "alpha not 0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_root", default="/kaggle/working/runs_cam_only_nomask")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ckpt_every", type=int, default=20)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        sys.exit("REFUSING: set CUDA_VISIBLE_DEVICES (e.g. =1) before launching")
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
    print("VARIANT", VARIANT, "| model class", R.SCANet.__name__, "| criterion", R.SCANetCriterion.__name__,
          "| alpha", R.SCANetCriterion(751, 0.3, 0.5).alpha, "| params excl. classifier (M)", nparam, flush=True)
    print("PLAN", json.dumps(p), "| DEVICE", json.dumps(dev), flush=True)
    if p["state"] == "finished":
        print("seed already finished, refusing:", p["dir"]); sys.exit(2)
    if a.dry_run:
        print("DRY RUN: nothing created"); return
    os.makedirs(seed_dir, exist_ok=True)
    cfg = {"variant": VARIANT, "model_class": R.SCANet.__name__, "criterion_class": R.SCANetCriterion.__name__,
           "alpha": 0.0, "params_M_excl_classifier": nparam,
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
