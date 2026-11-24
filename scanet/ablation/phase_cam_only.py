"""Notebook driver for the CAM Only ablation: check / start / status. No training logic here (= scanet.ablation.launch_cam_only).
  python -m scanet.ablation.phase_cam_only check
  python -m scanet.ablation.phase_cam_only start  --seeds 0 1     (seed i of the list -> GPU i)
  python -m scanet.ablation.phase_cam_only status --seeds 0 1
Reuses scanet.phase (check, log parsing) and scanet.launch.plan (resume / refuse-finished). Output: /kaggle/working/runs_cam_only."""
import argparse, glob, os, subprocess, sys, time
from .. import phase as P, launch as L

RUNS = "/kaggle/working/runs_cam_only"
MOD = "scanet.ablation.launch_cam_only"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _running(s):
    return subprocess.run(["pgrep", "-f", f"{MOD} --seed {s} --out_root"], capture_output=True, text=True).stdout.split()

def start(seeds, epochs, ckpt_every):
    ngpu = len(P._sh("nvidia-smi -L").splitlines())
    if len(seeds) > ngpu: sys.exit(f"REFUSING: {len(seeds)} seeds but {ngpu} GPU(s)")
    plans = []
    for s in seeds:
        if _running(s): sys.exit(f"REFUSING: seed {s} already running (pids {_running(s)})")
        p = L.plan(f"{RUNS}/seed{s}")
        if p["state"] == "finished": sys.exit(f"REFUSING: seed {s} finished: {p['dir']}")
        plans.append(p)
    for i, (s, p) in enumerate(zip(seeds, plans)):
        sd = f"{RUNS}/seed{s}"; os.makedirs(sd, exist_ok=True)
        lg = open(f"{sd}/launch_{time.strftime('%Y%m%d_%H%M%S')}.log", "x")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        pr = subprocess.Popen([sys.executable, "-m", MOD, "--seed", str(s), "--out_root", RUNS, "--epochs", str(epochs),
                               "--ckpt_every", str(ckpt_every)], cwd=ROOT, env=env, stdout=lg,
                              stderr=subprocess.STDOUT, start_new_session=True)
        print(f"started seed {s} on GPU {i} pid {pr.pid} | {p['state']} -> attempt{p['attempt']} resume_epoch={p['resume_epoch']} | log {lg.name}")

def status(seeds, total=60):
    for s in seeds:
        sd = f"{RUNS}/seed{s}"
        if not os.path.isdir(sd): print(f"seed {s}: not started"); continue
        p = L.plan(sd); r = P._recs(sd); run = _running(s)
        last = (f"ep {r[-1]['epoch']}/{total} lr {r[-1]['lr']:.1e} xent {r[-1]['xent']:.3f} tri {r[-1]['tri']:.3f} "
                f"mask {r[-1]['mask']:.3f} acc {r[-1]['acc']:.3f}") if r else "no epoch yet"
        eta = ""
        if r and run:
            t = sum(x["time_s"] for x in r[-5:]) / len(r[-5:]); eta = f" | ~{t:.0f}s/ep, ~{(total - r[-1]['epoch']) * t / 60:.0f} min left (+eval)"
        state = ("FINISHED" if p["state"] == "finished" else "RUNNING" if run else
                 f"STOPPED (resumable from ep {p['resume_epoch']})" if p["state"] == "resume" else "STOPPED (no checkpoint)")
        print(f"seed {s}: {state} | {last}{eta}")
        logs = sorted(glob.glob(f"{sd}/launch_*.log"))
        if state.startswith("STOPPED") and logs and "Traceback" in open(logs[-1], errors="replace").read():
            print("   ERROR in log, last line:", [l.strip() for l in open(logs[-1], errors="replace")][-1:])
    print(P._sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader").replace("\n", " || "))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "start", "status"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--ckpt_every", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "check": sys.exit(0 if P.check() else 1)
    elif a.cmd == "start": start(a.seeds, a.epochs, a.ckpt_every)
    else: status(a.seeds, a.epochs)
