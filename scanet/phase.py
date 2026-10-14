"""Notebook driver: check / start / status / pack.  No training logic here; training = scanet.launch.
  python -m scanet.phase check
  python -m scanet.phase start  --seeds 0 1      (seed i of the list -> GPU i, one background process each)
  python -m scanet.phase status [--seeds 0 1 2 3 4]
  python -m scanet.phase pack   --seeds 0 1      (eval-only ZIP per finished seed, verified 90-110 MB)"""
import argparse, glob, json, os, shutil, subprocess, sys, time, urllib.request
import torch
from . import data as D, launch as L, pack as P
from .run import MARKET, MASKS

RUNS = "/kaggle/working/runs"; ZIPS = "/kaggle/working/zips"
def _ok(c, m): print("PASS" if c else "FAIL", "|", m); return bool(c)
def _sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
def _running(s):
    return subprocess.run(["pgrep", "-f", f"scanet.launch --seed {s} --out_root"], capture_output=True, text=True).stdout.split()
def _recs(sd):
    d = {}
    for lg in sorted(glob.glob(f"{sd}/launch_*.log")):
        for l in open(lg, errors="replace"):
            if l.startswith("EPOCH "):
                try: r = json.loads(l[6:]); d[r["epoch"]] = r
                except Exception: pass
    return [d[k] for k in sorted(d)]

def check():
    import torchvision
    ok = True
    print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| python", sys.version.split()[0])
    gpus = _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").splitlines()
    ok &= _ok(len(gpus) >= 1, f"GPUs: {gpus}")
    ok &= _ok(os.path.isdir(MARKET) and os.path.isdir(MASKS), f"dataset dirs exist: {MARKET} | {MASKS}")
    if os.path.isdir(MARKET) and os.path.isdir(MASKS):
        tr, ncls, _ = D.process_dir(f"{MARKET}/bounding_box_train", MASKS, relabel=True)
        q, _, _ = D.process_dir(f"{MARKET}/query", None, relabel=False)
        g, _, _ = D.process_dir(f"{MARKET}/bounding_box_test", None, relabel=False)
        ok &= _ok((len(tr), ncls, len(q), len(g)) == (12936, 751, 3368, 15913), f"train/ids/query/gallery = {(len(tr), ncls, len(q), len(g))} (expected 12936/751/3368/15913)")
    free = shutil.disk_usage("/kaggle/working").free / 2**30
    ok &= _ok(free >= 12, f"free disk {free:.1f} GiB (>=12 needed for 2 seeds)")
    mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 2**20
    ok &= _ok(mem >= 24, f"available RAM {mem:.1f} GiB (>=24 for 2 parallel evals)")
    cached = os.path.exists("/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth")
    net = False
    if not cached:
        try: urllib.request.urlopen(urllib.request.Request("https://download.pytorch.org/models/resnet50-19c8e357.pth", method="HEAD"), timeout=10); net = True
        except Exception as e: print("   internet probe:", repr(e)[:120])
    ok &= _ok(cached or net, f"resnet50 weights cached={cached} or downloadable={net}")
    print("git:", _sh(f"git -C {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))} rev-parse --short HEAD"))
    print("CHECK:", "ALL PASS" if ok else "FAILED"); return ok

def start(seeds, epochs, ckpt_every):
    ngpu = len(_sh("nvidia-smi -L").splitlines())
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
        pr = subprocess.Popen([sys.executable, "-m", "scanet.launch", "--seed", str(s), "--out_root", RUNS, "--epochs", str(epochs),
                               "--ckpt_every", str(ckpt_every)], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              env=env, stdout=lg, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"started seed {s} on GPU {i} pid {pr.pid} | {p['state']} -> attempt{p['attempt']} resume_epoch={p['resume_epoch']} | log {lg.name}")

def status(seeds, total=60):
    for s in seeds:
        sd = f"{RUNS}/seed{s}"
        if not os.path.isdir(sd): print(f"seed {s}: not started"); continue
        p = L.plan(sd); r = _recs(sd); run = _running(s)
        last = f"ep {r[-1]['epoch']}/{total} lr {r[-1]['lr']:.1e} xent {r[-1]['xent']:.3f} tri {r[-1]['tri']:.3f} mask {r[-1]['mask']:.3f} acc {r[-1]['acc']:.3f}" if r else "no epoch yet"
        eta = ""
        if r and run:
            t = sum(x["time_s"] for x in r[-5:]) / len(r[-5:]); eta = f" | ~{t:.0f}s/ep, ~{(total - r[-1]['epoch']) * t / 60:.0f} min left (+eval)"
        state = "FINISHED" if p["state"] == "finished" else ("RUNNING" if run else f"STOPPED (resumable from ep {p['resume_epoch']})" if p["state"] == "resume" else "STOPPED (no checkpoint)")
        print(f"seed {s}: {state} | {last}{eta}")
        if state.startswith("STOPPED") and glob.glob(f"{sd}/launch_*.log"):
            tail = [l.strip() for l in open(sorted(glob.glob(f"{sd}/launch_*.log"))[-1], errors="replace")][-1:]
            if "Traceback" in "".join(open(sorted(glob.glob(f"{sd}/launch_*.log"))[-1], errors="replace")): print("   ERROR in log, last line:", tail)
    print(_sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader").replace("\n", " || "),
          "| RAM used", _sh("free -g | awk '/Mem:/{print $3}'"), "GiB")

def pack(seeds):
    os.makedirs(ZIPS, exist_ok=True)
    for s in seeds:
        out = f"{ZIPS}/seed{s}_final.zip"
        if os.path.exists(out): print(f"seed {s}: {out} already exists, skipping (never overwrite)"); continue
        rdir, ep = P.pack(f"{RUNS}/seed{s}", out)
        res = P.verify(out, os.path.join(rdir, f"ckpt_epoch{ep}.pt"))
        lines = [("PASS" if g else "FAIL") + " | " + m for g, m in res]
        print(f"seed {s}:", *lines, sep="\n  "); open(f"{ZIPS}/seed{s}_final.verify.txt", "x").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "start", "status", "pack"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--ckpt_every", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "check": sys.exit(0 if check() else 1)
    elif a.cmd == "start": start(a.seeds, a.epochs, a.ckpt_every)
    elif a.cmd == "status": status(a.seeds, a.epochs)
    else: pack(a.seeds)
