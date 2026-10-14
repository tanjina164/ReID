# RUNBOOK: SCANet reproduction (Experiment A), fresh Kaggle notebook

Recipe = authors' (60 epochs, Adam 3.5e-4, MultiStepLR [20,40], batch 64 = 16x4, FP32, final-epoch checkpoint is the primary
result, rerank ON/OFF reported separately, no model selection on the test set). 100-epoch runs are NOT part of Experiment A.
Seed order: (0,1) in parallel on 2 GPUs, then (2,3), then 4 alone. Each seed writes only to runs/seed<S>/attempt<N>/.

## Notebook settings (not code)
Add Data: market1501 and mask-file (paths in scanet/run.py). Accelerator: GPU T4 x2. Internet ON (ResNet50 weights download).

## Cells, in order. Run one phase at a time.
Cell 1: clone at the tagged commit
```python
import os, subprocess
R = "/kaggle/working/ReID"
if not os.path.exists(R):
    subprocess.run(["git", "clone", "https://github.com/tanjina164/ReID.git", R], check=True)
subprocess.run(["git", "-C", R, "fetch", "--tags"], check=True)
subprocess.run(["git", "-C", R, "checkout", "expA-train-v1"], check=True)
print(subprocess.run(["git", "-C", R, "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
```
Cell 2: checks (stop unless CHECK: ALL PASS)
```
!cd /kaggle/working/ReID && python -m scanet.phase check
```
Cell 3: start one phase (background processes, seed i of the list runs on GPU i)
```
!cd /kaggle/working/ReID && python -m scanet.phase start --seeds 0 1
```
Phase 2: --seeds 2 3. Phase 3: --seeds 4 (one GPU). Defaults: --epochs 60 --ckpt_every 10.
Cell 4: status (repeat as needed)
```
!cd /kaggle/working/ReID && python -m scanet.phase status --seeds 0 1
```
Cell 5: pack, only when the seeds are FINISHED (eval-only ZIP per seed, verified 90-110 MB)
```
!cd /kaggle/working/ReID && python -m scanet.phase pack --seeds 0 1
```
Cell 6: verify and download the ZIPs
```python
import glob, os, zipfile
from IPython.display import FileLink, display
for z in sorted(glob.glob("/kaggle/working/zips/seed*_final.zip")):
    mb = os.path.getsize(z) / 2**20
    with zipfile.ZipFile(z) as f:
        ok, names = f.testzip() is None, f.namelist()
    print(os.path.basename(z), f"{mb:.1f} MiB", "CRC ok" if ok else "CRC BAD", "| 90-110:", 90 <= mb <= 110, "|", names[:6])
    if ok and 90 <= mb <= 110:
        display(FileLink(z))
```

## Resume after an interruption
Re-run Cell 1, 2, 3 with the same seeds. scanet.launch resumes from the highest-epoch checkpoint in runs/seed<S>/ as a new
attempt<N+1> (RNG state restored). A finished seed is refused. This works only if runs/ still exists.

## Persistence (UNRESOLVED)
/kaggle/working can be wiped on a session restart. Where runs/ is kept outside the session (download, Kaggle output version,
dataset) is not decided and not tested. Full checkpoints (332.5 MiB each, 6 per seed, ~2 GiB per seed) are NOT in the ZIP and
are ignored by git (*.pt, *.zip). The ZIP is eval-only (final model + result.json + config + log + MANIFEST.json): it cannot
resume training.

## Outputs
runs/seed<S>/attempt<N>/{ckpt_epoch10..60.pt, result.json}, runs/seed<S>/attempt<N>_config.json, runs/seed<S>/launch_*.log,
zips/seed<S>_final.zip and .verify.txt. Nothing is overwritten; the tools refuse existing targets.

## Tested vs not tested (see results/)
Tested: 2-GPU parallel launch, pinning by CUDA_VISIBLE_DEVICES, kill + resume, refuse-while-running, refuse-finished, ZIP pack+verify
(2-epoch models: 103.1 MiB), parity of model/loss/data/eval/train loop vs the authors' code.
NOT tested: any 60-epoch run, ZIP size of a 60-epoch model, a fresh-notebook run of this runbook, resume after a real session
restart, a real LR-drop checkpoint resume (epochs 20/40).
