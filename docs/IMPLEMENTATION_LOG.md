# SCANet image-based reproduction (Experiment A) - implementation log

## Status
- Implemented and parity-tested against the authors' code: model, losses, data layer, evaluation layer.
- NOT done yet: training loop, LR-schedule trace test (step order), 1-epoch smoke test, any training run.
- No accuracy result exists yet. Nothing in this repository is a reproduction result.

## Reference
- Authors' repo: github.com/FatimaZulfiqar/Person-ReID-STCANet, pinned at commit 1b1eb9dd46c0765d84ebd20bc80eee048608595f
  ("Update README.md", 2025-08-22). Notebook: "Jupyter Notebook/Image Based Person ReID STCANet.ipynb"
  (28 code cells; used here: 1, 5, 7, 8, 9, 12, 19, 20, 21, 22, 24, 25, 27).
- That repo's README names the paper "Multi-Camera Person Re-Identification using Spatiotemporal Context Modeling"
  (STCANet). Whether it is the same paper as the SCANet paper (Neural Comput Appl 35:20117, 2023) is NOT confirmed.
- LICENSE: the authors' repository contains no LICENSE file. `scanet/*.py` follow the authors' code closely
  (same parameter names, so state dicts are interchangeable) and `scanet/eval.py` is stated in its own header to be
  a verbatim copy of parts of the authors' notebook. This repository is PUBLIC (owner's decision).
  Redistribution permission is UNRESOLVED; clarify with the authors before any paper release.

## Scope
- Image-based SCANet only: spatial attention only (CAM -> SAM inside IAT after layer2 and layer3).
- The authors' video / temporal parts are not used. A keyword scan of the image notebook found no temporal code.

## Authors' recipe (read from their code; not yet re-run here)
- Input 256x128, bicubic resize, ImageNet normalize. One flip decision (p=0.5) shared by the image and its 4 masks.
  RandomErasing(0.5, sl 0.02, sh 0.4, r1 0.3, value (0.4914, 0.4822, 0.4465)) on the image only, after normalize.
  Masks stay in [0, 1], bicubic-upscaled from 128x64.
- RandomIdentitySampler, K=4, batch 64. Batches do NOT guarantee 16 distinct IDs (measured mean 15.85, min 15).
- Model: ResNet-50 (ImageNet weights resnet50-19c8e357), last stride 1, IAT2(512), IAT3(1024),
  global avgpool -> BatchNorm1d(2048) -> Linear. Test feature = post-BN `f`. 27.464286 M params excl. classifier.
- Loss: CE label smoothing 0.1 + cosine hard-mining triplet (margin 0.3, dist = -cos)
  + 0.5 * BCE mask loss (4 parts, IAT2 and IAT3 maps averaged), alpha from iteration 0.
- Adam lr 3.5e-4, weight decay 5e-4, MultiStepLR [20, 40] gamma 0.1, 60 epochs, FP32.
  `scheduler.step()` is called at the START of each epoch (before training), so the LR drops one epoch early
  (derived from code, to be verified by an LR trace).
- Eval: feature = post-BN `f`, no test flip (flip_cnt = 1), distance 1 - cos, Market-1501 protocol, max_rank 50,
  re-ranking k1=20, k2=6, lambda=0.3 (off by default). Authors keep the best test Rank-1 checkpoint.

## Deviations from the authors (documented)
- Framework and environment: plain PyTorch standalone package (not FastReID); environment below
  (authors: python 3.7, torch 1.0.1).
- Masks: SCHP-generated masks (audit passed) instead of the authors' LIP masks.
- Final-epoch checkpoint is the primary result; no model selection on the test set.
- All RNGs (python, numpy, torch) are seeded; authors seeded torch only. Bitwise determinism is not claimed.
- Sorted file listing; no infinite read-retry loop; foreground.png not loaded (unused by the loss, no RNG use).
- 1 GPU, no DataParallel. DataLoader workers will be seeded explicitly (authors' behaviour unverified).
- Plain and re-ranked results are always computed from the same features and reported separately.

## Tests (all under tests/, outputs under results/)
| test | result file | outcome |
|---|---|---|
| test_parity_model_loss.py | parity_model_loss_run1.txt | ALL PASS, rel_err 0 (bitwise), random input and masks |
| test_parity_negative_control.py (v1) | parity_negative_control_run1.txt | WEAK: 2 of 9 injected bugs not detected (design flaw: thr 1e-4). Kept. |
| test_parity_negative_control_v2.py | parity_negative_control_v2_run1.txt | Sensitive: thr 1e-6, all 9 injected bugs detected |
| test_parity_data.py | parity_data_run2.txt | ALL PASS (re-run after the working tree was re-cloned; run1 predates a session restart, content not reviewed) |
| test_parity_eval.py | parity_eval_run1.txt | ALL PASS. Synthetic features are saturated (plain mAP 0.9993, rerank 1.0): array-level equality of re_ranking and CMC is the strong evidence. |

## Evaluation modules
- `scanet/evaluation.py`: parity-tested (tests/test_parity_eval.py); this is the module the training code will use.
- `scanet/eval.py`: committed from the owner's other Claude account (commit b16fb6e, with 1d9f61f).
  Its header says it is a verbatim copy of the authors' notebook cells plus `compute_minp`. The verbatim claim was
  NOT checked by a diff; `compute_minp` (mINP) has NO test and is not used yet. Kept for reference; not deleted.

## Not verified yet
- Training loop and LR trace; authors' notebook has not been run end-to-end here.
- Full-size re-ranking RAM/time; GPU memory at batch 64.
- Eval parity on real features (planned with the smoke test). mINP implementation.

## Environment at doc-write time
python 3.13.15 | torch 2.11.0+cu128 | torchvision 0.26.0+cu128 | numpy 2.1.3 | PIL 12.3.0 | GPUs: Tesla T4, Tesla T4

## Reproduce / recover (/kaggle/working can be wiped on restart; commit and push before leaving a session)
1. git clone https://github.com/tanjina164/ReID /kaggle/working/ReID  (do not rm -rf an existing working copy
   without pushing first: an unpushed working copy was lost once this way and had to be rebuilt)
2. git clone https://github.com/FatimaZulfiqar/Person-ReID-STCANet /kaggle/working/ref/Person-ReID-STCANet
   then `git checkout 1b1eb9dd46c0765d84ebd20bc80eee048608595f` inside it.
3. Attach Kaggle datasets: market1501 (/kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15)
   and mask-file (/kaggle/input/datasets/jiniyatanjina/mask-file). GPU accelerator on.
4. Run each tests/test_parity_*.py from /kaggle/working/ReID and compare with results/.

## Provenance
| file | sha256[:12] at doc-write time |
|---|---|
| scanet/__init__.py | 64b16c9e0473 |
| scanet/data.py | 90ba58db4358 |
| scanet/eval.py | 9764a50709d0 |
| scanet/evaluation.py | 77bb0b86ac93 |
| scanet/losses.py | 3fc233a53744 |
| scanet/model.py | df97ac5ff1b1 |
| tests/test_parity_data.py | 692d59feab45 |
| tests/test_parity_eval.py | 85915dcd83f6 |
| tests/test_parity_model_loss.py | 43df0366d188 |
| tests/test_parity_negative_control.py | dcb94d3034b1 |
| tests/test_parity_negative_control_v2.py | 2afba18d90ca |
| results/parity_data_run1.txt | 6de0d70354d4 |
| results/parity_data_run2.txt | bd150b5d6c9e |
| results/parity_eval_run1.txt | b48c9a9e3398 |
| results/parity_model_loss_run1.txt | cb6f75270a92 |
| results/parity_negative_control_run1.txt | db18ee34225c |
| results/parity_negative_control_v2_run1.txt | 1efdee052a53 |
