"""Driver of the SCANet reproduction: fit (authors' epoch order), checkpoints, resume, final-epoch evaluation.
The final-epoch model is the primary result (no selection on the test set). Plain and rerank metrics are
both returned, computed from the same features, and must be reported separately.
Tests: tests/test_run.py."""
import json
import os
import random
import resource
import time

import numpy as np
import torch

from . import data as D, train as TR, evaluation as EV
from .model import SCANet
from .losses import SCANetCriterion

MARKET = "/kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15"
MASKS = "/kaggle/input/datasets/jiniyatanjina/mask-file"


def _to_py(o):
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if torch.is_tensor(o):
        return o.tolist()
    return o


def save_ckpt(path, model, opt, sch, loader, epoch, meta):
    if os.path.exists(path):
        raise FileExistsError(f"REFUSING to overwrite {path}")
    st = {"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "epoch": epoch,
          "meta": meta, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
          "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
          "loader_gen": loader.generator.get_state()}
    torch.save(st, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_ckpt(path, model, opt, sch, loader, restore_rng=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    opt.load_state_dict(ck["opt"])
    sch.load_state_dict(ck["sch"])
    if restore_rng:
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        loader.generator.set_state(ck["loader_gen"])
    return ck["epoch"]


def evaluate_final(model, market=MARKET, batch_size=128, num_workers=4, rerank=True):
    tf = D.build_test_transform()
    q_items, _, _ = D.process_dir(f"{market}/query", None, relabel=False)
    g_items, _, _ = D.process_dir(f"{market}/bounding_box_test", None, relabel=False)
    mk = lambda it: torch.utils.data.DataLoader(D.ReIDEvalDataset(it, tf), batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers, pin_memory=True)
    t0 = time.time()
    qf, qp, qc = EV.extract_features(model, mk(q_items))
    gf, gp, gc = EV.extract_features(model, mk(g_items))
    t1 = time.time()
    res = EV.evaluate_features(qf, gf, qp, gp, qc, gc, rerank=rerank)
    t2 = time.time()
    return {"metrics": _to_py(res), "n_query": len(q_items), "n_gallery": len(g_items),
            "feat_dim": int(qf.shape[1]), "t_extract_s": t1 - t0, "t_eval_s": t2 - t1,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}


def fit(out_dir, seed=0, max_epoch=60, ckpt_every=10, resume=None, do_eval=True, deterministic=False,
        restore_rng=True, pretrained=True, num_workers=4, market=MARKET, masks=MASKS):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"REFUSING to reuse non-empty {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    TR.seed_everything(seed, deterministic)
    items, ncls, _ = D.process_dir(f"{market}/bounding_box_train", masks, relabel=True)
    loader = TR.build_train_loader(items, 64, 4, num_workers, seed)
    model = SCANet(ncls, pretrained=pretrained).cuda()
    crit = SCANetCriterion(ncls, 0.3, 0.5).cuda()
    opt = TR.build_optimizer(model)
    sch = TR.build_scheduler(opt)
    meta = {"seed": seed, "max_epoch": max_epoch, "deterministic": deterministic, "pretrained": pretrained}
    start = load_ckpt(resume, model, opt, sch, loader, restore_rng) if resume else 0
    log = []
    for ep in range(start + 1, max_epoch + 1):
        sch.step()                                   # authors' order: step at the START of the epoch
        lr = opt.param_groups[0]["lr"]
        torch.cuda.synchronize(); t0 = time.time()
        m = TR.train_one_epoch(model, crit, opt, loader)
        torch.cuda.synchronize(); dt = time.time() - t0
        rec = {"epoch": ep, "lr": lr, "time_s": dt, **m}
        log.append(rec)
        print("EPOCH", json.dumps(rec), flush=True)
        if ep % ckpt_every == 0 or ep == max_epoch:
            save_ckpt(os.path.join(out_dir, f"ckpt_epoch{ep}.pt"), model, opt, sch, loader, ep, meta)
    res = {"meta": meta, "resumed_from_epoch": start, "log": log}
    if do_eval:
        res["final_epoch_eval"] = evaluate_final(model, market)
    with open(os.path.join(out_dir, "result.json"), "x") as fh:
        json.dump(_to_py(res), fh, indent=1)
    return res
