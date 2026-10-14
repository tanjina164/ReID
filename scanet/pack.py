"""Pack the FINAL-epoch model of one seed into a compressed eval-only ZIP and verify it.
ZIP = model_final_epoch<N>.pt (fp32 state_dict) + result.json + attempt*_config.json + *.log + MANIFEST.json.
Optimizer / RNG state are NOT included: the ZIP cannot resume training (resume uses the full ckpt in runs/)."""
import argparse, glob, hashlib, io, json, os, zipfile
import torch

def _sha(b): return hashlib.sha256(b).hexdigest()

def pack(seed_dir, out_zip):
    if os.path.exists(out_zip):
        raise FileExistsError(f"REFUSING to overwrite {out_zip}")
    res = glob.glob(os.path.join(seed_dir, "attempt*", "result.json"))
    if len(res) != 1:
        raise RuntimeError(f"need exactly one result.json in {seed_dir}, found {res}")
    rdir = os.path.dirname(res[0]); r = json.load(open(res[0])); ep = r["meta"]["max_epoch"]
    ck = torch.load(os.path.join(rdir, f"ckpt_epoch{ep}.pt"), map_location="cpu", weights_only=False)
    assert ck["epoch"] == ep, (ck["epoch"], ep)
    buf = io.BytesIO(); torch.save(ck["model"], buf); mb = buf.getvalue()
    extra = [res[0]] + sorted(glob.glob(os.path.join(seed_dir, "attempt*_config.json"))) \
            + sorted(glob.glob(os.path.join(seed_dir, "**", "*.log"), recursive=True))
    manifest = {"seed": r["meta"]["seed"], "epoch": ep, "model_sha256": _sha(mb), "n_tensors": len(ck["model"]),
                "note": "eval-only: no optimizer/RNG state; final-epoch model is the primary result"}
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    with zipfile.ZipFile(out_zip + ".tmp", "x", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(f"model_final_epoch{ep}.pt", mb)
        for p in extra:
            z.write(p, os.path.relpath(p, seed_dir))
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
    os.replace(out_zip + ".tmp", out_zip)
    return rdir, ep

def verify(out_zip, ckpt_path, lo_mb=90, hi_mb=110):
    from .model import SCANet
    out = []
    size = os.path.getsize(out_zip) / 2**20
    out.append((lo_mb <= size <= hi_mb, f"zip size {size:.1f} MB in [{lo_mb},{hi_mb}]"))
    with zipfile.ZipFile(out_zip) as z:
        out.append((z.testzip() is None, "zip CRC test"))
        names = z.namelist()
        mname = [n for n in names if n.startswith("model_final_epoch")][0]
        sd = torch.load(io.BytesIO(z.read(mname)), map_location="cpu", weights_only=False)
        man = json.loads(z.read("MANIFEST.json"))
        out.append((_sha(z.read(mname)) == man["model_sha256"], "model sha256 == manifest"))
        for n in names:
            print("  ", n, f"{z.getinfo(n).file_size/2**20:.2f} MB (uncompressed)")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    same = set(sd) == set(ck) and all(torch.equal(sd[k], ck[k]) for k in ck)
    out.append((same, f"all {len(ck)} tensors bitwise equal to the full checkpoint"))
    cls = [k for k in sd if k.endswith("classifier.weight")]
    if cls:
        m = SCANet(sd[cls[0]].shape[0], pretrained=False); m.load_state_dict(sd, strict=True)
        out.append((True, f"strict load_state_dict into SCANet(ncls={sd[cls[0]].shape[0]})"))
    else:
        out.append((False, "no *classifier.weight key: strict-load check NOT run"))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed_dir", required=True); ap.add_argument("--out_zip", required=True)
    a = ap.parse_args()
    rdir, ep = pack(a.seed_dir, a.out_zip)
    ok = True
    for good, msg in verify(a.out_zip, os.path.join(rdir, f"ckpt_epoch{ep}.pt")):
        print("PASS" if good else "FAIL", "|", msg); ok &= good
    print("SUMMARY:", "ALL PASS" if ok else "FAILED")
