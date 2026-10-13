"""Pretrained-weight loading test of scanet.model.SCANet(pretrained=True) against the resnet50-19c8e357 file.
No hardcoded key counts: everything is derived from the file and the model."""
import sys, copy
import torch
sys.path.insert(0, "/kaggle/working/ReID")
from scanet.model import SCANet, RESNET50_URL

FAILS = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""), flush=True)
    if not ok: FAILS.append(name)
def neg(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'BAD '} | NEG-CONTROL {name}: {'DETECTED' if ok else 'NOT DETECTED'}" + (f" | {detail}" if detail else ""), flush=True)
    if not ok: FAILS.append("neg-control " + name)

NEW_PREFIX = ("IAT2.", "IAT3.", "bn.", "classifier.")   # layers that are new in SCANet (not in torchvision resnet50)

def compare(s, sd):
    file_bb = [k for k in sd if not k.startswith("fc.")]
    missing = [k for k in file_bb if k not in s]
    mism = [k for k in file_bb if k in s and not torch.equal(s[k], sd[k])]
    extra = [k for k in s if k not in sd]
    unexpected = [k for k in extra if not (k.endswith("num_batches_tracked") or k.startswith(NEW_PREFIX))]
    return file_bb, missing, mism, extra, unexpected

sd = torch.hub.load_state_dict_from_url(RESNET50_URL, progress=False, check_hash=True, weights_only=False)
m = SCANet(751, pretrained=True)
s = m.state_dict()
file_bb, missing, mism, extra, unexpected = compare(s, sd)
fc_keys = [k for k in sd if k.startswith("fc.")]
ntb_in_file = [k for k in sd if k.endswith("num_batches_tracked")]
print(f"file keys {len(sd)} | non-fc {len(file_bb)} | fc {fc_keys} | num_batches_tracked in file {len(ntb_in_file)} | model keys {len(s)} | model-only keys {len(extra)}", flush=True)

check("every non-fc file key exists in the model", len(missing) == 0, f"missing {missing[:3]}")
check("every non-fc file tensor equals the model tensor (shape+value)", len(mism) == 0 and len(file_bb) > 0, f"{len(file_bb)} compared, mismatches {len(mism)} {mism[:3]}")
check("model-only keys are only num_batches_tracked or new layers (IAT2/IAT3/bn/classifier)", len(unexpected) == 0, f"model-only {len(extra)}, unexpected {unexpected[:3]}")
check("fc.* is not in the model (classifier is a new Linear)", all(k not in s for k in fc_keys) and len(fc_keys) > 0)
ntb_model_only = [k for k in extra if k.endswith("num_batches_tracked")]
check("model-only num_batches_tracked are fresh (0)", all(int(s[k]) == 0 for k in ntb_model_only), f"{len(ntb_model_only)} keys")
bb_new = [k for k in s if k.startswith(NEW_PREFIX) and k in sd]
check("no new-layer key is overwritten by the file", len(bb_new) == 0, str(bb_new[:3]))

# negative controls
s1 = copy.deepcopy(s); k0 = file_bb[-3]; s1[k0].view(-1)[0] += 1e-3
neg("one element perturbed by 1e-3", compare(s1, sd)[2] == [k0], k0)
s2 = copy.deepcopy(s); del s2[file_bb[5]]
neg("a backbone key removed", compare(s2, sd)[1] == [file_bb[5]], file_bb[5])
s3 = copy.deepcopy(s); s3["layer9.weight"] = torch.zeros(1)
neg("unexpected extra layer in the model", compare(s3, sd)[4] == ["layer9.weight"])
m0 = SCANet(751, pretrained=False)
n0 = len(compare(m0.state_dict(), sd)[2])
neg("pretrained=False", n0 > 0, f"{n0} mismatching tensors")

print("\nSUMMARY:", "ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
