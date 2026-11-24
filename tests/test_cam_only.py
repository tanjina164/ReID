"""Tests for scanet/ablation/model_cam_only.py (CPU, pretrained=False). Plain asserts; prints PASS/FAIL per check."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scanet.model import SCANet, IAT, count_num_param
from scanet.ablation.model_cam_only import SCANetCamOnly, IATCamOnly
ok_all = True
def chk(c, m):
    global ok_all; ok_all &= bool(c); print("PASS" if c else "FAIL", "|", m)

torch.manual_seed(0)
full = SCANet(751, pretrained=False); cam = SCANetCamOnly(751, pretrained=False)
nf, nc = count_num_param(full), count_num_param(cam)
print(f"params excl. classifier: full {nf:.6f} M | cam_only {nc:.6f} M | diff {nf-nc:.6f} M")
chk(nc < nf, "cam_only has fewer params than full")
keys = list(cam.state_dict().keys())
bad = [k for k in keys if k.startswith(("IAT2.SAM", "IAT3.SAM", "IAT2.g.", "IAT3.g.", "IAT2.W2", "IAT3.W2"))]
chk(not bad, f"no SAM/g/W2 keys in cam_only state_dict (found: {bad[:3]})")
chk(all(k.startswith(("IAT2.sa", "IAT2.W1", "IAT3.sa", "IAT3.W1")) for k in keys if k.startswith(("IAT2", "IAT3"))), "IAT keys are only sa/W1")
chk(float(cam.IAT2.W1[1].weight.abs().sum()) == 0 and float(cam.IAT3.W1[1].bias.abs().sum()) == 0, "W1 BN zero-init kept")

cam.train(); x = torch.randn(4, 3, 256, 128)
out = cam(x)
chk(len(out) == 6, f"6 outputs (got {len(out)})")
y, f, ah, au, al, ash = out
chk(tuple(y.shape) == (4, 751) and tuple(f.shape) == (4, 2048), f"y {tuple(y.shape)} f {tuple(f.shape)}")
chk(tuple(ah[0].shape) == (4, 1, 32, 16) and tuple(ah[1].shape) == (4, 1, 16, 8), f"part maps IAT2 {tuple(ah[0].shape)} IAT3 {tuple(ah[1].shape)}")
loss = y.sum() + sum(a.sum() for g in (ah, au, al, ash) for a in g)
loss.backward()
nog = [n for n, p in cam.named_parameters() if p.grad is None]
chk(not nog, f"all params get grad (None: {nog[:3]})")

# CAM equivalence: copy W1+sa from a FULL IAT into IATCamOnly; full IAT's returned map a = sa(z) must equal cam_only's a.
for c in (512, 1024):
    torch.manual_seed(1)
    fi = IAT(c).eval(); ci = IATCamOnly(c).eval()
    with torch.no_grad():
        fi.W1[1].weight.normal_(1, 0.2); fi.W1[1].bias.normal_(0, 0.2)
        ci.W1.load_state_dict(fi.W1.state_dict()); ci.sa.load_state_dict(fi.sa.state_dict())
        xi = torch.randn(2, c, 8, 4)
        _, a_full = fi(xi); z, a_cam = ci(xi)
    chk(torch.allclose(a_full, a_cam, atol=1e-5), f"c={c}: sa(z) of cam_only == sa(z) of full IAT (max diff {(a_full-a_cam).abs().max():.2e})")
    chk(not torch.allclose(z, xi), f"c={c}: CAM changes the feature (z != x)")
print("TEST:", "ALL PASS" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
