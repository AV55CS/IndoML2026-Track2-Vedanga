"""Shared MP-SENet helpers: model loading, windowed inference with cross-fades, and shift-TTA.

MP-SENet (MIT licence): https://github.com/yxlu-0102/MP-SENet — cloned into third_party/MP-SENet by scripts/setup.sh.
"""
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPSENET_DIR = os.environ.get("MPSENET_DIR", os.path.join(ROOT, "third_party", "MP-SENet"))
sys.modules.setdefault("pesq", types.SimpleNamespace(pesq=lambda *a, **k: 0.0))  # PESQ is not needed (no metric discriminator)
if MPSENET_DIR not in sys.path:
    sys.path.append(MPSENET_DIR)  # append (not insert): the baseline repo also ships an inference.py

from env import AttrDict  # noqa: E402  (MP-SENet)
from dataset import mag_pha_stft, mag_pha_istft  # noqa: E402
from models.model import MPNet, phase_losses  # noqa: E402,F401

SR = 16000
H = AttrDict(json.load(open(os.path.join(MPSENET_DIR, "best_ckpt", "config.json"))))
DNS_CKPT = os.path.join(MPSENET_DIR, "best_ckpt", "g_best_dns")


def load_mp(path=DNS_CKPT, device="cuda"):
    m = MPNet(H).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)["generator"])
    return m.eval()


def mp_forward(m, wav):
    """wav [B, T] (RMS-normalised) -> enhanced wav [B, T] and (amp, pha, com)."""
    amp, pha, _ = mag_pha_stft(wav, H.n_fft, H.hop_size, H.win_size, H.compress_factor)
    a, p, c = m(amp, pha)
    return mag_pha_istft(a, p, H.n_fft, H.hop_size, H.win_size, H.compress_factor), (a, p, c)


@torch.no_grad()
def enhance_mp(m, x, win=8.0, hop=6.0, device="cuda"):
    """Enhance a 1-D float32 signal with overlapping windows and linear cross-fades."""
    x = np.asarray(x, np.float32)
    n = len(x)
    nf = float(np.sqrt(n / (np.sum(x ** 2) + 1e-9)))
    xn = torch.from_numpy(x * nf).to(device)
    W, Hp = int(win * SR), int(hop * SR)
    if n <= W:
        y = mp_forward(m, xn[None])[0][0][:n]
    else:
        y = torch.zeros(n, device=device)
        wsum = torch.zeros(n, device=device)
        r = W - Hp
        fade = torch.ones(W, device=device)
        fade[:r] = torch.linspace(0, 1, r, device=device)
        fade[-r:] = torch.linspace(1, 0, r, device=device)
        s = 0
        while True:
            e = min(s + W, n)
            seg = xn[s:e]
            if len(seg) < W:
                seg = F.pad(seg, (0, W - len(seg)))
            out = mp_forward(m, seg[None])[0][0][:W]
            L = e - s
            f = fade[:L].clone()
            if s == 0:
                f[:r] = 1
            if e == n:
                f[-min(r, L):] = 1
            y[s:e] += out[:L] * f
            wsum[s:e] += f
            if e == n:
                break
            s += Hp
        y = y / wsum.clamp_min(1e-6)
    y = (y / nf).cpu().numpy()
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def enhance_tta(m, x, shifts=(0, 12, 25, 37, 50, 62, 75, 87), win=8.0, hop=6.0, device="cuda"):
    """Shift-TTA: enhance copies delayed by sub-hop shifts (STFT hop = 100 samples), realign and average."""
    outs = [enhance_mp(m, np.pad(x, (s, 0)), win=win, hop=hop, device=device)[s:s + len(x)] for s in shifts]
    return np.mean(outs, 0).astype(np.float32)


def post_level(x, y):
    """Match the output RMS to the input RMS and avoid clipping (SI-SDR is scale-invariant; keeps ASR input level natural)."""
    ri, ro = np.sqrt((x ** 2).mean()), np.sqrt((y ** 2).mean())
    z = y * (ri / ro) if ro > 1e-8 else y
    pk = np.abs(z).max()
    return (z * (0.99 / pk) if pk > 0.99 else z).astype(np.float32)
