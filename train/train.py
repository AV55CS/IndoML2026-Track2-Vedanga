"""Fine-tune MP-SENet on simulated Vaani-like mixtures (manifest written by train/prepare_data.py).

Loss (MP-SENet generator terms, no discriminator):
  L = 0.9*MSE(|S|^c) + 0.3*(IP + GD + IAF phase losses) + 0.2*MSE(compressed complex) + 0.2*consistency + 0.2*L1(waveform)
Optimiser AdamW(betas 0.8/0.99), OneCycleLR (5% warm-up, cosine to ~0) sized to the actual number of steps
(--max_steps), gradient clipping 5.0, optional bf16 autocast for the network forward (A100), fp32 otherwise.
Selection metric: SI-SDR on leak-free held-out validation clips (unseen noise clusters).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.mpsenet_utils import H, DNS_CKPT, load_mp, enhance_mp, post_level, mag_pha_stft, mag_pha_istft, phase_losses  # noqa: E402
from train.prepare_data import import_baseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=DNS_CKPT, help="initial generator checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0, help="size the schedule to the real budget (measure steps/h first)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seg", type=int, default=48000, help="random crop length in samples")
    ap.add_argument("--nsamp", type=int, default=80000, help="max samples loaded per training clip")
    ap.add_argument("--bf16", type=int, default=0)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--keep", choices=["best", "last"], default="best")
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--held_clips", default=str(ROOT / "work" / "held_clips.json"))
    ap.add_argument("--baseline_dir", default=str(ROOT / "third_party" / "noise-event-detection-and-removal" / "track2_removal" / "versions" / "v-1_indoml2026_track2" / "modular"))
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); dev = torch.device("cuda")
    config, data, evaluate = import_baseline(a.baseline_dir, a.val_dir)
    data.N_SAMP = a.nsamp
    records = [json.loads(l) for l in open(config.MANIFEST)]
    tdl, _, _, _ = data.build_dataloaders(records, a.bs)
    _, items = data.resolve_validation(); byid = {v["id"]: v for v in items}
    held = [i for i in json.load(open(a.held_clips)) if i in byid]
    X = {i: data.read_audio(byid[i]["path"]) for i in held}; REF = {i: data.read_audio(byid[i]["ref"]) for i in held}

    def heldout(m):
        m.eval(); s = float(np.mean([evaluate.si_sdr(REF[i], post_level(X[i], enhance_mp(m, X[i], win=4.0, hop=3.0))) for i in held])); m.train(); return s

    m = load_mp(a.init); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.8, 0.99))
    total = a.epochs * len(tdl)
    if a.max_steps:
        total = min(total, a.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    best = heldout(m); print(f"init {os.path.basename(a.init)} held-out SI-SDR {best:.2f} | {len(records)} mixtures | {total} steps", flush=True)
    torch.save({"generator": m.state_dict()}, a.out)
    t0, step, stop = time.time(), 0, False
    for ep in range(a.epochs):
        run = None
        for b in tdl:
            mix, clean = b["mix"].to(dev), b["clean"].to(dev)
            if mix.shape[-1] > a.seg:
                st = np.random.randint(0, mix.shape[-1] - a.seg); mix, clean = mix[:, st:st + a.seg], clean[:, st:st + a.seg]
            nf = torch.sqrt(mix.shape[-1] / (mix.pow(2).sum(-1, keepdim=True) + 1e-9)); mix, clean = mix * nf, clean * nf
            c_amp, c_pha, c_com = mag_pha_stft(clean, H.n_fft, H.hop_size, H.win_size, H.compress_factor)
            n_amp, n_pha, _ = mag_pha_stft(mix, H.n_fft, H.hop_size, H.win_size, H.compress_factor)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.bf16)):
                a_g, p_g, co_g = m(n_amp, n_pha)
            a_g, p_g, co_g = a_g.float(), p_g.float(), co_g.float()
            y = mag_pha_istft(a_g, p_g, H.n_fft, H.hop_size, H.win_size, H.compress_factor)[:, :clean.shape[-1]]
            _, _, co_h = mag_pha_stft(y, H.n_fft, H.hop_size, H.win_size, H.compress_factor)
            ip, gd, iaf = phase_losses(c_pha, p_g)
            loss = 0.9 * F.mse_loss(c_amp, a_g) + 0.3 * (ip + gd + iaf) + 0.2 * F.mse_loss(c_com, co_g) + 0.2 * F.mse_loss(co_g, co_h) + 0.2 * F.l1_loss(clean, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            if torch.isfinite(gn):
                opt.step()
            sched.step(); step += 1; run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
            if step % 500 == 0:
                print(f"ep{ep} step{step}/{total} loss {run:.4f} | {(time.time() - t0) / 3600:.2f} h", flush=True)
            if time.time() - t0 > a.hours * 3600 or step >= total:
                stop = True; break
        s = heldout(m); print(f"epoch {ep} held-out SI-SDR {s:.2f} (best {best:.2f})", flush=True)
        if a.keep == "last" or s > best:
            torch.save({"generator": m.state_dict()}, a.out)
        best = max(best, s)
        if stop:
            break
    print("saved", a.out)


if __name__ == "__main__":
    main()
