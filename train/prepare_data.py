"""Build the training noise banks and the simulated (noisy, target) mixtures.

Modes
  standard : clean speech from Vaani Gold event-free spans + noise from (a) validation noise events of the
             TRAIN noise clusters only (leak-free), (b) ESC-50, (c) MUSAN noise, (d) one augmented copy of each.
             This is the data used for the submitted model D (E8 -> E9 -> E10-R).
  e13      : "event-only" recipe (novelty, not on the LB): target = natural Vaani clip as-is (its native ambience
             kept), input = target + 1-3 library events. Teaches the model to remove added events only.

The baseline repository (nagshubhadip/noise-event-detection-and-removal) supplies Vaani streaming, clean-bank
carving and the simulator; it is cloned to third_party/ by scripts/setup.sh.
"""
import argparse
import glob
import json
import os
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def import_baseline(baseline_dir, val_dir):
    """Patch the baseline data loader for our validation path / metadata key and import it."""
    d = Path(baseline_dir) / "data.py"
    s = d.read_text()
    if "VAL_DIR_ENV" not in s:
        s = s.replace('Path("/kaggle/input/indoml-validation"),',
                      'Path(__import__("os").environ.get("VAL_DIR_ENV", "/kaggle/input/indoml-validation")), Path("/kaggle/input/indoml-validation"),')
        s = s.replace('cid = (r.get("clip_id") or r.get("id")', 'cid = (r.get("segmentFileName") or r.get("clip_id") or r.get("id")')
        d.write_text(s)
    os.environ["VAL_DIR_ENV"] = str(val_dir)
    sys.path.insert(0, str(baseline_dir))
    import config, data, evaluate  # noqa: E402
    return config, data, evaluate


def fingerprint(seg):
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=seg, sr=SR, n_mels=48, n_fft=512, hop_length=160) + 1e-10)
    f = librosa.util.fix_length(mel, size=48, axis=1).ravel()
    f = (f - f.mean()) / (f.std() + 1e-6)
    return f / np.linalg.norm(f)


def validation_noise_split(data, items, val_dir, out, seed=7):
    """Isolate validation noise events (noisy - clean inside the timestamps), cluster near-duplicates
    (cosine > 0.95 on log-mel fingerprints; 98.8% of events are reused), hold out ~45% of clusters.
    Returns train-cluster noise segments and the held-out evaluation clip ids (leak-free protocol)."""
    meta = {m["segmentFileName"][:-4]: m for m in json.load(open(Path(val_dir) / "validationMetadata.json", encoding="utf-8"))}
    nv = out / "bank" / "val_noise"; nv.mkdir(parents=True, exist_ok=True)
    segs, fps, owner = [], [], []
    for v in items:
        if not v["syn"]:
            continue
        y = data.read_audio(v["path"]); c = data.read_audio(v["ref"]); L = min(len(y), len(c)); n = (y[:L] - c[:L]).astype(np.float32)
        for k, e in enumerate(meta.get(v["id"], {}).get("NoiseSubCategoryTimeStamp", [])):
            a, b = max(0, int((float(e["start"]) - 0.02) * SR)), min(L, int((float(e["end"]) + 0.02) * SR))
            if b - a < int(0.3 * SR) or np.abs(n[a:b]).max() < 1e-4:
                continue
            p = nv / f"{v['id'][-40:]}_{k}.wav"; sf.write(p, n[a:b], SR)
            segs.append({"path": str(p), "dur": (b - a) / SR}); fps.append(fingerprint(n[a:b])); owner.append(v["id"])
    F = np.stack(fps); cents, lab = [], np.zeros(len(F), int)
    for i, f in enumerate(F):
        if cents:
            s = np.stack(cents) @ f; j = int(s.argmax())
            if s[j] > 0.95:
                lab[i] = j; continue
        cents.append(f); lab[i] = len(cents) - 1
    rng = np.random.default_rng(seed); ncl = len(cents)
    for frac in (0.35, 0.45, 0.55):
        held = set(rng.permutation(ncl)[:int(frac * ncl)].tolist())
        clip_cl = {}
        for o, l in zip(owner, lab):
            clip_cl.setdefault(o, set()).add(int(l))
        held_clips = [o for o, s in clip_cl.items() if s <= held]
        if len(held_clips) >= 80:
            break
    train_noise = [s for s, l in zip(segs, lab) if int(l) not in held]
    print(f"validation noise events {len(F)} | clusters {ncl} | held-out clusters {len(held)} | held-out clips {len(held_clips)} | train noise {len(train_noise)}")
    return train_noise, held_clips


def esc50_bank(esc_dir, out):
    esc_dir = Path(esc_dir)
    if not (esc_dir / "meta" / "esc50.csv").exists():
        import urllib.request
        zp = out / "esc50.zip"
        urllib.request.urlretrieve("https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip", zp)
        zipfile.ZipFile(zp).extractall(out); esc_dir = out / "ESC-50-master"
    ne = out / "bank" / "esc50"; ne.mkdir(parents=True, exist_ok=True); bank = []
    for _, r in pd.read_csv(esc_dir / "meta" / "esc50.csv").iterrows():
        w, _ = librosa.load(esc_dir / "audio" / r.filename, sr=SR, mono=True)
        w, _ = librosa.effects.trim(w, top_db=30)
        if len(w) < int(0.3 * SR):
            continue
        w = w[:int(2.5 * SR)].astype(np.float32); p = ne / r.filename; sf.write(p, w, SR)
        bank.append({"path": str(p), "dur": len(w) / SR, "cat": r.category, "fold": int(r.fold)})
    print("ESC-50 segments", len(bank)); return bank


def musan_bank(musan_dir, out, seed=3):
    files = sorted(glob.glob(f"{musan_dir}/**/free-sound/*.wav", recursive=True) + glob.glob(f"{musan_dir}/**/sound-bible/*.wav", recursive=True))
    nm = out / "bank" / "musan"; nm.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed); bank = []
    for k, fp in enumerate(files):
        w, _ = librosa.load(fp, sr=SR, mono=True)
        if len(w) < int(0.5 * SR) or np.abs(w).max() < 1e-4:
            continue
        for j in range(min(4, max(1, len(w) // (3 * SR)))):
            L = int(rng.uniform(0.5, 2.5) * SR); st = int(rng.integers(0, max(1, len(w) - L))); seg = w[st:st + L].astype(np.float32)
            if np.sqrt((seg ** 2).mean()) < 1e-3:
                continue
            p = nm / f"{k:04d}_{j}.wav"; sf.write(p, seg, SR); bank.append({"path": str(p), "dur": len(seg) / SR})
    print("MUSAN files", len(files), "| segments", len(bank)); return bank


def augment_bank(base, out, seed=11):
    """One augmented copy per noise segment: speed 0.9/1.0/1.1, random low/high-pass EQ, synthetic exponential-decay reverb."""
    na = out / "bank" / "aug"; na.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed); bank = []
    for k, nb in enumerate(base):
        w, _ = sf.read(nb["path"], dtype="float32")
        f = rng.choice([0.9, 1.0, 1.1])
        if f != 1.0:
            w = librosa.resample(w, orig_sr=int(SR * f), target_sr=SR)
        if rng.random() < 0.6:
            sos = butter(4, rng.uniform(2000, 7000), "low", fs=SR, output="sos") if rng.random() < 0.5 else butter(2, rng.uniform(100, 500), "high", fs=SR, output="sos")
            w = sosfilt(sos, w)
        if rng.random() < 0.5:
            rt = rng.uniform(0.15, 0.7); t = np.arange(int(rt * SR)) / SR
            rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt); rir[0] = 1.0
            w = fftconvolve(w, rir / np.abs(rir).sum() * 4)[:len(w)]
        pk = np.abs(w).max(); w = (w / pk * 0.9 if pk > 0 else w).astype(np.float32)
        if len(w) < int(0.3 * SR):
            continue
        p = na / f"a{k:05d}.wav"; sf.write(p, w, SR); bank.append({"path": str(p), "dur": len(w) / SR})
    print("augmented copies", len(bank)); return bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True, help="validation folder containing validationMetadata.json")
    ap.add_argument("--musan_dir", required=True, help="folder containing MUSAN noise/ (free-sound, sound-bible)")
    ap.add_argument("--esc_dir", default="", help="ESC-50-master folder (downloaded if missing)")
    ap.add_argument("--out", default=str(ROOT / "work"))
    ap.add_argument("--n_mix", type=int, default=40000)
    ap.add_argument("--max_gold", type=int, default=11000)
    ap.add_argument("--mode", choices=["standard", "e13"], default="standard")
    ap.add_argument("--baseline_dir", default=str(ROOT / "third_party" / "noise-event-detection-and-removal" / "track2_removal" / "versions" / "v-1_indoml2026_track2" / "modular"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hf = os.environ["HF_TOKEN"]
    config, data, evaluate = import_baseline(a.baseline_dir, a.val_dir)
    config.ensure_dirs()
    _, items = data.resolve_validation()
    train_noise, held = validation_noise_split(data, items, a.val_dir, out)
    json.dump(held, open(out / "held_clips.json", "w"))
    esc = esc50_bank(a.esc_dir or out / "ESC-50-master", out)
    musan = musan_bank(a.musan_dir, out)
    aug = augment_bank(train_noise + esc + musan, out)
    for f in list(config.MIX_DIR.glob("*.wav")) + list(config.REF_DIR.glob("*.wav")):
        f.unlink()

    if a.mode == "standard":
        clean_idx, _ = data.build_banks(a.max_gold, hf_token=hf)
        config.SIM.update(snr_db=(-20.0, 5.0), events=(1, 4), max_noise_sec=2.5, min_noise_sec=0.3)
        records = data.simulate_dataset(clean_idx, train_noise + esc + musan + aug, a.n_mix)
        print("mixtures", len(records), "| manifest", config.MANIFEST)
        return

    # ---- e13: target = natural Vaani clip as-is, input = target + library events (events from ESC folds 1-4 + MUSAN + val train clusters + aug)
    from datasets import load_dataset, Audio
    rng = np.random.RandomState(0); nat = [v["id"] for v in items if not v["syn"]]; byid = {v["id"]: v for v in items}
    reserved = set(nat[i] for i in rng.choice(len(nat), 600, replace=False))
    targets = [data.read_audio(byid[i]["path"]) for i in nat if i not in reserved]
    raw = load_dataset(config.REPO, split="train", streaming=True, token=hf).cast_column("audio", Audio(decode=False)); skipped = 0
    for ex in raw:
        try:
            w = data.decode_audio(ex["audio"])
        except Exception:
            continue
        if w is None or len(w) < 2 * SR:
            continue
        if skipped < 400:          # first 400 clips reserved for the test-like evaluation
            skipped += 1; continue
        targets.append(w[:int(12 * SR)].astype(np.float32))
        if len(targets) >= 10200:
            break
    events = [e for e in esc if e["fold"] != 5] + musan[:2100] + train_noise + aug
    R = np.random.default_rng(42); records = []
    for i in range(a.n_mix):
        s = targets[R.integers(len(targets))]
        if len(s) < 6 * SR:
            s = np.concatenate([s, targets[R.integers(len(targets))]])
        s = s[:int(12 * SR)].astype(np.float32); L = len(s); ps = float((s ** 2).mean()) + 1e-10; x = s.copy(); ev = []
        for _ in range(int(R.choice([1, 1, 2, 2, 3]))):
            e, _ = sf.read(events[R.integers(len(events))]["path"], dtype="float32"); e = e[:int(2.5 * SR)]
            if len(e) < int(0.3 * SR) or len(e) >= L:
                continue
            on = int(R.integers(0, L - len(e))); snr = R.uniform(-20, 5)
            x[on:on + len(e)] += e * np.sqrt(ps / ((e ** 2).mean() + 1e-10) / 10 ** (snr / 10)); ev.append({"onset": on / SR, "offset": (on + len(e)) / SR})
        if not ev:
            continue
        g = 0.95 / max(np.abs(x).max(), 1e-6) if np.abs(x).max() > 0.95 else 1.0
        cid = f"e13_{i:06d}"; sf.write(config.MIX_DIR / f"{cid}.wav", x * g, SR); sf.write(config.REF_DIR / f"{cid}.wav", s * g, SR)
        records.append({"id": cid, "mix": str(config.MIX_DIR / f"{cid}.wav"), "clean": str(config.REF_DIR / f"{cid}.wav"), "events": ev})
    with open(config.MANIFEST, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("E13 mixtures", len(records), "| manifest", config.MANIFEST)


if __name__ == "__main__":
    main()
