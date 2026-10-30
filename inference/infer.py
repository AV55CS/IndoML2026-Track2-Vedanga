"""Phase-2 inference: enhance every test clip with the fine-tuned MP-SENet (model D) + 8 s windows + shift-TTA x8,
write 16 kHz mono PCM-16 WAVs with the original file names, transcribe them with the mandated SraVaani-1.0 ASR,
and pack `transcripts.jsonl` + WAVs at the root of a ZIP (Codabench Phase-2 format).

Usage:
  HF_TOKEN=hf_xxx python inference/infer.py --input_dir /path/to/test/audio --output_dir out/ \
      --ckpt checkpoints/D_e10r_anneal.pt --zip submission_track2.zip
"""
import argparse
import glob
import json
import os
import sys
import time
import zipfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.mpsenet_utils import SR, load_mp, enhance_tta, post_level  # noqa: E402


def read_mono16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="folder with the noisy test WAVs")
    ap.add_argument("--output_dir", required=True, help="folder for enhanced WAVs + transcripts.jsonl")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "D_e10r_anneal.pt"))
    ap.add_argument("--shifts", type=int, default=8, help="number of shift-TTA views (8 -> shifts 0..87 samples)")
    ap.add_argument("--win", type=float, default=8.0)
    ap.add_argument("--hop", type=float, default=6.0)
    ap.add_argument("--zip", default="submission_track2.zip")
    ap.add_argument("--baseline_dir", default=os.path.join(ROOT, "third_party", "noise-event-detection-and-removal",
                                                           "track2_removal", "versions", "v-1_indoml2026_track2", "modular"))
    ap.add_argument("--asr_batch", type=int, default=16)
    ap.add_argument("--skip_asr", action="store_true")
    a = ap.parse_args()

    shifts = tuple(int(round(k * 100 / a.shifts)) for k in range(a.shifts))  # STFT hop = 100 samples
    os.makedirs(a.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.input_dir, "*.wav")))
    assert files, f"no WAVs in {a.input_dir}"
    model = load_mp(a.ckpt)
    print(f"model {a.ckpt} | {len(files)} clips | win {a.win}s hop {a.hop}s | shifts {shifts}", flush=True)

    t0 = time.time()
    for k, p in enumerate(files):
        x = read_mono16k(p)
        y = post_level(x, enhance_tta(model, x, shifts=shifts, win=a.win, hop=a.hop))
        assert np.isfinite(y).all() and len(y) == len(x), p
        sf.write(os.path.join(a.output_dir, os.path.basename(p)), y, SR, subtype="PCM_16")
        if k % 250 == 0:
            print(f"{k}/{len(files)} enhanced | {(time.time() - t0) / 60:.1f} min", flush=True)
    names = [os.path.basename(p) for p in files]

    if not a.skip_asr:
        # The baseline repo provides the mandated-ASR wrapper (SraVaani-1.0). Load its inference.py by path so it cannot
        # be confused with this repo's inference/ folder or MP-SENet's inference.py.
        import importlib.util
        sys.path.insert(0, a.baseline_dir)
        spec = importlib.util.spec_from_file_location("baseline_inference", os.path.join(a.baseline_dir, "inference.py"))
        base_inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(base_inf)
        Transcriber = base_inf.Transcriber
        asr = Transcriber(os.environ["HF_TOKEN"])
        texts = {}
        for i in range(0, len(names), a.asr_batch):
            batch = names[i:i + a.asr_batch]
            for n, t in zip(batch, asr.transcribe_paths([os.path.join(a.output_dir, n) for n in batch])):
                texts[n[:-4]] = (t or "").strip()
        with open(os.path.join(a.output_dir, "transcripts.jsonl"), "w", encoding="utf-8") as f:
            for n in names:
                f.write(json.dumps({"clip_id": n[:-4], "text": texts.get(n[:-4], "")}, ensure_ascii=False) + "\n")
        with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(a.output_dir, "transcripts.jsonl"), "transcripts.jsonl")
            for n in names:
                z.write(os.path.join(a.output_dir, n), n)
        nl = zipfile.ZipFile(a.zip).namelist()
        print(f"ZIP {a.zip}: {len(nl)} entries ({sum(x.endswith('.wav') for x in nl)} wavs) | "
              f"empty transcripts {sum(1 for v in texts.values() if not v)}", flush=True)


if __name__ == "__main__":
    main()
