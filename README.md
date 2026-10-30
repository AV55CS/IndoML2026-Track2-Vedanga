# IndoML 2026 Datathon — Track 2 (Noise Event Removal) — Team **Vedanga**

**Author:** Avinash Kumar Sharma (solo) — Wadhwani School of Data Science and AI, IIT Madras · Codabench: `avics55`

Phase-2 selected submission: **Codabench Combined 15.21** (SI-SDR 13.18 dB, ΔWER +2.02 %: synthetic +5.68 %, natural +0.30 %, audit PASS).

**Method in one line:** phase-aware **MP-SENet** (2.26 M parameters) initialised from the official DNS-Challenge checkpoint, fine-tuned in four warm-started stages on leak-free simulated Vaani mixtures (Vaani Gold event-free speech + validation noise of *training* clusters only + ESC-50 + MUSAN noise + augmentation), followed by 8-s windowed inference with **shift test-time augmentation ×8**; transcripts from the mandated **SraVaani-1.0** ASR.

The full experiment log (every run, local and leaderboard numbers, negative results) is in `docs/experiment_log.md`; the technical report is `docs/technical_report.pdf`.

---
## Repository layout
```
README.md
requirements_train.txt         requirements_inference.txt
common/mpsenet_utils.py        # model loading, windowed inference, shift-TTA, level matching
inference/infer.py             # Phase-2 inference: enhance -> SraVaani transcripts -> ZIP
train/prepare_data.py          # noise banks + simulated mixtures (modes: standard | e13)
train/train.py                 # MP-SENet fine-tuning (budget-sized OneCycle, bf16 option)
train/config.yaml              # training lineage of the submitted model D
configs/model_config.yaml      # architecture + inference settings
scripts/setup.sh               # clones MP-SENet + organisers' baseline into third_party/, installs deps
scripts/run_inference.sh       # reproduces the submission
scripts/run_training.sh        # rebuilds model D (and optional E13)
checkpoints/D_e10r_anneal.pt   # FINAL model used for the leaderboard (9 MB)
checkpoints/e13_D.pt           # optional "event-only" model (novelty, not on the leaderboard)
notebooks/                     # original Kaggle / RunPod notebooks of the key runs
docs/technical_report.pdf      docs/experiment_log.md
```

## Setup
```bash
git clone <this repo> && cd <repo>
bash scripts/setup.sh requirements_inference.txt     # or requirements_train.txt for training
export HF_TOKEN=hf_...                               # SraVaani-1.0 and the Vaani dataset are gated on Hugging Face
```

## Inference (reproduces the Phase-2 submission)
```bash
bash scripts/run_inference.sh /path/to/indoml-track2-test/audio outputs/enhanced submission_track2.zip
# = python inference/infer.py --input_dir <test_audio> --output_dir outputs/enhanced \
#       --ckpt checkpoints/D_e10r_anneal.pt --shifts 8 --win 8 --hop 6 --zip submission_track2.zip
```
* **Input:** a folder of noisy WAV clips (any sample rate; converted to 16 kHz mono).
* **Output:** one enhanced **16 kHz mono PCM-16 WAV per input with the original file name**, `transcripts.jsonl`
  (`{"clip_id": <stem>, "text": <SraVaani transcript of the enhanced WAV>}` per line), and a ZIP with all files at its root.
* Every clip is processed by the model (no clip-level gating). Processing: RMS-normalise → MP-SENet on 8-s windows
  (6-s hop, linear cross-fades) → repeat for 8 sub-hop delays (0, 12, …, 87 samples; STFT hop = 100) and average the
  realigned outputs → rescale to the input RMS, peak-limit at 0.99.
* Runtime on the 3,178 test clips: ~2.4 h on one A100 (8 TTA passes), ~13 min per pass.

## Training
```bash
export VAL_DIR=/path/to/validation      # organisers' validation folder (validationMetadata.json, noisy + syntheticCleanRefAudio)
export MUSAN_DIR=/path/to/musan/noise   # free-sound/ and sound-bible/ (MUSAN: https://www.openslr.org/17/)
bash scripts/run_training.sh
```
Stages (details in `train/config.yaml` and the report):
| stage | init | data | crop × batch | LR (OneCycle) | hardware |
|---|---|---|---|---|---|
| E8 | MP-SENet DNS | 8k mixtures (val train-cluster noise + ESC-50) | 1 s × 2 | 2e-4, 5 ep | Kaggle T4, fp32 |
| E9-B | E8 | 16k mixtures (+ MUSAN subset + augmentation) | 1 s × 2 | 1e-4, 3 ep | Kaggle T4, fp32 |
| E10-R run 1 | E9-B | 40k mixtures (8,733 clean spans, 10,192 noise segments) | 3 s × 4 | 1e-4, epoch-0 kept | A100 80 GB, bf16 |
| anneal → **D** | E10-R run 1 | same 40k | 3 s × 4 | 5e-5, exactly 1 epoch | A100 80 GB, bf16 |

Leak-free validation: validation noise events are isolated (noisy − clean inside the timestamps), near-duplicates are clustered
(cosine > 0.95 on log-mel fingerprints; 98.8 % of events are reused), and ~45 % of clusters are held out; held-out clips are
never used for training noise. The held-out SI-SDR is the selection metric.

## Data used
* **ARTPARK-IISc/Vaani-Noise-Event-Dataset** (Gold split, ~11k clips streamed) — clean speech spans; Vaani training clips also used as natural targets in E13.
* **Datathon validation set** — noise events of the training clusters only (leak-free); held-out clips for evaluation.
* **ESC-50** (Piczak, 2015; CC BY-NC 3.0) — 1,998 event segments.
* **MUSAN** noise subset (Snyder et al., 2015; CC BY 4.0) — 930 files → 2,456 crops.
* Augmentation: speed 0.9/1.0/1.1, random low/high-pass EQ, synthetic exponential-decay reverb (one copy per segment).

## Pretrained models
| model | role | source / version |
|---|---|---|
| MP-SENet (DNS checkpoint `g_best_dns`) | initialisation of the enhancer | https://github.com/yxlu-0102/MP-SENet (`best_ckpt/`, MIT) |
| SraVaani-1.0 | mandated ASR for transcripts and evaluation | https://huggingface.co/ARTPARK-IISc/SraVaani-1.0 (gated; snapshot f5dd5358325a5208775b91dad98918e079ea2b27) |
| Organisers' baseline repo | Vaani streaming, clean-bank carving, simulator, ASR wrapper | https://github.com/nagshubhadip/noise-event-detection-and-removal |

## Environment and hardware
| | training | inference |
|---|---|---|
| Final stages (E10-R, anneal), inference, E13 | RunPod **NVIDIA A100-SXM4 80 GB**, 16 vCPU, 117 GB RAM, Python 3.12, PyTorch 2.8.0+cu128 (CUDA 12.8) | same |
| Early stages (E8, E9), evaluations | Kaggle **2× NVIDIA T4 16 GB**, Python 3.13, Kaggle default PyTorch/CUDA 12.x image | Kaggle T4 also works (~75 min per TTA pass) |

System dependencies: `git`, `libsndfile` (for soundfile), internet access for the gated Hugging Face downloads.

## Results summary (Codabench Phase 2)
| submission | SI-SDR | ΔWER (syn / nat) | Combined |
|---|---|---|---|
| baseline mask-net, realistic simulation (E4) | — | — | 3.40 |
| MP-SENet DNS zero-shot (E6) | — | — | 10.00 |
| MP-SENet fine-tuned (E8) | 12.37 | +4.65 / +0.20 | 13.99 |
| avg(E9-A, E9-B) | 12.73 | +5.05 / +0.27 | 14.53 |
| B + D ensemble, win4 | 12.86 | +5.41 / +0.31 | 14.81 |
| D, win8 + shift-TTA×4 | 13.15 | +5.69 / +0.30 | 15.18 |
| **D, win8 + shift-TTA×8 (selected)** | **13.18** | **+5.68 / +0.30** | **15.21** |

## Licence and acknowledgements
Code in this repository: MIT (see `LICENSE`). MP-SENet and the organisers' baseline are used under their own licences. ESC-50 is CC BY-NC 3.0
(non-commercial research use), MUSAN CC BY 4.0, Vaani per its dataset terms.
