# Experiments summary

This page summarises the experiments behind the submission. "Local" numbers use our leak-free validation split
(228 held-out synthetic clips whose noise comes only from held-out noise clusters, plus natural clips);
"LB" numbers are Codabench Phase-2 scores (Combined = SI-SDR on synthetic clips + 100 × ΔWER of SraVaani-1.0).

## 1. Evaluation protocol
- **Noise reuse in the validation set.** About 98.8% of the validation noise events have a near-identical copy in
  another clip. A model fine-tuned on a random validation split scored 12.3 dB locally but 0.83 on the leaderboard.
- **Leak-free split.** Noise events are isolated (noisy − clean inside the timestamps, 1,819 events) and clustered by
  log-mel fingerprints (cosine > 0.95, 139 clusters). 62 clusters are held out; the 228 synthetic clips whose events all
  belong to held-out clusters form the evaluation set. Only the 644 events of the training clusters are used for training.
- **Local score.** Held-out SI-SDR plus ΔWER on the held-out synthetic clips and on a fixed sample of natural clips,
  weighted like the leaderboard.

## 2. Leaderboard submissions
| submission | SI-SDR | ΔWER synthetic / natural (%) | Combined |
|---|---|---|---|
| baseline mask network, realistic simulation | — | — | 3.40 |
| + ASR-feature loss (λ = 5) | — | — | 4.24 |
| MP-SENet DNS checkpoint, no fine-tuning | — | — | 10.00 |
| MP-SENet, stage 1 | 12.37 | 4.65 / 0.20 | 13.99 |
| average of two stage-2 models | 12.73 | 5.05 / 0.27 | 14.53 |
| stage-2 + final model, 4-s windows | 12.86 | 5.41 / 0.31 | 14.81 |
| final model, 8-s windows + shift-TTA ×4 | 13.15 | 5.69 / 0.30 | 15.18 |
| **final model, 8-s windows + shift-TTA ×8 (selected)** | **13.18** | **5.68 / 0.30** | **15.21** |
| + output rescaled to peak 0.95 | 13.18 | 5.09 / 0.38 | 15.07 |
| + residual gating (keep input where little noise is removed) | 12.51 | 5.26 / 0.26 | 14.37 |

## 3. Training
| change | local held-out SI-SDR |
|---|---|
| stage 1 (DNS init, 8k mixtures) | 16.68 dB |
| stage 2 (16k mixtures, more noise, augmentation) | 16.93 dB |
| stage 3 (40k mixtures, 3-s crops, batch 4, bf16), first-epoch checkpoint | 17.00 dB |
| stage 4 (one-epoch annealing to LR ≈ 0) → final model | **17.26 dB** |

Notes:
- fp16 mixed precision on the T4 silently skipped optimiser steps (gradient-scaler overflow); all T4 training is fp32.
- In stage 3 the learning-rate schedule was planned for eight epochs but the time budget allowed about three, so the
  learning rate never decayed. A separate one-epoch annealing run recovered +0.26 dB. Schedules are now sized from a
  measured steps-per-hour figure.
- ASR-feature loss (L1 on layer-normalised SraVaani encoder features): λ = 0.5 / 1.5 changed the local score from 18.56
  to 18.29 / 18.16 (synthetic ΔWER +0.35 / +0.62, SI-SDR −0.39 / −0.64 dB). Not used.
- Noise bootstrapped from Vaani recordings (input minus model output inside annotated events) lowered the local score
  (18.44 → 18.25). Not used.

## 4. Inference
| setting (final model) | local combined |
|---|---|
| 4-s windows | 18.44 |
| 8-s windows | 18.66 |
| 8-s windows + shift-TTA ×2 | 19.22 |
| 8-s windows + shift-TTA ×4 | 19.65 |
| 8-s windows + shift-TTA ×8 | 19.83 |
| time-reversal views instead of extra shifts (equal compute) | 18.66 / 18.97 (worse) |
| averaging with zero-shot MossFormerGAN / FRCRN / MossFormer2 | 17.62 / 18.53 / 18.28 (worse) |
| output high-pass 80 Hz / soft limiter / mixing the input back | all worse (SI-SDR −1 to −6 dB) |

Shift-TTA transferred to the leaderboard (+0.37 for ×4, +0.03 more for ×8). Output post-processing that helped locally
(loudness) hurt on the leaderboard, so it was not used.

## 5. Test-set diagnosis (analysis only; no test labels used)
| statistic (median) | test | validation synthetic | validation natural |
|---|---|---|---|
| clip length | 11.9 s | 4.4 s | 4.4 s |
| share of frames dominated by removed noise | 10% | 39% | 0% |
| estimated SNR (kept speech vs removed noise) | 8.2 dB | −3.2 dB | 40.4 dB |

Test clips are about three times longer and mostly free of loud events, and keeping the input where the model removes
little lowered the test SI-SDR. A local set built from **noisy Vaani clips + unseen library events, with the Vaani clip
itself as the target**, gives our final model 12.66 dB, close to its leaderboard SI-SDR of 13.18 dB, and reproduces its
ASR pattern (ΔWER +5.1 against a ceiling of +12.0). This suggests that the test references keep the native Vaani audio
and that only the added events should be removed.

## 6. Optional event-only model (`checkpoints/e13_D.pt`, not used on the leaderboard)
Trained from the final model on 30k mixtures whose target is a natural Vaani clip as recorded and whose input adds 1–3
library events (6-s crops, batch 2, 13,775 steps, learning rate 7e-5 annealed).

| evaluation set (8-s windows + shift-TTA ×2) | final model | event-only model |
|---|---|---|
| validation natural clips + unseen events (SI-SDR) | 15.41 | 15.82 |
| 12-s concatenated natural clips + events (SI-SDR) | 16.06 | 17.50 |
| noisy Vaani clips + unseen events (SI-SDR) | 12.66 | 12.95 |
| validation held-out synthetic (SI-SDR) | 17.93 | 15.38 |

It could not be scored on the leaderboard before the deadline.
