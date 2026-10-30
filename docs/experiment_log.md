# Datathon@IndoML 2026 — Track 2: Noise Event Removal — Problem Statement & Plan

**Team:** Vedanga (solo) · Leader: Avinash Kumar Sharma, WSAI, IIT Madras · Codabench: `avics55` · Email: avics2020@gmail.com
**Status (5 Oct 2026, 22:18 IST — 11.5 days left):** registered on Codabench Track 2; missed the Google-Form registration (closed 9 Sep); organisers replied and asked for an Excel registration for award eligibility → sent on 5 Oct with team name "Vedanga".
**Final deadline:** **17 Oct 2026, 12:00 noon IST** (extended from 15 Oct due to Codabench downtime)
**Sources:** official page indoml.in/datathon, Codabench Track 2 page (competition 17835), organisers' baseline repo. Items marked **[VERIFY]** must be checked on Codabench/Discord before relying on them.

---

## 1. Links
- Official page: https://indoml.in/datathon/
- Track 2 on Codabench: https://www.codabench.org/competitions/17835/ (Track 1: …/17825/)
- Training data (Hugging Face): `ARTPARK-IISc/Vaani-Noise-Event-Dataset`
- Sample preview: `PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp`
- Mandated ASR: `ARTPARK-IISc/SraVaani-1.0` (frozen; only the HF call is needed)
- Baselines & code: github.com/nagshubhadip/noise-event-detection-and-removal (`track2_removal/versions/v-1_indoml2026_track2`)
- AMA #1 (9 Sep): recording youtu.be/6-YFTtaF7Nw · slides on Google Drive (linked from the official page)
- Test data mirror if Codabench download fails: Google Drive folder linked on the official page
- Discord (fastest answers; tag **Shubhadip Nag**): discord.gg/xderWnjSZT · Email: datathon@indoml.in

---

## 2. Problem statement

Real-world Indic speech recorded in Indian homes and streets (Vaani corpus) contains **bursty, non-stationary noise events**: vehicle horns, dogs barking, children crying, doorbells, ringtones, pressure-cooker whistles, TV/devotional music. These events hurt downstream ASR, and standard denoisers either smear them or treat them as speech.

**Track 2 task:** given a noisy clip **and the noise-event timeline** (onset/offset pairs), **suppress the noise events while preserving the speech**, and produce clean, intelligible audio.

- **Input at test time (checked 5 Oct): ONLY noisy 16 kHz WAVs** — no metadata, no timestamps, no `syntheticData` flag, no language. This is consistent with the problem statement once read carefully: Codabench says Track 2 receives event timestamps "either from Track 1 predictions or provided ground truth", and the official page describes a **two-stage system** where the "event timeline from Track 1 is passed as conditioning input to Track 2". Ground-truth timestamps are provided for training/validation; at test time they are withheld (they are Track 1's answers), so **timestamps must come from our own detector (Track-1-style)**. → Intended design: **detector → event-conditioned enhancer** (a blind enhancer is the simpler fallback).
- **Output (submission):**
  1. Enhanced **16 kHz mono WAV**, one per test clip, **same filename** as the input.
  2. `transcripts.jsonl` produced by the frozen ASR **SraVaani-1.0** on our enhanced audio. **[VERIFY]** exact JSONL schema on Codabench.
- Phase 2 uses a **new Track 2 test set**: Codabench → Get Started → Files → `input_data [Final Submission (Phase 2)]`. Submissions built on the old set fail.
- Submission limits: **5 per day, 100 total.**

Framing from organisers: Responsible AI — robustness (real recordings, scored by real ASR improvement), inclusivity (language-/class-balanced eval), transparency (top-5 release code, frozen ASR identified).

---

## 3. Evaluation

**Leaderboard (ranking) score:**
```
Combined = SI-SDR (dB)  +  100 × ΔWER (fraction)
ΔWER = WER(noisy) − WER(enhanced)        → higher is better; negative = we made ASR worse
```
- **SI-SDR**: scale-invariant SDR between enhanced audio and a **synthetic clean reference** → computed **only on synthetic clips** (the only ones with a clean reference).
- **ΔWER**: frozen SraVaani-1.0 run on noisy and enhanced audio → computed on **all** clips (natural + synthetic).
- Leaderboard shows ΔWER as a **percentage** (e.g. −1.17); ranking still uses 100 × ΔWER(fraction). So +1 point of ΔWER% = +1 point of Combined, same weight as +1 dB SI-SDR.
- **PESQ**: computed only for the **top-5** initial entries.
- **Final standings:** **80% leaderboard + 15% novelty + 5% report.** Novelty applied by an expert panel to the top-5 entries (rewards new architectures, novel training regimes, principled event-conditioning, semi-supervised use of unlabelled Vaani audio).

**Implication:** aggressive denoising can raise SI-SDR but lower ASR accuracy (negative ΔWER), especially on natural clips. Best Combined comes from suppressing **only inside event segments** and leaving speech untouched.

---

## 4. Data

### 4.1 Training data (Hugging Face, ~154.6 h, 90,637 segments)
| Tier | Hours | Segments | Annotation |
|---|---|---|---|
| 🥇 Gold | 21.8 | 11,111 | Precise timestamps, multi-annotator agreement verified |
| 🥈 Silver | 100.32 | 61,642 | Timestamps, agreement **not** verified (noisy labels) |
| 🥉 Bronze | 32.42 | 17,884 | Only noise tags in transcript, **no** timestamps |

- No clean references in training data → for supervised enhancement, **training pairs must be constructed** (e.g. clean-ish speech segments + noise-event segments mixed synthetically). Mirror the organisers' synthetic-mix recipe as closely as possible (see baseline repo). Image fields in the HF dataset can be ignored.

### 4.2 Evaluation splits (Codabench Track 2 table)
| Split | Total | Natural | Synthetic | Availability |
|---|---|---|---|---|
| Train | 20 h | 20 h | — | open |
| Validation | 4 h | 3 h | 1 h | Codabench Files (`input_data`) |
| Test (Phase 2) | 11 h | 7 h | 4 h | withheld, blind |

- The official page also mentions a 10 h gold test set; the Codabench Track 2 page (11 h = 7 h natural + 4 h synthetic) is treated as authoritative. **[VERIFY]**
- A `syntheticData` field in test metadata marks the subset of each clip.
- **[VERIFY]** whether validation synthetic clips ship clean references (needed for a local SI-SDR harness).

### 4.3 Actual files (inspected 5 Oct 2026)

**Validation** (`validation/`, 498 MB, ~4 h, Codabench Files → "Validation Dataset", Public Data, 445 MB zip):
| Folder | Files | Content |
|---|---|---|
| `naturalNoisyAudio/` | 1,806 | real noisy clips (ΔWER only) |
| `syntheticNoiseAudio/` | 695 | synthetic **noisy mixtures** (speech + inserted noise events) — model input |
| `syntheticCleanRefAudio/` | 695 | **clean references** for the synthetic clips (same filenames) — enables local SI-SDR |
| `validationMetadata.json` | 1 | per-clip metadata (below) |

Metadata fields per clip: `segmentFileName`, `imageFileName` (ignore), `state`, `district`, `duration` (≈3–8 s clips), `language` (Assamese, Hindi, Kannada, … multilingual), `annotationQuality` (e.g. `verified_timestamps`), `isTranscriptionAvailable`, `transcript` (speech text with inline noise tags, e.g. `<horn> … </horn>`, `[breathing]`), `NoiseCategory` (e.g. `animal`, `vehicle_traffic`, `baby_child`, `human_non_speech`), `NoiseSubCategoryTimeStamp` (list of `{category, tag, start, end}` in seconds, stored as strings), `syntheticData` (bool).

Filenames encode source recording + clip start/end in ms (e.g. `…_19520_23198.wav`).

**Test, Phase 2** (`input_data_track-2_phase-2/audio/`, 1,199 MB ≈ 11 h): **only WAV files**, no metadata at all.

**Critical observation — noise overlaps speech.** Transcripts show words *inside* noise tags (`<horn> words </horn>`): noise events occur **while people are speaking**. Muting/attenuating event regions also removes speech → hurts ASR (a likely reason some teams have negative ΔWER). The task is genuine **speech/noise separation inside events**, not gating.

### 4.4 Submission format & official scorer (from organisers' baseline repo, `track2_removal/v-1`)
- ZIP with **`transcripts.jsonl` + every enhanced WAV at the ZIP root** (no subfolder). WAV: 16 kHz, mono, **PCM 16-bit**, same filename as input, same length.
- `transcripts.jsonl`: one line per clip, `{"clip_id": <filename without .wav>, "text": <SraVaani transcript>}`.
- ASR: `ARTPARK-IISc/SraVaani-1.0` (gated — accept terms on HF), loaded with `AutoModel.from_pretrained(path, trust_remote_code=True)`, called as `asr.transcribe(list_of_wav_paths, return_hypotheses=True)`. The **audit re-runs this ASR on our WAVs**, so transcripts must come from exactly this model on exactly the submitted audio.
- Scorer (verbatim in baseline `evaluate.py`): SI-SDR (scale-invariant, clipped ±100 dB) on synthetic clips; **pooled** WER via `jiwer` (total edits / total words, not a mean of ratios); `normalize()` lower-cases, strips `<…>`/`[…]` tags and punctuation; empty hypothesis → "@". Combined = SI-SDR + 100·ΔWER.
- Baseline system: STFT magnitude-mask net (BiGRU) with optional SraVaani/WavLM features via FiLM; trained on simulated mixtures (clean spans + isolated noise spans cut from Gold clips, event-local SNR −5…12 dB, 1–2 events per 5 s window), SI-SDR loss, then fine-tuned on real validation pairs; output = 0.85·enhanced + 0.15·mixture, then **RMS restored to input level** (SI-SDR-neutral, important for ASR); 30 s windows with cross-fade for long clips. Budget switch tiny/fast(~1.5 h)/full(~3 h) on Kaggle.
- Baseline expects Kaggle inputs `indoml-validation` and `indoml-track2-test`.

---

## 5. Rules & eligibility

### 5.0 Organisers' explicit guidance for Track 2 (posted after the Phase-1 audit) — MUST follow
- **No noise-dictionary / lookup systems.** A fixed library of noise clips cut from validation data + correlation-search subtraction is declared non-scalable and not acceptable. The system must be **a model that learns to denoise in general** (works on unseen noise events).
- **Every clip contains at least one noise event** → outputting a clip **unchanged is never valid**.
- **No deterministic gating of whether to enhance** (by filename pattern, amplitude/clipping threshold, or classifier score). Guessing natural vs synthetic is not the task. **Every clip must be genuinely processed by the enhancement model.**
- **One general pipeline for natural and synthetic clips** — no separate pipelines (e.g. dictionary subtraction for one type, neural model for the other).
- Track 2 Phase-1 entries were audited by reproducing submitted models/code; an **Audit** column now appears on the leaderboard (1.0 for all current entries). Expect the same audit for final rankings → keep `main.py <in> <out>` + model files + requirements + README reproducible at all times.

**Impact on our plan:** identity output (E0) is only a *local* format/harness check, never a real submission. Event-conditioned enhancement is still in the spirit of the task (organisers describe the Track 1 timeline as conditioning input), but the model must process every clip; soft, model-driven suppression around events is safer than a hard "copy everything outside segments" rule. **[ASK on Discord]** whether masking model output to event regions (with cross-fades) is acceptable.

- Open to people **based in India**; each team needs ≥1 member affiliated with an Indian institution/organisation (we qualify).
- **One Codabench account per team submits.**
- **Open model & data policy:** "Teams may use any publicly available, closed-source, or proprietary models, along with additional data or augmentation strategies." → pretrained enhancers and external noise/speech data are allowed.
- **Top-5 per track:** release training + inference code under a permissive open-source licence and submit a **≤4-page system description**.
- Phase 1 verification required a ZIP named `Team_name/` with model files, `requirements.txt`, `main.py <input_dir> <output_dir>` writing enhanced WAVs (same names) + transcript JSON, and a README with contact. The ASR model itself is not included (just the HF call). Build this package format **from day 1** — likely needed again for final verification. **[VERIFY]**
- Award eligibility: Excel registration sent 5 Oct (Team "Vedanga"). **[VERIFY]** acknowledgement.

---

## 6. Prizes (Track 2)
- **Final stage:** 1st ₹35,000 · 2nd ₹15,000 · 3rd ₹10,000 · 4th–8th ₹3,000 each.
- **Spotlight Solution Award ₹10,000** (Track 2 only, for exceptional solutions).
- Novelty Award (expert panel) and invitation to present at **IndoML 2026, IIT Kharagpur, 18–21 Dec 2026**; travel details TBA.
- Phase 1 half-marathon prizes already decided (Track 2 results pending organiser verification).

---

## 7. Competitive picture (Phase 2 leaderboard, 5 Oct 2026, 22:18 IST — 12 teams)

| # | Team | SI-SDR (dB) | ΔWER (%) | ΔWER synth (%) | ΔWER raw/natural (%) | Combined | Runtime (s) |
|---|---|---|---|---|---|---|---|
| 1 | liyingfa | 14.89 | 5.90 | 17.39 | 0.49 | **20.80** | 102 |
| 2 | marchmello01 | 14.91 | 4.49 | 13.38 | 0.30 | 19.40 | 748 |
| 3 | Hydra | **15.42** | 3.74 | 10.60 | 0.50 | 19.16 | 136 |
| 4 | sharadhnaidu | 13.21 | 2.53 | 6.92 | 0.46 | 15.74 | 113 |
| 5 | ayushm | 13.20 | 2.11 | 6.28 | 0.14 | 15.31 | 755 |
| 6 | sajidjavid | 12.92 | 2.28 | 6.70 | 0.20 | 15.20 | 113 |
| 7 | speechie | 9.56 | 5.28 | 15.57 | 0.43 | 14.84 | 208 |
| 8 | nilan | 9.24 | 0.77 | 2.28 | 0.06 | 10.01 | 125 |
| 9 | roujaz | 9.87 | 0.01 | 0.30 | −0.12 | 9.88 | 98 |
| 10 | sayandas | 7.59 | 0.51 | 1.34 | 0.12 | 8.10 | 106 |
| 11 | syehsyoh | 8.20 | −0.89 | −1.59 | −0.56 | 7.32 | 143 |
| 12 | asymptoticchase | 7.74 | −0.68 | −1.57 | −0.25 | 7.06 | 151 |

Phase-1 Track 2 audit-verified top 8 (5 Oct): CodeAMU, marchmello01, speechie, Hydra, ARCLY INDIA, bumblebee, RareMajor, liyingfa (leaderboard not reset).

**Reading the leaderboard**
- **Combined = SI-SDR (dB) + ΔWER (%)** exactly (e.g. 14.8945 + 5.9041 = 20.7986).
- Overall ΔWER ≈ **0.32 × synthetic ΔWER + 0.68 × natural ΔWER** (fits liyingfa and others) → natural clips carry ~2/3 of the weight, but **natural ΔWER is ≈0–0.5% for everyone**; the big ΔWER gains come from the synthetic clips. On natural clips the job is mainly *do no harm* (two teams are negative there).
- **Paid ranks (top 8)** currently need Combined ≳ 10; **top 5** ≳ 15.3; **top 3** ≳ 19.2; **1st** > 20.8.
- SI-SDR ceiling seen so far ≈ 15.4 dB; the leader wins on ΔWER (synthetic 17.4%), not on SI-SDR → preserving/restoring ASR-relevant speech matters as much as noise removal.
- Several teams submitted today → active field; scores will move before 17 Oct.
- Runtime varies 100–750 s → heavier models exist; runtime is not scored but must be reproducible by organisers.

## 8. Key insights driving the approach
1. **No timestamps at test time** → either a **blind enhancer** (whole clip) or a **two-stage system**: own noise-event detector (trained on gold/silver timestamps; Track-1 baselines exist) → event-conditioned enhancer. The two-stage design matches the organisers' framing and the novelty criteria ("principled event-conditioning").
2. **Noise overlaps speech** → gating/attenuation is wrong; need separation that keeps the speech under the noise.
3. **SI-SDR only on synthetic, ΔWER on everything** → natural clips (7 of 11 h, ~2/3 of ΔWER weight) mostly reward *not hurting* ASR; synthetic clips give SI-SDR and most ΔWER.
4. **One method for all clips** — organiser rule; also impossible to branch anyway (no flag in test).
5. **Equal weights:** +1 dB SI-SDR = +1 ΔWER% → a method that gains +3 dB but loses 4% ΔWER is worse.
6. **Local harness is possible:** 695 synthetic validation pairs (noisy + clean) for SI-SDR, transcripts for WER on all 2,501 validation clips. Use part for tuning, keep part held out (Zindi-style folds).
7. **Training-pair recipe:** clean vs noisy synthetic validation pairs reveal how organisers mixed noise (SNR, placement) → reproduce it to build large synthetic training sets from training audio.

---

## 9. Plan — experiment ladder (one change at a time, logged)

Local harness first: SI-SDR on validation synthetic clips (if clean refs exist) + WER via SraVaani on validation clips → no wasted daily submissions.

| Step | Experiment | Purpose |
|---|---|---|
| E0 | **Local harness**: SI-SDR on synthetic val pairs + SraVaani WER on all val clips; identity output as floor (never submitted) | Baseline numbers, format checks |
| E1 | Organisers' baseline (repo v-1) | Reference point |
| E2 | **Blind** pretrained enhancer on the whole clip (DeepFilterNet-class) + dry/wet blend | First valid submission; checks over-denoising on natural clips |
| E3 | **Two-stage**: own event detector (gold/silver timestamps) → event-conditioned enhancement (soft emphasis on detected regions, model processes every clip) | Protect ΔWER on natural clips, raise SI-SDR; novelty |
| E4 | Dry/wet blend tuning of E3 inside segments | Trade SI-SDR vs ΔWER explicitly |
| E5 | Fine-tune enhancer on Vaani synthetic mixtures (gold segments; external noise/speech allowed) | Domain adaptation |
| E6+ | Novelty: event-conditioned gating as model input; semi-supervised use of silver/bronze; lightweight/quantised enhancer that preserves ΔWER (links to compression research) | 15% novelty + report |

Logging: CSV per experiment (id, change, local SI-SDR, local ΔWER, Combined, LB result, notes). Only move forward on measured improvement.

---

## 10. Lessons carried over from the Zindi HTR competition
- Define the problem from data (listen to clips, inspect timestamps/segments) before choosing models.
- Trust the local harness over a small public leaderboard; check that local and LB agree early.
- Guard against single catastrophic outputs (e.g. silence, clipping, wrong length/sample rate) before comparing methods.
- Verify every claim: loading logs, file formats, rules wording, time estimates.
- Protect artefacts (persistent storage, download before stopping instances).
- Keep the reproducible `main.py` package and a running decision log from day 1 (needed for verification and the 4-page report).

---

## 11. Open questions
- ✅ Test timestamps? **No** — test input is audio only (checked Codabench + Drive).
- ✅ Validation clean references? **Yes** — 695 synthetic pairs.
- ✅ `transcripts.jsonl` schema and SraVaani call — see 4.4 (no language field needed; model is called on paths).
- ✅ WER references: tags + punctuation stripped, lower-cased, pooled WER (see 4.4).
- ⏳ Final-stage verification package required again? (likely; keep `main.py` ready)
- ⏳ Acknowledgement of award registration (Excel sent 5 Oct).
- ✅ Why no test timestamps: by design — test-time timestamps should come from our own Track-1-style detector (problem statement: "Track 1 predictions or provided ground truth").

## 12. Compute
- Kaggle T4 (free, ~30 GPU-h/week, 12 h/session) as default; small enhancers (DeepFilterNet-class) fit comfortably.
- RunPod: ~$4 credit left — reserve for a final heavy run if needed.
- SraVaani ASR inference on 11 h of audio also needs GPU time — budget it.

---

## 13. Experiment log & findings

| # | Experiment | Local (validation) | Codabench Phase 2 |
|---|---|---|---|
| E0 | No processing (floor) | WER noisy: all 0.290 · natural 0.282 · synthetic 0.315; synthetic input SI-SDR −0.33 dB; ASR RTF 0.0155 (≈10 min for 11 h test) | — |
| E1 | Organisers' baseline v-1, `fast` (BiGRU mask + SraVaani FiLM), sim-trained → fine-tuned on 591 validation synthetic pairs, blend 1.0 | sim val SI-SDRi +5.7 dB; real held-out before FT +2.9 dB; **after FT held-out 12.3 dB**; natural ΔWER (400 clips) −0.36%; est. Combined ≈ 11.3 | **SI-SDR 3.67 · ΔWER −2.84% (syn −6.47, raw −1.12) · Combined 0.83 · rank 13/13 · audit PASS** |

Submission details: 3,178 test clips (2,079 natural + 1,099 synthetic), ZIP 1,050 MB, 19 clips flagged by the baseline's level check (peak-limited loud inputs, harmless), 2 empty transcripts. Kaggle quirk: `.pt` files are auto-extracted on upload → rebuild the zip archive in the notebook.

### 13.1 Diagnostic (why local 12.3 dB ≠ test 3.7 dB) — LEAKAGE CONFIRMED
- **Noise library reuse:** of 2,213 noise events in the 695 synthetic validation pairs, **98.8%** have a near-identical noise (log-mel fingerprint cosine > 0.98) in another clip. Validation synthetic mixtures are built from a **small, heavily reused noise library**.
- **Held-out vs fine-tune:** **97.7%** of held-out noise events have a near-identical noise in the fine-tune set → the baseline's random 85/15 split leaks; fine-tuning memorised the library. Recording overlap adds a little (23/104 held-out clips share a recording with fine-tune; 590 recordings for 695 clips overall).
- Test synthetic noise is evidently different (the Phase-2 test was changed after dictionary-style exploits) → the fine-tuned gains did not transfer; test SI-SDR ≈ the pre-fine-tune level.
- **Real SNRs are far harsher than the baseline simulation:** global SNR of real pairs median −2.6 dB (5–95%: −9.2…+16.5); **event-local SNR median −9.5 dB** (25%: ≈ −94 dB, i.e. many events have no speech underneath; 75%: −5.2; 95%: +2.5). Baseline simulated event-local −5…+12 dB (input SI-SDR +12.5 dB vs real −0.3 dB) and its noise bank had only **141** spans.
- Top synthetic noise tags: `<horn>` 591, `[breathing]` 544, `<bird sound>` 289, `<bird squawking>` 140, `<child talking>` 108, `[inhaling]` 94, `<child yelling>` 66, `<barking>` 50, `<vehicle noise>` 35, `<animal sound>` 34, `<machine noise>` 29, `<phone ringing>` 23.
- Natural ΔWER is negative for every blend → the model also removes speech-relevant content on real recordings.

### 13.2 Revised plan (validation first)
1. **Leak-free validation:** cluster noise fingerprints; hold out **entire noise clusters** (and recordings) → measures performance on unseen noise, like the test. Natural clips: ΔWER check on a fixed subset.
2. **No direct fine-tuning on the reused validation noise** (memorisation). Validation pairs are for evaluation (cluster-held-out), at most for light adaptation on disjoint clusters.
3. **Training data that generalises:** many diverse noises (training-set events + open external noise datasets — allowed by the open-data policy), **real-like SNRs** (event-local ≈ −20…+5 dB, plus noise-only events), clean speech spans from Vaani training audio.

### 13.3 E2 — pretrained enhancers zero-shot (leak-free: never saw validation)
| Model | blend | SI-SDR (syn) | ΔWER all | ΔWER syn | ΔWER nat | Combined est |
|---|---|---|---|---|---|---|
| identity | 1.0 | −0.33 | 0.01 | 0.01 | 0.01 | −0.32 |
| SepFormer WHAM16k | 1.0 | −0.30 | −4.78 | −13.56 | −2.06 | −5.08 |
| SepFormer WHAM16k | 0.85 | −0.44 | −5.00 | −14.27 | −2.13 | −5.44 |
| SepFormer WHAM16k | 0.70 | −0.72 | −4.98 | −14.31 | −2.08 | −5.70 |

DeepFilterNet3 and FRCRN were skipped (install/load failures on Kaggle Python 3.13; installing them normally also broke numpy → install with `--no-deps`). **Lesson:** generic enhancers trained on stationary/babble noise and English speech neither remove bursty events nor preserve Indic speech (breathing / child sounds look like speech) → no readymade shortcut; task-specific training needed.

### 13.4 E3 — ORACLE event suppression with ground-truth timestamps + classic DSP (no training)
| Variant | SI-SDR | ΔWER all | ΔWER syn | ΔWER nat | Combined |
|---|---|---|---|---|---|
| gate g=0.3 (20 ms fades) | 4.24 | −0.43 | −1.76 | −0.02 | **3.81** |
| gate g=0.1 | 5.61 | −2.00 | −8.14 | −0.09 | 3.61 |
| gate g=0.1, dilate 50 ms | 4.79 | −1.61 | −6.09 | −0.22 | 3.18 |
| gate g=0.1, dilate 150 ms | 2.27 | −0.61 | −1.70 | −0.28 | 1.66 |
| Wiener in events, floor 0.1 | 4.76 | −5.94 | −15.14 | −3.09 | −1.18 |
| Wiener in events, floor 0.03 | 4.78 | −10.24 | −18.31 | −7.74 | −5.46 |
| gate g=0 (mute) | 4.44 | −19.81 | −20.76 | −19.52 | −15.37 |

**Lessons:** even perfect timestamps + suppression give only ~4–5.6 dB / Combined ≤ 3.8 → "detect + suppress" is not the winning recipe (hypothesis falsified). ~25–35% of speech energy sits under events; muting removes words (ΔWER −20%); classic Wiener creates ASR-hostile artefacts. Natural clips are barely affected by mild gating. → the core task is **separating speech from loud overlapping noise**; a detector is at most auxiliary.
Team overlap observation: 5 teams are top-8 in both Track 1 and Track 2 (CodeAMU, liyingfa, Hydra, ARCLY INDIA, bumblebee).

### 13.5 E4 — realistic simulation (one change vs E1: training data)
- Noise pool: **isolated validation noise** (noisy − clean, per event, fingerprint-clustered; only train clusters) **+ ESC-50** (2,000 clips, 50 categories; CC BY-NC 3.0 — documented). Clean: Vaani Gold gaps (baseline bank). SNR event-local −20…+5 dB, 1–4 events / 5 s, 6,000 mixtures, baseline MaskNet + SraVaani FiLM, 20 epochs. Model saved as `t2_e4.pt`.
- **Leak-free eval:** 228 synthetic validation clips whose events all belong to held-out noise clusters (input SI-SDR −1.03 dB, noisy WER 0.338) + all 1,806 natural clips (noisy WER 0.282).

| blend | SI-SDR held-out | ΔWER syn held-out | ΔWER nat | Combined est (0.32·syn + 0.68·nat) |
|---|---|---|---|---|
| **1.0** | **7.51** | −5.00 | −0.33 | **5.69** |
| 0.85 | 6.27 | −2.71 | −0.11 | 5.33 |
| 0.70 | 4.49 | −1.78 | −0.06 | 3.88 |

**Lessons:** data alone lifted unseen-noise SI-SDR from ~2.5 dB (E1 pre-fine-tune) to 7.5 dB (+8.5 dB over input) → realistic simulation is the main lever. **ASR is now the bottleneck** (synthetic ΔWER −5%). Go/no-go: SI-SDR ≥ 6 ✓, ΔWER ≥ 0 ✗ → **partial go**.

### 13.6 Next (decision by 8 Oct)
1. **Calibration submission** with `t2_e4.pt` (attach the E4 notebook *output* directly — not as a re-uploaded dataset — so Kaggle does not auto-extract the `.pt`). Expected ≈ 5.7 if the leak-free protocol is reliable.
2. **E5 — ASR-aware loss:** SI-SDR loss + feature-matching loss on frozen SraVaani encoder (enhanced vs clean), same realistic simulation. Target on held-out: ΔWER ≥ 0 with SI-SDR ≈ 7+.
3. If E5 fails → stop IndoML and move to the Bias Bounty challenge (31 Oct).

### 13.7 Calibration submission — E4 model (`t2_e4.pt`, blend 1.0), 6 Oct 2026 20:52
| | Local leak-free (E4) | **Codabench Phase 2** | Gap |
|---|---|---|---|
| SI-SDR (synthetic) | 7.51 | **5.91** | −1.6 dB |
| ΔWER synthetic | −5.00% | **−5.62%** | ≈ match |
| ΔWER natural | −0.33% | **−1.04%** | test ~3× worse |
| ΔWER overall | — | −2.51% | (= 0.32·syn + 0.68·nat, confirms weighting) |
| Combined | 5.69 (est.) | **3.40** | −2.3 |
Audit 1.0; runtime 766 s.

**Lessons:**
- The leak-free protocol is **directionally reliable but optimistic**: old leaky estimate 11.3 vs LB 0.83; leak-free estimate 5.7 vs LB 3.40. Working rule: **LB ≈ local Combined − ~2.3** (SI-SDR −1.6, natural ΔWER more negative on test).
- E4's data change was real: LB 0.83 → 3.40, SI-SDR 3.67 → 5.91.
- **ASR damage costs ~2.5 points** on LB (synthetic ≈ 1.8, natural ≈ 0.7). Bringing ΔWER to 0 alone would give ≈ 5.9 → E5 (ASR-aware loss) targets exactly this. Watch natural ΔWER closely; local natural underestimates harm.

### 13.8 E5 — ASR-aware loss (SI-SDR + λ·L1 on layer-normed frozen SraVaani-encoder features), same data/eval as E4
Training: 20 epochs, bs 4 × accum 2, ~5.2 min/epoch per T4 (encoder backward), 2 variants in parallel. Sim val SI-SDR (input −5.63 dB): λ5 → 6.15, λ20 → 5.76 (still rising slowly → capacity-limited).

| λ | blend | SI-SDR held-out | ΔWER syn held-out | ΔWER nat | Combined est | ≈ LB (est − 2.3) |
|---|---|---|---|---|---|---|
| 0 (E4) | 1.0 | 7.51 | −5.00 | −0.33 | 5.69 | 3.40 (actual) |
| **5** | 1.0 | 7.48 | −4.22 | −0.18 | **6.00** | ≈ 3.7 |
| 20 | 1.0 | 7.25 | −4.03 | −0.20 | 5.82 | ≈ 3.5 |
| 5 | 0.85 | 6.26 | −2.32 | −0.06 | 5.48 | — |
| 20 | 0.85 | 5.94 | −1.63 | −0.07 | 5.38 | — |

**Lessons:** the ASR-feature loss helps slightly (+0.3 est., ΔWER less negative) but does **not** remove the ASR damage; within held-out noise (~228 clips). The bottleneck is the model itself (magnitude mask with noisy phase, small BiGRU) rather than the loss. Go/no-go criteria (ΔWER ≥ 0 and SI-SDR ≥ 7 on held-out) **not met** → per the agreed rule, **no-go**; remaining gap to a paid rank (LB ≳ 10 vs ≈ 3.7) needs a new architecture, unlikely in the remaining time on T4 solo.

### 13.9 E5 submission + survey + E6 MP-SENet (phase-aware) — BREAKTHROUGH
**E5 λ=5 on Codabench (8 Oct 00:10):** SI-SDR 5.82 · ΔWER −1.57 (syn −3.67, nat −0.58) · **Combined 4.24** (local est 6.0; ASR loss helped more on test than locally).
**ASR-ceiling diagnostic:** synthetic validation WER noisy 0.315 vs **clean 0.194** → max ΔWER_syn ≈ **+12.1%** (big headroom; leader +17.4% on test).
**Survey (literature):** SE front-ends often fail to help or hurt modern ASR mainly through **artifacts** (non-linear error component); remedies: observation adding (fixed/time-varying blend), artifact-boosted SDR loss, joint/ASR-aware training (often lowers SDR); phase-aware models reduce artifacts. Candidate architecture: **MP-SENet** (parallel magnitude + phase denoising, ~2.3 M params, MIT, training code + **DNS-pretrained checkpoint** in repo). SEMamba/CS-Mamba strong but Mamba kernels are an install burden; generative SE risky for ASR (hallucinated words).

**E6 Phase A — MP-SENet DNS checkpoint zero-shot on ALL validation (leak-free, never trained on it):**
| blend | SI-SDR | ΔWER syn | ΔWER nat | Combined est |
|---|---|---|---|---|
| **1.0** | **14.24** | **+0.72** | −0.30 | **14.26** |
| 0.85 | 11.54 | +3.50 | −0.12 | 12.58 |
Enhancement of 2,501 clips (3.5 h) took 24 min on T4 (4 s windows, 3 s hop, cross-fade; RMS-normalised in/out).

**E6 submission (8 Oct 03:40) — MP-SENet zero-shot, blend 1.0:** **SI-SDR 9.56 · ΔWER +0.44 (syn +1.88, nat −0.24) · Combined 10.00 · rank 10** (top-8 boundary ≈ 10.0). First positive ΔWER.
- Local → LB gap now −4.3 (SI-SDR 14.2 → 9.6): test synthetic noise is harder/different than validation's library → **domain adaptation (fine-tuning on harsh, diverse simulation) is the next lever**.
- **E6 Phase B smoke fine-tune failed with CUDA OOM on T4** (MP-SENet time/frequency GRU+attention over 201 bins × frames at bs 4 × 2 s, plus SraVaani encoder for ASR loss, plus ASR and zero-shot models resident). Fix: free other models, AMP, bs 2, 1–1.5 s segments, or run on RunPod A100 80 GB.

**Lesson:** the real bottleneck was the **model class** (magnitude mask + noisy phase). A phase-aware model pretrained on large, diverse noise (DNS) generalises to unseen bursty events and preserves ASR; WHAM-trained SepFormer did not.

### 13.10 Next
1. **Fine-tune MP-SENet** (DNS init) on the realistic simulation (isolated val noise train-clusters + ESC-50 + optionally DNS noise; event SNR −20…+5 dB), losses = MP-SENet (mag + phase + complex + consistency + time) ± ASR-feature loss; leak-free held-out eval; RunPod A100 for the real run after a fixed Kaggle smoke test.
2. Natural-clip protection (time-varying mixing) — natural ΔWER still −0.24.

### 13.11 E7 / E7b — MP-SENet fine-tune smoke tests (Kaggle T4)
- **E7 (fp16 AMP, bs 2, 1.5 s):** ran without OOM (peak 7.2 GB, 2.4 it/s) but zero-shot and fine-tuned scores were **identical** → fp16 overflow made GradScaler skip every update (silent failure; same lesson as LightOnOCR fp16 in the Zindi HTR competition). Always log weight change.
- **E7b (fp32, bs 2, 1 s, 1,000 mixtures × 2 epochs, λ=0):** weight change 3.26%, loss 0.72 → ~0.5, peak 5.7 GB, 2.15 it/s.

| model | SI-SDR held-out | ΔWER syn held-out | ΔWER nat (300) | Combined est |
|---|---|---|---|---|
| MP-SENet zero-shot | 14.39 | +0.62 | −0.53 | 14.23 |
| **MP-SENet smoke FT** | **14.74** | **+2.36** | **+0.06** | **15.53** |
Even a tiny fine-tune helps (+1.3), and natural ΔWER turns positive.

### 13.12 E8 (running) — full fine-tune on Kaggle 2×T4 (free; RunPod not needed so far)
8,000 mixtures, 5 epochs, fp32, OneCycle LR 2e-4; GPU 0 λ=0, GPU 1 λ=5 (ASR-feature loss); best epoch by held-out SI-SDR; final leak-free eval on held-out synthetic + all natural (blends 1.0 / 0.9).

### 13.13 E8 results — full MP-SENet fine-tune (Kaggle 2×T4, 3 h 37 min)
| run | best epoch held-out SI-SDR (zero-shot 14.39) | blend | ΔWER syn held-out | ΔWER nat (all) | Combined est |
|---|---|---|---|---|---|
| **λ=0** | **16.68** (ep 3; per epoch 15.25 / 15.96 / 16.46 / 16.68 / 16.66) | **1.0** | **+3.25** | −0.02 | **17.71** |
| λ=0 | — | 0.9 | +5.35 | −0.02 | 15.31 |
| λ=5 | never beat zero-shot (12.45 → 13.61 → 13.38) → saved ckpt = zero-shot | 1.0 | +0.62 | −0.28 | 14.40 |
**Lessons:** plain MP-SENet losses on realistic simulation give +2.3 dB on unseen noise and better ASR; the ASR-feature loss at λ=5 dominated MP-SENet's small loss scale (~0.5 vs ~1.2) and hurt SI-SDR → drop it or use a much smaller λ. Next: submit λ=0 (blend 1.0); expected LB ≈ 13 if the zero-shot gap (−4.3) holds.

### 13.14 E8 submission — fine-tuned MP-SENet (λ=0, blend 1.0), 10 Oct 10:07
**SI-SDR 12.37 · ΔWER +1.62 (syn +4.65, nat +0.20) · Combined 13.99 · rank 10/12** (zero-shot was 10.00). Local 17.71 → LB 13.99 (gap ≈ −3.7, stable with zero-shot's −4.3 → leak-free protocol now reliable for ranking changes). Natural ΔWER positive for the first time.

Leaderboard 10 Oct: 1 liyingfa 20.80 · 2 Hydra 19.83 · 3 marchmello01 19.40 · 4 Raj Patil 16.25 · 5 sharadhnaidu 15.74 · 6 sajidjavid 15.67 · 7 ayushm 15.31 · **8 asymptoticchase 15.08** · 9 speechie 14.84 · **10 avics55 13.99** · 11 nilan 10.01 · 12 syehsyoh 9.89.
Gap to paid zone (top 8): **+1.1**; typical 4–8th entries have SI-SDR ≈ 13–14 and ΔWER syn ≈ 6–7% (ours 12.4 / 4.65).

### 13.15 E9 plan — more noise diversity + more data (Kaggle 2×T4)
- Add **MUSAN noise** (CC BY 4.0 / public domain, documented) + noise augmentation (speed 0.9/1.1, random EQ, synthetic reverb) to the val-train + ESC-50 pool; 16,000 mixtures.
- GPU 0: fine-tune from DNS checkpoint, 1.5 s segments, LR 2e-4, 4 epochs. GPU 1: warm-start from E8 checkpoint, LR 1e-4, 3 epochs. Best epoch by held-out SI-SDR; final leak-free ASR eval.

### 13.16 E9 results (Kaggle 2×T4; MUSAN download truncated at the 40-min limit → only 397 MUSAN segments; noise pool 3,039 base + 3,035 augmented; 16,000 mixtures)
| run | held-out SI-SDR per epoch | best | ΔWER syn | ΔWER nat (all) | Combined est |
|---|---|---|---|---|---|
| A: DNS init, 1.5 s seg, LR 2e-4, 4 ep | 15.53 / 16.71 / 17.05 / 17.13 | **17.13** | +3.37 | −0.08 | **18.15** |
| B: warm start E8, 1 s seg, LR 1e-4, 3 ep | 16.83 / 16.64 / 16.93 | 16.93 | +3.41 | −0.02 | 18.01 |
vs E8 16.68 / 17.71 → +0.3–0.45 local (more noise diversity + longer segments help; full MUSAN still pending).
Lesson: never download 11 GB inside a timed notebook — MUSAN noise is being uploaded as a Kaggle dataset (`musan-noise`).
Next: submission notebook picks the best of A / B / average(A,B) on held-out, then runs the test set.

### 13.17 E9 submission — ensemble avg(A, B), 13 Oct 08:32
Local (held-out + 400 natural): A 18.11, B 17.99, **avg(A,B) 18.25** (SI-SDR 17.14, ΔWER syn +3.80, nat −0.16) → chosen. First upload attempt failed on Codabench ("child task failed") — likely the page was refreshed during the 1 GB upload; the same ZIP re-uploaded without touching the page scored fine.
**Codabench: SI-SDR 12.73 · ΔWER +1.80 (syn +5.05, nat +0.27) · Combined 14.53 · rank 11** (predicted ≈ 14.5 from local − 3.7 → calibration holds).
Leaderboard 13 Oct: 1 liyingfa 20.80 · 2 Hydra 19.90 · 3 rounders 19.53 · 4 marchmello01 19.40 · 5 Raj Patil 17.02 · 6 asymptoticchase 16.48 · 7 sajidjavid 16.18 · **8 sharadhnaidu 15.74** · 9 ayushm 15.31 · 10 speechie 14.84 · 11 avics55 14.53. Top-8 line moved up; +1.2 needed.
Ensemble lesson: averaging two differently trained MP-SENets keeps SI-SDR and improves ASR (fewer artefacts).

### 13.18 Decision to use RunPod (14 Oct)
- After E9 (LB 14.53, rank 11; top-8 line 15.74) the remaining lever was *more training at scale* (curves still rising). Kaggle T4 limits: batch 2, ≤ 2 s segments, 12 h sessions, weekly quota and long GPU queues (one submission waited at queue position 34).
- An E10 Kaggle notebook (full MUSAN from the new `musan-noise` dataset, 2 s segments, warm start from E9-A) was prepared but **not run**; the big run moved to RunPod instead.
- MUSAN noise was uploaded once as a private Kaggle dataset `musan-noise` (930 WAVs, free-sound 845 + sound-bible 88, ~717 MB) after the in-notebook download had been truncated by a 40-min time limit (E9 got only 397 MUSAN segments). Lesson: never download multi-GB data inside a timed notebook.
- Second Kaggle account for extra GPU quota was considered and rejected (Kaggle ToS: one account per person; risk to the main account).

### 13.19 E10-R on RunPod A100 SXM 80 GB ($1.79–1.80/h, container disk 60 GB, no network volume) — chronological log
**Setup**
- Pod: RunPod PyTorch 2.8.0 template, A100 SXM (A100 was briefly unavailable; became available). Notebook `indoml_t2_e10r_runpod.ipynb` reuses the Kaggle code unchanged by symlinking `/kaggle/working → /workspace/work` and `/kaggle/input → /workspace/input`.
- Kaggle now issues **API tokens (`KGAT_…`) instead of `kaggle.json`**: set `KAGGLE_API_TOKEN` and write `~/.kaggle/access_token`; Kaggle CLI 2.2.4.
- Data pulled with the Kaggle API: `indoml-validation`, `indoml-track2-test` (3,178 WAVs), `musan-noise` (930 WAVs), `indoml-t2-ckpt` (noisy ASR JSON). Pitfalls: `dataset_list(mine=True)` returns only the first page (fix: download by explicit `owner/slug`); some datasets live under `avin55/…`, newer ones under `avics55/…`; `kaggle kernels output --file-pattern` fetched only the notebook log, not the `.pt` files.
- E9 checkpoints had to be uploaded by hand. Kaggle's Output page delivers a `.pt` as a zip of the extracted torch archive (folder with `data.pkl`) → re-zipped into a `.pt` on the pod. **E9-B (`e9_warm_e8`) arrived; E9-A (`e9_dns_init`) did not** (the same B file had been downloaded twice) → to avoid idle GPU cost, **B was copied under A's name** and used as the warm start.
- RunPod image lacked `matplotlib` (MP-SENet utils), `datasets` (Vaani streaming) and later `transformers` (SraVaani ASR) — installed on demand.

**Data**
- Clean bank from **11,000 Vaani Gold clips → 8,733 clean spans** (13.5 min; Kaggle runs had 3,968). Baseline noise bank 276 (unused).
- Noise pool: isolated validation noise (train clusters) 644 + ESC-50 1,998 + **MUSAN 2,456 crops** = 5,098 base + 5,094 augmented (speed 0.9/1.1, random EQ, synthetic reverb) = **10,192**.
- **40,000 mixtures** (2.5 min), event SNR −20…+5 dB, input SI-SDR −5.41 dB. Held-out split identical to earlier runs (139 noise clusters, 62 held out, 228 eval clips).
- Ordering slip: the augmentation cell was skipped once (`NameError: aug_noise`); re-ran only the missing steps without rebuilding the clean bank.

**Memory probe (bf16 autocast for the MP-SENet forward, fp32 losses/iSTFT)**
- bs 8 × 3 s → OOM; bs 6 × 3 s → OOM; **bs 4 × 3 s → 67.7 GB** (chosen). MP-SENet's time/frequency attention dominates memory.

**Training run 1 (warm start from B, LR 1e-4 OneCycle over 8 epochs, 5.5 h cap)**
- Speed ≈ 5,900 steps/h (1 epoch = 9,500 steps ≈ 1.6 h).
- Held-out SI-SDR: start 16.93 → **epoch 0 17.00** → epoch 1 16.95. Stopped manually at ~26k steps (~4.4 h).
- **Mistake:** the LR schedule was set for 8 epochs while the time cap allowed only ~3.3, so the LR never decayed and the model could not settle. Should have measured speed first and sized epochs to the budget (as done in the Zindi work).

**Annealing run (from epoch-0 checkpoint, 1 epoch, LR 5e-5 OneCycle → ~0, 1.61 h)**
- Held-out SI-SDR **17.00 → 17.26** (saved as `e10r_anneal.pt`, model **D**). The final in-script ASR eval crashed only because `transformers` was missing (model unaffected).

**Selection on held-out (228 synthetic) + 400 natural clips** (on the pod "A" = copy of B)
| combo | SI-SDR | ΔWER syn | ΔWER nat | Combined est |
|---|---|---|---|---|
| B only | 16.93 | +3.45 | −0.02 | 18.02 |
| C (run-1 epoch 0) | 17.00 | +4.07 | −0.12 | 18.22 |
| B + C | 17.17 | +3.99 | −0.12 | 18.36 |
| B + C (+B dup) | 17.14 | +3.80 | −0.07 | 18.31 |
| D (annealed) | 17.26 | +4.15 | −0.23 | 18.43 |
| **B + D** | **17.28** | +4.07 | −0.05 | **18.55** ← chosen |
| B + C + D | 17.27 | +3.99 | −0.05 | 18.51 |
(E9 ensemble avg(A,B) was 18.25 → B+D is +0.30 local ⇒ expected LB ≈ 14.8–14.9.)

**Test submission on the pod (B + D)**: enhancement 26.8 min for 3,178 clips (2 models, A100), 2 level flags, 0 empty transcripts, ZIP 1,079 MB → saved as `submission_BD.zip`.

**Real A check (after re-upload, verified A-vs-B weight difference 8.92%)**: A+D 18.52, A+B+D 18.55 → **no gain over B+D** → no re-run; pod terminated.

**Artefacts downloaded**: `submission_BD.zip` and `all_models_bundle.zip` (32 MB: A_e9_dns_init.pt, B_e9_warm_e8.pt, C_e10r_warmA.pt, D_e10r_anneal.pt, both logs, mp_ft.py, held_clips.json, transcripts, notebook with tokens removed).

**Cost**: ≈ 10 A100-hours (setup ~2 h, run 1 ~4.4 h, anneal ~1.8 h, selection/test ~1.2 h) ≈ $18; balance left ≈ $2.7.

### 13.20 Lessons from the RunPod phase
1. **Size the LR schedule to the real compute budget** (measure steps/h first). A short annealing run recovered most of the lost value (+0.26 dB).
2. Kaggle → RunPod transfer: use API tokens and explicit dataset refs; download notebook-output checkpoints manually; verify every checkpoint by loading it and by a weight-difference check (caught the A/B mix-up).
3. Plan credits for idle time after training ends (the pod keeps billing); terminate only after artefacts are downloaded.
4. Ensembling saturates: B+D = A+B+D; a third model adds cost, not score.
5. Pod base images miss common libraries (matplotlib, datasets, transformers) — install up front.

### 13.21 Status
- Pending: Codabench score of `submission_BD.zip` (expected ≈ 14.8–14.9).
- Deadline: 17 Oct 2026, 12:00 IST.

## 14. Mistakes and misses — what happened, why, and how to avoid it (whole Track 2 effort)
| # | What we missed / got wrong | Why it happened | Cost | Prevention next time |
|---|---|---|---|---|
| 1 | Fine-tuned the baseline on validation pairs and trusted a random 85/15 split (local 12.3 dB vs LB 3.67) | Did not check whether validation noise clips were reused; assumed validation ≈ test | 1 submission, ~1 day | Inspect data reuse (fingerprints) before any split; split by noise cluster/recording |
| 2 | Assumed "detect + suppress" (oracle gating) would win | Reasoned from leaderboard team overlap without measuring how much speech sits under events | ~1 h | Run cheap oracle/upper-bound experiments before building pipelines (we did, and it falsified the idea quickly) |
| 3 | Spent E1–E5 on the baseline magnitude-mask model | Model class not questioned early; noisy phase + small BiGRU capped SI-SDR and created ASR-hurting artefacts | ~2 days | Survey architectures first; test a strong phase-aware pretrained model (MP-SENet) zero-shot on day 1 |
| 4 | fp16 AMP silently skipped every update (E7) | GradScaler skips steps on inf/NaN gradients; loss still printed | 1 run | Always log relative weight change; fp32 on T4, bf16 on A100 |
| 5 | ASR-feature loss at λ=5 hurt MP-SENet | Loss scales differ (MP-SENet loss ~0.5 vs ASR term ~1.2) → ASR term dominated | half of E8 | Normalise loss terms / sweep λ on a short run first |
| 6 | Module-name clash (`inference.py` in both repos) killed E8 at start | `sys.path.insert(0, …)` put MP-SENet ahead of the baseline | ~15 min | Append third-party repos to the end of `sys.path`; smoke-test the script, not only the notebook |
| 7 | Committed an outdated E9 notebook version (checkpoint finder fix missing) | Re-imported the older file | ~16 min | Re-download the latest notebook before each commit; put a version string in cell 1 |
| 8 | MUSAN download truncated at a 40-min limit (397 instead of 2,456 segments) | Downloading 11 GB inside a timed notebook on a slow day | weaker E9 data | Upload big external data once as a Kaggle dataset |
| 9 | Kaggle notebook outputs overwritten by later commits / attaching the wrong version | Reused one notebook for training and submission | confusion, re-uploads | One notebook per experiment; checkpoints into datasets |
| 10 | `.pt` uploads auto-extracted by Kaggle | Kaggle treats zip-format files (torch archives) as archives | small | Loader that rebuilds a `.pt` from the extracted folder (now standard) |
| 11 | Codabench submission failed once | Page refreshed during the 1 GB upload | 1 attempt | Never touch the page until upload completes |
| 12 | E9-A checkpoint never reached the pod; B was downloaded twice | Similar filenames (`e9_dns_init` vs `e9_warm_e8`); no verification after upload | ensemble limited to B+D (later verified: real A gave no gain) | Verify every checkpoint by loading it and by a weight-difference check |
| 13 | Skipped the augmentation cell on the pod (`NameError`) | Running cells out of order in a long notebook | ~0 (clean bank kept in memory) | Single "prepare data" script or one cell per stage with asserts |
| 14 | Pod base image missing `matplotlib`, `datasets`, `transformers` | Kaggle image preinstalls them; RunPod image does not | minutes | Install the full dependency list in the setup cell |
| 15 | **LR schedule sized for 8 epochs while the 5.5 h cap allowed ~3.3** (run 1 stalled at 17.00) | Did not measure steps/h before fixing epochs; OneCycle never reached its decay phase | ~$4–5 of A100 time with little gain | Measure speed in the first 10 min, then set epochs/schedule to the budget; annealing recovered +0.26 dB |
| 16 | Idle-pod billing risk overnight | Training end ≠ pod stop; credits could run out and wipe the container disk | near-loss of the trained model | Budget idle hours, auto-upload + auto-stop watcher, or top up before sleeping |

**Correction (training volume):** before the RunPod run the plan was quoted as "~250 h of audio, ~10× Kaggle", which assumed all 8 epochs. Actual: run 1 ≈ 26,000 steps × 4 × 3 s ≈ 87 h; annealing 9,500 × 4 × 3 s ≈ 32 h; **total ≈ 119 h ≈ 4.7× E9-A** (30,400 × 2 × 1.5 s ≈ 25 h). The over-estimate came from the same planning error as mistake #15 (schedule/epochs not sized to the time cap).

### 13.22 E10-R submission — ensemble B + D (15 Oct 07:22)
**Codabench: SI-SDR 12.86 · ΔWER +1.95 (syn +5.41, nat +0.31) · Combined 14.81 · rank 11** (E9: 14.53). Predicted 14.8–14.9 from local 18.55 − 3.7 → calibration holds again (gap −3.74).
Leaderboard 15 Oct: 1 sajidjavid **26.49** (jumped from 16.18; SI-SDR 20.34, ΔWER syn 18.09) · 2 liyingfa 20.80 · 3 Hydra 19.90 · 4 marchmello01 19.55 · 5 rounders 19.53 · 6 Raj Patil 17.23 · 7 asymptoticchase 16.58 · **8 sharadhnaidu 15.74** · 9 ayushm 15.31 · 10 speechie 14.84 · **11 avics55 14.81**.
Reading: our steps now add ~+0.3 each; the paid line (15.74) is +0.93 away and top-5 ≥ 19.5. The persistent local→LB gap (~3.7) says test noise differs from anything we simulate; the leader's jump to SI-SDR 20 implies a different method class, not more of our recipe.

### 13.23 E11 smoke (Kaggle T4) — bootstrap real Vaani noise with D + noise-match diagnostic + quick wins
- **Bootstrapping:** 1,500 Vaani gold/silver clips → noise_hat = noisy − D(noisy) inside annotated event windows (0.3–2.5 s) → only **535 events** (many events are longer than 2.5 s) + 1,130 gold clean gaps.
- **Diagnostic** (which pool does TEST residual noise resemble? mean max-cosine of log-mel fingerprints): MUSAN 0.731, ESC-50 0.727, Vaani-bootstrapped 0.671, validation library 0.592; nearest-pool share MUSAN 48%, ESC-50 35%, Vaani 7%, val-library 10%. Test noise looks **library-like (Freesound-derived: ESC-50 + MUSAN free-sound)**, not Vaani-recording-like. Caveat: crude metric (val-synthetic residuals also matched MUSAN more than their own library) and bigger pools get an advantage.
- **Mini fine-tune with Vaani noise** (3,000 mixtures, 1 epoch, LR 5e-5 annealed, weight change 0.58%): held-out 18.44 → **18.25 (worse)** → Vaani-noise idea dropped (E11-R RunPod run cancelled, money saved).
- **Quick wins on D (no training)** — held-out + 400 natural:
| variant | SI-SDR | ΔWER syn | ΔWER nat | Combined |
|---|---|---|---|---|
| win4 (current) | 17.25 | +4.18 | −0.23 | 18.44 |
| win8 | 17.48 | +3.84 | −0.07 | 18.66 |
| shift-TTA×2 (0, 50 samples) | 17.72 | +4.69 | −0.09 | 19.16 |
| **win8 + TTA×2** | **17.92** | +4.22 | −0.09 | **19.22** |
60% of clips are longer than 4 s (median 4.3 s), so the 4 s windowing + cross-fade was costing quality.

### 13.24 Quick-win submission on RunPod A100 (no training)
- Pitfall: `mpsenet-e10r` lives on the second Kaggle account (avics55) → 403 with the avin55 API token → bundle uploaded by hand (32 MB).
- Selection (held-out + 400 natural): D win8+TTA2 19.27 · **D win8+TTA4 19.65** · B+D win8+TTA2 19.13 · B+D win8+TTA4 19.55 → TTA makes the B+D ensemble unnecessary (TTA is itself a self-ensemble).
- Test with D win8+TTA4: 71.5 min (4 passes), 0 fails, ZIP 1,077 MB.
- Extra check **D win8+TTA8 (8 shifts 0…87): 19.83** (+0.18 over TTA4) → test re-run with TTA8 (8 passes, ~2.4 h). The kernel/UI looked frozen near the end; the WAVs (13:39) and transcripts/ZIP (13:54 UTC) were in fact written — verified via file timestamps and a zip-integrity check (3,179 entries). Pod terminated afterwards.

### 13.25 Codabench submissions on 15–16 Oct
| submission | LB SI-SDR | ΔWER (syn / nat) | **Combined** |
|---|---|---|---|
| B+D win4 (E10-R) | 12.86 | 1.95 (5.41 / 0.31) | 14.81 |
| **D win8 + TTA×4** | 13.15 | 2.03 (5.69 / 0.30) | **15.18** |
| **D win8 + TTA×8** | 13.18 | 2.02 (5.68 / 0.30) | **15.21** ← best, rank 10 |
| D win8 + TTA×8, rescaled to peak 0.95 | 13.18 | 1.89 (5.09 / 0.38) | 15.07 |
- Submission handling: two uploads sat in "Preparing" with "No logs" for hours (queue/upload issue on Codabench, file verified fine); cancel + re-upload, or wait — they eventually ran. Email to datathon@indoml.in drafted. Local files identified by exact byte size (TTA4 1,076,526,137; TTA8 1,076,348,305).
- **Transfer warning:** TTA gave +1.1–1.3 local but only **+0.37–0.40 on LB**; loudness (peak 0.95) gave +0.09 local but **−0.14 on LB** (synthetic ΔWER fell 5.68 → 5.09 while natural rose 0.30 → 0.38).

### 13.26 ASR-output tweak tests (local, D win8+TTA2, held-out 228 + 600 natural)
Round 1 (SI-SDR is scale-invariant): rms 0.03 18.62 · rms 0.06 18.98 · rms 0.10 19.02 · rms = input (current) 19.20 · **peak 0.95 19.29** · blend 0.97 18.47 · blend 0.94 17.45.
Round 2: peak 0.95 19.29 · hpf80 + peak 17.41 (SI-SDR −1.9 dB) · softclip rms 0.20 15.13 · softclip 0.30 13.22 · hpf80 + softclip 14.94.
**Conclusion:** post-processing of the output level/shape does not help on the LB → path closed. Rule adopted: only submit changes whose effect type has already transferred to the LB (model/training changes, TTA), not output post-processing.

### 13.27 Gap analysis vs 8th place (15.74) and current bet
| | ours (15.21) | 8th | Δ → Combined |
|---|---|---|---|
| SI-SDR | 13.18 | 13.21 | ≈ 0 |
| ΔWER syn | 5.68 | 6.92 | −0.40 |
| ΔWER nat | 0.30 | 0.46 | −0.11 |
The whole gap is **ASR on synthetic clips**, not noise removal. Running: **ASR-aware fine-tune of D** (Kaggle 2×T4, λ = 0.5 and 1.5, 1 epoch annealed, last checkpoint kept, reference D row with identical evaluation). Rationale: on the baseline the ASR-feature loss transferred to the LB (+0.84, mostly ΔWER); on MP-SENet λ=5 dominated the loss (scale ~1.2 vs ~0.5), so small λ.

## 15. Research notes — open hypotheses (16 Oct)
1. **Local→LB transfer is partial** because the local synthetic held-out uses the validation noise library + validation speakers, while test synthetic noise is library-like but different (and SNR/speaker mix may differ). Gains that reduce generic artefacts (TTA) transfer partly; gains tied to the local distribution (loudness) do not.
2. **Synthetic ΔWER is the bottleneck** (ours 5.7%, 8th 6.9%, leaders 10–18%). Likely cause: speech under loud events is partly suppressed or distorted → missing/garbled words. Candidates: ASR-aware loss (running), less aggressive suppression inside events, models that restore rather than only mask.
3. **Leader (26.5; SI-SDR 20.3, ΔWER syn 18%)** implies a different method class or training data closely matching the test noise source (e.g. very large Freesound-type corpora such as FSD50K/DNS, larger separation models, or generative restoration).
4. Natural ΔWER is small for every team (0.1–0.5%) → protect, don't chase.

### 13.28 Diagnostic: test vs validation (D, win8, 800 test / 300+300 validation clips) — KEY FINDING
| stat (p10 / p50 / p90) | TEST | VAL synthetic | VAL natural |
|---|---|---|---|
| clip duration (s) | 8.6 / **11.9** / 17.3 | 3.2 / 4.4 / 7.5 | 2.9 / 4.4 / 8.6 |
| noise occupancy (frames where removed noise > kept speech) | 0.00 / **0.10** / 0.38 | 0.17 / 0.39 / 0.60 | 0 / 0 / 0.03 |
| clearly audible noise share | 0.00 / 0.04 / 0.32 | 0.02 / 0.32 / 0.54 | 0 / 0 / 0 |
| est. SNR (dB) | −8.0 / 8.2 / 42.1 | −8.6 / −3.2 / 12.9 | 20.0 / 40.4 / 46.2 |
| clips with est. SNR < 0 dB | 30% | 66% | 0% |
**Test clips are ~3× longer and mostly noise-free.** All tuning (windows, TTA) was done on 4 s clips. On test, artefacts in the long clean stretches dominate both SI-SDR and ASR → explains the ~5 dB local→LB SI-SDR gap and why leaders reach clean-level synthetic ΔWER (~17–18%) at similar SI-SDR. Next: (a) model-driven residual gating — keep the input where the removed noise ≈ 0 (oracle with GT timestamps as upper bound); (b) test-like long-clip evaluation (1 noisy + 2 clean held-out clips concatenated, ~13 s) for window size 4/8/12 s and gating.

### 13.29 Time-reversal TTA (local, D win8, held-out + 600 natural) — rejected
Equal compute: fwd×2 19.20 vs fwd+rev 18.66; fwd×4 19.64 vs fwd×2+rev×2 18.97. A reversed view is a poor view for a model trained on forward speech (fwd+rev ≈ single pass) → only time-shift TTA is useful.
### 13.30 ASR-aware fine-tune of D (Kaggle 2×T4, 1 epoch, LR 3e-5 annealed, last checkpoint)
Reference D (held-out + ALL natural, win4): SI-SDR 17.25, ΔWER syn 4.18, nat −0.05 → 18.56. λ=0.5: 16.86 / 4.53 / −0.02 → 18.29. λ=1.5: 16.61 / 4.80 / +0.01 → 18.16. The ASR loss raises synthetic ΔWER (+0.35/+0.62) but costs about as much SI-SDR locally; LB effect uncertain (SI-SDR gains have transferred ~30%, ΔWER-syn gains ~70% in the TTA submissions).

### 13.31 Residual gating (local, D win8+TTA2, held-out 228 + 600 natural)
current 17.92 / 4.22 / −0.11 → 19.20 · **ORACLE GT-timestamp gating 18.98 / 4.18 / 0.00 → 20.32 (+1.12)** · **residual gate T=−25 dB 18.45 / 3.68 / −0.05 → 19.60 (+0.40)** · T=−20 19.39 · T=−15 18.89 · T=−10 18.42. Artefacts in noise-free regions are real (+1 dB SI-SDR with oracle); model-driven gating recovers part of it; lower thresholds better (try −30/−35). Risk: LB natural ΔWER (+0.30) may shrink because gated natural clips stay closer to the input.

### 13.32 LB: TTA×8 + residual gate T=−25 dB → 14.37 (worse)
SI-SDR 12.51 (TTA×8: 13.18), ΔWER 1.86 (syn 5.26 / nat 0.26). Local gating gain (+0.53 dB SI-SDR) reversed on test (−0.67 dB). Most likely the TEST synthetic clips contain noise outside the annotated "event" regions (e.g. a continuous background layer) and/or are built differently from the validation synthetic clips, so keeping the input where D removes little re-inserts noise. Gating path closed; TTA×8 (15.21) stays the selected submission. This also supports the view that the ~5 dB local→LB SI-SDR gap comes from a different synthetic recipe on test.

### 13.33 Long-clip test (test-like 1 noisy + 2 clean, ~13 s) — stopped
Ran 6.9 h on T4 without finishing (12 s windows likely near T4 memory limit → very slow); stopped after the gating LB result made it moot.

## 16. Fresh synthesis (16 Oct evening) — what the evidence says about the TEST synthetic recipe
1. **Test clips are long (median 11.9 s vs 4.4 s) and mostly free of loud events** (noise occupancy median 10% vs 39%).
2. **Keeping the input outside model-detected noise HURT on test** (gate: SI-SDR −0.67 dB) although it helped on validation (+0.53) → in TEST synthetic clips the "quiet" stretches are **not clean**: most likely a continuous low-level background layer is present in the noisy input but not in the reference. Our simulation never had this (residual outside events = 0).
3. Our training mixtures are short (≤3 s crops), event-dense (~40% occupancy) and background-free; the test is long, event-sparse and (likely) has a continuous background. This mismatch explains the persistent ~5 dB local→LB SI-SDR gap and why tuning on validation-like data transfers poorly.
4. Historical LB gains came mainly from **training data realism** (+4 from realistic sim, +0.5 more data/longer segments) and model class; inference tricks gave +0.4 (TTA) and post-processing hurt.
**→ Most promising remaining lever: fine-tune D on TEST-LIKE mixtures** — long (6–10 s) clean speech, 1–2 events (low occupancy), plus a continuous background layer at moderate SNR — with a budget-sized LR schedule, then TTA×8.

### 13.34 Two final bets prepared
- **E12 (RunPod):** fine-tune D on test-like mixtures (long 6–10 s, sparse events, continuous background), schedule sized from a measured speed probe; two LB shots (E12 TTA×4; E12+D TTA×2 each).
- **Cross-architecture ensemble (Kaggle, free):** D + ClearerVoice MossFormerGAN_SE_16K / FRCRN_SE_16K (Apache-2.0, zero-shot). Rationale: model-class changes gave the largest LB jumps, and averaging different views (TTA) transferred; a different architecture decorrelates artefacts more than time shifts.

### 13.35 Cross-architecture ensembles (local, held-out 228 + 600 natural) — rejected
D win8+TTA2 19.20 · MossFormerGAN_SE_16K 14.97 (SI-SDR 14.30) · FRCRN_SE_16K 14.88 · MossFormer2_SE_48K 14.69 (ΔWER syn −0.39) · D+FRCRN 18.53 · D+MossFormerGAN 17.62 · D+MossFormer2 18.28 · D+both 17.70. DeepFilterNet3 failed to install on Kaggle (no wheel). Large generic pretrained SE models are 3–5 dB behind our in-domain fine-tuned 2.3M-param D and hurt ASR on Indic speech; averaging with a much weaker model drags D down. **Lesson: our advantage is in-domain training, not model size** → remaining lever is training D on test-like data (E12 / E12-lite).

### 13.36 Literature check (16 Oct night) — confirms direction, no shortcut
- A 2025 systematic study found that SE pre-processing can *degrade* modern ASR, because modern ASR models are internally noise-robust and SE can remove acoustic cues → explains why large generic SE models (MossFormer2/FRCRN/MossFormerGAN) hurt SraVaani on our data and why natural-clip ΔWER is small for every team.
- Iwamoto et al. (Interspeech 2022) "How bad are artifacts?": the artefact component of SE errors drives ASR degradation; observation adding trades SDR for ASR (our blend tests confirm the trade is negative under this metric). Later work (ICASSP 2025) suppresses enhancement distortion during training → fix artefacts at training time, not by post-processing.
- URGENT-challenge lessons ("Less is more: data curation matters in scaling SE", 2025): matched, curated data beats more data → supports E12 (test-like mixtures).
- Searched for public solutions of top teams: none published.

### 13.37 Status at the end of 16 Oct
- Best LB: **15.21** (D win8 + shift-TTA×8), rank 10; selected on Codabench.
- Running: **E12-lite on Kaggle 2×T4** (free): test-like mixtures (6–10 s clean speech, continuous background at SNR 5–30 dB in 85% of clips, 1–3 events), 2 LR variants (7e-5, 3e-5), 10k steps each, then test-like long-clip evaluation vs D.
- Deadline: 17 Oct 2026, 12:00 IST.

### 13.38 Full experiment ledger since E10-R (LB-verified effects only)
| change | local Δ | LB Δ |
|---|---|---|
| B+D win4 (E10-R) | — | 14.81 |
| D win8 + shift-TTA×4 | +1.1 | **+0.37** (15.18) |
| shift-TTA×8 | +0.18 | +0.03 (15.21) |
| output peak 0.95 | +0.09 | −0.14 (15.07) |
| residual gate T=−25 dB | +0.40 | −0.84 (14.37) |
Rejected locally (not submitted): Vaani-noise bootstrapping, ASR-feature loss λ 0.5/1.5, reverse TTA, high-pass/soft-limiter, observation adding, cross-architecture ensembles (MossFormerGAN, FRCRN, MossFormer2).

## 17. Notebook index (all notebooks of the final phase, with results)
| notebook | where | what it tested | result |
|---|---|---|---|
| `indoml_t2_submit_e9_kaggle.ipynb` | Kaggle | A / B / avg(A,B) selection + test | avg(A,B) local 18.25 → **LB 14.53** |
| `indoml_t2_e10_fullmusan_kaggle.ipynb` | Kaggle | full MUSAN + 2 s segments | prepared, not run (moved to RunPod) |
| `indoml_t2_e10r_runpod.ipynb` | RunPod A100 | 40k mixtures, bf16 bs4×3 s, warm start B; annealing | held-out 17.00 → 17.26 (anneal); B+D local 18.55 → **LB 14.81** |
| `indoml_t2_e11_vaaniboot_smoke_kaggle.ipynb` | Kaggle | Vaani-noise bootstrapping, noise-match diagnostic, quick wins | Vaani FT 18.25 < 18.44 (rejected); test noise library-like; win8+TTA2 19.22 |
| `indoml_t2_e11r_runpod.ipynb` | RunPod | Vaani-noise at scale | cancelled after smoke (saved money) |
| `indoml_t2_quickwin_tta_runpod.ipynb` | RunPod A100 | {D, B+D} × {TTA×2, TTA×4}, then TTA×8 | D TTA4 19.65, TTA8 19.83 → **LB 15.18 / 15.21** |
| `indoml_t2_tta8_backup_kaggle.ipynb` | Kaggle 2×T4 | TTA×8 backup for test | not needed |
| `indoml_t2_asr_level_kaggle.ipynb` | Kaggle | output level / blend (round 1) | peak 0.95 best local (19.29) |
| `indoml_t2_tta8_peak_kaggle.ipynb` | Kaggle | peak 0.95 on TTA×8 outputs + ASR | **LB 15.07** (worse) |
| `indoml_t2_asr_tweaks2_kaggle.ipynb` | Kaggle | HPF 80 Hz, soft-limiter (round 2) | all worse (17.41 / 15.13 / 13.22 / 14.94) |
| `indoml_t2_tta8_apply_variant_kaggle.ipynb` | Kaggle | apply a post-processing variant to TTA×8 | not used (post-processing closed) |
| `indoml_t2_asrft_kaggle.ipynb` | Kaggle 2×T4 | ASR-feature loss λ 0.5 / 1.5 | 18.29 / 18.16 vs D 18.56 (rejected) |
| `indoml_t2_ens_asr_kaggle.ipynb` | Kaggle | D + ASR-model ensembles | not run (ASR-ft rejected) |
| `indoml_t2_reverse_tta_kaggle.ipynb` | Kaggle | time-reversal TTA | worse at equal compute (18.66 vs 19.20; 18.97 vs 19.64) |
| `indoml_t2_test_vs_val_diag_kaggle.ipynb` | Kaggle | test vs validation statistics | **test clips 11.9 s, noise occupancy 10%** (val 4.4 s, 39%) |
| `indoml_t2_resid_gate_kaggle.ipynb` | Kaggle | oracle + residual gating | oracle 20.32, gate T=−25 19.60 (local) |
| `indoml_t2_tta8_gate_kaggle.ipynb` | Kaggle | residual gate on TTA×8 outputs + ASR | **LB 14.37** (worse) |
| `indoml_t2_longclip_kaggle.ipynb` | Kaggle | test-like long clips: window size + gate | stuck 6.9 h on T4 (12 s windows), stopped |
| `indoml_t2_crossarch_kaggle.ipynb` | Kaggle | D + MossFormerGAN / FRCRN | MFG 14.97, FRCRN 14.88, D+FRCRN 18.53, D+MFG 17.62, D+both 17.70 (rejected) |
| `indoml_t2_crossarch2_kaggle.ipynb` | Kaggle | D + MossFormer2_48K / DeepFilterNet3 | MF2 14.69, D+MF2 18.28; DeepFilterNet install failed (rejected) |
| `indoml_t2_e12lite_kaggle.ipynb` | Kaggle 2×T4 | test-like mixtures (background + sparse events), LR 7e-5 / 3e-5 | **running** |
| `indoml_t2_e12_testlike_runpod.ipynb` | RunPod A100 | full E12 + background check + test-like eval + 2 LB shots | **pending (pod availability)** |
Reference for all local numbers: D win8 + TTA×2 on held-out 228 synthetic + 600 natural = 19.20 (D win4 = 18.44).

## 18. Late insight (16 Oct night): "noise EVENT removal" — the reference probably keeps Vaani-native noise
Evidence assembled: test clips' median est. SNR is 8 dB while validation natural clips sit at ~40 dB → test natural clips are noisy; validation synthetic has zero residual outside events → the reference is the base clip itself, not clean speech; leaders share a synthetic ΔWER ceiling (~17–18) ≈ output equal to the reference; test residual noise resembles Freesound-type libraries. Consistent reading: **test synthetic = noisy natural Vaani clip + added library events, reference = the natural clip (its own noise kept)**. D strips Vaani-native noise too → the persistent ~5 dB local→LB SI-SDR gap. The baseline's clean bank (gold gaps) and all our simulations trained "remove everything".
**E13 recipe (`indoml_t2_e13_eventonly_runpod.ipynb`):** target = natural Vaani clip as-is (validation natural minus 600 reserved + ~9k Vaani training clips), input = target + 1–3 library events (ESC-50 folds 1–4, MUSAN subset, validation library, augmented copies without leakage), fine-tune D with a speed-probe-sized schedule, evaluate on the 600 reserved natural clips + unseen events (ESC fold 5 + held MUSAN), then TTA×4 test submission (+ optional E13+D ensemble). This supersedes E12 on RunPod (E12 teaches background removal, the opposite direction); E12-lite on Kaggle continues as a separate bet.

### 18.1 Contradiction found (stress-thinking, 16 Oct night)
- Gate on LB (keep input where D removes little) **lowered** test SI-SDR → in TEST synthetic clips something outside the events is absent from the reference → removing it is right (**E12 direction**).
- E12-lite (learn to remove background) **lowered validation** held-out SI-SDR by 2.7–3.5 dB → validation references contain native ambience → removing it is wrong there (**E13 direction**).
- Both hold if the **test synthetic recipe adds a background layer that validation synthetic lacks**. Then E12-lite can lose on validation yet win on test, and E13 may not help on test.
- The free hypothesis check uses validation natural clips, which are much quieter than test natural clips (est. SNR ~40 vs ~8 dB) → it can give a false "no".
- **Decisive, free test: submit E12-lite to the LB** (`indoml_t2_e12lite_submit_kaggle.ipynb`, TTA×4). LB up → E12 direction; LB down → E13 direction (RunPod).

### 18.2 E12-lite results (Kaggle 2×T4) — E12 direction rejected
Validation held-out (win4): D 17.25 · lr7e-5 13.78 · lr3e-5 14.57 (combined 18.56 / 14.82 / 15.57). Test-like long clips WITH added background (unseen ESC fold 5), win8+TTA2: **D 19.98** · lr7e-5 18.99 · lr3e-5 19.54. E12-lite loses everywhere → not submitted.
**Key inference:** with an added background layer D still scores ~20 dB, so "test has extra background" cannot explain the LB SI-SDR of ~13. The only reading that predicts ~13 is the E13 one: noisy Vaani base clips whose native noise stays in the reference, which D strips. E13 eval updated to use 400 reserved NOISY Vaani training clips + unseen library events; if D lands near 13 dB there, the set mimics the test and E13's gain on it should transfer.

### 18.3 E13 on RunPod A100 (17 Oct early morning)
- Free hypothesis check (Kaggle): D on validation references SI-SDR(ref, D(ref)) ≈ 46 dB; on validation natural clips p25 30 / p50 39 dB; natural clip + unseen library events (target = natural clip): **D 15.41 dB**, ΔWER(D) +5.14 vs ceiling +11.99 (D captures ~43% of the ceiling, like the LB ratio 5.7 vs ~17–18).
- E13 data: 10,206 natural targets (1,206 validation natural + 9,000 Vaani training clips; 600 + 400 reserved for eval), 30,000 mixtures (input SI-SDR −3.19 dB), events: train 8,680 / unseen eval 756.
- Training: bs 2 × 6 s, bf16, speed probe 5,009 steps/h → 13,775 steps (ran faster, 2.2 h), LR 7e-5 annealed. Validation synthetic held-out: 17.26 → 14.76 (validation events appear Vaani-like, so an event-only model keeps part of them).
- Test-like evaluations (win8 + TTA2):
| set | D | E13 |
|---|---|---|
| validation natural + library events (SI-SDR / ΔWER) | 15.41 / +5.13 | **15.82** / +4.88 |
| 12-s concatenated natural clips + events | 16.06 (win16 15.95) | **17.50** (win16 17.48) |
| **noisy Vaani clips + unseen events (mimics test: D 12.66 vs LB 13.18)** | 12.66 | **12.95** (+0.28) |
| avg(D, E13) on the noisy-Vaani set | | 12.84 |
- ASR batched vs one-by-one: identical WER (0.2672) → batch 16 kept.
- Test run: E13, win8, shift-TTA×4. Expected LB ≈ 15.4–15.6 (from the +0.28 dB on the test-mimicking set). LB at the time: 8th 16.61, 9th 15.81, 10th 15.31; ours 15.21 (rank 11).
- First local evaluation that reproduces the LB for D (noisy Vaani + library events, target = Vaani clip as-is).

## 19. Final outcome (17 Oct 2026, deadline 12:00 IST)
- E13 test run on RunPod finished enhancement (3,178 WAVs, E13 + win8 + shift-TTA×4, 68 min). The script stopped at a stale print line (`BEST` undefined); ASR + ZIP were re-run in a separate cell. `transcripts.jsonl` and `e13_D.pt` were downloaded, but **the RunPod credit ran out during the ZIP download** → pod terminated, container disk erased, `submission_e13.zip` lost. Cause: overnight idle time + longer-than-estimated eval/ASR were not budgeted (estimate said ~$4.5 would remain).
- Kaggle rescue (`indoml_t2_e13_submit_kaggle.ipynb`, E13 + win8 + TTA×2, 2×T4, `e13_D.pt` uploaded as dataset `e13-model`): the first version had a duplicated `WORKER = r'''` wrapper (bug) → fixed, but by then it was ~11:00 and the run (~75–85 min) could not finish before the deadline.
- **Final selected submission: 15.21 (D win8 + shift-TTA×8), rank 11.** Final LB snapshot (17 Oct morning): 1 sajidjavid 26.49 · 2 liyingfa 20.80 · 3 syehsyoh 20.03 · 4 Hydra 19.90 · 5 marchmello01 19.85 · 6 rounders 19.53 · 7 Raj Patil 17.23 · 8 asymptoticchase 16.61 · 9 sharadhnaidu 15.81 · 10 ayushm 15.31 · 11 avics55 15.21.
- E13 expectation (not verified on LB): +0.28 dB on the noisy-Vaani set that reproduced D's LB SI-SDR (12.66 vs 13.18) → ~15.4–15.6.

### 19.1 Final lessons (operational)
1. Budget idle time and add ≥30% margin to every cloud estimate; download each artefact the moment it exists; never let a pod run near zero credit.
2. Never edit notebooks under deadline pressure without a dry run; one stray wrapper cost the last hour.
3. Keep the final submission path short and rehearsed before the last day (one tested notebook per platform).
4. Validate that a local evaluation reproduces the LB for the current best model *before* investing in new training — the noisy-Vaani + library-events set was the first that did, and it came too late.

### 19.2 Assets kept
`e13_D.pt`, `all_models_bundle.zip` (A, B, C, D), all notebooks (section 17), this MD with every result and the research narrative (sections 13–19).
