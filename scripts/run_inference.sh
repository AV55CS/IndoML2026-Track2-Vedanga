#!/usr/bin/env bash
# Reproduce the Phase-2 submission (model D + 8 s windows + shift-TTA x8 + SraVaani transcripts).
#   HF_TOKEN must be set (SraVaani-1.0 is gated on Hugging Face).
#   bash scripts/run_inference.sh /path/to/test/audio
set -e
TEST_DIR=${1:?usage: run_inference.sh <test_audio_dir> [out_dir] [zip]}
OUT_DIR=${2:-outputs/enhanced}
ZIP=${3:-submission_track2.zip}
python inference/infer.py --input_dir "$TEST_DIR" --output_dir "$OUT_DIR" --ckpt checkpoints/D_e10r_anneal.pt --shifts 8 --win 8 --hop 6 --zip "$ZIP"
