#!/usr/bin/env bash
# Rebuild the final model from the MP-SENet DNS checkpoint (see train/config.yaml). Requires HF_TOKEN (Vaani is gated), the
# validation set and MUSAN noise. Stage hyper-parameters follow the original runs; times are for the hardware listed in README.
set -e
VAL=${VAL_DIR:?set VAL_DIR to the validation folder}
MUSAN=${MUSAN_DIR:?set MUSAN_DIR to the MUSAN noise folder}
mkdir -p work
python train/prepare_data.py --val_dir "$VAL" --musan_dir "$MUSAN" --out work --n_mix 40000 --max_gold 11000 --mode standard
# Stage 1 (from DNS) -> stage 2 -> stage 3 (large-scale) -> stage 4 (annealing = final model). On one A100 the first two stages can reuse the same 40k data.
python train/train.py --val_dir "$VAL" --out work/stage1.pt   --lr 2e-4 --epochs 5 --bs 2 --seg 16000
python train/train.py --val_dir "$VAL" --out work/stage2.pt  --init work/stage1.pt  --lr 1e-4 --epochs 3 --bs 2 --seg 16000
python train/train.py --val_dir "$VAL" --out work/stage3.pt --init work/stage2.pt --lr 1e-4 --epochs 1 --bs 4 --seg 48000 --bf16 1
python train/train.py --val_dir "$VAL" --out checkpoints/D_retrained.pt --init work/stage3.pt --lr 5e-5 --epochs 1 --bs 4 --seg 48000 --bf16 1 --keep last
# Optional event-only model (E13): target = natural Vaani clip as-is, input = + library events
#   python train/prepare_data.py --val_dir "$VAL" --musan_dir "$MUSAN" --out work --n_mix 30000 --mode e13
#   python train/train.py --val_dir "$VAL" --out checkpoints/e13_D.pt --init checkpoints/D_e10r_anneal.pt --lr 7e-5 --bs 2 --seg 96000 --nsamp 160000 --bf16 1 --max_steps 13775 --keep last
