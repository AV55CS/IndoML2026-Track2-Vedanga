#!/usr/bin/env bash
# Clone third-party code (MP-SENet, MIT; organisers' baseline for Vaani loading + mandated ASR wrapper) and install deps.
set -e
mkdir -p third_party
[ -d third_party/MP-SENet ] || git clone --depth 1 https://github.com/yxlu-0102/MP-SENet third_party/MP-SENet
[ -d third_party/noise-event-detection-and-removal ] || git clone --depth 1 https://github.com/nagshubhadip/noise-event-detection-and-removal third_party/noise-event-detection-and-removal
pip install -r "${1:-requirements_inference.txt}"
# baseline's own dependencies (mandated-ASR wrapper / Vaani loader), if it ships a requirements file
BASE=third_party/noise-event-detection-and-removal
REQ=$(find "$BASE" -maxdepth 6 -name "requirements*.txt" | head -n 1)
[ -n "$REQ" ] && pip install -r "$REQ" || true
echo "setup done"
