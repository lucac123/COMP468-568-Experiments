#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dconv2d"
SPATIAL=(256 512 1024)
FILTERS=(32 64 128)
IMPLS=(tiled naive)
CIN=32
K=3
STRIDE=1
PADDING=1

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_conv_sweep.csv"
echo "impl,height,width,cin,cout,k,stride,padding,time_ms,gflops" > "$LOG"

for h in "${SPATIAL[@]}"; do
  for cout in "${FILTERS[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Running $impl H=$h Cout=$cout"
      # TODO(student): parse stdout from the binary and append to the CSV.
      "$BIN" --height "$h" --width "$h" --channels "$CIN" --filters "$cout" \
        --ksize "$K" --stride "$STRIDE" --padding "$PADDING" --impl "$impl" --no-verify
    done
  done
done

echo "Results stored in $LOG"
