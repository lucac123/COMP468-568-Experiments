#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dmlp"
LAYERS=("512,512,512" "1024,2048,1024" "2048,2048,2048")
BATCHES=(64 128 256 512)
IMPLS=(baseline activation_fused)
ACTIVATION="relu"

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_mlp_sweep.csv"
echo "impl,layers,batch,activation,time_ms,gflops" > "$LOG"

for layers in "${LAYERS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Running $impl layers=$layers batch=$batch"
      # TODO(student): parse stdout from the binary and append to the CSV.
      OUT="$("$BIN" --layers "$layers" --batch "$batch" --activation "$ACTIVATION" --impl "$impl" --no-verify)"
      echo $OUT

      TIME="$(echo $OUT | sed "s/.*Time(ms)=\([0-9.]\+\).*/\1/")"
      GFLOP="$(echo $OUT | sed "s/.*GFLOP\/s=\([0-9.]\+\).*/\1/")"

      echo "Parsed time(ms)=$TIME, GFLOP\/s=$GFLOP"

      layers_clean="$(echo $layers | sed "s/,/_/g")"

      echo "$impl,$layers_clean,$batch,$ACTIVATION,$TIME,$GFLOP" >> "$LOG"

    done
  done
done

echo "Results stored in $LOG"
