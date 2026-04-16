#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dlenet"
BATCHES=(32 64 128)
ALGOS=(implicit_gemm implicit_precomp)
IMPLS=(baseline fused)

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_lenet_sweep.csv"
echo "impl,batch,algo,time_ms,gflops" > "$LOG"

for batch in "${BATCHES[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Running $impl batch=$batch algo=$algo"
      # TODO(student): parse stdout and append to CSV (e.g., grep GFLOP/s, awk fields).
      "$BIN" --batch "$batch" --algo "$algo" --impl "$impl" --no-verify
    done
  done
done

echo "Results stored in $LOG"
