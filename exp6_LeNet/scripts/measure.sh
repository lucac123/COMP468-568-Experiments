#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dlenet"
BATCHES=(32 64 128)
ALGOS=(implicit_gemm implicit_precomp gemm winograd)
IMPLS=(baseline fused)
WARMUP=2
TRIALS=5

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_lenet_sweep.csv"
echo "impl,batch,algo,trial,time_ms,gflops" > "$LOG"

for batch in "${BATCHES[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "=== Running impl=$impl batch=$batch algo=$algo ==="
      # Warmup runs (not logged)
      for ((i=0; i<WARMUP; i++)); do
        "$BIN" --batch "$batch" --algo "$algo" --impl "$impl" --no-verify >/dev/null 2>&1 || true
      done
      # Measurement trials
      for ((trial=0; trial<TRIALS; trial++)); do
        out=$("$BIN" --batch "$batch" --algo "$algo" --impl "$impl" --no-verify 2>&1) || {
          echo "  trial $trial FAILED (algo may be unsupported for this shape)"
          echo "$impl,$batch,$algo,$trial,NA,NA" >> "$LOG"
          continue
        }
        # Expected stdout line: "Impl=... Batch=... Algo=... Time(ms)=X.XX GFLOP/s=YYYY.YY"
        time_ms=$(echo "$out" | awk -F'Time\\(ms\\)=' '/Time\(ms\)/{print $2}' | awk '{print $1}')
        gflops=$(echo  "$out" | awk -F'GFLOP/s=' '/GFLOP/{print $2}' | awk '{print $1}')
        echo "  trial $trial: time=${time_ms}ms  gflops=${gflops}"
        echo "$impl,$batch,$algo,$trial,${time_ms:-NA},${gflops:-NA}" >> "$LOG"
      done
    done
  done
done

echo ""
echo "Results stored in $LOG"
echo "Quick summary:"
awk -F',' 'NR>1 && $5!="NA" {key=$1","$2","$3; sum[key]+=$5; n[key]++} END {for (k in sum) printf "  %-30s mean_time=%.3f ms  (n=%d)\n", k, sum[k]/n[k], n[k]}' "$LOG" | sort
