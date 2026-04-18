#!/usr/bin/env bash
set -euo pipefail

# Work from the project root so relative paths (data/cora, bin/dgcn) match
# the ones the user passes when running the binary directly. This makes the
# script behave the same whether invoked as ./scripts/measure.sh from the
# root or ./measure.sh from inside scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BIN="./bin/dgcn"
GRAPHS=("data/cora" "data/citeseer")
HIDDENS=(64 128 256)
IMPLS=(baseline fused)
LAYERS=2
WARMUPS=1   # runs discarded (JIT + cache warm-up)
REPEATS=3   # measurement runs per config (each gets its own CSV row)

mkdir -p data
LOG="data/$(date +%Y%m%d_%H%M%S)_gcn_sweep.csv"
echo "graph,hidden,impl,run,time_ms,edges_per_s" > "$LOG"

# Extract the numeric value following `Key=` in one line of stdout. Uses
# awk with a fixed-string split so parens/slashes in the key need no escaping.
extract_after_eq() {
    # $1 = full output, $2 = key (e.g. "Time(ms)")
    # We look for the literal "<key>=" and take the numeric field after it.
    echo "$1" | awk -v key="$2" '
        {
            for (i = 1; i <= NF; i++) {
                n = length(key) + 1
                if (substr($i, 1, n) == key "=") {
                    print substr($i, n + 1)
                    exit
                }
            }
        }
    '
}

for graph in "${GRAPHS[@]}"; do
  for hidden in "${HIDDENS[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "=== $impl graph=$graph hidden=$hidden ==="

      # Warm-up runs (output discarded). If they fail, skip the whole config
      # since the measurement runs would fail too.
      warmup_ok=1
      for ((w = 0; w < WARMUPS; ++w)); do
        if ! "$BIN" --graph "$graph" --hidden "$hidden" --layers "$LAYERS" --impl "$impl" --no-verify >/dev/null 2>&1; then
          echo "  warmup failed — skipping this config" >&2
          warmup_ok=0
          break
        fi
      done
      [[ $warmup_ok -eq 0 ]] && continue

      # Measurement runs. Each one contributes a CSV row so downstream analysis
      # can compute mean/std/min over repeats.
      for ((r = 1; r <= REPEATS; ++r)); do
        if ! output=$("$BIN" --graph "$graph" --hidden "$hidden" --layers "$LAYERS" --impl "$impl" --no-verify 2>&1); then
          echo "  run $r: FAILED (exit nonzero)" >&2
          continue
        fi
        time_ms=$(extract_after_eq "$output" "Time(ms)")
        edges_s=$(extract_after_eq "$output" "Edges/s")
        if [[ -z "$time_ms" || -z "$edges_s" ]]; then
          echo "  run $r: could not parse output:" >&2
          echo "$output" | sed 's/^/    /' >&2
          continue
        fi
        echo "  run $r: time=${time_ms}ms edges/s=${edges_s}"
        echo "$graph,$hidden,$impl,$r,$time_ms,$edges_s" >> "$LOG"
      done
    done
  done
done

echo "Results stored in $LOG"
