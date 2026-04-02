#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_benchmarks.sh
#   ./run_benchmarks.sh 32 64 128 256
#
# If no D values are provided, use a default sweep.
if [ "$#" -gt 0 ]; then
  D_VALUES=("$@")
else
  D_VALUES=(32 64 128 256 512 1024)
fi

OUTPUT_CSV="spmm_results.csv"

# Write CSV header
cat > "$OUTPUT_CSV" <<'EOF'
D,baseline_sddmm_ms,baseline_spmm_ms,optimized_sddmm_ms,optimized_spmm_ms,sddmm_speedup,spmm_speedup
EOF

echo "Building executables..."
make

extract_time() {
  local pattern="$1"
  local text="$2"
  echo "$text" | awk -F': ' -v pat="$pattern" '$0 ~ pat {print $2}' | tail -n 1
}

for D in "${D_VALUES[@]}"; do
  echo "Running benchmarks for D=$D ..."

  baseline_output=$(./spmm_baseline "$D")
  optimized_output=$(./spmm_opt "$D")

  baseline_sddmm=$(extract_time "Baseline SDDMM avg time" "$baseline_output")
  baseline_spmm=$(extract_time "Baseline SpMM  avg time" "$baseline_output")
  optimized_sddmm=$(extract_time "Optimized SDDMM avg time" "$optimized_output")
  optimized_spmm=$(extract_time "Optimized SpMM  avg time" "$optimized_output")

  if [ -z "$baseline_sddmm" ] || [ -z "$baseline_spmm" ] || \
     [ -z "$optimized_sddmm" ] || [ -z "$optimized_spmm" ]; then
    echo "Error: failed to parse timing output for D=$D"
    echo
    echo "Baseline output:"
    echo "$baseline_output"
    echo
    echo "Optimized output:"
    echo "$optimized_output"
    exit 1
  fi

  sddmm_speedup=$(awk -v b="$baseline_sddmm" -v o="$optimized_sddmm" 'BEGIN { printf "%.6f", b / o }')
  spmm_speedup=$(awk -v b="$baseline_spmm" -v o="$optimized_spmm" 'BEGIN { printf "%.6f", b / o }')

  echo "$D,$baseline_sddmm,$baseline_spmm,$optimized_sddmm,$optimized_spmm,$sddmm_speedup,$spmm_speedup" >> "$OUTPUT_CSV"
done

echo "Done. Results written to $OUTPUT_CSV"
