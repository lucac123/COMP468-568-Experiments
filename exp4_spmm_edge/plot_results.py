#!/usr/bin/env python3

import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def read_results(csv_path):
    rows = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "D",
            "baseline_sddmm_ms",
            "baseline_spmm_ms",
            "optimized_sddmm_ms",
            "optimized_spmm_ms",
            "sddmm_speedup",
            "spmm_speedup",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "D": int(row["D"]),
                    "baseline_sddmm_ms": float(row["baseline_sddmm_ms"]),
                    "baseline_spmm_ms": float(row["baseline_spmm_ms"]),
                    "optimized_sddmm_ms": float(row["optimized_sddmm_ms"]),
                    "optimized_spmm_ms": float(row["optimized_spmm_ms"]),
                    "sddmm_speedup": float(row["sddmm_speedup"]),
                    "spmm_speedup": float(row["spmm_speedup"]),
                }
            )

    rows.sort(key=lambda r: r["D"])
    return rows


def plot_runtime(x, baseline, optimized, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, baseline, marker="o", label="Baseline")
    plt.plot(x, optimized, marker="o", label="Optimized")
    plt.xlabel("Embedding dimension D")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_speedup(x, speedup, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, speedup, marker="o")
    plt.xlabel("Embedding dimension D")
    plt.ylabel("Speedup (baseline / optimized)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    if len(sys.argv) > 2:
        print(f"Usage: {Path(sys.argv[0]).name} [results.csv]")
        sys.exit(1)

    csv_path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("spmm_results.csv")

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    rows = read_results(csv_path)

    x = [r["D"] for r in rows]

    baseline_sddmm = [r["baseline_sddmm_ms"] for r in rows]
    optimized_sddmm = [r["optimized_sddmm_ms"] for r in rows]
    baseline_spmm = [r["baseline_spmm_ms"] for r in rows]
    optimized_spmm = [r["optimized_spmm_ms"] for r in rows]

    sddmm_speedup = [r["sddmm_speedup"] for r in rows]
    spmm_speedup = [r["spmm_speedup"] for r in rows]

    plot_runtime(
        x,
        baseline_sddmm,
        optimized_sddmm,
        ylabel="Runtime (ms)",
        title="SDDMM Runtime vs Embedding Dimension",
        output_path="sddmm_runtime_vs_d.png",
    )

    plot_runtime(
        x,
        baseline_spmm,
        optimized_spmm,
        ylabel="Runtime (ms)",
        title="SpMM Runtime vs Embedding Dimension",
        output_path="spmm_runtime_vs_d.png",
    )

    plot_speedup(
        x,
        sddmm_speedup,
        title="SDDMM Speedup vs Embedding Dimension",
        output_path="sddmm_speedup_vs_d.png",
    )

    plot_speedup(
        x,
        spmm_speedup,
        title="SpMM Speedup vs Embedding Dimension",
        output_path="spmm_speedup_vs_d.png",
    )

    print("Generated plots:")
    print("  sddmm_runtime_vs_d.png")
    print("  spmm_runtime_vs_d.png")
    print("  sddmm_speedup_vs_d.png")
    print("  spmm_speedup_vs_d.png")


if __name__ == "__main__":
    main()
