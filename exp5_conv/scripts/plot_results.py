#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(csv_path):
    rows = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "impl": row["impl"],
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                    "cin": int(row["cin"]),
                    "cout": int(row["cout"]),
                    "k": int(row["k"]),
                    "stride": int(row["stride"]),
                    "padding": int(row["padding"]),
                    "time_ms": float(row["time_ms"]),
                    "gflops": float(row["gflops"]),
                }
            )

    return rows


def group_by_cout(rows):
    grouped = {}
    for row in rows:
        cout = row["cout"]
        impl = row["impl"]
        grouped.setdefault(cout, {}).setdefault(impl, []).append(row)

    for cout in grouped:
        for impl in grouped[cout]:
            grouped[cout][impl].sort(key=lambda r: r["height"])

    return grouped


def make_runtime_plot(grouped, output_dir):
    for cout, impl_data in grouped.items():
        plt.figure(figsize=(8, 5))

        for impl in ["naive", "tiled"]:
            if impl not in impl_data:
                continue
            x = [r["height"] for r in impl_data[impl]]
            y = [r["time_ms"] for r in impl_data[impl]]
            plt.plot(x, y, marker="o", label=impl)

        plt.xlabel("Spatial size (H = W)")
        plt.ylabel("Runtime (ms)")
        plt.title(f"Convolution Runtime vs Spatial Size (Cout={cout})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"runtime_cout_{cout}.png", dpi=300)
        plt.close()


def make_gflops_plot(grouped, output_dir):
    for cout, impl_data in grouped.items():
        plt.figure(figsize=(8, 5))

        for impl in ["naive", "tiled"]:
            if impl not in impl_data:
                continue
            x = [r["height"] for r in impl_data[impl]]
            y = [r["gflops"] for r in impl_data[impl]]
            plt.plot(x, y, marker="o", label=impl)

        plt.xlabel("Spatial size (H = W)")
        plt.ylabel("Throughput (GFLOP/s)")
        plt.title(f"Convolution Throughput vs Spatial Size (Cout={cout})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"gflops_cout_{cout}.png", dpi=300)
        plt.close()


def make_speedup_plot(grouped, output_dir):
    plt.figure(figsize=(8, 5))

    for cout, impl_data in sorted(grouped.items()):
        if "naive" not in impl_data or "tiled" not in impl_data:
            continue

        naive_by_h = {r["height"]: r for r in impl_data["naive"]}
        tiled_by_h = {r["height"]: r for r in impl_data["tiled"]}

        heights = sorted(set(naive_by_h) & set(tiled_by_h))
        speedups = [naive_by_h[h]["time_ms"] / tiled_by_h[h]["time_ms"] for h in heights]

        plt.plot(heights, speedups, marker="o", label=f"Cout={cout}")

    plt.xlabel("Spatial size (H = W)")
    plt.ylabel("Speedup (naive / tiled)")
    plt.title("Tiled Speedup vs Spatial Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_vs_spatial.png", dpi=300)
    plt.close()


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {Path(sys.argv[0]).name} <results.csv> [output_dir]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("plots")

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(csv_path)
    grouped = group_by_cout(rows)

    make_runtime_plot(grouped, output_dir)
    make_gflops_plot(grouped, output_dir)
    make_speedup_plot(grouped, output_dir)

    print(f"Plots written to: {output_dir}")


if __name__ == "__main__":
    main()
