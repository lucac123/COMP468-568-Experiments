#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_and_average_results(csv_dir):
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    grouped = defaultdict(lambda: {"time_ms_sum": 0.0, "gflops_sum": 0.0, "count": 0})

    for csv_path in csv_files:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row["impl"],
                    int(row["height"]),
                    int(row["width"]),
                    int(row["cin"]),
                    int(row["cout"]),
                    int(row["k"]),
                    int(row["stride"]),
                    int(row["padding"]),
                )

                grouped[key]["time_ms_sum"] += float(row["time_ms"])
                grouped[key]["gflops_sum"] += float(row["gflops"])
                grouped[key]["count"] += 1

    rows = []
    for key, values in grouped.items():
        impl, height, width, cin, cout, k, stride, padding = key
        count = values["count"]
        rows.append(
            {
                "impl": impl,
                "height": height,
                "width": width,
                "cin": cin,
                "cout": cout,
                "k": k,
                "stride": stride,
                "padding": padding,
                "time_ms": values["time_ms_sum"] / count,
                "gflops": values["gflops_sum"] / count,
                "count": count,
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


def write_averaged_csv(rows, output_dir):
    out_csv = output_dir / "averaged_results.csv"
    rows = sorted(
        rows,
        key=lambda r: (
            r["cout"],
            r["impl"],
            r["height"],
            r["width"],
            r["cin"],
            r["k"],
            r["stride"],
            r["padding"],
        ),
    )

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "impl",
                "height",
                "width",
                "cin",
                "cout",
                "k",
                "stride",
                "padding",
                "time_ms",
                "gflops",
                "samples_averaged",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["impl"],
                    r["height"],
                    r["width"],
                    r["cin"],
                    r["cout"],
                    r["k"],
                    r["stride"],
                    r["padding"],
                    r["time_ms"],
                    r["gflops"],
                    r["count"],
                ]
            )

    return out_csv


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {Path(sys.argv[0]).name} <csv_folder> [output_dir]")
        sys.exit(1)

    csv_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("plots")

    if not csv_dir.exists() or not csv_dir.is_dir():
        print(f"Error: CSV folder not found or is not a directory: {csv_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_and_average_results(csv_dir)
    grouped = group_by_cout(rows)

    averaged_csv = write_averaged_csv(rows, output_dir)
    make_runtime_plot(grouped, output_dir)
    make_gflops_plot(grouped, output_dir)
    make_speedup_plot(grouped, output_dir)

    print(f"Averaged CSV written to: {averaged_csv}")
    print(f"Plots written to: {output_dir}")


if __name__ == "__main__":
    main()
