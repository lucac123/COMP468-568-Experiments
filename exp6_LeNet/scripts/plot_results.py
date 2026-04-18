#!/usr/bin/env python3
"""Generate plots 1, 2, and 3 from the measure.sh CSV output.

Usage:
    python3 plot_results.py ../data/20260418_120000_lenet_sweep.csv

Plot 1: Latency vs batch size, grouped by (impl, algo).
Plot 2: Throughput (GFLOP/s) vs batch size, grouped by (impl, algo),
        with a 70%-of-peak reference line per the assignment rubric.
Plot 3: Baseline-vs-fused speedup ratio per algo, faceted by batch size.
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC",
           "#CA9161", "#949494", "#ECE133", "#56B4E9"]

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "legend.frameon": False,
})


def load_sweep_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", ""])
    initial_rows = len(df)
    df = df.dropna(subset=["time_ms", "gflops"])
    df["time_ms"] = pd.to_numeric(df["time_ms"])
    df["gflops"] = pd.to_numeric(df["gflops"])
    dropped = initial_rows - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with NA (unsupported algo/shape combinations)")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["impl", "batch", "algo"]).agg(
        time_ms_mean=("time_ms", "mean"),
        time_ms_std=("time_ms", "std"),
        gflops_mean=("gflops", "mean"),
        gflops_std=("gflops", "std"),
        n=("trial", "count"),
    ).reset_index()
    return agg


def plot_1_latency(agg: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots()
    batches = sorted(agg["batch"].unique())
    combos = agg[["impl", "algo"]].drop_duplicates().values.tolist()
    combos.sort(key=lambda x: (x[0], x[1]))

    for i, (impl, algo) in enumerate(combos):
        sub = agg[(agg["impl"] == impl) & (agg["algo"] == algo)].sort_values("batch")
        if sub.empty:
            continue
        color = PALETTE[i % len(PALETTE)]
        linestyle = "-" if impl == "baseline" else "--"
        marker = "o" if impl == "baseline" else "s"
        ax.errorbar(
            sub["batch"], sub["time_ms_mean"],
            yerr=sub["time_ms_std"],
            label=f"{impl} / {algo}",
            color=color, linestyle=linestyle, marker=marker,
            capsize=3, linewidth=1.5, markersize=5,
        )

    ax.set_xlabel("batch size")
    ax.set_ylabel("time (ms)")
    ax.set_xticks(batches)
    ax.set_title("LeNet forward latency vs batch size")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_3_speedup(agg: pd.DataFrame, out_path: pathlib.Path) -> None:
    batches = sorted(agg["batch"].unique())
    algos = sorted(agg["algo"].unique())

    fig, axes = plt.subplots(1, len(batches), figsize=(4 * len(batches), 5), sharey=True)
    if len(batches) == 1:
        axes = [axes]

    for ax, batch in zip(axes, batches):
        sub = agg[agg["batch"] == batch]
        ratios, labels = [], []
        for algo in algos:
            baseline = sub[(sub["impl"] == "baseline") & (sub["algo"] == algo)]
            fused    = sub[(sub["impl"] == "fused")    & (sub["algo"] == algo)]
            if baseline.empty or fused.empty:
                continue
            ratio = fused["time_ms_mean"].values[0] / baseline["time_ms_mean"].values[0]
            ratios.append(ratio)
            labels.append(algo)

        x = np.arange(len(labels))
        colors = [PALETTE[1] if r > 1.0 else PALETTE[2] for r in ratios]
        bars = ax.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)

        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{ratio:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"batch = {batch}")
        if ax is axes[0]:
            ax.set_ylabel("fused time / baseline time")

    fig.suptitle("Fusion speedup (values < 1.0 mean fused is faster)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_2_throughput(agg: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Plot 2: GFLOP/s vs batch size, one line per (impl, algo).
    Adds a dotted reference line at 70% of the max IMPLICIT_GEMM throughput
    observed — the soft target from the assignment rubric.
    """
    fig, ax = plt.subplots()
    batches = sorted(agg["batch"].unique())
    combos = agg[["impl", "algo"]].drop_duplicates().values.tolist()
    combos.sort(key=lambda x: (x[0], x[1]))

    for i, (impl, algo) in enumerate(combos):
        sub = agg[(agg["impl"] == impl) & (agg["algo"] == algo)].sort_values("batch")
        if sub.empty:
            continue
        color = PALETTE[i % len(PALETTE)]
        linestyle = "-" if impl == "baseline" else "--"
        marker = "o" if impl == "baseline" else "s"
        ax.errorbar(
            sub["batch"], sub["gflops_mean"],
            yerr=sub["gflops_std"],
            label=f"{impl} / {algo}",
            color=color, linestyle=linestyle, marker=marker,
            capsize=3, linewidth=1.5, markersize=5,
        )

    # Reference line at 70% of max IMPLICIT_GEMM throughput (rubric soft target)
    implicit_peak = agg[agg["algo"] == "implicit_gemm"]["gflops_mean"].max()
    if pd.notna(implicit_peak):
        target = 0.7 * implicit_peak
        ax.axhline(target, color="black", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(batches[-1], target, f"  70% of peak ({target:.0f})",
                va="center", ha="left", fontsize=8, color="black", alpha=0.7)

    ax.set_xlabel("batch size")
    ax.set_ylabel("throughput (GFLOP/s)")
    ax.set_xticks(batches)
    ax.set_title("LeNet forward throughput vs batch size")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=pathlib.Path,
                   help="sweep CSV from measure.sh (impl,batch,algo,trial,time_ms,gflops)")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("./plots"),
                   help="output directory (default: ./plots)")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_sweep_csv(args.csv)
    agg = aggregate(df)
    print(f"Loaded {len(df)} trials, {len(agg)} unique (impl, batch, algo) configs")

    plot_1_latency(agg, args.out / "plot1_latency.pdf")
    plot_2_throughput(agg, args.out / "plot2_throughput.pdf")
    plot_3_speedup(agg, args.out / "plot3_speedup.pdf")


if __name__ == "__main__":
    main()
