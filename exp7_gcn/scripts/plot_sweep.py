#!/usr/bin/env python3
"""Generate plots from a measure.sh sweep CSV.

Usage:
    python3 scripts/plot_sweep.py                      # auto-find latest CSV
    python3 scripts/plot_sweep.py --csv path/to.csv    # explicit file
    python3 scripts/plot_sweep.py --out plots          # output directory

Produces two PNGs:
    throughput.png  — edges/s bar chart across all (graph, hidden, impl) configs
    scaling.png     — time_ms vs hidden_dim, one line per (graph, impl)

Notes:
    - DGL throughput is NOT plotted because measure.sh doesn't collect it.
      For a DGL comparison figure, extract throughputs from compare_with_dgl.py
      output separately and add them to the report manually.
    - Bars show the MIN over the repeats (least noisy), with error bars up
      to the MAX. This is more honest than std on n=3 samples.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict


def find_latest_csv(data_dir: str) -> str:
    pattern = os.path.join(data_dir, "*_gcn_sweep.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"ERROR: no CSV matching {pattern}. Run ./scripts/measure.sh first.",
              file=sys.stderr)
        sys.exit(1)
    return matches[-1]


def load_csv(path: str):
    """Read the CSV and group rows by (graph, hidden, impl).

    Returns a dict keyed by (graph, hidden, impl) -> list of (time_ms, edges_s).
    """
    rows = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row["graph"], int(row["hidden"]), row["impl"])
                rows[key].append((float(row["time_ms"]), float(row["edges_per_s"])))
            except (KeyError, ValueError) as exc:
                print(f"WARN: skipping malformed row {row}: {exc}", file=sys.stderr)
    return dict(rows)


def aggregate(vals):
    """Return (min, max, mean) for a list of floats."""
    if not vals:
        return (0.0, 0.0, 0.0)
    return (min(vals), max(vals), sum(vals) / len(vals))


def graph_short(g: str) -> str:
    """Turn 'data/cora' into 'cora' for plot labels."""
    return os.path.basename(g)


def plot_throughput(data, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    # Build a consistent ordering of configs.
    graphs = sorted({g for (g, _, _) in data.keys()})
    hiddens = sorted({h for (_, h, _) in data.keys()})
    impls = sorted({i for (_, _, i) in data.keys()})

    # One subplot per graph, one group per hidden, two bars per group (baseline/fused).
    fig, axes = plt.subplots(1, len(graphs), figsize=(5 * len(graphs), 4), sharey=True)
    if len(graphs) == 1:
        axes = [axes]

    bar_width = 0.38
    x = np.arange(len(hiddens))

    for ax, graph in zip(axes, graphs):
        for j, impl in enumerate(impls):
            mins, maxs = [], []
            for h in hiddens:
                key = (graph, h, impl)
                if key not in data:
                    mins.append(0)
                    maxs.append(0)
                    print(f"WARN: no data for {key}, bar will be empty", file=sys.stderr)
                    continue
                edges = [e for (_, e) in data[key]]
                lo, hi, _ = aggregate(edges)
                mins.append(lo)
                maxs.append(hi)
            mins = np.array(mins)
            maxs = np.array(maxs)
            yerr = np.vstack([np.zeros_like(mins), maxs - mins])  # asymmetric: only up
            ax.bar(x + (j - 0.5) * bar_width, mins,
                   width=bar_width, label=impl,
                   yerr=yerr, capsize=3,
                   edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in hiddens])
        ax.set_xlabel("hidden dim")
        ax.set_title(graph_short(graph))
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("throughput (edges / sec, min across repeats)")
    axes[-1].legend(loc="upper right")
    fig.suptitle("GCN forward throughput by implementation\n"
                 "(bar = min across repeats, error bar up to max)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_scaling(data, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    graphs = sorted({g for (g, _, _) in data.keys()})
    hiddens = sorted({h for (_, h, _) in data.keys()})
    impls = sorted({i for (_, _, i) in data.keys()})

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Style: linestyle encodes graph, color encodes impl.
    linestyles = {"data/cora": "-", "data/citeseer": "--"}
    markers = {"data/cora": "o", "data/citeseer": "s"}

    for graph in graphs:
        for impl in impls:
            xs, ys, errs_lo, errs_hi = [], [], [], []
            for h in hiddens:
                key = (graph, h, impl)
                if key not in data:
                    continue
                times = [t for (t, _) in data[key]]
                lo, hi, mean = aggregate(times)
                xs.append(h)
                ys.append(mean)
                errs_lo.append(mean - lo)
                errs_hi.append(hi - mean)
            if not xs:
                continue
            label = f"{graph_short(graph)} / {impl}"
            ax.errorbar(xs, ys,
                        yerr=[errs_lo, errs_hi],
                        label=label,
                        linestyle=linestyles.get(graph, "-"),
                        marker=markers.get(graph, "o"),
                        capsize=3,
                        linewidth=1.5)

    ax.set_xlabel("hidden dim")
    ax.set_ylabel("forward time (ms, mean across repeats)")
    ax.set_xticks(hiddens)
    ax.set_title("Forward pass time scaling with hidden_dim\n"
                 "(error bars = min/max across repeats)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="Path to sweep CSV (defaults to newest in data/)")
    p.add_argument("--data-dir", default="data",
                   help="Directory to search for the newest CSV (default: data)")
    p.add_argument("--out", default="plots",
                   help="Output directory for PNGs (default: plots)")
    args = p.parse_args()

    # Lazy import so the --help message works even if matplotlib is missing.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("ERROR: matplotlib is required. pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    csv_path = args.csv or find_latest_csv(args.data_dir)
    print(f"Reading {csv_path}")
    data = load_csv(csv_path)
    if not data:
        print("ERROR: no rows parsed from CSV", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    plot_throughput(data, os.path.join(args.out, "throughput.png"))
    plot_scaling(data, os.path.join(args.out, "scaling.png"))


if __name__ == "__main__":
    main()
