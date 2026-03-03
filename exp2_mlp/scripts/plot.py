#!/usr/bin/env python3
"""
Plot MLP latency + throughput vs batch size from the sweep CSV.

Expected CSV columns:
  impl,layers,batch,activation,time_ms,gflops

Creates (by default) 2 figures per layer-shape:
  - latency_vs_batch_<layers>.png
  - throughput_vs_batch_<layers>.png
Each figure has two lines:
  - baseline
  - activation_fused
"""

import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt


LAYER_KEYS = ["512_512_512", "1024_2048_1024", "2048_2048_2048"]
IMPLS = ["baseline", "activation_fused"]


def plot_one_layer(df_layer: pd.DataFrame, layers: str, outdir: str) -> None:
    title_layers = layers

    # Ensure batches are sorted numerically
    df_layer = df_layer.copy()
    df_layer["batch"] = df_layer["batch"].astype(int)
    df_layer["time_ms"] = df_layer["time_ms"].astype(float)
    df_layer["gflops"] = df_layer["gflops"].astype(float)

    # --- Latency plot ---
    plt.figure()
    for impl in IMPLS:
        d = df_layer[df_layer["impl"] == impl].sort_values("batch")
        if d.empty:
            continue
        plt.plot(d["batch"], d["time_ms"], marker="o", label=impl)
    plt.xlabel("Batch size")
    plt.ylabel("Latency (ms)")
    plt.title(f"Latency vs Batch — Layers: {title_layers}")
    plt.legend()
    plt.tight_layout()
    latency_path = os.path.join(outdir, f"latency_vs_batch_{layers}.png")
    plt.savefig(latency_path, dpi=200)
    plt.close()

    # --- Throughput plot ---
    plt.figure()
    for impl in IMPLS:
        d = df_layer[df_layer["impl"] == impl].sort_values("batch")
        if d.empty:
            continue
        plt.plot(d["batch"], d["gflops"], marker="o", label=impl)
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (GFLOP/s)")
    plt.title(f"Throughput vs Batch — Layers: {title_layers}")
    plt.legend()
    plt.tight_layout()
    thr_path = os.path.join(outdir, f"throughput_vs_batch_{layers}.png")
    plt.savefig(thr_path, dpi=200)
    plt.close()

    print(f"Wrote:\n  {latency_path}\n  {thr_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to sweep CSV (e.g., ../data/<timestamp>_mlp_sweep.csv)")
    ap.add_argument("--outdir", default="plots", help="Directory to write PNGs")
    ap.add_argument("--activation", default=None, help="Optional: filter to a specific activation (e.g., relu)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = {"impl", "layers", "batch", "activation", "time_ms", "gflops"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if args.activation is not None:
        df = df[df["activation"] == args.activation]

    # Make sure we only plot the two impls we care about
    df = df[df["impl"].isin(IMPLS)]

    # Plot each requested layer combo (only if present)
    for layers in LAYER_KEYS:
        df_layer = df[df["layers"] == layers]
        if df_layer.empty:
            print(f"Skipping layers={layers} (no rows found in CSV)")
            continue
        plot_one_layer(df_layer, layers, args.outdir)


if __name__ == "__main__":
    main()
