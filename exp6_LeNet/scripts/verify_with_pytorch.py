#!/usr/bin/env python3
"""Compare C++ LeNet logits with a PyTorch reference implementation."""

import argparse
import pathlib
import sys
from typing import Tuple

import numpy as np
import torch

IN_CHANNELS = 1
IN_HEIGHT = 32
IN_WIDTH = 32
CONV1_OUT = 6
CONV2_OUT = 16
KERNEL = 5
POOL_STRIDE = 2
FC1_OUT = 120
FC2_OUT = 84
FC3_OUT = 10

DTYPE = torch.float32
DEVICE = torch.device("cpu")


def seeded_tensor(numel: int, scale: float) -> torch.Tensor:
    idx = torch.arange(numel, dtype=DTYPE, device=DEVICE)
    return scale * torch.sin(0.017 * idx)


class TorchLeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(IN_CHANNELS, CONV1_OUT, kernel_size=5, bias=True)
        self.conv2 = torch.nn.Conv2d(CONV1_OUT, CONV2_OUT, kernel_size=5, bias=True)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=POOL_STRIDE)
        self.act = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(CONV2_OUT * 5 * 5, FC1_OUT)
        self.fc2 = torch.nn.Linear(FC1_OUT, FC2_OUT)
        self.fc3 = torch.nn.Linear(FC2_OUT, FC3_OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def split_weights() -> Tuple[torch.Tensor, torch.Tensor]:
    # Matches seed_tensor usage in C++ harness.
    total_weight = (
        CONV1_OUT * IN_CHANNELS * KERNEL * KERNEL
        + CONV2_OUT * CONV1_OUT * KERNEL * KERNEL
        + FC1_OUT * CONV2_OUT * 5 * 5
        + FC2_OUT * FC1_OUT
        + FC3_OUT * FC2_OUT
    )
    total_bias = CONV1_OUT + CONV2_OUT + FC1_OUT + FC2_OUT + FC3_OUT
    return seeded_tensor(total_weight, 0.05), seeded_tensor(total_bias, 0.01)


def load_reference_parameters(model: TorchLeNet) -> None:
    weights, biases = split_weights()
    cursor_w = 0
    cursor_b = 0
    with torch.no_grad():
        conv1_elems = CONV1_OUT * IN_CHANNELS * KERNEL * KERNEL
        model.conv1.weight.copy_(weights[cursor_w:cursor_w + conv1_elems].view_as(model.conv1.weight))
        cursor_w += conv1_elems
        model.conv1.bias.copy_(biases[cursor_b:cursor_b + CONV1_OUT])
        cursor_b += CONV1_OUT

        conv2_elems = CONV2_OUT * CONV1_OUT * KERNEL * KERNEL
        model.conv2.weight.copy_(weights[cursor_w:cursor_w + conv2_elems].view_as(model.conv2.weight))
        cursor_w += conv2_elems
        model.conv2.bias.copy_(biases[cursor_b:cursor_b + CONV2_OUT])
        cursor_b += CONV2_OUT

        fc1_elems = FC1_OUT * CONV2_OUT * 5 * 5
        model.fc1.weight.copy_(weights[cursor_w:cursor_w + fc1_elems].view_as(model.fc1.weight))
        cursor_w += fc1_elems
        model.fc1.bias.copy_(biases[cursor_b:cursor_b + FC1_OUT])
        cursor_b += FC1_OUT

        fc2_elems = FC2_OUT * FC1_OUT
        model.fc2.weight.copy_(weights[cursor_w:cursor_w + fc2_elems].view_as(model.fc2.weight))
        cursor_w += fc2_elems
        model.fc2.bias.copy_(biases[cursor_b:cursor_b + FC2_OUT])
        cursor_b += FC2_OUT

        fc3_elems = FC3_OUT * FC2_OUT
        model.fc3.weight.copy_(weights[cursor_w:cursor_w + fc3_elems].view_as(model.fc3.weight))
        cursor_w += fc3_elems
        model.fc3.bias.copy_(biases[cursor_b:cursor_b + FC3_OUT])
        cursor_b += FC3_OUT

    assert cursor_w == weights.numel()
    assert cursor_b == biases.numel()


def build_reference_inputs(batch: int) -> torch.Tensor:
    numel = batch * IN_CHANNELS * IN_HEIGHT * IN_WIDTH
    seeded = seeded_tensor(numel, 1.0)
    return seeded.view(batch, IN_CHANNELS, IN_HEIGHT, IN_WIDTH)


def load_cpp_output(path: pathlib.Path, batch: int) -> torch.Tensor:
    expected = batch * FC3_OUT
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != expected:
        raise ValueError(f"Expected {expected} floats in {path}, found {raw.size}")
    return torch.from_numpy(raw.copy()).view(batch, FC3_OUT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate C++ LeNet outputs with PyTorch")
    parser.add_argument("--output", required=True, help="Binary file written by dlenet --dump")
    parser.add_argument("--batch", type=int, required=True, help="Batch size used for the run")
    parser.add_argument("--tol", type=float, default=1e-3, help="Allowed max-abs error")
    args = parser.parse_args()

    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        raise FileNotFoundError(f"Missing output file: {output_path}")

    model = TorchLeNet().to(DEVICE)
    load_reference_parameters(model)
    inputs = build_reference_inputs(args.batch)

    with torch.no_grad():
        torch_logits = model(inputs)

    cpp_logits = load_cpp_output(output_path, args.batch)
    diff = (torch_logits - cpp_logits).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    print(f"Max abs diff: {max_diff:.6f}  Mean abs diff: {mean_diff:.6f}")
    if max_diff > args.tol:
        print("Mismatch exceeds tolerance", file=sys.stderr)
        sys.exit(1)
    print("Outputs match within tolerance")


if __name__ == "__main__":
    main()
