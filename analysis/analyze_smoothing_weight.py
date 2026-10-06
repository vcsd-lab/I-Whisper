#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch

import i_whisper


def compute_weight_column_max(weight):
    """
    MLP2 weight:
        [out_features, in_features]

    Each input column corresponds to one post-GeLU channel.

    Return:
        [in_features]
    """
    weight = (
        weight
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    return np.abs(weight).max(axis=0)


def compute_summary(before, after):
    eps = 1e-12

    increase = (
        after
        / np.maximum(before, eps)
    )

    return {
        "before_median": float(
            np.median(before)
        ),
        "before_p99": float(
            np.percentile(before, 99)
        ),
        "before_max": float(
            np.max(before)
        ),

        "after_median": float(
            np.median(after)
        ),
        "after_p99": float(
            np.percentile(after, 99)
        ),
        "after_max": float(
            np.max(after)
        ),

        "increase_median": float(
            np.median(increase)
        ),
        "increase_p90": float(
            np.percentile(increase, 90)
        ),
        "increase_p99": float(
            np.percentile(increase, 99)
        ),
        "increase_max": float(
            np.max(increase)
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="base",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--download-root",
        default="CKPT",
    )

    parser.add_argument(
        "--scale-dir",
        type=Path,
        default=Path(
            "ACTIVATION_DUMP/"
            "smoothing_base_regression10/"
            "smoothing_scales"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Fresh unsmoothed quantizable FP model
    # --------------------------------------------------------

    print("Loading unsmoothed I-Whisper model...")

    model = i_whisper.load_model_quantized(
        args.model,
        device=args.device,
        download_root=args.download_root,
        quantize=False,
    )

    model.eval()

    print()
    print("=" * 108)
    print("MLP2 COLUMN-WISE WEIGHT RANGE BEFORE / AFTER SMOOTHING")
    print("Each column corresponds to one post-GeLU activation channel")
    print("=" * 108)

    print(
        f"{'Block':<10}"
        f"{'Med(B)':>11}"
        f"{'P99(B)':>11}"
        f"{'Max(B)':>11}"
        f"{'Med(A)':>11}"
        f"{'P99(A)':>11}"
        f"{'Max(A)':>11}"
        f"{'ScaleMed':>11}"
        f"{'ScaleP99':>11}"
        f"{'ScaleMax':>11}"
    )

    print("-" * 108)

    for block in [3, 4, 5]:

        layer_name = (
            f"encoder.blocks.{block}.mlp.1"
        )

        scale_path = (
            args.scale_dir
            / f"{layer_name}.npy"
        )

        if not scale_path.exists():
            raise FileNotFoundError(
                scale_path
            )

        scales = np.load(
            scale_path,
            allow_pickle=False,
        ).astype(np.float32)

        mlp2 = (
            model.encoder
            .blocks[block]
            .mlp[2]
        )

        weight_before = (
            mlp2.weight
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if weight_before.shape[1] != scales.size:
            raise RuntimeError(
                f"Block {block}: "
                f"weight columns "
                f"{weight_before.shape[1]} "
                f"!= scales {scales.size}"
            )

        # Exact transformation used by smooth_gelu_fcs:
        #
        # W_after[:, c]
        # =
        # W_before[:, c] * scale[c]
        #
        weight_after = (
            weight_before
            * scales.reshape(1, -1)
        )

        before_colmax = np.abs(
            weight_before
        ).max(axis=0)

        after_colmax = np.abs(
            weight_after
        ).max(axis=0)

        summary = compute_summary(
            before_colmax,
            after_colmax,
        )

        print(
            f"{'Block ' + str(block):<10}"

            f"{summary['before_median']:11.5f}"
            f"{summary['before_p99']:11.5f}"
            f"{summary['before_max']:11.5f}"

            f"{summary['after_median']:11.5f}"
            f"{summary['after_p99']:11.5f}"
            f"{summary['after_max']:11.5f}"

            f"{summary['increase_median']:11.2f}"
            f"{summary['increase_p99']:11.2f}"
            f"{summary['increase_max']:11.2f}"
        )


if __name__ == "__main__":
    main()