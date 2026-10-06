#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AGGREGATE_NAME = "__aggregate__"


def load_activation_run(run_dir: Path):
    """
    Load one activation dump produced by collect_activation.py.

    Expected structure:

        run_dir/
        ├── metadata.json
        ├── encoder.blocks.0.mlp.1.npy
        ├── encoder.blocks.1.mlp.1.npy
        └── ...

    Returns
    -------
    metadata : dict
    activations : dict[str, np.ndarray]
    """
    metadata_path = run_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "layers" not in metadata:
        raise RuntimeError(
            f"'layers' field missing in {metadata_path}"
        )

    activations = {}

    for layer_name, layer_info in metadata["layers"].items():
        file_name = layer_info["file"]
        path = run_dir / file_name

        if not path.exists():
            raise FileNotFoundError(
                f"Activation file not found: {path}"
            )

        # mmap avoids loading every layer into RAM immediately.
        activations[layer_name] = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

    return metadata, activations


def flatten_abs(x):
    """
    Convert activation tensor into 1-D absolute activation magnitude.

    Quantization dynamic range depends on |activation|.
    """
    x = np.asarray(x, dtype=np.float32)

    return np.abs(
        x.reshape(-1)
    )


def compute_stats(values):
    """
    Calculate tail statistics of |activation|.
    """
    return {
        "n": int(values.size),
        "p99": float(
            np.percentile(values, 99.0)
        ),
        "p99.9": float(
            np.percentile(values, 99.9)
        ),
        "max": float(
            np.max(values)
        ),
    }


def collect_layer_stats(activations):
    rows = []

    for layer_name, activation in activations.items():
        values = flatten_abs(activation)
        stats = compute_stats(values)

        rows.append({
            "scope": layer_name,
            **stats,
        })

    return rows


def write_stats_csv(rows, output_path):
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "n",
                "p99",
                "p99.9",
                "max",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def print_stats(rows):
    print()
    print("=" * 90)
    print("FP32 WHISPER POST-GeLU ACTIVATION STATISTICS")
    print("Statistics are calculated on |activation|")
    print("=" * 90)

    print(
        f"{'Layer':40s}"
        f"{'P99':>12s}"
        f"{'P99.9':>12s}"
        f"{'Max':>12s}"
        f"{'Max/P99':>12s}"
        f"{'Max/P99.9':>14s}"
    )

    print("-" * 90)

    for row in rows:
        p99 = row["p99"]
        p999 = row["p99.9"]
        max_value = row["max"]

        ratio99 = (
            max_value / p99
            if p99 > 0
            else float("inf")
        )

        ratio999 = (
            max_value / p999
            if p999 > 0
            else float("inf")
        )

        print(
            f"{row['scope']:40s}"
            f"{p99:12.6f}"
            f"{p999:12.6f}"
            f"{max_value:12.6f}"
            f"{ratio99:12.1f}"
            f"{ratio999:14.1f}"
        )

    print()


def empirical_ccdf(values):
    """
    Exact empirical complementary CDF:

        CCDF(x) = P(X >= x)

    values must contain absolute activation magnitudes.
    """
    x = np.sort(values)

    n = x.size

    ccdf = (
        n - np.arange(n, dtype=np.float64)
    ) / n

    return x, ccdf


def short_layer_name(name):
    """
    encoder.blocks.3.mlp.1
        ->
    Block 3
    """
    prefix = "encoder.blocks."
    suffix = ".mlp.1"

    if name.startswith(prefix) and name.endswith(suffix):
        index = name[
            len(prefix):
            -len(suffix)
        ]

        return f"Block {index}"

    return name


def plot_layer_ccdf(
    activations,
    output_path,
    ccdf_min=1e-6,
    log_x=False,
):
    """
    Plot one CCDF curve per encoder Transformer block.
    """
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for layer_name in sorted(activations.keys()):

        print(
            f"Preparing CCDF: {layer_name}"
        )

        values = flatten_abs(
            activations[layer_name]
        )

        x, ccdf = empirical_ccdf(values)

        mask = (
            ccdf >= ccdf_min
        )

        if log_x:
            mask &= x > 0

        ax.plot(
            x[mask],
            ccdf[mask],
            linewidth=1.8,
            label=short_layer_name(layer_name),
        )

        del values
        del x
        del ccdf

    if log_x:
        ax.set_xscale("log")

    ax.set_yscale("log")

    ax.set_xlabel(
        "|Post-GeLU activation|"
    )

    ax.set_ylabel(
        r"CCDF  $P(|A| \geq x)$"
    )

    if log_x:
        ax.set_title(
            "FP32 Whisper Encoder Post-GeLU Activation Tail "
            "(Log–Log)"
        )
    else:
        ax.set_title(
            "FP32 Whisper Encoder Post-GeLU Activation Tail"
        )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
        linewidth=0.6,
    )

    ax.legend(
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_tail_summary(
    rows,
    output_path,
):
    """
    Plot P99, P99.9 and Max together on a logarithmic y-axis.

    The log scale is necessary because the measured values span
    multiple orders of magnitude.
    """
    rows = sorted(
        rows,
        key=lambda r: r["scope"],
    )

    labels = [
        short_layer_name(row["scope"])
        for row in rows
    ]

    p99 = np.array([
        row["p99"]
        for row in rows
    ])

    p999 = np.array([
        row["p99.9"]
        for row in rows
    ])

    max_values = np.array([
        row["max"]
        for row in rows
    ])

    x = np.arange(len(rows))

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        x,
        p99,
        marker="o",
        linewidth=1.8,
        label="P99",
    )

    ax.plot(
        x,
        p999,
        marker="o",
        linewidth=1.8,
        label="P99.9",
    )

    ax.plot(
        x,
        max_values,
        marker="o",
        linewidth=1.8,
        label="Max",
    )

    ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel(
        "Encoder Transformer block"
    )

    ax.set_ylabel(
        "|Post-GeLU activation|"
    )

    ax.set_title(
        "FP32 Whisper Encoder Activation Tail Statistics"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
        linewidth=0.6,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def compute_channel_max(activation):
    """
    activation:
        [N, T, C]

    return:
        [C]
    """
    activation = np.asarray(
        activation,
        dtype=np.float32,
    )

    return np.abs(
        activation
    ).max(
        axis=(0, 1)
    )


def plot_sorted_channel_max(
    activation,
    layer_name,
    output_path,
):
    channel_max = compute_channel_max(
        activation
    )

    order = np.argsort(
        channel_max
    )[::-1]

    channel_max = channel_max[order]

    x = np.arange(
        channel_max.size
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        x,
        channel_max,
        linewidth=1.8,
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Channel rank (sorted by maximum)"
    )

    ax.set_ylabel(
        "Maximum |post-GeLU activation|"
    )

    ax.set_title(
        f"FP32 Whisper {short_layer_name(layer_name)} "
        "Channel-wise Activation Range"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
        linewidth=0.6,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_channel_summary_csv(
    summaries,
    output_path,
):
    fieldnames = [
        "layer",
        "num_channels",
        "median_channel_max",
        "p90_channel_max",
        "p99_channel_max",
        "max_channel_max",
        "max_over_median",
        "count_ge_10",
        "count_ge_50",
        "count_ge_100",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for layer_name, summary in summaries.items():
            writer.writerow({
                "layer": layer_name,
                **summary,
            })


def compute_channel_summary(activation):
    """
    activation shape: [N, T, C]

    First compute per-channel maximum:
        M_c = max_{n,t} |A[n,t,c]|

    Then summarize the distribution of {M_c}.
    """
    channel_max = compute_channel_max(activation)

    median = float(np.median(channel_max))
    p90 = float(np.percentile(channel_max, 90))
    p99 = float(np.percentile(channel_max, 99))
    maximum = float(np.max(channel_max))

    return {
        "num_channels": int(channel_max.size),

        "median_channel_max": median,
        "p90_channel_max": p90,
        "p99_channel_max": p99,
        "max_channel_max": maximum,

        "max_over_median": (
            maximum / median
            if median > 0
            else float("inf")
        ),

        "count_ge_10": int(
            np.sum(channel_max >= 10.0)
        ),

        "count_ge_50": int(
            np.sum(channel_max >= 50.0)
        ),

        "count_ge_100": int(
            np.sum(channel_max >= 100.0)
        ),
    }


def print_channel_summary(
    activations,
    layer_names,
):
    print()
    print("=" * 100)
    print("CHANNEL-WISE POST-GeLU ACTIVATION RANGE")
    print(
        "Each channel value is "
        "max_{sample,token} |activation|"
    )
    print("=" * 100)

    print(
        f"{'Layer':12s}"
        f"{'Median':>12s}"
        f"{'P90':>12s}"
        f"{'P99':>12s}"
        f"{'Max':>12s}"
        f"{'Max/Med':>12s}"
        f"{'>=10':>8s}"
        f"{'>=50':>8s}"
        f"{'>=100':>8s}"
    )

    print("-" * 100)

    summaries = {}

    for layer_name in layer_names:
        summary = compute_channel_summary(
            activations[layer_name]
        )

        summaries[layer_name] = summary

        print(
            f"{short_layer_name(layer_name):12s}"
            f"{summary['median_channel_max']:12.4f}"
            f"{summary['p90_channel_max']:12.4f}"
            f"{summary['p99_channel_max']:12.4f}"
            f"{summary['max_channel_max']:12.4f}"
            f"{summary['max_over_median']:12.1f}"
            f"{summary['count_ge_10']:8d}"
            f"{summary['count_ge_50']:8d}"
            f"{summary['count_ge_100']:8d}"
        )

    print()

    return summaries


def plot_smoothing_channel_profile(
    before_activation,
    after_activation,
    layer_name,
    output_path,
):
    """
    Compare per-channel maximum activation before and after smoothing.

    Channel order is determined ONLY from the before-smoothing
    activation, so each x position refers to the same physical channel.
    """

    before_max = compute_channel_max(
        before_activation
    )

    after_max = compute_channel_max(
        after_activation
    )

    if before_max.shape != after_max.shape:
        raise RuntimeError(
            f"Channel mismatch: "
            f"{before_max.shape} vs {after_max.shape}"
        )

    order = np.argsort(
        before_max
    )[::-1]

    before_sorted = before_max[order]
    after_sorted = after_max[order]

    x = np.arange(
        before_sorted.size
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        x,
        before_sorted,
        linewidth=1.8,
        label="Before smoothing",
    )

    ax.plot(
        x,
        after_sorted,
        linewidth=1.8,
        label="After smoothing",
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Channel rank "
        "(sorted by before-smoothing maximum)"
    )

    ax.set_ylabel(
        "Maximum |post-GeLU activation|"
    )

    ax.set_title(
        f"{short_layer_name(layer_name)} "
        "Channel-wise Activation Range"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
        linewidth=0.6,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_smoothing_channel_scatter(
    before_activation,
    after_activation,
    layer_name,
    output_path,
):
    before_max = compute_channel_max(
        before_activation
    )

    after_max = compute_channel_max(
        after_activation
    )

    if before_max.shape != after_max.shape:
        raise RuntimeError(
            f"Channel mismatch: "
            f"{before_max.shape} vs {after_max.shape}"
        )

    # log axis cannot represent zero
    valid = (
        (before_max > 0)
        & (after_max > 0)
    )

    x = before_max[valid]
    y = after_max[valid]

    fig, ax = plt.subplots(
        figsize=(5.8, 5.5)
    )

    ax.scatter(
        x,
        y,
        s=12,
        alpha=0.55,
    )

    # --------------------------------------------------------
    # Identity line y = x
    # --------------------------------------------------------

    plot_min = min(
        float(x.min()),
        float(y.min()),
    )

    plot_max = max(
        float(x.max()),
        float(y.max()),
    )

    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--",
        linewidth=1.4,
        label="No change (y = x)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        "Before smoothing\n"
        "Maximum |post-GeLU activation|"
    )

    ax.set_ylabel(
        "After smoothing\n"
        "Maximum |post-GeLU activation|"
    )

    ax.set_title(
        f"{short_layer_name(layer_name)} "
        "Channel Range Before vs. After Smoothing"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
        linewidth=0.6,
    )

    ax.legend(
        frameon=False,
    )

    # Same limits make distance from y=x visually meaningful.
    ax.set_xlim(
        plot_min,
        plot_max,
    )

    ax.set_ylim(
        plot_min,
        plot_max,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def compute_smoothing_reduction(
    before_activation,
    after_activation,
):
    before_max = compute_channel_max(
        before_activation
    )

    after_max = compute_channel_max(
        after_activation
    )

    eps = 1e-12

    reduction = (
        before_max
        / np.maximum(after_max, eps)
    )

    return {
        "median": float(
            np.median(reduction)
        ),
        "p90": float(
            np.percentile(reduction, 90)
        ),
        "p99": float(
            np.percentile(reduction, 99)
        ),
        "max": float(
            np.max(reduction)
        ),

        "ge_2x": float(
            np.mean(reduction >= 2.0) * 100
        ),
        "ge_5x": float(
            np.mean(reduction >= 5.0) * 100
        ),
        "ge_10x": float(
            np.mean(reduction >= 10.0) * 100
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze FP32 Whisper encoder post-GeLU "
            "activation outliers."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Activation run directory produced by "
            "collect_activation.py"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/figures/fp32"
        ),
    )

    parser.add_argument(
        "--ccdf-min",
        type=float,
        default=1e-6,
        help=(
            "Minimum CCDF probability shown. "
            "Default: 1e-6"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Load activation run
    # ------------------------------------------------------------

    print(
        f"Loading activation run: {args.input}"
    )

    metadata, activations = (
        load_activation_run(args.input)
    )

    print()
    print(
        f"Model      : {metadata.get('model')}"
    )
    print(
        f"Condition  : {metadata.get('condition')}"
    )
    print(
        f"Dataset    : {metadata.get('dataset')}"
    )
    print(
        f"Num audio  : {metadata.get('num_audio')}"
    )

    print()
    print("Layers:")

    for name in activations:
        print(
            f"  {name} "
            f"{tuple(activations[name].shape)}"
        )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    rows = collect_layer_stats(
        activations
    )

    print_stats(rows)

    stats_path = (
        args.output_dir
        / "fp32_post_gelu_stats.csv"
    )

    write_stats_csv(
        rows,
        stats_path,
    )
    

    # ------------------------------------------------------------
    # plot smooting channel profile for selected layers
    # ------------------------------------------------------------
    before_meta, before_acts = load_activation_run(
        Path(
            "ACTIVATION_DUMP/"
            "smoothing_base_regression10/before"
        )
    )

    after_meta, after_acts = load_activation_run(
        Path(
            "ACTIVATION_DUMP/"
            "smoothing_base_regression10/after"
        )
    )
    for block in [3, 4, 5]:
        layer_name = (
            f"encoder.blocks.{block}.mlp.1"
        )

        # --------------------------------------------------------
        # Sorted channel profile
        # --------------------------------------------------------

        plot_smoothing_channel_profile(
            before_acts[layer_name],
            after_acts[layer_name],
            layer_name,
            args.output_dir
            / f"smoothing_block{block}_channel_profile.png",
        )

        # --------------------------------------------------------
        # Before-vs-after channel scatter
        # --------------------------------------------------------

        plot_smoothing_channel_scatter(
            before_acts[layer_name],
            after_acts[layer_name],
            layer_name,
            args.output_dir
            / f"smoothing_block{block}_channel_scatter.png",
        )
    
    
    
    # ------------------------------------------------------------
    # plot per-channel maximum for selected layers
    # ------------------------------------------------------------
    
    channel_layers = [
        "encoder.blocks.3.mlp.1",
        "encoder.blocks.4.mlp.1",
        "encoder.blocks.5.mlp.1",
    ]

    channel_summaries = print_channel_summary(
        activations,
        channel_layers,
    )
    
    write_channel_summary_csv(
        channel_summaries,
        args.output_dir
        / "fp32_channel_summary.csv",
    )
    
    
    # ------------------------------------------------------------
    # plot before vs after smoothing for selected layers
    # ------------------------------------------------------------
    
    print()
    print("=" * 86)
    print("PER-CHANNEL RANGE REDUCTION AFTER SMOOTHING")
    print("=" * 86)

    print(
        f"{'Block':<10}"
        f"{'Median':>10}"
        f"{'P90':>10}"
        f"{'P99':>10}"
        f"{'Max':>10}"
        f"{'>=2x':>10}"
        f"{'>=5x':>10}"
        f"{'>=10x':>10}"
    )

    for block in [3, 4, 5]:
        layer_name = (
            f"encoder.blocks.{block}.mlp.1"
        )

        r = compute_smoothing_reduction(
            before_acts[layer_name],
            after_acts[layer_name],
        )

        print(
            f"{'Block ' + str(block):<10}"
            f"{r['median']:10.2f}"
            f"{r['p90']:10.2f}"
            f"{r['p99']:10.2f}"
            f"{r['max']:10.2f}"
            f"{r['ge_2x']:9.1f}%"
            f"{r['ge_5x']:9.1f}%"
            f"{r['ge_10x']:9.1f}%"
        )
    
    

    # ------------------------------------------------------------
    # Tail summary
    # ------------------------------------------------------------

    summary_path = (
        args.output_dir
        / "fp32_post_gelu_tail_summary.png"
    )

    plot_tail_summary(
        rows,
        summary_path,
    )

    # ------------------------------------------------------------
    # Linear-x CCDF
    # ------------------------------------------------------------

    ccdf_linear_path = (
        args.output_dir
        / "fp32_post_gelu_ccdf_linear_x.png"
    )

    plot_layer_ccdf(
        activations,
        ccdf_linear_path,
        ccdf_min=args.ccdf_min,
        log_x=False,
    )

    # ------------------------------------------------------------
    # Log-log CCDF
    # ------------------------------------------------------------

    ccdf_loglog_path = (
        args.output_dir
        / "fp32_post_gelu_ccdf_loglog.png"
    )

    plot_layer_ccdf(
        activations,
        ccdf_loglog_path,
        ccdf_min=args.ccdf_min,
        log_x=True,
    )

    # ------------------------------------------------------------
    # Done
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Statistics : {stats_path}"
    )

    print(
        f"Tail stats : {summary_path}"
    )

    print(
        f"CCDF       : {ccdf_linear_path}"
    )

    print(
        f"CCDF log   : {ccdf_loglog_path}"
    )


if __name__ == "__main__":
    main()