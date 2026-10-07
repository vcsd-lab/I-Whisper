#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTEXT_BINS = [
    (1, 8),
    (9, 16),
    (17, 24),
    (25, 32),
    (33, 40),
    (41, 52),
]


def load_rows(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "context_len":
                    int(row["context_len"]),

                "naive_lost_mass":
                    float(row["naive_lost_mass"]),

                "fusion_lost_mass":
                    float(row["fusion_lost_mass"]),

                "naive_l1_error":
                    float(row["naive_l1_error"]),

                "fusion_l1_error":
                    float(row["fusion_l1_error"]),
            })

    return rows


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "median":
            float(np.median(values)),

        "p25":
            float(np.percentile(values, 25)),

        "p75":
            float(np.percentile(values, 75)),

        "mean":
            float(np.mean(values)),
    }


def build_bin_stats(rows):
    stats = []

    for low, high in CONTEXT_BINS:

        group = [
            row
            for row in rows
            if low <= row["context_len"] <= high
        ]

        if not group:
            continue

        naive_mass = np.asarray(
            [
                row["naive_lost_mass"]
                for row in group
            ],
            dtype=np.float64,
        )

        fusion_mass = np.asarray(
            [
                row["fusion_lost_mass"]
                for row in group
            ],
            dtype=np.float64,
        )

        naive_l1 = np.asarray(
            [
                row["naive_l1_error"]
                for row in group
            ],
            dtype=np.float64,
        )

        fusion_l1 = np.asarray(
            [
                row["fusion_l1_error"]
                for row in group
            ],
            dtype=np.float64,
        )

        nm = summarize(naive_mass)
        fm = summarize(fusion_mass)

        nl = summarize(naive_l1)
        fl = summarize(fusion_l1)

        stats.append({
            "context_bin":
                f"{low}-{high}",

            "context_low":
                low,

            "context_high":
                high,

            "num_rows":
                len(group),

            "naive_lost_mass_median":
                nm["median"],

            "naive_lost_mass_p25":
                nm["p25"],

            "naive_lost_mass_p75":
                nm["p75"],

            "naive_lost_mass_mean":
                nm["mean"],

            "fusion_lost_mass_median":
                fm["median"],

            "fusion_lost_mass_p25":
                fm["p25"],

            "fusion_lost_mass_p75":
                fm["p75"],

            "fusion_lost_mass_mean":
                fm["mean"],

            "lost_mass_fusion_better_fraction":
                float(np.mean(
                    fusion_mass < naive_mass
                )),

            "naive_l1_median":
                nl["median"],

            "naive_l1_p25":
                nl["p25"],

            "naive_l1_p75":
                nl["p75"],

            "naive_l1_mean":
                nl["mean"],

            "fusion_l1_median":
                fl["median"],

            "fusion_l1_p25":
                fl["p25"],

            "fusion_l1_p75":
                fl["p75"],

            "fusion_l1_mean":
                fl["mean"],

            "l1_fusion_better_fraction":
                float(np.mean(
                    fusion_l1 < naive_l1
                )),
        })

    return stats


def write_stats(stats, path):
    if not stats:
        raise RuntimeError(
            "No context-bin statistics."
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                stats[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(stats)


def plot_metric(
    stats,
    naive_prefix,
    fusion_prefix,
    ylabel,
    title,
    output_path,
):
    x = np.arange(
        len(stats)
    )

    labels = [
        row["context_bin"]
        for row in stats
    ]

    naive_median = np.asarray([
        row[f"{naive_prefix}_median"]
        for row in stats
    ])

    naive_p25 = np.asarray([
        row[f"{naive_prefix}_p25"]
        for row in stats
    ])

    naive_p75 = np.asarray([
        row[f"{naive_prefix}_p75"]
        for row in stats
    ])

    fusion_median = np.asarray([
        row[f"{fusion_prefix}_median"]
        for row in stats
    ])

    fusion_p25 = np.asarray([
        row[f"{fusion_prefix}_p25"]
        for row in stats
    ])

    fusion_p75 = np.asarray([
        row[f"{fusion_prefix}_p75"]
        for row in stats
    ])

    fig, ax = plt.subplots(
        figsize=(7.0, 4.5)
    )

    ax.plot(
        x,
        naive_median,
        marker="o",
        linewidth=2.0,
        label="Direct Q1.7(P)",
    )

    ax.fill_between(
        x,
        naive_p25,
        naive_p75,
        alpha=0.18,
    )

    ax.plot(
        x,
        fusion_median,
        marker="o",
        linewidth=2.0,
        label="Q1.7(E) + normalization",
    )

    ax.fill_between(
        x,
        fusion_p25,
        fusion_p75,
        alpha=0.18,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel(
        "Decoder Context Length"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_stats(stats):
    print()
    print("=" * 116)
    print("CONTEXT-BIN SUMMARY")
    print("=" * 116)

    print(
        f"{'Bin':>8}"
        f"{'N':>8}"
        f"{'NaiveMass':>14}"
        f"{'FusionMass':>14}"
        f"{'Mass Better':>14}"
        f"{'NaiveL1':>14}"
        f"{'FusionL1':>14}"
        f"{'L1 Better':>12}"
    )

    for row in stats:

        print(
            f"{row['context_bin']:>8}"
            f"{row['num_rows']:8d}"
            f"{row['naive_lost_mass_median']:14.6f}"
            f"{row['fusion_lost_mass_median']:14.6f}"
            f"{row['lost_mass_fusion_better_fraction'] * 100:13.1f}%"
            f"{row['naive_l1_median']:14.6f}"
            f"{row['fusion_l1_median']:14.6f}"
            f"{row['l1_fusion_better_fraction'] * 100:11.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot A1 attention Softmax "
            "representation analysis."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "ATTENTION_DUMP/"
            "fp32_base_regression10/"
            "rows.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/figures/attention"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows(
        args.input
    )

    stats = build_bin_stats(
        rows
    )

    print_stats(
        stats
    )

    stats_path = (
        args.output_dir
        / "a1_context_bin_stats.csv"
    )

    write_stats(
        stats,
        stats_path,
    )

    lost_mass_path = (
        args.output_dir
        / "a1_lost_mass_vs_context.png"
    )

    plot_metric(
        stats=stats,
        naive_prefix="naive_lost_mass",
        fusion_prefix="fusion_lost_mass",
        ylabel="Lost FP32 Attention Mass",
        title=(
            "Attention Mass Lost by "
            "Q1.7 Quantization"
        ),
        output_path=lost_mass_path,
    )

    l1_path = (
        args.output_dir
        / "a1_l1_error_vs_context.png"
    )

    plot_metric(
        stats=stats,
        naive_prefix="naive_l1",
        fusion_prefix="fusion_l1",
        ylabel=r"$\|P-\hat{P}\|_1$",
        title=(
            "Attention Distribution "
            "Quantization Error"
        ),
        output_path=l1_path,
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Statistics : {stats_path}"
    )

    print(
        f"Lost mass : {lost_mass_path}"
    )

    print(
        f"L1 error  : {l1_path}"
    )


if __name__ == "__main__":
    main()
