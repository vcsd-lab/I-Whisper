#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
    "naive_lost_mass",
    "fusion_lost_mass",
    "naive_l1_error",
    "fusion_l1_error",
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
                "audio_id": row["audio_id"],
                "layer": int(row["layer"]),
                "call_index": int(row["call_index"]),
                "head": int(row["head"]),
                "query_index": int(row["query_index"]),
                "context_len": int(row["context_len"]),
                "p_max": float(row["p_max"]),
                "effective_support":
                    float(row["effective_support"]),
                "naive_zero_fraction":
                    float(row["naive_zero_fraction"]),
                "naive_lost_mass":
                    float(row["naive_lost_mass"]),
                "naive_l1_error":
                    float(row["naive_l1_error"]),
                "naive_probability_sum":
                    float(row["naive_probability_sum"]),
                "fusion_zero_fraction":
                    float(row["fusion_zero_fraction"]),
                "fusion_lost_mass":
                    float(row["fusion_lost_mass"]),
                "fusion_l1_error":
                    float(row["fusion_l1_error"]),
                "fusion_probability_sum":
                    float(row["fusion_probability_sum"]),
            })

    return rows


def arr(rows, name):
    return np.asarray(
        [row[name] for row in rows],
        dtype=np.float64,
    )


def describe(x):
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "p75": np.percentile(x, 75),
        "p90": np.percentile(x, 90),
        "p95": np.percentile(x, 95),
        "p99": np.percentile(x, 99),
        "max": np.max(x),
    }


def print_metric(name, x):
    s = describe(x)

    print(
        f"{name:<24}"
        f"mean={s['mean']:.6f}  "
        f"median={s['median']:.6f}  "
        f"P90={s['p90']:.6f}  "
        f"P99={s['p99']:.6f}  "
        f"max={s['max']:.6f}"
    )


def paired_summary(
    rows,
    naive_name,
    fusion_name,
    title,
):
    naive = arr(
        rows,
        naive_name,
    )

    fusion = arr(
        rows,
        fusion_name,
    )

    delta = naive - fusion

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    print_metric(
        "Naive",
        naive,
    )

    print_metric(
        "Fusion",
        fusion,
    )

    print_metric(
        "Naive - Fusion",
        delta,
    )

    better = np.mean(
        fusion < naive
    )

    equal = np.mean(
        np.isclose(
            fusion,
            naive,
            atol=1e-12,
        )
    )

    worse = np.mean(
        fusion > naive
    )

    print()
    print(
        f"Fusion better : "
        f"{better * 100:.2f}%"
    )

    print(
        f"Equal         : "
        f"{equal * 100:.2f}%"
    )

    print(
        f"Fusion worse  : "
        f"{worse * 100:.2f}%"
    )


def print_context_stats(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[
            row["context_len"]
        ].append(row)

    print()
    print("=" * 104)
    print("BY CONTEXT LENGTH")
    print("=" * 104)

    print(
        f"{'L':>4} "
        f"{'N':>6} "
        f"{'Pmax':>9} "
        f"{'Neff':>9} "
        f"{'NaiveMass':>12} "
        f"{'FusionMass':>12} "
        f"{'Delta':>12} "
        f"{'NaiveL1':>12} "
        f"{'FusionL1':>12}"
    )

    for context_len in sorted(groups):

        group = groups[
            context_len
        ]

        naive_mass = arr(
            group,
            "naive_lost_mass",
        )

        fusion_mass = arr(
            group,
            "fusion_lost_mass",
        )

        naive_l1 = arr(
            group,
            "naive_l1_error",
        )

        fusion_l1 = arr(
            group,
            "fusion_l1_error",
        )

        p_max = arr(
            group,
            "p_max",
        )

        neff = arr(
            group,
            "effective_support",
        )

        print(
            f"{context_len:4d} "
            f"{len(group):6d} "
            f"{np.median(p_max):9.4f} "
            f"{np.median(neff):9.3f} "
            f"{np.median(naive_mass):12.6f} "
            f"{np.median(fusion_mass):12.6f} "
            f"{np.median(naive_mass - fusion_mass):12.6f} "
            f"{np.median(naive_l1):12.6f} "
            f"{np.median(fusion_l1):12.6f}"
        )


def print_support_bins(
    rows,
    num_bins=10,
):
    support = arr(
        rows,
        "effective_support",
    )

    edges = np.quantile(
        support,
        np.linspace(
            0.0,
            1.0,
            num_bins + 1,
        ),
    )

    print()
    print("=" * 102)
    print("BY EFFECTIVE SUPPORT QUANTILE")
    print("=" * 102)

    print(
        f"{'Bin':>4} "
        f"{'N':>6} "
        f"{'Neff median':>12} "
        f"{'Pmax':>9} "
        f"{'NaiveMass':>12} "
        f"{'FusionMass':>12} "
        f"{'Delta':>12}"
    )

    for i in range(num_bins):

        low = edges[i]
        high = edges[i + 1]

        if i == num_bins - 1:
            group = [
                row for row in rows
                if (
                    row["effective_support"]
                    >= low
                    and
                    row["effective_support"]
                    <= high
                )
            ]
        else:
            group = [
                row for row in rows
                if (
                    row["effective_support"]
                    >= low
                    and
                    row["effective_support"]
                    < high
                )
            ]

        if not group:
            continue

        neff = arr(
            group,
            "effective_support",
        )

        pmax = arr(
            group,
            "p_max",
        )

        naive = arr(
            group,
            "naive_lost_mass",
        )

        fusion = arr(
            group,
            "fusion_lost_mass",
        )

        print(
            f"{i + 1:4d} "
            f"{len(group):6d} "
            f"{np.median(neff):12.3f} "
            f"{np.median(pmax):9.4f} "
            f"{np.median(naive):12.6f} "
            f"{np.median(fusion):12.6f} "
            f"{np.median(naive - fusion):12.6f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "ATTENTION_DUMP/"
            "fp32_base_regression10/"
            "rows.csv"
        ),
    )

    args = parser.parse_args()

    rows = load_rows(
        args.input
    )

    print()
    print("=" * 88)
    print("ATTENTION SOFTMAX A1 ANALYSIS")
    print("=" * 88)

    print(
        f"Rows            : {len(rows)}"
    )

    context = arr(
        rows,
        "context_len",
    )

    support = arr(
        rows,
        "effective_support",
    )

    pmax = arr(
        rows,
        "p_max",
    )

    print(
        f"Context length  : "
        f"{int(context.min())} .. "
        f"{int(context.max())}"
    )

    print(
        f"Effective support: "
        f"{support.min():.3f} .. "
        f"{support.max():.3f}"
    )

    print(
        f"Pmax            : "
        f"{pmax.min():.6f} .. "
        f"{pmax.max():.6f}"
    )

    paired_summary(
        rows,
        "naive_lost_mass",
        "fusion_lost_mass",
        "LOST ATTENTION MASS",
    )

    paired_summary(
        rows,
        "naive_l1_error",
        "fusion_l1_error",
        "ATTENTION DISTRIBUTION L1 ERROR",
    )

    print_context_stats(
        rows
    )

    print_support_bins(
        rows
    )


if __name__ == "__main__":
    main()
