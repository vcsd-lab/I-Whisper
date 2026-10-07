#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CHECKPOINTS = [16, 32, 64, 128, 256, 448]


def per_step_stats(max_context, head_dim):
    context = np.arange(1, max_context + 1, dtype=np.int64)
    conventional = context.copy()
    reform = np.full_like(context, head_dim)
    ratio = conventional.astype(np.float64) / reform.astype(np.float64)
    saved = conventional - reform
    return {
        "context": context,
        "conventional": conventional,
        "reform": reform,
        "ratio": ratio,
        "saved": saved,
    }


def cumulative_stats(max_context, head_dim):
    seq_len = np.arange(1, max_context + 1, dtype=np.int64)
    conventional = seq_len * (seq_len + 1) // 2
    reform = seq_len * head_dim
    ratio = conventional.astype(np.float64) / reform.astype(np.float64)
    saved = conventional - reform
    return {
        "seq_len": seq_len,
        "conventional": conventional,
        "reform": reform,
        "ratio": ratio,
        "saved": saved,
    }


def checkpoint_rows(checkpoints, head_dim, num_heads, num_layers, bytes_per_attention_value):
    rows = []

    for s in checkpoints:
        step_direct = s
        step_reform = head_dim

        cumulative_direct = s * (s + 1) // 2
        cumulative_reform = s * head_dim

        total_direct = cumulative_direct * num_heads * num_layers
        total_reform = cumulative_reform * num_heads * num_layers

        normalized_p_bytes = (
            cumulative_direct
            * num_heads
            * num_layers
            * bytes_per_attention_value
        )

        rows.append({
            "sequence_length": s,
            "head_dim": head_dim,
            "per_step_direct_normalized_values": step_direct,
            "per_step_reform_normalized_values": step_reform,
            "per_step_direct_over_reform": step_direct / step_reform,
            "cumulative_direct_normalized_values_per_head": cumulative_direct,
            "cumulative_reform_normalized_values_per_head": cumulative_reform,
            "cumulative_direct_over_reform": cumulative_direct / cumulative_reform,
            "cumulative_direct_values_all_heads_layers": total_direct,
            "cumulative_reform_values_all_heads_layers": total_reform,
            "cumulative_values_saved_all_heads_layers": total_direct - total_reform,
            "direct_normalized_p_materialization_bytes": normalized_p_bytes,
        })

    return rows


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_per_step(stats, head_dim, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        stats["context"],
        stats["conventional"],
        linewidth=2.2,
        label="Direct Softmax",
    )

    ax.plot(
        stats["context"],
        stats["reform"],
        linewidth=2.2,
        label="Attention Reform",
    )

    ax.axvline(head_dim, linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Decoder Context Length K")
    ax.set_ylabel("Values Requiring Normalization per Head")
    ax.set_title("Per-Step Normalization Workload")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative(stats, head_dim, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        stats["seq_len"],
        stats["conventional"],
        linewidth=2.2,
        label="Direct Softmax",
    )

    ax.plot(
        stats["seq_len"],
        stats["reform"],
        linewidth=2.2,
        label="Attention Reform",
    )

    crossover = 2 * head_dim - 1
    ax.axvline(crossover, linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Decoded Sequence Length S")
    ax.set_ylabel("Cumulative Normalized Values per Head")
    ax.set_title("Causal Decoder Normalization Workload")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ratio(per_step, cumulative, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        per_step["context"],
        per_step["ratio"],
        linewidth=2.2,
        label="Per decoding step",
    )

    ax.plot(
        cumulative["seq_len"],
        cumulative["ratio"],
        linewidth=2.2,
        label="Cumulative causal sequence",
    )

    ax.axhline(1.0, linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Context / Sequence Length")
    ax.set_ylabel("Direct Softmax / Reform")
    ax.set_title("Normalization Workload Reduction")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_bytes(n):
    n = float(n)
    if n < 1024:
        return f"{n:.0f} B"
    if n < 1024**2:
        return f"{n / 1024:.2f} KiB"
    return f"{n / 1024**2:.2f} MiB"


def print_summary(rows, head_dim, num_heads, num_layers):
    print()
    print("=" * 132)
    print("ATTENTION REFORM STRUCTURAL BENEFIT")
    print("=" * 132)

    print(f"Head dimension       : {head_dim}")
    print(f"Decoder heads        : {num_heads}")
    print(f"Decoder layers       : {num_layers}")

    print()
    print("Per-step comparison:")
    print("  Direct Softmax normalizes K attention weights.")
    print("  Reform normalizes D output values after EV.")
    print(f"  Per-step crossover: K = D = {head_dim}.")

    print()
    print("Cumulative causal comparison:")
    print("  Direct Softmax: S(S+1)/2 normalized values per head.")
    print("  Reform        : S*D normalized values per head.")

    cumulative_crossover = 2 * head_dim - 1
    print(
        f"  Cumulative crossover: equal at S={cumulative_crossover}; "
        f"reform wins for S>{cumulative_crossover}."
    )

    print()
    print("-" * 132)

    print(
        f"{'S':>6}"
        f"{'Step Direct':>14}"
        f"{'Step Reform':>14}"
        f"{'Step Ratio':>13}"
        f"{'Causal Direct':>16}"
        f"{'Causal Reform':>16}"
        f"{'Causal Ratio':>14}"
        f"{'P Values':>14}"
    )

    for row in rows:
        print(
            f"{row['sequence_length']:6d}"
            f"{row['per_step_direct_normalized_values']:14d}"
            f"{row['per_step_reform_normalized_values']:14d}"
            f"{row['per_step_direct_over_reform']:13.3f}x"
            f"{row['cumulative_direct_normalized_values_per_head']:16d}"
            f"{row['cumulative_reform_normalized_values_per_head']:16d}"
            f"{row['cumulative_direct_over_reform']:14.3f}x"
            f"{format_bytes(row['direct_normalized_p_materialization_bytes']):>14}"
        )

    print()
    print("Interpretation:")
    print("  - Ratio > 1 means the reform applies normalization to fewer values.")
    print(
        "  - The K*D weighted-sum MAC count is unchanged; the saving is "
        "specifically in normalization workload and normalized-P materialization."
    )
    print(
        "  - 'P Values' is cumulative quantized normalized-P data over all "
        "decoder heads/layers, assuming 1 byte/value. It is NOT peak RAM usage."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quantify the structural benefit of moving attention "
            "normalization after the EV weighted sum."
        )
    )

    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--max-context", type=int, default=448)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
    )
    parser.add_argument("--bytes-per-attention-value", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/figures/attention/reform"),
    )

    args = parser.parse_args()

    if args.head_dim <= 0:
        raise ValueError("--head-dim must be positive.")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be positive.")
    if args.num_layers <= 0:
        raise ValueError("--num-layers must be positive.")
    if args.max_context <= 0:
        raise ValueError("--max-context must be positive.")

    checkpoints = sorted(set(args.checkpoints))
    for s in checkpoints:
        if s <= 0 or s > args.max_context:
            raise ValueError(
                f"Checkpoint {s} must be in [1, {args.max_context}]."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_step = per_step_stats(args.max_context, args.head_dim)
    cumulative = cumulative_stats(args.max_context, args.head_dim)

    rows = checkpoint_rows(
        checkpoints=checkpoints,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        bytes_per_attention_value=args.bytes_per_attention_value,
    )

    print_summary(
        rows=rows,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )

    stats_path = args.output_dir / "attention_reform_cost.csv"
    per_step_path = args.output_dir / "normalization_per_step.png"
    cumulative_path = args.output_dir / "normalization_cumulative.png"
    ratio_path = args.output_dir / "normalization_reduction_ratio.png"

    write_csv(rows, stats_path)
    plot_per_step(per_step, args.head_dim, per_step_path)
    plot_cumulative(cumulative, args.head_dim, cumulative_path)
    plot_ratio(per_step, cumulative, ratio_path)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Statistics : {stats_path}")
    print(f"Per-step  : {per_step_path}")
    print(f"Cumulative: {cumulative_path}")
    print(f"Ratio     : {ratio_path}")


if __name__ == "__main__":
    main()
