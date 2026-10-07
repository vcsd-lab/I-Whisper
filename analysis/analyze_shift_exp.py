#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FRAC_BITS = [9, 10, 11, 12]


def shift_exp_scalar(x_sub_int: int, frac_bits: int) -> int:
    """
    Scalar reproduction of MultiHeadAttention_int.shift_exp()
    for one VALID attention entry.

    Important:
      model_int.py uses:
          qk_row_max = max(qk) + 1

      Therefore, for the maximum-score entry:
          x_sub_int = qk - qk_row_max = -1

      not zero.

    The final output is reduced to at most Q1.7 when frac_bits > 7.
    """

    if x_sub_int > 0:
        return 0

    x_mul_fix = (
        x_sub_int
        + (x_sub_int >> 1)
        - (x_sub_int >> 4)
    )

    x_mul_int = (
        x_mul_fix >> frac_bits
    )

    x_mul_frac = (
        x_mul_fix
        - (x_mul_int << frac_bits)
    )

    shift_amount = -x_mul_int

    if shift_amount < 0:
        return 0

    x_exp = (
        (
            (1 << frac_bits)
            + (x_mul_frac >> 1)
        )
        >> shift_amount
    )

    if frac_bits > 7:
        x_exp = (
            x_exp
            >> (frac_bits - 7)
        )

    if not (0 <= x_exp <= 255):
        raise RuntimeError(
            f"ShiftExp output outside uint8 range: "
            f"x_sub={x_sub_int}, frac_bits={frac_bits}, "
            f"x_exp={x_exp}"
        )

    return int(x_exp)


def build_transfer(frac_bits: int, xmin: float, xmax: float):
    """
    Sweep score differences relative to the true row maximum.

    delta = qk - max(qk) <= 0

    model_int.py internally uses:
        row_max = max(qk) + 1
        x_sub   = qk - row_max

    Hence:
        x_sub_int = delta_int - 1

    Both ideal and ShiftExp weights are normalized by the maximum-score
    entry so common row scaling is removed.
    """

    scale = 1 << frac_bits

    delta_int_min = int(np.floor(xmin * scale))
    delta_int_max = int(np.ceil(xmax * scale))

    if delta_int_max > 0:
        delta_int_max = 0

    delta_int = np.arange(
        delta_int_min,
        delta_int_max + 1,
        dtype=np.int64,
    )

    delta_real = (
        delta_int.astype(np.float64)
        / float(scale)
    )

    x_sub_int = delta_int - 1

    shift_code = np.asarray(
        [
            shift_exp_scalar(
                int(x),
                frac_bits,
            )
            for x in x_sub_int
        ],
        dtype=np.float64,
    )

    max_code = float(
        shift_exp_scalar(
            -1,
            frac_bits,
        )
    )

    if max_code <= 0:
        raise RuntimeError(
            f"Invalid ShiftExp max code for Q{frac_bits}: {max_code}"
        )

    shift_norm = (
        shift_code / max_code
    )

    exact_norm = np.exp(
        delta_real
    )

    # Decomposition helper:
    # x_mul_fix approximates x_sub * log2(e) by 23/16.
    # Here 2^x is still evaluated exactly in floating point.
    x_mul_fix = (
        x_sub_int
        + (x_sub_int >> 1)
        - (x_sub_int >> 4)
    )

    log2_arg = (
        x_mul_fix.astype(np.float64)
        / float(scale)
    )

    slope_exact2 = np.exp2(
        log2_arg
    )

    max_index = np.where(
        delta_int == 0
    )[0]

    if len(max_index) != 1:
        raise RuntimeError(
            "delta=0 is missing from transfer sweep."
        )

    max_index = int(
        max_index[0]
    )

    slope_exact2_norm = (
        slope_exact2
        / slope_exact2[max_index]
    )

    abs_error = np.abs(
        shift_norm - exact_norm
    )

    signed_error = (
        shift_norm - exact_norm
    )

    slope_abs_error = np.abs(
        slope_exact2_norm - exact_norm
    )

    shift_vs_slope_abs_error = np.abs(
        shift_norm - slope_exact2_norm
    )

    nonzero_mask = (
        shift_code > 0
    )

    zero_mask = ~nonzero_mask

    if np.any(nonzero_mask):
        most_negative_nonzero = float(
            delta_real[
                np.where(nonzero_mask)[0][0]
            ]
        )
    else:
        most_negative_nonzero = float("nan")

    if np.any(zero_mask):
        closest_zero = float(
            delta_real[
                np.where(zero_mask)[0][-1]
            ]
        )
    else:
        closest_zero = float("nan")

    # Relative error is not useful once the exact exponential is tiny.
    relevant = (
        exact_norm >= (1.0 / 128.0)
    )

    relative_error = np.zeros_like(
        exact_norm
    )

    relative_error[relevant] = (
        abs_error[relevant]
        / exact_norm[relevant]
    )

    return {
        "frac_bits": frac_bits,
        "delta_int": delta_int,
        "delta_real": delta_real,
        "x_sub_int": x_sub_int,
        "shift_code": shift_code,
        "max_code": max_code,
        "shift_norm": shift_norm,
        "exact_norm": exact_norm,
        "slope_exact2_norm": slope_exact2_norm,
        "signed_error": signed_error,
        "abs_error": abs_error,
        "slope_abs_error": slope_abs_error,
        "shift_vs_slope_abs_error": shift_vs_slope_abs_error,
        "relative_error": relative_error,
        "relevant_mask": relevant,
        "most_negative_nonzero": most_negative_nonzero,
        "closest_zero": closest_zero,
    }


def summarize_transfer(data):
    relevant = data["relevant_mask"]

    abs_error = data["abs_error"]
    signed_error = data["signed_error"]
    relative_error = data["relative_error"]
    slope_abs_error = data["slope_abs_error"]
    shift_vs_slope_abs_error = data["shift_vs_slope_abs_error"]

    return {
        "frac_bits":
            data["frac_bits"],

        "shift_max_code":
            int(data["max_code"]),

        "most_negative_nonzero_delta":
            data["most_negative_nonzero"],

        "closest_zero_delta":
            data["closest_zero"],

        "mean_abs_error":
            float(np.mean(abs_error)),

        "max_abs_error":
            float(np.max(abs_error)),

        "mean_signed_error":
            float(np.mean(signed_error)),

        "mean_abs_error_relevant":
            float(np.mean(
                abs_error[relevant]
            )),

        "max_abs_error_relevant":
            float(np.max(
                abs_error[relevant]
            )),

        "mean_relative_error_relevant":
            float(np.mean(
                relative_error[relevant]
            )),

        "p90_relative_error_relevant":
            float(np.percentile(
                relative_error[relevant],
                90,
            )),

        "mean_slope_only_abs_error_relevant":
            float(np.mean(
                slope_abs_error[relevant]
            )),

        "mean_fraction_quant_abs_error_relevant":
            float(np.mean(
                shift_vs_slope_abs_error[
                    relevant
                ]
            )),
    }


def write_summary_csv(
    summaries,
    path,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )


def write_transfer_csv(
    transfers,
    path,
):
    fields = [
        "frac_bits",
        "delta_real",
        "delta_int",
        "x_sub_int",
        "shift_code",
        "exact_norm",
        "shift_norm",
        "slope_exact2_norm",
        "signed_error",
        "abs_error",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for data in transfers:
            n = len(
                data["delta_real"]
            )

            for i in range(n):
                writer.writerow({
                    "frac_bits":
                        data["frac_bits"],

                    "delta_real":
                        float(
                            data["delta_real"][i]
                        ),

                    "delta_int":
                        int(
                            data["delta_int"][i]
                        ),

                    "x_sub_int":
                        int(
                            data["x_sub_int"][i]
                        ),

                    "shift_code":
                        int(
                            data["shift_code"][i]
                        ),

                    "exact_norm":
                        float(
                            data["exact_norm"][i]
                        ),

                    "shift_norm":
                        float(
                            data["shift_norm"][i]
                        ),

                    "slope_exact2_norm":
                        float(
                            data["slope_exact2_norm"][i]
                        ),

                    "signed_error":
                        float(
                            data["signed_error"][i]
                        ),

                    "abs_error":
                        float(
                            data["abs_error"][i]
                        ),
                })


def plot_transfer(
    transfers,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ref = transfers[0]

    ax.plot(
        ref["delta_real"],
        ref["exact_norm"],
        linewidth=2.4,
        label="Exact exp",
    )

    for data in transfers:
        ax.plot(
            data["delta_real"],
            data["shift_norm"],
            linewidth=1.5,
            label=(
                f"ShiftExp "
                f"(Q{data['frac_bits']})"
            ),
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Score Difference from Row Maximum"
    )

    ax.set_ylabel(
        "Normalized Exponential Weight"
    )

    ax.set_title(
        "ShiftExp Transfer Function"
    )

    ax.grid(
        True,
        alpha=0.25,
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


def plot_error(
    transfers,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for data in transfers:
        ax.plot(
            data["delta_real"],
            data["signed_error"],
            linewidth=1.5,
            label=(
                f"Q{data['frac_bits']}"
            ),
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Score Difference from Row Maximum"
    )

    ax.set_ylabel(
        "ShiftExp - Exact"
    )

    ax.set_title(
        "ShiftExp Signed Transfer Error"
    )

    ax.grid(
        True,
        alpha=0.25,
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


def plot_decomposition(
    data,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        data["delta_real"],
        data["exact_norm"],
        linewidth=2.4,
        label="Exact exp",
    )

    ax.plot(
        data["delta_real"],
        data["slope_exact2_norm"],
        linewidth=1.8,
        label="23/16 log2(e), exact 2^x",
    )

    ax.plot(
        data["delta_real"],
        data["shift_norm"],
        linewidth=1.8,
        label="Actual ShiftExp",
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Score Difference from Row Maximum"
    )

    ax.set_ylabel(
        "Normalized Exponential Weight"
    )

    ax.set_title(
        f"ShiftExp Error Decomposition "
        f"(Q{data['frac_bits']})"
    )

    ax.grid(
        True,
        alpha=0.25,
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


def print_summary(
    summaries,
):
    print()
    print("=" * 132)
    print("SHIFTEXP TRANSFER-FUNCTION AUDIT")
    print("=" * 132)

    print(
        f"{'Q':>4}"
        f"{'MaxCode':>10}"
        f"{'NonzeroMin':>14}"
        f"{'MeanAbs':>12}"
        f"{'MaxAbs':>12}"
        f"{'MeanSigned':>14}"
        f"{'RelErrMean':>14}"
        f"{'SlopeOnly':>14}"
        f"{'Frac+Quant':>14}"
    )

    for row in summaries:
        print(
            f"{row['frac_bits']:4d}"
            f"{row['shift_max_code']:10d}"
            f"{row['most_negative_nonzero_delta']:14.4f}"
            f"{row['mean_abs_error_relevant']:12.6f}"
            f"{row['max_abs_error_relevant']:12.6f}"
            f"{row['mean_signed_error']:14.6f}"
            f"{row['mean_relative_error_relevant']:14.6f}"
            f"{row['mean_slope_only_abs_error_relevant']:14.6f}"
            f"{row['mean_fraction_quant_abs_error_relevant']:14.6f}"
        )

    print()
    print("Notes:")
    print(
        "  NonzeroMin = most negative score difference whose "
        "actual ShiftExp code remains nonzero."
    )
    print(
        "  RelErrMean is evaluated only where exact exp >= 1/128."
    )
    print(
        "  SlopeOnly isolates the 23/16 approximation of log2(e)."
    )
    print(
        "  Frac+Quant measures the remaining gap from the "
        "fractional 2^x approximation and final output quantization."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit the integer ShiftExp transfer function used by "
            "I-Whisper decoder attention."
        )
    )

    parser.add_argument(
        "--frac-bits",
        type=int,
        nargs="+",
        default=DEFAULT_FRAC_BITS,
        help=(
            "QK fractional-bit settings to audit. "
            "Default: 9 10 11 12"
        ),
    )

    parser.add_argument(
        "--xmin",
        type=float,
        default=-10.0,
        help=(
            "Minimum score difference from row maximum. "
            "Default: -10"
        ),
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=0.0,
        help=(
            "Maximum score difference from row maximum. "
            "Must be <= 0. Default: 0"
        ),
    )

    parser.add_argument(
        "--representative-frac-bits",
        type=int,
        default=11,
        help=(
            "Fractional-bit setting used for the decomposition plot. "
            "Default: 11"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/figures/attention/shift_exp"
        ),
    )

    args = parser.parse_args()

    if args.xmax > 0:
        raise ValueError(
            "--xmax must be <= 0."
        )

    if args.xmin >= args.xmax:
        raise ValueError(
            "--xmin must be smaller than --xmax."
        )

    frac_bits_list = sorted(
        set(args.frac_bits)
    )

    if (
        args.representative_frac_bits
        not in frac_bits_list
    ):
        raise ValueError(
            "--representative-frac-bits must be included "
            "in --frac-bits."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transfers = [
        build_transfer(
            frac_bits=f,
            xmin=args.xmin,
            xmax=args.xmax,
        )
        for f in frac_bits_list
    ]

    summaries = [
        summarize_transfer(data)
        for data in transfers
    ]

    print_summary(
        summaries
    )

    summary_path = (
        args.output_dir
        / "shift_exp_summary.csv"
    )

    transfer_path = (
        args.output_dir
        / "shift_exp_transfer.csv"
    )

    transfer_plot_path = (
        args.output_dir
        / "shift_exp_transfer.png"
    )

    error_plot_path = (
        args.output_dir
        / "shift_exp_signed_error.png"
    )

    decomposition_plot_path = (
        args.output_dir
        / "shift_exp_decomposition.png"
    )

    write_summary_csv(
        summaries,
        summary_path,
    )

    write_transfer_csv(
        transfers,
        transfer_path,
    )

    plot_transfer(
        transfers,
        transfer_plot_path,
    )

    plot_error(
        transfers,
        error_plot_path,
    )

    representative = next(
        data
        for data in transfers
        if (
            data["frac_bits"]
            == args.representative_frac_bits
        )
    )

    plot_decomposition(
        representative,
        decomposition_plot_path,
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Summary       : {summary_path}"
    )
    print(
        f"Transfer CSV  : {transfer_path}"
    )
    print(
        f"Transfer plot : {transfer_plot_path}"
    )
    print(
        f"Error plot    : {error_plot_path}"
    )
    print(
        f"Decomposition : {decomposition_plot_path}"
    )


if __name__ == "__main__":
    main()
