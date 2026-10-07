#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import i_whisper

from i_whisper.kernels import _C
from i_whisper.modules.functional import (
    Rshift,
    Batch_Head_u8s8s32_GeMM,
)


DEFAULT_CONTEXTS = [
    16,
    32,
    64,
    128,
    256,
    448,
]


def pin_cpu(cpu):
    if cpu is None:
        return

    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(
            "CPU affinity is not supported on this platform."
        )

    os.sched_setaffinity(
        0,
        {cpu},
    )


def percentile(values, q):
    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def summarize_samples(
    samples_us,
):
    return {
        "median_us":
            float(
                statistics.median(
                    samples_us
                )
            ),

        "mean_us":
            float(
                statistics.mean(
                    samples_us
                )
            ),

        "p10_us":
            percentile(
                samples_us,
                10,
            ),

        "p90_us":
            percentile(
                samples_us,
                90,
            ),

        "min_us":
            float(
                min(samples_us)
            ),

        "max_us":
            float(
                max(samples_us)
            ),
    }


def time_block(
    fn,
    inner,
):
    start = time.perf_counter_ns()

    for _ in range(inner):
        fn()

    end = time.perf_counter_ns()

    return (
        (end - start)
        / inner
        / 1000.0
    )


def benchmark_single(
    fn,
    warmup,
    repeats,
    inner,
):
    for _ in range(warmup):
        fn()

    samples = []

    gc_was_enabled = (
        gc.isenabled()
    )

    gc.disable()

    try:
        for _ in range(repeats):
            samples.append(
                time_block(
                    fn,
                    inner,
                )
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    return summarize_samples(
        samples
    )


def benchmark_pair(
    fn_a,
    fn_b,
    warmup,
    repeats,
    inner,
):
    """
    Alternate execution order to reduce bias from thermal drift.

    Returns:
        stats_a,
        stats_b
    """

    for _ in range(warmup):
        fn_a()
        fn_b()

    samples_a = []
    samples_b = []

    gc_was_enabled = (
        gc.isenabled()
    )

    gc.disable()

    try:
        for r in range(repeats):

            if r % 2 == 0:
                a = time_block(
                    fn_a,
                    inner,
                )

                b = time_block(
                    fn_b,
                    inner,
                )

            else:
                b = time_block(
                    fn_b,
                    inner,
                )

                a = time_block(
                    fn_a,
                    inner,
                )

            samples_a.append(a)
            samples_b.append(b)

    finally:
        if gc_was_enabled:
            gc.enable()

    return (
        summarize_samples(
            samples_a
        ),
        summarize_samples(
            samples_b
        ),
    )


def make_inputs(
    *,
    layer,
    context_len,
    n_head,
    head_dim,
    decoder_mask,
    seed,
):
    """
    Build deterministic synthetic tensors with the same shapes/dtypes
    as one autoregressive decoder self-attention step.

    QK:
        [1, H, 1, K] INT16

    V:
        [1, H, D, K] INT8

    Mask:
        [1, K], taken from the real decoder causal mask.

    The benchmark measures dense kernels whose execution cost is
    shape-dependent; synthetic score values avoid requiring a model
    decode trace while preserving the actual integer formats.
    """

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    frac_bits = int(
        layer.qk_frac_bits.item()
    )

    scale = float(
        2 ** frac_bits
    )

    # Typical bounded score distribution.
    #
    # Clamp before INT16 conversion so qk_row_max + 1 cannot overflow.
    qk_fp = (
        torch.randn(
            (
                1,
                n_head,
                1,
                context_len,
            ),
            generator=generator,
            dtype=torch.float32,
        )
        * 1.5
    )

    qk = (
        qk_fp
        .mul(scale)
        .round()
        .clamp(
            torch.iinfo(
                torch.int16
            ).min + 1,
            torch.iinfo(
                torch.int16
            ).max - 1,
        )
        .to(torch.int16)
        .contiguous()
    )

    qk_row_max = (
        qk.max(
            dim=-1,
            keepdim=True,
        )
        .values
        .to(torch.int32)
        .add(1)
        .clamp(
            max=torch.iinfo(
                torch.int16
            ).max
        )
        .to(torch.int16)
        .contiguous()
    )

    v = torch.randint(
        low=-127,
        high=128,
        size=(
            1,
            n_head,
            head_dim,
            context_len,
        ),
        generator=generator,
        dtype=torch.int8,
    ).contiguous()

    # For Q=1 at the newest decoder position, this is the actual
    # causal-mask row corresponding to context length K.
    attn_mask = (
        decoder_mask[
            context_len - 1
            :
            context_len,
            :
            context_len
        ]
        .cpu()
        .contiguous()
    )

    return (
        qk,
        qk_row_max,
        v,
        attn_mask,
    )


def make_naive_functions(
    *,
    qk,
    v,
    attn_mask,
    frac_bits,
    av_req_multiplier,
    av_req_shiftmount,
):
    """
    Baseline:

        INT16 QK
          -> dequantize to FP32
          -> apply causal mask
          -> F.softmax
          -> Q1.7 probability quantization
          -> uint8 x int8 WV
          -> >> 7
          -> output requantization
          -> INT8 output

    This intentionally mirrors the direct probability-quantization
    baseline previously present in model_quant.py.
    """

    scale = float(
        2 ** frac_bits
    )

    mask4 = (
        attn_mask
        .unsqueeze(0)
        .unsqueeze(0)
    )

    av_mul_i32 = (
        av_req_multiplier
        .int()
        .contiguous()
    )

    av_shift = (
        av_req_shiftmount
        .contiguous()
    )

    def softmax_quant():
        score = (
            qk.float()
            .div(scale)
        )

        score = (
            score + mask4
        )

        p = F.softmax(
            score,
            dim=-1,
        )

        p_q = (
            p
            .mul(128.0)
            .round()
            .clamp(
                0,
                255,
            )
            .to(torch.uint8)
        )

        return p_q

    # Fixed P for isolated WV/requant timing.
    p_q_fixed = (
        softmax_quant()
        .contiguous()
    )

    def wv_requant_from_fixed_p():
        out = (
            Batch_Head_u8s8s32_GeMM(
                p_q_fixed,
                v,
            )
        )

        out = Rshift(
            out,
            7,
        )

        out = Rshift(
            out
            .mul(
                av_mul_i32
            )
            .int(),
            av_shift,
        )

        out = (
            out
            .clamp(
                -127,
                127,
            )
            .to(torch.int8)
        )

        return (
            out
            .permute(
                0,
                2,
                1,
                3,
            )
            .flatten(
                start_dim=2
            )
        )

    def total():
        p_q = softmax_quant()

        out = (
            Batch_Head_u8s8s32_GeMM(
                p_q,
                v,
            )
        )

        out = Rshift(
            out,
            7,
        )

        out = Rshift(
            out
            .mul(
                av_mul_i32
            )
            .int(),
            av_shift,
        )

        out = (
            out
            .clamp(
                -127,
                127,
            )
            .to(torch.int8)
        )

        return (
            out
            .permute(
                0,
                2,
                1,
                3,
            )
            .flatten(
                start_dim=2
            )
        )

    return (
        softmax_quant,
        wv_requant_from_fixed_p,
        total,
    )


def make_iwhisper_functions(
    *,
    qk,
    qk_row_max,
    v,
    attn_mask,
    qk_frac_bits,
    av_req_multiplier,
    av_req_shiftmount,
    rsqrt_offset,
):
    """
    Production I-Whisper post-QK path:

        INT16 QK + fused row max
          -> shift_exp_simd
          -> UINT8 E + sum(E)
          -> pv_norm_requant_simd
          -> INT8 output
    """

    def shift_exp():
        return _C.shift_exp_simd(
            qk,
            qk_row_max,
            attn_mask,
            qk_frac_bits,
        )

    qk_exp_fixed, qk_exp_sum_fixed = (
        shift_exp()
    )

    qk_exp_fixed = (
        qk_exp_fixed.contiguous()
    )

    qk_exp_sum_fixed = (
        qk_exp_sum_fixed.contiguous()
    )

    def pv_norm_from_fixed_exp():
        return _C.pv_norm_requant_simd(
            qk_exp_fixed,
            v,
            qk_exp_sum_fixed,
            av_req_multiplier,
            av_req_shiftmount,
            rsqrt_offset,
        )

    def total():
        qk_exp, qk_exp_sum = (
            _C.shift_exp_simd(
                qk,
                qk_row_max,
                attn_mask,
                qk_frac_bits,
            )
        )

        return _C.pv_norm_requant_simd(
            qk_exp.contiguous(),
            v,
            qk_exp_sum.contiguous(),
            av_req_multiplier,
            av_req_shiftmount,
            rsqrt_offset,
        )

    return (
        shift_exp,
        pv_norm_from_fixed_exp,
        total,
    )


def write_csv(
    rows,
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
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def aggregate_context(
    detail_rows,
    context_len,
):
    rows = [
        row
        for row in detail_rows
        if (
            row["context_len"]
            == context_len
        )
    ]

    naive_sum = sum(
        row["naive_total_median_us"]
        for row in rows
    )

    iwhisper_sum = sum(
        row["iwhisper_total_median_us"]
        for row in rows
    )

    return {
        "context_len":
            context_len,

        "num_layers":
            len(rows),

        "naive_six_layer_post_qk_us":
            naive_sum,

        "iwhisper_six_layer_post_qk_us":
            iwhisper_sum,

        "speedup":
            (
                naive_sum
                / iwhisper_sum
            ),

        "naive_softmax_quant_sum_us":
            sum(
                row[
                    "naive_softmax_quant_median_us"
                ]
                for row in rows
            ),

        "naive_wv_requant_sum_us":
            sum(
                row[
                    "naive_wv_requant_median_us"
                ]
                for row in rows
            ),

        "iwhisper_shift_exp_sum_us":
            sum(
                row[
                    "iwhisper_shift_exp_median_us"
                ]
                for row in rows
            ),

        "iwhisper_pv_norm_sum_us":
            sum(
                row[
                    "iwhisper_pv_norm_median_us"
                ]
                for row in rows
            ),
    }


def print_detail_row(
    row,
):
    print(
        f"  Layer {row['layer']:d} "
        f"Q{row['qk_frac_bits']:d}: "
        f"Naive={row['naive_total_median_us']:.3f} us, "
        f"I-Whisper={row['iwhisper_total_median_us']:.3f} us, "
        f"speedup={row['speedup']:.3f}x"
    )


def print_summary(
    summary_rows,
):
    print()
    print("=" * 108)
    print("POST-QK ATTENTION LATENCY SUMMARY")
    print("=" * 108)

    print(
        f"{'K':>6}"
        f"{'Naive 6-layer':>20}"
        f"{'I-Whisper 6-layer':>22}"
        f"{'Speedup':>12}"
        f"{'Naive Softmax+Q':>20}"
        f"{'I-W ShiftExp':>18}"
    )

    for row in summary_rows:
        print(
            f"{row['context_len']:6d}"
            f"{row['naive_six_layer_post_qk_us']:19.3f} us"
            f"{row['iwhisper_six_layer_post_qk_us']:21.3f} us"
            f"{row['speedup']:11.3f}x"
            f"{row['naive_softmax_quant_sum_us']:19.3f} us"
            f"{row['iwhisper_shift_exp_sum_us']:17.3f} us"
        )


def plot_latency(
    summary_rows,
    output_path,
):
    x = [
        row["context_len"]
        for row in summary_rows
    ]

    naive = [
        row[
            "naive_six_layer_post_qk_us"
        ]
        for row in summary_rows
    ]

    iwhisper = [
        row[
            "iwhisper_six_layer_post_qk_us"
        ]
        for row in summary_rows
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        x,
        naive,
        marker="o",
        linewidth=2.0,
        label=(
            "Dequant + Softmax + Q1.7 + WV"
        ),
    )

    ax.plot(
        x,
        iwhisper,
        marker="o",
        linewidth=2.0,
        label=(
            "I-Whisper fused integer path"
        ),
    )

    ax.set_xlabel(
        "Decoder Context Length K"
    )

    ax.set_ylabel(
        "Post-QK Latency for 6 Decoder Layers (us)"
    )

    ax.set_title(
        "Post-QK Attention Latency"
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


def plot_speedup(
    summary_rows,
    output_path,
):
    x = [
        row["context_len"]
        for row in summary_rows
    ]

    speedup = [
        row["speedup"]
        for row in summary_rows
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        x,
        speedup,
        marker="o",
        linewidth=2.0,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.set_xlabel(
        "Decoder Context Length K"
    )

    ax.set_ylabel(
        "Speedup over Direct Softmax Baseline"
    )

    ax.set_title(
        "I-Whisper Post-QK Attention Speedup"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark direct dequantize-Softmax-Q1.7-WV "
            "against the I-Whisper SIMD post-QK attention path."
        )
    )

    parser.add_argument(
        "--model",
        default="base",
    )

    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=DEFAULT_CONTEXTS,
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=None,
        help=(
            "Optional Linux CPU ID for sched_setaffinity. "
            "You may instead use taskset externally."
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--inner",
        type=int,
        default=50,
        help=(
            "Invocations per timed block."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261006,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/figures/attention/benchmark"
        ),
    )

    args = parser.parse_args()

    pin_cpu(
        args.cpu
    )

    # Set before model execution.
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if not hasattr(
        _C,
        "shift_exp_simd",
    ):
        raise RuntimeError(
            "shift_exp_simd kernel is unavailable."
        )

    if not hasattr(
        _C,
        "pv_norm_requant_simd",
    ):
        raise RuntimeError(
            "pv_norm_requant_simd kernel is unavailable."
        )

    print()
    print(
        "Loading integer-only model..."
    )

    model = i_whisper.load_int_model(
        args.model,
        device="cpu",
    )

    model.eval()

    n_head = int(
        model.dims.n_text_head
    )

    n_state = int(
        model.dims.n_text_state
    )

    head_dim = (
        n_state // n_head
    )

    n_text_ctx = int(
        model.dims.n_text_ctx
    )

    contexts = sorted(
        set(args.contexts)
    )

    for k in contexts:
        if (
            k <= 0
            or k > n_text_ctx
        ):
            raise ValueError(
                f"Context {k} must be in "
                f"[1, {n_text_ctx}]."
            )

    decoder_mask = (
        model.decoder.mask.detach()
    )

    layers = [
        block.attn
        for block in model.decoder.blocks
    ]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 88)
    print("BENCHMARK CONFIGURATION")
    print("=" * 88)

    print(
        f"Machine       : {platform.machine()}"
    )

    print(
        f"Platform      : {platform.platform()}"
    )

    print(
        f"PyTorch       : {torch.__version__}"
    )

    print(
        f"Model         : {args.model}"
    )

    print(
        f"Decoder layers: {len(layers)}"
    )

    print(
        f"Heads         : {n_head}"
    )

    print(
        f"Head dimension: {head_dim}"
    )

    print(
        f"Contexts      : {contexts}"
    )

    print(
        f"Torch threads : {torch.get_num_threads()}"
    )

    if hasattr(
        os,
        "sched_getaffinity",
    ):
        print(
            f"CPU affinity  : "
            f"{sorted(os.sched_getaffinity(0))}"
        )

    print(
        f"Warmup        : {args.warmup}"
    )

    print(
        f"Repeats       : {args.repeats}"
    )

    print(
        f"Inner         : {args.inner}"
    )

    detail_rows = []

    with torch.inference_mode():

        for context_len in contexts:

            print()
            print(
                f"Context K={context_len}"
            )

            for layer_idx, layer in enumerate(
                layers
            ):

                (
                    qk,
                    qk_row_max,
                    v,
                    attn_mask,
                ) = make_inputs(
                    layer=layer,
                    context_len=context_len,
                    n_head=n_head,
                    head_dim=head_dim,
                    decoder_mask=decoder_mask,
                    seed=(
                        args.seed
                        + context_len * 100
                        + layer_idx
                    ),
                )

                (
                    naive_softmax_quant,
                    naive_wv_requant,
                    naive_total,
                ) = make_naive_functions(
                    qk=qk,
                    v=v,
                    attn_mask=attn_mask,
                    frac_bits=int(
                        layer.qk_frac_bits.item()
                    ),
                    av_req_multiplier=(
                        layer.av_req_multiplier
                    ),
                    av_req_shiftmount=(
                        layer.av_req_shiftmount
                    ),
                )

                (
                    iwhisper_shift_exp,
                    iwhisper_pv_norm,
                    iwhisper_total,
                ) = make_iwhisper_functions(
                    qk=qk,
                    qk_row_max=qk_row_max,
                    v=v,
                    attn_mask=attn_mask,
                    qk_frac_bits=(
                        layer.qk_frac_bits
                    ),
                    av_req_multiplier=(
                        layer.av_req_multiplier
                    ),
                    av_req_shiftmount=(
                        layer.av_req_shiftmount
                    ),
                    rsqrt_offset=(
                        layer.rsqrt_offset
                    ),
                )

                # One-time contract sanity check.
                naive_out = (
                    naive_total()
                )

                iwhisper_out = (
                    iwhisper_total()
                )

                if (
                    naive_out.dtype
                    != torch.int8
                ):
                    raise RuntimeError(
                        "Naive output is not INT8."
                    )

                if (
                    iwhisper_out.dtype
                    != torch.int8
                ):
                    raise RuntimeError(
                        "I-Whisper output is not INT8."
                    )

                if (
                    naive_out.shape
                    != iwhisper_out.shape
                ):
                    raise RuntimeError(
                        "Output shape mismatch: "
                        f"Naive {tuple(naive_out.shape)} "
                        f"vs I-Whisper "
                        f"{tuple(iwhisper_out.shape)}"
                    )

                # Total-path timing is paired and alternates
                # execution order to reduce thermal/order bias.
                (
                    naive_total_stats,
                    iwhisper_total_stats,
                ) = benchmark_pair(
                    naive_total,
                    iwhisper_total,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    inner=args.inner,
                )

                # Stage timings are diagnostic only.
                naive_sq_stats = (
                    benchmark_single(
                        naive_softmax_quant,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        inner=args.inner,
                    )
                )

                naive_wv_stats = (
                    benchmark_single(
                        naive_wv_requant,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        inner=args.inner,
                    )
                )

                iwhisper_se_stats = (
                    benchmark_single(
                        iwhisper_shift_exp,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        inner=args.inner,
                    )
                )

                iwhisper_pv_stats = (
                    benchmark_single(
                        iwhisper_pv_norm,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        inner=args.inner,
                    )
                )

                row = {
                    "context_len":
                        context_len,

                    "layer":
                        layer_idx,

                    "qk_frac_bits":
                        int(
                            layer.qk_frac_bits.item()
                        ),

                    "naive_total_median_us":
                        naive_total_stats[
                            "median_us"
                        ],

                    "naive_total_p10_us":
                        naive_total_stats[
                            "p10_us"
                        ],

                    "naive_total_p90_us":
                        naive_total_stats[
                            "p90_us"
                        ],

                    "iwhisper_total_median_us":
                        iwhisper_total_stats[
                            "median_us"
                        ],

                    "iwhisper_total_p10_us":
                        iwhisper_total_stats[
                            "p10_us"
                        ],

                    "iwhisper_total_p90_us":
                        iwhisper_total_stats[
                            "p90_us"
                        ],

                    "speedup":
                        (
                            naive_total_stats[
                                "median_us"
                            ]
                            /
                            iwhisper_total_stats[
                                "median_us"
                            ]
                        ),

                    "naive_softmax_quant_median_us":
                        naive_sq_stats[
                            "median_us"
                        ],

                    "naive_wv_requant_median_us":
                        naive_wv_stats[
                            "median_us"
                        ],

                    "iwhisper_shift_exp_median_us":
                        iwhisper_se_stats[
                            "median_us"
                        ],

                    "iwhisper_pv_norm_median_us":
                        iwhisper_pv_stats[
                            "median_us"
                        ],
                }

                detail_rows.append(
                    row
                )

                print_detail_row(
                    row
                )

    summary_rows = [
        aggregate_context(
            detail_rows,
            context_len,
        )
        for context_len in contexts
    ]

    print_summary(
        summary_rows
    )

    detail_path = (
        args.output_dir
        / "post_qk_latency_detail.csv"
    )

    summary_path = (
        args.output_dir
        / "post_qk_latency_summary.csv"
    )

    latency_plot_path = (
        args.output_dir
        / "post_qk_latency_vs_context.png"
    )

    speedup_plot_path = (
        args.output_dir
        / "post_qk_speedup_vs_context.png"
    )

    metadata_path = (
        args.output_dir
        / "metadata.json"
    )

    write_csv(
        detail_rows,
        detail_path,
    )

    write_csv(
        summary_rows,
        summary_path,
    )

    plot_latency(
        summary_rows,
        latency_plot_path,
    )

    plot_speedup(
        summary_rows,
        speedup_plot_path,
    )

    metadata = {
        "benchmark":
            "post_qk_attention",

        "model":
            args.model,

        "machine":
            platform.machine(),

        "platform":
            platform.platform(),

        "python":
            platform.python_version(),

        "torch":
            torch.__version__,

        "contexts":
            contexts,

        "decoder_layers":
            len(layers),

        "heads":
            n_head,

        "head_dim":
            head_dim,

        "torch_num_threads":
            torch.get_num_threads(),

        "cpu_affinity":
            (
                sorted(
                    os.sched_getaffinity(0)
                )
                if hasattr(
                    os,
                    "sched_getaffinity",
                )
                else None
            ),

        "warmup":
            args.warmup,

        "repeats":
            args.repeats,

        "inner":
            args.inner,

        "naive_path": (
            "INT16 QK -> FP32 dequant -> causal mask -> "
            "F.softmax -> Q1.7 uint8 -> "
            "Batch_Head_u8s8s32_GeMM -> >>7 -> "
            "output requant -> INT8"
        ),

        "iwhisper_path": (
            "INT16 QK + row max -> shift_exp_simd -> "
            "uint8 E + sum(E) -> pv_norm_requant_simd -> INT8"
        ),

        "scope_note": (
            "Timing starts after QK score generation. "
            "It excludes Q/K/V projections and QK GEMM. "
            "I-Whisper row max is assumed available from the "
            "production fused QK-requant-rowmax kernel."
        ),

        "baseline_note": (
            "The direct Softmax baseline uses PyTorch FP32 Softmax "
            "and the repository's CPU Batch_Head_u8s8s32_GeMM helper. "
            "Therefore measured speedup reflects both the attention "
            "reform and the optimized fused SIMD implementation; "
            "it must not be described as reform-only speedup."
        ),

        "input_note": (
            "Dense kernels are benchmarked with deterministic synthetic "
            "QK/V values using actual model layer quantization parameters. "
            "Tensor shapes and dtypes match one autoregressive decoder "
            "self-attention step."
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Detail   : {detail_path}"
    )

    print(
        f"Summary  : {summary_path}"
    )

    print(
        f"Latency  : {latency_plot_path}"
    )

    print(
        f"Speedup  : {speedup_plot_path}"
    )

    print(
        f"Metadata : {metadata_path}"
    )


if __name__ == "__main__":
    main()
