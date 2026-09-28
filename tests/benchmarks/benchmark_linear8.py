#!/usr/bin/env python3

import argparse
import csv
import gc
import os
import statistics
import time
from pathlib import Path

import torch

from i_whisper.kernels import _C


def make_case(
    M: int,
    N: int,
    K: int,
    has_bias: bool,
    asymmetric: bool,
    seed: int,
):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x = torch.randint(
        -127, 128,
        (M, K),
        dtype=torch.int8,
        generator=g,
    ).contiguous()

    weight = torch.randint(
        -127, 128,
        (N, K),
        dtype=torch.int8,
        generator=g,
    ).contiguous()

    bias = None
    if has_bias:
        bias = torch.randint(
            -1000, 1001,
            (N,),
            dtype=torch.int32,
            generator=g,
        ).contiguous()

    weight_zero = None
    if asymmetric:
        weight_zero = torch.randint(
            -4, 5,
            (N,),
            dtype=torch.int8,
            generator=g,
        ).contiguous()

    # Generate in int64 first because randint support for uint32
    # can vary between PyTorch versions.
    multiplier = torch.randint(
        1 << 29,
        1 << 31,
        (N,),
        dtype=torch.int64,
        generator=g,
    ).to(torch.uint32).contiguous()

    shift = torch.randint(
        5, 16,
        (N,),
        dtype=torch.int8,
        generator=g,
    ).contiguous()

    return x, weight, bias, weight_zero, multiplier, shift


def time_call(fn):
    t0 = time.perf_counter_ns()
    output = fn()
    t1 = time.perf_counter_ns()

    # Keep output alive until after timestamp.
    # This measures function latency, but not Tensor destruction.
    _ = output

    return t1 - t0


def benchmark_pair(scalar_fn, simd_fn, warmup: int, repeat: int):
    # Warm both implementations.
    for _ in range(warmup):
        scalar_fn()
        simd_fn()

    scalar_ns = []
    simd_ns = []

    # Alternate order to reduce bias from thermal/DVFS drift.
    for i in range(repeat):
        if i % 2 == 0:
            scalar_ns.append(time_call(scalar_fn))
            simd_ns.append(time_call(simd_fn))
        else:
            simd_ns.append(time_call(simd_fn))
            scalar_ns.append(time_call(scalar_fn))

    return scalar_ns, simd_ns


def summarize(times_ns):
    return {
        "median_us": statistics.median(times_ns) / 1e3,
        "mean_us": statistics.mean(times_ns) / 1e3,
        "min_us": min(times_ns) / 1e3,
        "max_us": max(times_ns) / 1e3,
    }


def gemm_gops(M, N, K, latency_us):
    # GEMM-equivalent operation count:
    # one multiply + one add = 2 operations.
    ops = 2 * M * N * K
    seconds = latency_us * 1e-6
    return ops / seconds / 1e9


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 16, 64, 256],
        help="List of M dimensions",
    )
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)

    parser.add_argument(
        "--label",
        type=str,
        default="unknown",
        help="Build label, e.g. dotprod or baseline",
    )

    parser.add_argument(
        "--no-bias",
        action="store_true",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Disable weight_zero correction",
    )

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--csv",
        type=str,
        default="benchmark_linear8.csv",
    )

    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    gc.disable()

    has_bias = not args.no_bias
    asymmetric = not args.symmetric

    print("=" * 72)
    print("I-Whisper Linear8 Benchmark")
    print("=" * 72)
    print(f"backend label : {args.label}")
    print(f"extension     : {_C.__file__}")
    print(f"CPU affinity  : {sorted(os.sched_getaffinity(0))}")
    print(f"torch threads : {torch.get_num_threads()}")
    print(f"N, K          : {args.n}, {args.k}")
    print(f"bias          : {has_bias}")
    print(f"asymmetric    : {asymmetric}")
    print(f"warmup        : {args.warmup}")
    print(f"repeat        : {args.repeat}")
    print()

    for M in args.m:
        tensors = make_case(
            M=M,
            N=args.n,
            K=args.k,
            has_bias=has_bias,
            asymmetric=asymmetric,
            seed=args.seed + M,
        )

        x, weight, bias, weight_zero, multiplier, shift = tensors

        scalar_fn = lambda: _C.linear8_scalar(
            x,
            weight,
            bias,
            weight_zero,
            multiplier,
            shift,
        )

        simd_fn = lambda: _C.linear8_simd(
            x,
            weight,
            bias,
            weight_zero,
            multiplier,
            shift,
        )

        # One correctness guard before timing.
        scalar_out = scalar_fn()
        simd_out = simd_fn()

        if not torch.equal(scalar_out, simd_out):
            raise RuntimeError(
                f"Scalar/SIMD mismatch for M={M}, N={args.n}, K={args.k}"
            )

        scalar_ns, simd_ns = benchmark_pair(
            scalar_fn,
            simd_fn,
            args.warmup,
            args.repeat,
        )

        scalar = summarize(scalar_ns)
        simd = summarize(simd_ns)

        speedup = scalar["median_us"] / simd["median_us"]

        scalar_gops = gemm_gops(
            M, args.n, args.k, scalar["median_us"]
        )
        simd_gops = gemm_gops(
            M, args.n, args.k, simd["median_us"]
        )

        print(
            f"M={M:4d} N={args.n:4d} K={args.k:4d}"
        )
        print(
            f"  Scalar : "
            f"{scalar['median_us']:10.3f} us median | "
            f"{scalar_gops:8.3f} GOPS"
        )
        print(
            f"  SIMD   : "
            f"{simd['median_us']:10.3f} us median | "
            f"{simd_gops:8.3f} GOPS"
        )
        print(
            f"  Speedup: {speedup:8.3f}x"
        )
        print(
            f"  Scalar range: "
            f"{scalar['min_us']:.3f} - {scalar['max_us']:.3f} us"
        )
        print(
            f"  SIMD range  : "
            f"{simd['min_us']:.3f} - {simd['max_us']:.3f} us"
        )
        print()

        row = {
            "label": args.label,
            "M": M,
            "N": args.n,
            "K": args.k,
            "bias": has_bias,
            "asymmetric": asymmetric,
            "repeat": args.repeat,
            "scalar_median_us": scalar["median_us"],
            "simd_median_us": simd["median_us"],
            "speedup": speedup,
            "scalar_gops": scalar_gops,
            "simd_gops": simd_gops,
        }

        append_csv(args.csv, row)


if __name__ == "__main__":
    main()
