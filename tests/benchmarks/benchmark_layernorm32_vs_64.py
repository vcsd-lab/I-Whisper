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
    rows: int,
    hidden_size: int,
    seed: int,
):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # INT15 activation stored in INT16.
    x = torch.randint(
        -16383,
        16384,
        (rows, hidden_size),
        dtype=torch.int16,
        generator=g,
    ).contiguous()

    weight = torch.randint(
        -127,
        128,
        (hidden_size,),
        dtype=torch.int8,
        generator=g,
    ).contiguous()

    bias = torch.randint(
        -20000,
        20001,
        (hidden_size,),
        dtype=torch.int32,
        generator=g,
    ).contiguous()

    ln_multiplier = torch.tensor(
        [96],
        dtype=torch.uint8,
    )

    ln_shift = torch.tensor(
        [9],
        dtype=torch.int8,
    )

    out_multiplier = torch.tensor(
        [0x60000000],
        dtype=torch.uint32,
    )

    out_shift = torch.tensor(
        [8],
        dtype=torch.int8,
    )

    return (
        x,
        weight,
        bias,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )


def time_call(fn):
    t0 = time.perf_counter_ns()
    output = fn()
    t1 = time.perf_counter_ns()

    # Keep output alive through timestamp.
    _ = output

    return t1 - t0


def benchmark_pair(
    fn32,
    fn64,
    warmup,
    repeat,
):
    for _ in range(warmup):
        fn32()
        fn64()

    times32 = []
    times64 = []

    # Alternate order to reduce thermal / DVFS bias.
    for i in range(repeat):
        if i % 2 == 0:
            times32.append(
                time_call(fn32)
            )
            times64.append(
                time_call(fn64)
            )
        else:
            times64.append(
                time_call(fn64)
            )
            times32.append(
                time_call(fn32)
            )

    return times32, times64


def summarize(times_ns):
    return {
        "median_us":
            statistics.median(times_ns) / 1e3,
        "mean_us":
            statistics.mean(times_ns) / 1e3,
        "min_us":
            min(times_ns) / 1e3,
        "max_us":
            max(times_ns) / 1e3,
    }


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    with path.open(
        "a",
        newline="",
    ) as f:
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
        "--rows",
        type=int,
        nargs="+",
        default=[
            1,
            16,
            64,
            256,
            1500,
        ],
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--label",
        type=str,
        default="rpi5",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=(
            "benchmark_layernorm32_vs_64.csv"
        ),
    )

    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    gc.disable()

    print("=" * 72)
    print(
        "I-Whisper LayerNorm32 vs LayerNorm64 SIMD Benchmark"
    )
    print("=" * 72)

    print(
        f"label         : {args.label}"
    )
    print(
        f"extension     : {_C.__file__}"
    )
    print(
        f"CPU affinity  : "
        f"{sorted(os.sched_getaffinity(0))}"
    )
    print(
        f"torch threads : "
        f"{torch.get_num_threads()}"
    )
    print(
        f"hidden size   : "
        f"{args.hidden_size}"
    )
    print(
        f"warmup        : {args.warmup}"
    )
    print(
        f"repeat        : {args.repeat}"
    )
    print()

    for rows in args.rows:
        (
            x,
            weight,
            bias,
            ln_multiplier,
            ln_shift,
            out_multiplier,
            out_shift,
        ) = make_case(
            rows=rows,
            hidden_size=args.hidden_size,
            seed=args.seed + rows,
        )

        stats32_fn = lambda: (
            _C.layernorm32_stats_simd(
                x
            )
        )

        stats64_fn = lambda: (
            _C.layernorm64_stats_simd(
                x
            )
        )
        
        mean32, var32, shift32 = stats32_fn()
        mean64, var64 = stats64_fn()

        if not torch.equal(mean32, mean64):
            raise RuntimeError(
                "LayerNorm32/64 mean mismatch"
            )
            
        stats32_ns, stats64_ns = benchmark_pair(
            stats32_fn,
            stats64_fn,
            args.warmup,
            args.repeat,
        )

        stats32 = summarize(stats32_ns)
        stats64 = summarize(stats64_ns)

        stats_speedup = (
            stats64["median_us"]
            / stats32["median_us"]
        )
        
        
        
        

        fn32 = lambda: (
            _C.layernorm32_simd(
                x,
                weight,
                bias,
                ln_multiplier,
                ln_shift,
                out_multiplier,
                out_shift,
            )
        )

        fn64 = lambda: (
            _C.layernorm64_simd(
                x,
                weight,
                bias,
                ln_multiplier,
                ln_shift,
                out_multiplier,
                out_shift,
            )
        )

        # Smoke test only.
        out32 = fn32()
        out64 = fn64()

        if out32.shape != x.shape:
            raise RuntimeError(
                "LayerNorm32 output shape mismatch"
            )

        if out64.shape != x.shape:
            raise RuntimeError(
                "LayerNorm64 output shape mismatch"
            )

        if out32.dtype != torch.int8:
            raise RuntimeError(
                "LayerNorm32 output dtype mismatch"
            )

        if out64.dtype != torch.int8:
            raise RuntimeError(
                "LayerNorm64 output dtype mismatch"
            )

        times32, times64 = (
            benchmark_pair(
                fn32,
                fn64,
                args.warmup,
                args.repeat,
            )
        )

        stat32 = summarize(times32)
        stat64 = summarize(times64)

        speedup = (
            stat64["median_us"]
            / stat32["median_us"]
        )

        ns_per_row32 = (
            stat32["median_us"]
            * 1000.0
            / rows
        )

        ns_per_row64 = (
            stat64["median_us"]
            * 1000.0
            / rows
        )

        raw32_ns, raw64_ns = (
            _C.layernorm_stats_raw_benchmark(
                x,
                args.warmup,
                args.repeat,
            )
        )

        raw32_us = raw32_ns / 1e3
        raw64_us = raw64_ns / 1e3

        raw_speedup = (
            raw64_us
            / raw32_us
        )



        

        
        

        print(
            f"rows={rows:4d} "
            f"H={args.hidden_size}"
        )

        print(
            f"  INT32 row-shift : "
            f"{stat32['median_us']:10.3f} us "
            f"median | "
            f"{ns_per_row32:10.3f} ns/row"
        )

        print(
            f"  INT64 baseline  : "
            f"{stat64['median_us']:10.3f} us "
            f"median | "
            f"{ns_per_row64:10.3f} ns/row"
        )

        print(
            f"  INT32 speedup   : "
            f"{speedup:8.3f}x"
        )

        print(
            f"  INT32 range     : "
            f"{stat32['min_us']:.3f} - "
            f"{stat32['max_us']:.3f} us"
        )

        print(
            f"  INT64 range     : "
            f"{stat64['min_us']:.3f} - "
            f"{stat64['max_us']:.3f} us"
        )

        print(
            f"  Stats INT32      : "
            f"{stats32['median_us']:10.3f} us median"
        )

        print(
            f"  Stats INT64      : "
            f"{stats64['median_us']:10.3f} us median"
        )

        print(
            f"  Stats INT32 speedup: "
            f"{stats_speedup:8.3f}x"
        )
        
        print(
            f"  Raw stats INT32  : "
            f"{raw32_us:10.3f} us"
        )

        print(
            f"  Raw stats INT64  : "
            f"{raw64_us:10.3f} us"
        )

        print(
            f"  Raw INT32 speedup: "
            f"{raw_speedup:8.3f}x"
        )
        print()
        t32_scalar, t64_scalar = benchmark_pair(
            lambda: _C.layernorm32_stats_scalar(
                x,
            ),
            lambda: _C.layernorm64_stats_scalar(
                x,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
    
        
        stats32 = summarize(t32_scalar)
        stats64 = summarize(t64_scalar)

        speedup = (
            stat64["median_us"]
            / stat32["median_us"]
        )

        ns_per_row32 = (
            stat32["median_us"]
            * 1000.0
            / rows
        )

        ns_per_row64 = (
            stat64["median_us"]
            * 1000.0
            / rows
        )

        raw32_ns, raw64_ns = (
            _C.layernorm_stats_raw_benchmark(
                x,
                args.warmup,
                args.repeat,
            )
        )

        raw32_us = raw32_ns / 1e3
        raw64_us = raw64_ns / 1e3

        raw_speedup = (
            raw64_us
            / raw32_us
        )
        print()
        print("Scalar")
        print(
            f"  Stats INT32      : "
            f"{stats32['median_us']:10.3f} us median"
        )

        print(
            f"  Stats INT64      : "
            f"{stats64['median_us']:10.3f} us median"
        )

        print(
            f"  Stats INT32 speedup: "
            f"{stats_speedup:8.3f}x"
        )
        

        row = {
            "label":
                args.label,
            "rows":
                rows,
            "hidden_size":
                args.hidden_size,
            "warmup":
                args.warmup,
            "repeat":
                args.repeat,

            "int32_median_us":
                stat32["median_us"],
            "int64_median_us":
                stat64["median_us"],

            "int32_mean_us":
                stat32["mean_us"],
            "int64_mean_us":
                stat64["mean_us"],

            "int32_ns_per_row":
                ns_per_row32,
            "int64_ns_per_row":
                ns_per_row64,

            "int32_speedup":
                speedup,
        }
        row.update(
            {
                "stats32_median_us":
                    stats32["median_us"],

                "stats64_median_us":
                    stats64["median_us"],

                "stats32_speedup":
                    stats_speedup,
            }
        )
        row.update(
            {
                "raw_stats32_us":
                    raw32_us,
                "raw_stats64_us":
                    raw64_us,
                "raw_stats32_speedup":
                    raw_speedup,
            }
        )
        append_csv(
            args.csv,
            row,
        )
        
        


if __name__ == "__main__":
    main()
