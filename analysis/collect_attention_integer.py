#!/usr/bin/env python3

import argparse
import csv
import json
import types
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import i_whisper

from i_whisper.modules.functional import (
    Rshift,
    Fast_Inverse_Square_Root,
    Batch_Head_u8s8s32_GeMM,
)
from i_whisper.modules.linear import (
    set_linear8_backend,
    set_linear32_backend,
)
from i_whisper.model_int import (
    set_residual_backend,
    set_layernorm_backend,
    set_gelu_backend,
    set_attention_backend,
    set_conv1d_backend,
    set_audio_encoder_backend,
    set_text_decoder_backend,
)


REGRESSION10 = [
    f"LibriSpeech.test-other/test_other/1688/142285/"
    f"1688-142285-{i:04d}.flac"
    for i in range(10)
]


CSV_FIELDS = [
    "audio_id",
    "layer",
    "call_index",
    "batch_index",
    "head",
    "query_index",
    "context_len",
    "qk_frac_bits",
    "exp_frac_bits",
    "p_max",
    "effective_support",
    # Lost attention mass
    "naive_lost_mass",
    "exact_fusion_lost_mass",
    "fusion_lost_mass",
    # Distribution error
    "naive_l1_error",
    "exact_fusion_l1_error",
    "fusion_l1_error",
    "shift_exp_extra_l1",
    # Distribution-only weighted-sum error
    "naive_float_out_rel_l2",
    "exact_fusion_float_out_rel_l2",
    "fusion_float_out_rel_l2",
    # Matched integer attention-output error
    "naive_int_out_rel_l2",
    "fusion_int_out_rel_l2",
    # Extra error from integer GEMM + FISR normalization/rounding,
    # holding the actual ShiftExp weights fixed.
    "integer_normalization_extra_rel_l2",
]


class IntegerAttentionCollector:
    """
    A2.1 error-decomposition collector for decoder causal self-attention.

    The model source is not modified. Decoder self-attention modules are
    temporarily wrapped at runtime while using backend="check".

    The check backend already verifies, bit-exactly:
        SIMD QK/requant/rowmax == torch reference
        SIMD ShiftExp          == torch reference
        SIMD PV/norm/requant    == torch reference

    For the same actual INT16 QK scores, this collector compares:

      1) Naive direct probability quantization
           P_ref = softmax(QK / 2^f)
           P_naive_q = round(P_ref * 128)

      2) Exact-E representation at the same output precision as ShiftExp
           E_exact = exp((QK - actual_rowmax) / 2^f)
           E_exact_q = round(E_exact * 2^min(f,7))
           P_exact_E = E_exact_q / sum(E_exact_q)

      3) Actual I-Whisper ShiftExp
           E_shift = actual integer ShiftExp(QK)
           P_shift = E_shift / sum(E_shift)

    It also performs a matched integer-output comparison:

      Naive:
           uint8 Q1.7(P) x int8 V -> int32 -> rounded >> 7

      I-Whisper:
           uint8 ShiftExp x int8 V -> int32
           -> Fast_Inverse_Square_Root twice + rounded shifts

    Both are compared against P_ref @ V in V-integer units, before the
    final attention-output requantization and output Linear layer.
    """

    def __init__(self):
        self.audio_id = None
        self.call_count = {}
        self.active_call = {}
        self.shift_capture = {}
        self.rows = []
        self.restore_entries = []

    def start_audio(self, audio_id):
        self.audio_id = str(audio_id)
        self.call_count.clear()
        self.active_call.clear()
        self.shift_capture.clear()

    def wrap_layer(self, module, layer_idx):
        if module.backend != "check":
            raise RuntimeError(
                f"decoder layer {layer_idx}: backend must be 'check', "
                f"got '{module.backend}'"
            )

        original_shift_exp = module.shift_exp
        original_qkv_attention = module.qkv_attention
        collector = self

        def shift_exp_wrapper(this, qk, row_max, mask):
            # This calls the torch/reference ShiftExp. In backend="check",
            # model_int.py immediately compares these outputs bit-exactly
            # against the SIMD ShiftExp outputs.
            qk_exp, qk_exp_sum = original_shift_exp(
                qk,
                row_max,
                mask,
            )

            if layer_idx not in collector.active_call:
                raise RuntimeError(
                    f"layer {layer_idx}: shift_exp called without an "
                    "active qkv_attention call"
                )

            collector.shift_capture[layer_idx] = {
                "qk": qk.detach(),
                "qk_row_max": row_max.detach(),
                "qk_exp": qk_exp.detach(),
                "qk_exp_sum": qk_exp_sum.detach(),
                "mask": mask.detach() if mask is not None else None,
            }

            return qk_exp, qk_exp_sum

        def qkv_attention_wrapper(
            this,
            q,
            k,
            v,
            mask=None,
            kv_cache=None,
            is_cross_attention=False,
        ):
            if is_cross_attention:
                raise RuntimeError(
                    "A2.1 wrapper must only be installed on decoder "
                    "self-attention"
                )

            call_index = collector.call_count.get(layer_idx, 0)
            collector.call_count[layer_idx] = call_index + 1
            collector.active_call[layer_idx] = call_index

            # qkv_attention internally reshapes V, so keep the original
            # [B, K, C] integer tensor for analysis.
            v_original = v.detach()

            try:
                out = original_qkv_attention(
                    q,
                    k,
                    v,
                    mask,
                    kv_cache=kv_cache,
                    is_cross_attention=False,
                )

                if layer_idx not in collector.shift_capture:
                    raise RuntimeError(
                        f"layer {layer_idx}: no ShiftExp capture produced"
                    )

                capture = collector.shift_capture.pop(layer_idx)

                collector.collect(
                    module=this,
                    layer_idx=layer_idx,
                    call_index=call_index,
                    v=v_original,
                    capture=capture,
                )

                return out

            finally:
                collector.active_call.pop(layer_idx, None)

        module.shift_exp = types.MethodType(
            shift_exp_wrapper,
            module,
        )
        module.qkv_attention = types.MethodType(
            qkv_attention_wrapper,
            module,
        )

        self.restore_entries.append(
            (
                module,
                original_shift_exp,
                original_qkv_attention,
            )
        )

    def restore(self):
        for module, shift_exp, qkv_attention in self.restore_entries:
            module.shift_exp = shift_exp
            module.qkv_attention = qkv_attention

        self.restore_entries.clear()

    @staticmethod
    def _build_valid_mask(mask, shape, device):
        B, H, Q, K = shape

        if mask is None:
            return torch.ones(
                (B, H, Q, K),
                dtype=torch.bool,
                device=device,
            )

        masked = mask

        if masked.dtype == torch.bool:
            pass
        elif torch.is_floating_point(masked):
            masked = torch.isneginf(masked)
        else:
            masked = masked != 0

        if masked.ndim == 2:
            masked = masked.unsqueeze(0).unsqueeze(0)
        elif masked.ndim != 4:
            raise RuntimeError(
                f"Unsupported attention mask shape: {tuple(masked.shape)}"
            )

        masked = masked.expand(B, H, Q, K)
        return ~masked

    @torch.no_grad()
    def collect(
        self,
        module,
        layer_idx,
        call_index,
        v,
        capture,
    ):
        qk = capture["qk"]
        qk_row_max = capture["qk_row_max"]
        qk_exp = capture["qk_exp"]
        qk_exp_sum = capture["qk_exp_sum"]
        mask = capture["mask"]

        if qk.ndim != 4:
            raise RuntimeError(
                f"Expected QK [B,H,Q,K], got {tuple(qk.shape)}"
            )

        B, H, Q, K = qk.shape

        frac_bits = int(module.qk_frac_bits.item())
        if frac_bits < 0:
            raise RuntimeError(
                f"layer {layer_idx}: invalid qk_frac_bits={frac_bits}"
            )

        # shift_exp keeps min(qk_frac_bits, 7) fractional bits in its
        # uint8 output representation.
        exp_frac_bits = min(frac_bits, 7)
        exp_scale = float(2 ** exp_frac_bits)

        valid = self._build_valid_mask(
            mask=mask,
            shape=(B, H, Q, K),
            device=qk.device,
        )

        context_len = valid.sum(dim=-1)
        if torch.any(context_len == 0):
            raise RuntimeError(
                f"layer {layer_idx}: found attention row with zero valid keys"
            )

        # ------------------------------------------------------------
        # Reference probability distribution from the ACTUAL INT16 QK.
        # ------------------------------------------------------------
        score = qk.float() / float(2 ** frac_bits)
        score = score.masked_fill(~valid, float("-inf"))

        p_ref = torch.softmax(score, dim=-1)

        # ------------------------------------------------------------
        # Branch 1: naive direct Q1.7 probability quantization.
        # This matches the commented naive path in model_quant.py.
        # ------------------------------------------------------------
        p_naive_q = (
            p_ref.mul(128.0)
            .round()
            .clamp(0, 255)
        )
        p_naive = p_naive_q / 128.0

        # ------------------------------------------------------------
        # Branch 2: exact exponential representation using the SAME
        # integer QK, SAME actual row-max convention (max + 1 LSB),
        # and SAME output precision as actual ShiftExp.
        # ------------------------------------------------------------
        exact_exp_input = (
            qk.float() - qk_row_max.float()
        ) / float(2 ** frac_bits)
        exact_exp_input = exact_exp_input.masked_fill(
            ~valid,
            float("-inf"),
        )

        e_exact = torch.exp(exact_exp_input)
        e_exact_q = (
            e_exact.mul(exp_scale)
            .round()
            .clamp(0, 255)
        )
        e_exact_q = e_exact_q.masked_fill(~valid, 0.0)

        e_exact_q_sum = e_exact_q.sum(dim=-1, keepdim=True)
        if torch.any(e_exact_q_sum == 0):
            raise RuntimeError(
                f"layer {layer_idx}: exact-E quantized row has zero sum"
            )

        p_exact_fusion = e_exact_q / e_exact_q_sum

        # ------------------------------------------------------------
        # Branch 3: actual integer ShiftExp representation.
        # The check backend guarantees this captured reference output is
        # bit-exact to the SIMD ShiftExp output.
        # ------------------------------------------------------------
        if torch.any(qk_exp_sum <= 0):
            raise RuntimeError(
                f"layer {layer_idx}: actual ShiftExp row has non-positive sum"
            )

        p_fusion = qk_exp.float() / qk_exp_sum.float()

        # ------------------------------------------------------------
        # Lost FP32 attention mass.
        # ------------------------------------------------------------
        naive_zero = (p_naive_q == 0) & valid
        exact_fusion_zero = (e_exact_q == 0) & valid
        fusion_zero = (qk_exp == 0) & valid

        naive_lost_mass = (
            p_ref.masked_fill(~naive_zero, 0.0).sum(dim=-1)
        )
        exact_fusion_lost_mass = (
            p_ref.masked_fill(~exact_fusion_zero, 0.0).sum(dim=-1)
        )
        fusion_lost_mass = (
            p_ref.masked_fill(~fusion_zero, 0.0).sum(dim=-1)
        )

        # ------------------------------------------------------------
        # Attention-distribution error.
        # ------------------------------------------------------------
        naive_l1_error = (
            (p_ref - p_naive).abs().sum(dim=-1)
        )
        exact_fusion_l1_error = (
            (p_ref - p_exact_fusion).abs().sum(dim=-1)
        )
        fusion_l1_error = (
            (p_ref - p_fusion).abs().sum(dim=-1)
        )
        shift_exp_extra_l1 = (
            (p_exact_fusion - p_fusion).abs().sum(dim=-1)
        )

        # ------------------------------------------------------------
        # Prepare actual integer V.
        # qkv_attention receives V as [B, K, C].
        # ------------------------------------------------------------
        if v.ndim != 3:
            raise RuntimeError(
                f"Expected V [B,K,C], got {tuple(v.shape)}"
            )

        if v.shape[0] != B:
            raise RuntimeError(
                f"V batch {v.shape[0]} != QK batch {B}"
            )

        if v.shape[1] != K:
            raise RuntimeError(
                f"V key length {v.shape[1]} != QK key length {K}"
            )

        n_state = v.shape[2]
        if n_state % H != 0:
            raise RuntimeError(
                f"V state {n_state} is not divisible by heads {H}"
            )

        head_dim = n_state // H

        # [B,K,C] -> [B,H,K,D]
        v_bhkd = (
            v.view(B, K, H, head_dim)
            .permute(0, 2, 1, 3)
            .float()
        )

        # Reference weighted sum in V-integer units.
        out_ref = torch.matmul(p_ref, v_bhkd)

        # Distribution-only weighted-sum effects. These do NOT include
        # integer GEMM/normalization rounding.
        out_naive_float = torch.matmul(p_naive, v_bhkd)
        out_exact_fusion_float = torch.matmul(p_exact_fusion, v_bhkd)
        out_fusion_float = torch.matmul(p_fusion, v_bhkd)

        eps = 1e-12
        ref_norm = torch.linalg.vector_norm(
            out_ref,
            dim=-1,
        ).clamp_min(eps)

        naive_float_out_rel_l2 = (
            torch.linalg.vector_norm(
                out_naive_float - out_ref,
                dim=-1,
            ) / ref_norm
        )
        exact_fusion_float_out_rel_l2 = (
            torch.linalg.vector_norm(
                out_exact_fusion_float - out_ref,
                dim=-1,
            ) / ref_norm
        )
        fusion_float_out_rel_l2 = (
            torch.linalg.vector_norm(
                out_fusion_float - out_ref,
                dim=-1,
            ) / ref_norm
        )

        # ------------------------------------------------------------
        # Matched integer attention-output comparison.
        # GEMM expects V as [B,H,D,K].
        # ------------------------------------------------------------
        v_bhdk = (
            v.view(B, K, H, head_dim)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        # Naive integer path:
        #   Q1.7(P) @ V -> int32 -> rounded >> 7
        p_naive_q_u8 = p_naive_q.to(torch.uint8)
        naive_int_out = Batch_Head_u8s8s32_GeMM(
            p_naive_q_u8,
            v_bhdk,
        )
        naive_int_out = Rshift(
            naive_int_out,
            7,
        )

        # I-Whisper integer path before the final av_req requantization:
        #   ShiftExp @ V -> int32 -> reciprocal via FISR twice
        fusion_numerator = Batch_Head_u8s8s32_GeMM(
            qk_exp,
            v_bhdk,
        )

        sum_inv_mul = Fast_Inverse_Square_Root(
            qk_exp_sum.int(),
            module.rsqrt_offset,
            Newton_iter=2,
        )

        fusion_int_out = Rshift(
            fusion_numerator.mul(sum_inv_mul).int(),
            module.rsqrt_offset,
        )
        fusion_int_out = Rshift(
            fusion_int_out.mul(sum_inv_mul).int(),
            module.rsqrt_offset,
        )

        naive_int_out_rel_l2 = (
            torch.linalg.vector_norm(
                naive_int_out.float() - out_ref,
                dim=-1,
            ) / ref_norm
        )
        fusion_int_out_rel_l2 = (
            torch.linalg.vector_norm(
                fusion_int_out.float() - out_ref,
                dim=-1,
            ) / ref_norm
        )

        # Extra error caused by integer GEMM + FISR normalization +
        # rounded shifts while holding the actual ShiftExp weights fixed.
        fusion_float_norm = torch.linalg.vector_norm(
            out_fusion_float,
            dim=-1,
        ).clamp_min(eps)

        integer_normalization_extra_rel_l2 = (
            torch.linalg.vector_norm(
                fusion_int_out.float() - out_fusion_float,
                dim=-1,
            ) / fusion_float_norm
        )

        # ------------------------------------------------------------
        # Distribution descriptors.
        # ------------------------------------------------------------
        p_max = p_ref.max(dim=-1).values
        effective_support = (
            1.0
            / p_ref.square().sum(dim=-1).clamp_min(eps)
        )

        # ------------------------------------------------------------
        # Move only row-level summary tensors to CPU.
        # ------------------------------------------------------------
        summary = {
            "context_len": context_len.cpu(),
            "p_max": p_max.cpu(),
            "effective_support": effective_support.cpu(),
            "naive_lost_mass": naive_lost_mass.cpu(),
            "exact_fusion_lost_mass": exact_fusion_lost_mass.cpu(),
            "fusion_lost_mass": fusion_lost_mass.cpu(),
            "naive_l1_error": naive_l1_error.cpu(),
            "exact_fusion_l1_error": exact_fusion_l1_error.cpu(),
            "fusion_l1_error": fusion_l1_error.cpu(),
            "shift_exp_extra_l1": shift_exp_extra_l1.cpu(),
            "naive_float_out_rel_l2": naive_float_out_rel_l2.cpu(),
            "exact_fusion_float_out_rel_l2": exact_fusion_float_out_rel_l2.cpu(),
            "fusion_float_out_rel_l2": fusion_float_out_rel_l2.cpu(),
            "naive_int_out_rel_l2": naive_int_out_rel_l2.cpu(),
            "fusion_int_out_rel_l2": fusion_int_out_rel_l2.cpu(),
            "integer_normalization_extra_rel_l2": (
                integer_normalization_extra_rel_l2.cpu()
            ),
        }

        for b in range(B):
            for h in range(H):
                for q in range(Q):
                    idx = (b, h, q)

                    self.rows.append({
                        "audio_id": self.audio_id,
                        "layer": layer_idx,
                        "call_index": call_index,
                        "batch_index": b,
                        "head": h,
                        "query_index": q,
                        "context_len": int(summary["context_len"][idx]),
                        "qk_frac_bits": frac_bits,
                        "exp_frac_bits": exp_frac_bits,
                        "p_max": float(summary["p_max"][idx]),
                        "effective_support": float(
                            summary["effective_support"][idx]
                        ),
                        "naive_lost_mass": float(
                            summary["naive_lost_mass"][idx]
                        ),
                        "exact_fusion_lost_mass": float(
                            summary["exact_fusion_lost_mass"][idx]
                        ),
                        "fusion_lost_mass": float(
                            summary["fusion_lost_mass"][idx]
                        ),
                        "naive_l1_error": float(
                            summary["naive_l1_error"][idx]
                        ),
                        "exact_fusion_l1_error": float(
                            summary["exact_fusion_l1_error"][idx]
                        ),
                        "fusion_l1_error": float(
                            summary["fusion_l1_error"][idx]
                        ),
                        "shift_exp_extra_l1": float(
                            summary["shift_exp_extra_l1"][idx]
                        ),
                        "naive_float_out_rel_l2": float(
                            summary["naive_float_out_rel_l2"][idx]
                        ),
                        "exact_fusion_float_out_rel_l2": float(
                            summary["exact_fusion_float_out_rel_l2"][idx]
                        ),
                        "fusion_float_out_rel_l2": float(
                            summary["fusion_float_out_rel_l2"][idx]
                        ),
                        "naive_int_out_rel_l2": float(
                            summary["naive_int_out_rel_l2"][idx]
                        ),
                        "fusion_int_out_rel_l2": float(
                            summary["fusion_int_out_rel_l2"][idx]
                        ),
                        "integer_normalization_extra_rel_l2": float(
                            summary["integer_normalization_extra_rel_l2"][idx]
                        ),
                    })


def validate_audio_files(paths):
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing audio files:\n"
            + "\n".join(f"  {path}" for path in missing)
        )


def prepare_mel(audio_path, model):
    audio = i_whisper.load_audio(str(audio_path))
    audio = i_whisper.pad_or_trim(audio)

    mel = i_whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels,
    )

    return mel.to(model.device)


def save_csv(rows, path):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    x = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def print_stat_line(name, values):
    s = describe(values)

    print(
        f"  {name:<12} "
        f"mean={s['mean']:.6f} "
        f"median={s['median']:.6f} "
        f"P90={s['p90']:.6f} "
        f"P99={s['p99']:.6f}"
    )


def print_three_way(rows, names, title):
    print()
    print(title)

    for label, field in names:
        print_stat_line(
            label,
            [row[field] for row in rows],
        )


def print_pair(rows, first, second, title):
    first_label, first_field = first
    second_label, second_field = second

    first_values = np.asarray(
        [row[first_field] for row in rows],
        dtype=np.float64,
    )
    second_values = np.asarray(
        [row[second_field] for row in rows],
        dtype=np.float64,
    )

    print()
    print(title)
    print_stat_line(first_label, first_values)
    print_stat_line(second_label, second_values)

    print(
        f"  {second_label} better than {first_label}: "
        f"{np.mean(second_values < first_values) * 100:.2f}%"
    )


def print_qk_frac_bits(rows):
    per_layer = {}

    for row in rows:
        layer = row["layer"]
        frac_bits = row["qk_frac_bits"]
        per_layer.setdefault(layer, set()).add(frac_bits)

    print()
    print("QK fractional bits by decoder self-attention layer")

    for layer in sorted(per_layer):
        values = sorted(per_layer[layer])
        text = ", ".join(str(v) for v in values)
        print(f"  Layer {layer}: {text}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect A2.1 integer decoder self-attention error "
            "decomposition statistics."
        )
    )

    parser.add_argument(
        "--model",
        default="base",
        help="I-Whisper integer model name. Default: base",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device used by the integer model. Default: cpu. "
            "The SIMD/check kernels are expected to be available "
            "in the current environment."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root relative to which Regression-10 paths are resolved.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ATTENTION_DUMP/int_base_regression10_a21"
        ),
    )

    args = parser.parse_args()

    audio_paths = [
        args.dataset_root / path
        for path in REGRESSION10
    ]
    validate_audio_files(audio_paths)

    print(f"Device : {args.device}")
    print(f"Model  : {args.model}")
    print()
    print("Regression-10:")
    for path in audio_paths:
        print(f"  {path}")

    print()
    print("Loading integer-only I-Whisper...")

    model = i_whisper.load_int_model(
        args.model,
        device=args.device,
    )
    model.eval()

    # Match the deployment/integer test setup.
    set_linear8_backend(model, backend="simd")
    set_linear32_backend(model, backend="simd")
    set_residual_backend(model, backend="simd")
    set_layernorm_backend(model, backend="simd")
    set_gelu_backend(model, backend="simd")
    set_attention_backend(model, backend="simd")
    set_conv1d_backend(model, backend="simd")
    set_audio_encoder_backend(model, backend="simd")
    set_text_decoder_backend(model, backend="simd")

    # Only decoder causal self-attention uses CHECK.
    # Encoder attention and decoder cross-attention stay SIMD.
    for block in model.decoder.blocks:
        block.attn.backend = "check"

    collector = IntegerAttentionCollector()

    for layer_idx, block in enumerate(model.decoder.blocks):
        collector.wrap_layer(
            block.attn,
            layer_idx,
        )

    options = i_whisper.DecodingOptions(
        fp16=False,
        language="en",
    )

    decode_metadata = []

    try:
        loop = tqdm(
            audio_paths,
            desc="Collecting A2.1 integer attention",
        )

        for audio_path in loop:
            loop.set_postfix(file=audio_path.name)

            collector.start_audio(audio_path.name)
            mel = prepare_mel(audio_path, model)

            with torch.inference_mode():
                result = i_whisper.decode(
                    model,
                    mel,
                    options,
                )

            decode_metadata.append({
                "audio_id": audio_path.name,
                "decoded_tokens": len(result.tokens),
                "text": result.text,
            })

    finally:
        collector.restore()

    rows = collector.rows
    if not rows:
        raise RuntimeError("No attention rows collected.")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_path = args.output / "rows.csv"
    save_csv(rows, rows_path)

    context = np.asarray(
        [row["context_len"] for row in rows],
        dtype=np.int32,
    )

    print()
    print("=" * 96)
    print("INTEGER ATTENTION A2.1 ERROR DECOMPOSITION")
    print("=" * 96)
    print(f"Rows                 : {len(rows)}")
    print(
        f"Context length range : "
        f"{int(context.min())} .. {int(context.max())}"
    )

    print_qk_frac_bits(rows)

    print_three_way(
        rows,
        [
            ("Naive", "naive_lost_mass"),
            ("Exact-E", "exact_fusion_lost_mass"),
            ("ShiftExp", "fusion_lost_mass"),
        ],
        "Lost attention mass",
    )

    print_three_way(
        rows,
        [
            ("Naive", "naive_l1_error"),
            ("Exact-E", "exact_fusion_l1_error"),
            ("ShiftExp", "fusion_l1_error"),
        ],
        "Distribution L1 error",
    )

    print()
    print("ShiftExp extra distribution L1")
    print_stat_line(
        "ShiftExp",
        [row["shift_exp_extra_l1"] for row in rows],
    )

    print_pair(
        rows,
        ("Naive", "naive_int_out_rel_l2"),
        ("I-Whisper", "fusion_int_out_rel_l2"),
        "Matched integer output relative L2",
    )

    print()
    print("Integer normalization extra relative L2")
    print_stat_line(
        "I-Whisper",
        [
            row["integer_normalization_extra_rel_l2"]
            for row in rows
        ],
    )

    # A few additional paired percentages are useful for diagnosing
    # which stage overturns the representation-only advantage.
    naive_l1 = np.asarray(
        [row["naive_l1_error"] for row in rows],
        dtype=np.float64,
    )
    exact_l1 = np.asarray(
        [row["exact_fusion_l1_error"] for row in rows],
        dtype=np.float64,
    )
    shift_l1 = np.asarray(
        [row["fusion_l1_error"] for row in rows],
        dtype=np.float64,
    )

    print()
    print("Paired distribution comparison")
    print(
        f"  Exact-E better than Naive : "
        f"{np.mean(exact_l1 < naive_l1) * 100:.2f}%"
    )
    print(
        f"  ShiftExp better than Naive: "
        f"{np.mean(shift_l1 < naive_l1) * 100:.2f}%"
    )
    print(
        f"  ShiftExp worse than Exact-E: "
        f"{np.mean(shift_l1 > exact_l1) * 100:.2f}%"
    )

    qk_frac_bits_by_layer = {}
    for row in rows:
        key = str(row["layer"])
        qk_frac_bits_by_layer.setdefault(key, set()).add(
            int(row["qk_frac_bits"])
        )

    qk_frac_bits_by_layer = {
        layer: sorted(values)
        for layer, values in qk_frac_bits_by_layer.items()
    }

    metadata = {
        "experiment": "integer_attention_a2_1_error_decomposition",
        "model": args.model,
        "region": "decoder",
        "attention_type": "causal_self_attention",
        "dataset": "LibriSpeech test-other Regression-10",
        "num_audio": len(audio_paths),
        "audio_files": [str(path) for path in audio_paths],
        "num_rows": len(rows),
        "integer_attention_backend": "check",
        "check_semantics": (
            "Decoder self-attention SIMD QK/rowmax, ShiftExp, and "
            "PV-normalize-requant outputs are checked bit-exactly "
            "against the torch/reference implementation inside "
            "model_int.py. Encoder and decoder cross-attention remain SIMD."
        ),
        "qk_frac_bits_by_layer": qk_frac_bits_by_layer,
        "reference_distribution": (
            "softmax(actual INT16 QK / 2**qk_frac_bits) over valid keys"
        ),
        "naive_distribution": (
            "round(reference_probability * 128) / 128; no renormalization"
        ),
        "exact_fusion_distribution": (
            "exact exp((INT16 QK - actual ShiftExp rowmax) / "
            "2**qk_frac_bits), quantized with scale "
            "2**min(qk_frac_bits,7), then normalized"
        ),
        "actual_fusion_distribution": (
            "actual integer ShiftExp output divided by actual ShiftExp sum"
        ),
        "matched_naive_integer_output": (
            "uint8 Q1.7 probability GEMM with actual int8 V, followed by "
            "rounded right shift by 7"
        ),
        "matched_iwhisper_integer_output": (
            "actual ShiftExp uint8 GEMM with actual int8 V, followed by "
            "Fast_Inverse_Square_Root reciprocal normalization using the "
            "same two rounded shifts as model_int.py; measured before "
            "av_req final requantization"
        ),
        "integer_normalization_extra_rel_l2": (
            "relative output error between the integer GEMM+FISR+rounded-"
            "shift result and exact floating normalization of the same "
            "actual ShiftExp weights and same integer V"
        ),
        "decode_options": {
            "fp16": False,
            "language": "en",
        },
        "decode_results": decode_metadata,
    }

    metadata_path = args.output / "metadata.json"
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
    print(f"Rows     : {rows_path}")
    print(f"Metadata : {metadata_path}")


if __name__ == "__main__":
    main()
