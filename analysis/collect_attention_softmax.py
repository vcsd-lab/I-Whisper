#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import i_whisper


REGRESSION10 = [
    f"LibriSpeech.test-other/test_other/1688/142285/"
    f"1688-142285-{i:04d}.flac"
    for i in range(10)
]


Q_FRAC_BITS = 7
Q_SCALE = float(2 ** Q_FRAC_BITS)
UINT8_MIN = 0
UINT8_MAX = 255


CSV_FIELDS = [
    "audio_id",
    "layer",
    "call_index",
    "batch_index",
    "head",
    "query_index",
    "query_len",
    "key_len",
    "context_len",
    "p_max",
    "effective_support",
    "naive_zero_fraction",
    "naive_lost_mass",
    "naive_l1_error",
    "naive_probability_sum",
    "fusion_zero_fraction",
    "fusion_lost_mass",
    "fusion_l1_error",
    "fusion_probability_sum",
]


class AttentionCollector:
    """
    Collect row-level decoder self-attention statistics.

    No model source code is modified.

    A1 compares:

        Naive:
            P = softmax(A)
            P_q = round(P * 128)
            P_hat = P_q / 128

        Fusion representation:
            E = exp(A - rowmax(A))
            E_q = round(E * 128)
            P_hat = E_q / sum(E_q)

    This experiment intentionally isolates representation
    quantization and does NOT include ShiftExp/FISR approximation.
    """

    def __init__(self):
        self.rows = []
        self.audio_id = None
        self.call_count = {}

    def start_audio(self, audio_id):
        self.audio_id = str(audio_id)
        self.call_count.clear()

    def hook(self, layer_idx):

        def fn(module, inputs, output):

            # MultiHeadAttention.forward() in model.py returns:
            #
            #   (
            #       self.out(wv),
            #       qk
            #   )
            #
            qk = (
                output[1]
                .detach()
                .float()
            )

            call_index = self.call_count.get(
                layer_idx,
                0,
            )

            self.call_count[layer_idx] = (
                call_index + 1
            )

            self.collect_qk(
                qk=qk,
                layer_idx=layer_idx,
                call_index=call_index,
            )

        return fn

    @torch.no_grad()
    def collect_qk(
        self,
        qk,
        layer_idx,
        call_index,
    ):
        """
        qk:
            [B, H, Q, K]

        qk already contains the causal mask from model.py.
        Masked locations are -inf.
        """

        if qk.ndim != 4:
            raise RuntimeError(
                f"Expected qk shape [B,H,Q,K], "
                f"got {tuple(qk.shape)}"
            )

        batch_size, n_head, query_len, key_len = (
            qk.shape
        )

        # ----------------------------------------------------
        # Valid attention positions
        # ----------------------------------------------------

        valid = torch.isfinite(qk)

        context_len = (
            valid.sum(dim=-1)
        )

        if torch.any(context_len == 0):
            raise RuntimeError(
                "Found attention row with no valid keys."
            )

        # ----------------------------------------------------
        # FP32 reference probability
        # ----------------------------------------------------

        p = torch.softmax(
            qk,
            dim=-1,
        )

        # ----------------------------------------------------
        # FP32 unnormalized exponential representation
        #
        # E_j = exp(A_j - max(A))
        #
        # Therefore max(E) = 1 for every valid row.
        # ----------------------------------------------------

        row_max = qk.max(
            dim=-1,
            keepdim=True,
        ).values

        e = torch.exp(
            qk - row_max
        )

        # ----------------------------------------------------
        # Naive Q1.7 probability quantization
        #
        # Matches the commented naive implementation in
        # model_quant.py:
        #
        #   softmax * 2**7
        #   round
        #   clamp(0,255)
        # ----------------------------------------------------

        p_q = (
            p.mul(Q_SCALE)
            .round()
            .clamp(
                UINT8_MIN,
                UINT8_MAX,
            )
        )

        p_naive = (
            p_q / Q_SCALE
        )

        # ----------------------------------------------------
        # Fusion representation-only Q1.7
        #
        # This is NOT yet the integer ShiftExp implementation.
        # It isolates the effect of quantizing E instead of P.
        # ----------------------------------------------------

        e_q = (
            e.mul(Q_SCALE)
            .round()
            .clamp(
                UINT8_MIN,
                UINT8_MAX,
            )
        )

        e_q_sum = e_q.sum(
            dim=-1,
            keepdim=True,
        )

        if torch.any(e_q_sum == 0):
            raise RuntimeError(
                "Quantized E row has zero sum."
            )

        p_fusion = (
            e_q / e_q_sum
        )

        # ----------------------------------------------------
        # Distribution properties
        # ----------------------------------------------------

        p_max = (
            p.max(
                dim=-1
            ).values
        )

        effective_support = (
            1.0
            / p.square()
            .sum(dim=-1)
            .clamp_min(1e-12)
        )

        # ----------------------------------------------------
        # Naive metrics
        # ----------------------------------------------------

        naive_zero = (
            (p_q == 0)
            & valid
        )

        naive_zero_fraction = (
            naive_zero
            .sum(dim=-1)
            .float()
            / context_len.float()
        )

        naive_lost_mass = (
            p.masked_fill(
                ~naive_zero,
                0.0,
            )
            .sum(dim=-1)
        )

        naive_l1_error = (
            (p - p_naive)
            .abs()
            .sum(dim=-1)
        )

        naive_probability_sum = (
            p_naive.sum(
                dim=-1
            )
        )

        # ----------------------------------------------------
        # Fusion metrics
        # ----------------------------------------------------

        fusion_zero = (
            (e_q == 0)
            & valid
        )

        fusion_zero_fraction = (
            fusion_zero
            .sum(dim=-1)
            .float()
            / context_len.float()
        )

        fusion_lost_mass = (
            p.masked_fill(
                ~fusion_zero,
                0.0,
            )
            .sum(dim=-1)
        )

        fusion_l1_error = (
            (p - p_fusion)
            .abs()
            .sum(dim=-1)
        )

        fusion_probability_sum = (
            p_fusion.sum(
                dim=-1
            )
        )

        # ----------------------------------------------------
        # Move only summary tensors to CPU.
        #
        # Do not dump the full attention matrices.
        # ----------------------------------------------------

        summary = {
            "context_len":
                context_len.cpu(),

            "p_max":
                p_max.cpu(),

            "effective_support":
                effective_support.cpu(),

            "naive_zero_fraction":
                naive_zero_fraction.cpu(),

            "naive_lost_mass":
                naive_lost_mass.cpu(),

            "naive_l1_error":
                naive_l1_error.cpu(),

            "naive_probability_sum":
                naive_probability_sum.cpu(),

            "fusion_zero_fraction":
                fusion_zero_fraction.cpu(),

            "fusion_lost_mass":
                fusion_lost_mass.cpu(),

            "fusion_l1_error":
                fusion_l1_error.cpu(),

            "fusion_probability_sum":
                fusion_probability_sum.cpu(),
        }

        # ----------------------------------------------------
        # Flatten [B,H,Q] into CSV rows.
        # ----------------------------------------------------

        for batch_idx in range(
            batch_size
        ):
            for head_idx in range(
                n_head
            ):
                for query_idx in range(
                    query_len
                ):

                    idx = (
                        batch_idx,
                        head_idx,
                        query_idx,
                    )

                    self.rows.append({
                        "audio_id":
                            self.audio_id,

                        "layer":
                            layer_idx,

                        "call_index":
                            call_index,

                        "batch_index":
                            batch_idx,

                        "head":
                            head_idx,

                        "query_index":
                            query_idx,

                        "query_len":
                            query_len,

                        "key_len":
                            key_len,

                        "context_len":
                            int(
                                summary[
                                    "context_len"
                                ][idx]
                            ),

                        "p_max":
                            float(
                                summary[
                                    "p_max"
                                ][idx]
                            ),

                        "effective_support":
                            float(
                                summary[
                                    "effective_support"
                                ][idx]
                            ),

                        "naive_zero_fraction":
                            float(
                                summary[
                                    "naive_zero_fraction"
                                ][idx]
                            ),

                        "naive_lost_mass":
                            float(
                                summary[
                                    "naive_lost_mass"
                                ][idx]
                            ),

                        "naive_l1_error":
                            float(
                                summary[
                                    "naive_l1_error"
                                ][idx]
                            ),

                        "naive_probability_sum":
                            float(
                                summary[
                                    "naive_probability_sum"
                                ][idx]
                            ),

                        "fusion_zero_fraction":
                            float(
                                summary[
                                    "fusion_zero_fraction"
                                ][idx]
                            ),

                        "fusion_lost_mass":
                            float(
                                summary[
                                    "fusion_lost_mass"
                                ][idx]
                            ),

                        "fusion_l1_error":
                            float(
                                summary[
                                    "fusion_l1_error"
                                ][idx]
                            ),

                        "fusion_probability_sum":
                            float(
                                summary[
                                    "fusion_probability_sum"
                                ][idx]
                            ),
                    })


def get_decoder_self_attention_layers(
    model,
):
    """
    Only collect decoder causal self-attention.

    Explicitly exclude decoder cross-attention.
    """

    layers = {}

    for i, block in enumerate(
        model.decoder.blocks
    ):
        name = (
            f"decoder.blocks.{i}.attn"
        )

        layers[name] = block.attn

    return layers


def validate_audio_files(paths):

    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        message = "\n".join(
            f"  {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Missing audio files:\n"
            + message
        )


def prepare_mel(
    audio_path,
    model,
    device,
):
    """
    Follow the same FP32 decoding path used by
    whisper_transcribe.py.
    """

    audio = i_whisper.load_audio(
        str(audio_path)
    )

    audio = torch.from_numpy(
        i_whisper.pad_or_trim(
            audio
        )
    )

    mel = i_whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels,
    )

    return mel.to(device)


def save_csv(
    rows,
    output_path,
):

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Collect decoder self-attention "
            "Softmax representation statistics."
        )
    )

    parser.add_argument(
        "--model",
        default="base",
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--download-root",
        default="CKPT",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ATTENTION_DUMP/"
            "fp32_base_regression10"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(
        f"Device : {device}"
    )

    print(
        f"Model  : {args.model}"
    )

    # --------------------------------------------------------
    # Regression-10
    # --------------------------------------------------------

    audio_paths = [
        args.dataset_root / path
        for path in REGRESSION10
    ]

    validate_audio_files(
        audio_paths
    )

    print()
    print("Regression-10:")

    for path in audio_paths:
        print(
            f"  {path}"
        )

    # --------------------------------------------------------
    # Load original FP32 Whisper
    # --------------------------------------------------------

    print()
    print(
        "Loading original FP32 Whisper..."
    )

    model = i_whisper.load_model(
        args.model,
        device=device,
        download_root=args.download_root,
    )

    model.eval()

    # --------------------------------------------------------
    # Decode configuration
    #
    # Match the existing FP32 WER evaluation path.
    # Explicit language avoids language-detection forwards.
    # --------------------------------------------------------

    options = i_whisper.DecodingOptions(
        fp16=False,
        language="en",
    )

    # --------------------------------------------------------
    # Hooks
    # --------------------------------------------------------

    collector = (
        AttentionCollector()
    )

    attention_layers = (
        get_decoder_self_attention_layers(
            model
        )
    )

    handles = []

    print()
    print("Hooked layers:")

    for layer_idx, (
        name,
        module,
    ) in enumerate(
        attention_layers.items()
    ):

        print(
            f"  {name}"
        )

        handles.append(
            module.register_forward_hook(
                collector.hook(
                    layer_idx
                )
            )
        )

    # --------------------------------------------------------
    # Decode Regression-10
    # --------------------------------------------------------

    decode_metadata = []

    try:

        loop = tqdm(
            audio_paths,
            desc=(
                "Collecting decoder "
                "self-attention"
            ),
        )

        for audio_path in loop:

            loop.set_postfix(
                file=audio_path.name
            )

            collector.start_audio(
                audio_path.name
            )

            mel = prepare_mel(
                audio_path=audio_path,
                model=model,
                device=device,
            )

            with torch.inference_mode():

                result = i_whisper.decode(
                    model,
                    mel,
                    options,
                )

            decode_metadata.append({
                "audio_id":
                    audio_path.name,

                "decoded_tokens":
                    len(result.tokens),

                "text":
                    result.text,
            })

    finally:

        for handle in handles:
            handle.remove()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        args.output
        / "rows.csv"
    )

    save_csv(
        collector.rows,
        csv_path,
    )

    # --------------------------------------------------------
    # Basic sanity summary
    # --------------------------------------------------------

    num_rows = len(
        collector.rows
    )

    context_lengths = np.asarray(
        [
            row["context_len"]
            for row in collector.rows
        ],
        dtype=np.int32,
    )

    naive_mass = np.asarray(
        [
            row["naive_lost_mass"]
            for row in collector.rows
        ],
        dtype=np.float64,
    )

    fusion_mass = np.asarray(
        [
            row["fusion_lost_mass"]
            for row in collector.rows
        ],
        dtype=np.float64,
    )

    naive_sum = np.asarray(
        [
            row[
                "naive_probability_sum"
            ]
            for row in collector.rows
        ],
        dtype=np.float64,
    )

    fusion_sum = np.asarray(
        [
            row[
                "fusion_probability_sum"
            ]
            for row in collector.rows
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 72)
    print("ATTENTION COLLECTION SUMMARY")
    print("=" * 72)

    print(
        f"Rows                  : "
        f"{num_rows}"
    )

    print(
        f"Context length range  : "
        f"{context_lengths.min()} "
        f".. "
        f"{context_lengths.max()}"
    )

    print(
        f"Naive lost mass mean  : "
        f"{naive_mass.mean():.6f}"
    )

    print(
        f"Fusion lost mass mean : "
        f"{fusion_mass.mean():.6f}"
    )

    print(
        f"Naive P sum mean      : "
        f"{naive_sum.mean():.6f}"
    )

    print(
        f"Fusion P sum mean     : "
        f"{fusion_sum.mean():.6f}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "experiment":
            "attention_softmax_representation",

        "model":
            args.model,

        "condition":
            "fp32_score_representation_probe",

        "region":
            "decoder",

        "attention_type":
            "causal_self_attention",

        "dataset":
            (
                "LibriSpeech test-other "
                "Regression-10"
            ),

        "num_audio":
            len(audio_paths),

        "audio_files":
            [
                str(path)
                for path in audio_paths
            ],

        "num_rows":
            num_rows,

        "quantization": {
            "fractional_bits":
                Q_FRAC_BITS,

            "scale":
                Q_SCALE,

            "naive": (
                "round(softmax(A) * 128), "
                "clamp [0,255]"
            ),

            "fusion": (
                "round(exp(A-rowmax(A)) * 128), "
                "clamp [0,255], "
                "then normalize"
            ),
        },

        "important_note": (
            "Representation-only A1 experiment. "
            "Fusion branch uses exact FP32 exp before "
            "Q1.7 quantization and therefore does not "
            "include integer ShiftExp or FISR approximation."
        ),

        "context_length_definition": (
            "Number of finite entries in each "
            "masked attention-score row."
        ),

        "decode_options": {
            "fp16": False,
            "language": "en",
        },

        "decode_results":
            decode_metadata,
    }

    metadata_path = (
        args.output
        / "metadata.json"
    )

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
        f"Rows     : {csv_path}"
    )

    print(
        f"Metadata : {metadata_path}"
    )


if __name__ == "__main__":
    main()