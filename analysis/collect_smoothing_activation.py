#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

import i_whisper
from i_whisper.model_quant import (
    GELU_quantized,
    Linear_quantized,
    ResidualAttentionBlock_quantized,
)


REGRESSION10 = [
    f"LibriSpeech.test-other/test_other/1688/142285/"
    f"1688-142285-{i:04d}.flac"
    for i in range(10)
]


GELU_FCS_ALPHA = {
    "tiny": 0.5,
    "base": 0.3,
    "base.en": 0.3,
    "small": 0.2,
    "small.en": 0.7,
    "medium": 0.2,
    "large": 0.4,
}


class ActivationCollector:
    def __init__(self):
        self.current = {}

    def hook(self, name):
        def fn(module, inputs, output):
            x = output.detach().float().cpu()

            if x.ndim == 3 and x.shape[0] == 1:
                x = x.squeeze(0)

            self.current[name] = x.numpy()

        return fn

    def clear(self):
        self.current.clear()


def get_encoder_gelu_layers(model):
    layers = {}

    for i, block in enumerate(model.encoder.blocks):
        name = f"encoder.blocks.{i}.mlp.1"
        layers[name] = block.mlp[1]

    return layers


def prepare_mel_from_array(audio, model, device):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = i_whisper.pad_or_trim(audio)

    mel = i_whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels,
    )

    return mel.unsqueeze(0).to(device)


def prepare_mel_from_file(audio_path, model, device):
    audio = i_whisper.load_audio(
        str(audio_path)
    )

    audio = i_whisper.pad_or_trim(audio)

    mel = i_whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels,
    )

    return mel.unsqueeze(0).to(device)


@torch.no_grad()
def smooth_gelu_fcs(
    mlp0,
    gelu,
    mlp2,
    act_scales,
    alpha,
):
    """
    Same GeLU smoothing transformation used in quantize.py.
    """

    assert isinstance(
        gelu,
        GELU_quantized,
    )

    assert isinstance(
        mlp0,
        Linear_quantized,
    )

    assert isinstance(
        mlp2,
        Linear_quantized,
    )

    H = mlp0.out_features

    assert H == act_scales.numel()

    device = mlp2.weight.device
    dtype = mlp2.weight.dtype

    act_scales = act_scales.to(
        device=device,
        dtype=dtype,
    )

    weight_scales = (
        mlp2.weight
        .abs()
        .amax(dim=0)
        .clamp_min(1e-12)
    )

    scales = (
        act_scales.pow(alpha)
        / weight_scales.pow(1 - alpha)
    )

    scales = (
        scales
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # Keep exactly the same scale fake-quantization
    # behavior as quantize.py.
    scale_max = scales.max()

    scales.div_(
        scale_max / 255.0
    ).round_().mul_(
        scale_max / 255.0
    )

    mlp0.weight.div_(
        scales.view(-1, 1)
    )

    if mlp0.bias is not None:
        mlp0.bias.div_(scales)

    gelu.de_smooth_vec.mul_(scales)

    mlp2.weight.mul_(
        scales.view(1, -1)
    )

    # Keep model's smoothing metadata consistent.
    mlp2.running_max_c.div_(scales)

    return scales.detach().cpu()


def run_smoothing_calibration(
    model,
    dataset,
    device,
    num_samples,
    seed,
):
    """
    Observe per-channel activation maxima used to derive
    channel-wise smoothing factors.

    This uses the same deterministic random sampling policy
    as quantize.py.
    """

    rng = random.Random(seed)

    model.observe_channel_max()

    indices = []

    print()
    print("Smoothing calibration")
    print("=" * 72)

    loop = tqdm(
        range(num_samples),
        desc="Smooth calibration",
    )

    with torch.no_grad():
        for _ in loop:
            index = rng.randint(
                0,
                len(dataset["train"]) - 1,
            )

            indices.append(index)

            audio = dataset["train"][index][
                "audio"
            ]["array"]

            mel = prepare_mel_from_array(
                audio,
                model,
                device,
            )

            # Encoder-only evaluation:
            # sufficient for encoder MLP smoothing statistics.
            _ = model.encoder(mel)

    model.stop_observe_channel_max()

    return indices


def apply_encoder_gelu_smoothing(
    model,
    alpha,
):
    """
    Apply GeLU-to-MLP2 smoothing to every encoder
    Transformer block.

    Returns per-layer smoothing vectors.
    """

    smoothing_vectors = {}

    for i, block in enumerate(
        model.encoder.blocks
    ):
        assert isinstance(
            block,
            ResidualAttentionBlock_quantized,
        )

        mlp0 = block.mlp[0]
        gelu = block.mlp[1]
        mlp2 = block.mlp[2]

        act_scales = (
            mlp2.running_max_c
        )

        scales = smooth_gelu_fcs(
            mlp0=mlp0,
            gelu=gelu,
            mlp2=mlp2,
            act_scales=act_scales,
            alpha=alpha,
        )

        layer_name = (
            f"encoder.blocks.{i}.mlp.1"
        )

        smoothing_vectors[layer_name] = (
            scales.numpy()
        )

    return smoothing_vectors


def collect_regression10(
    model,
    audio_paths,
    collector,
    gelu_layers,
    device,
):
    per_layer = {
        name: []
        for name in gelu_layers
    }

    loop = tqdm(
        audio_paths,
        desc="Collect Regression-10",
    )

    with torch.inference_mode():
        for audio_path in loop:
            loop.set_postfix(
                file=audio_path.name
            )

            collector.clear()

            mel = prepare_mel_from_file(
                audio_path,
                model,
                device,
            )

            _ = model.encoder(mel)

            expected = set(
                gelu_layers.keys()
            )

            observed = set(
                collector.current.keys()
            )

            if expected != observed:
                raise RuntimeError(
                    "Hook mismatch.\n"
                    f"Missing: "
                    f"{sorted(expected - observed)}\n"
                    f"Extra: "
                    f"{sorted(observed - expected)}"
                )

            for name in gelu_layers:
                per_layer[name].append(
                    collector.current[name]
                )

    return per_layer


def save_activation_run(
    output_dir,
    per_layer,
    condition,
    model_name,
    audio_paths,
    extra_metadata=None,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    layer_metadata = {}

    print()
    print(
        f"Saving {condition} activations..."
    )

    for name, arrays in per_layer.items():

        values = np.stack(
            arrays,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        path = output_dir / f"{name}.npy"

        np.save(
            path,
            values,
            allow_pickle=False,
        )

        abs_values = np.abs(
            values.reshape(-1)
        )

        layer_metadata[name] = {
            "file": path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "num_values": int(
                values.size
            ),
            "p99_abs": float(
                np.percentile(
                    abs_values,
                    99.0,
                )
            ),
            "p99.9_abs": float(
                np.percentile(
                    abs_values,
                    99.9,
                )
            ),
            "max_abs": float(
                np.max(abs_values)
            ),
        }

        stats = layer_metadata[name]

        print(
            f"{name}: "
            f"P99={stats['p99_abs']:.6f}, "
            f"P99.9={stats['p99.9_abs']:.6f}, "
            f"Max={stats['max_abs']:.6f}"
        )

    metadata = {
        "experiment": (
            "iwhisper_gelu_smoothing_activation"
        ),
        "model": model_name,
        "condition": condition,
        "region": "encoder",
        "activation": "post_gelu",
        "statistic_domain": (
            "absolute_activation"
        ),
        "dataset": (
            "LibriSpeech test-other "
            "Regression-10"
        ),
        "num_audio": len(audio_paths),
        "audio_files": [
            str(p)
            for p in audio_paths
        ],
        "layers": layer_metadata,
    }

    if extra_metadata:
        metadata.update(
            extra_metadata
        )

    with (
        output_dir / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect I-Whisper encoder "
            "post-GeLU activations before "
            "and after GeLU smoothing."
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
        "--output-root",
        type=Path,
        default=Path(
            "ACTIVATION_DUMP/"
            "smoothing_base_regression10"
        ),
    )

    parser.add_argument(
        "--smooth-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if args.model not in GELU_FCS_ALPHA:
        raise RuntimeError(
            f"No GeLU smoothing alpha "
            f"for model '{args.model}'"
        )

    alpha = GELU_FCS_ALPHA[
        args.model
    ]

    print(
        f"Device        : {device}"
    )
    print(
        f"Model         : {args.model}"
    )
    print(
        f"GeLU alpha    : {alpha}"
    )
    print(
        f"Smooth samples: "
        f"{args.smooth_samples}"
    )
    print(
        f"Random seed   : {args.seed}"
    )

    # --------------------------------------------------------
    # Regression-10
    # --------------------------------------------------------

    audio_paths = [
        args.dataset_root / path
        for path in REGRESSION10
    ]

    missing = [
        p for p in audio_paths
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "\n".join(
                str(p)
                for p in missing
            )
        )

    # --------------------------------------------------------
    # Load quantizable FP model
    # --------------------------------------------------------

    print()
    print(
        "Loading I-Whisper quantizable "
        "FP model..."
    )

    model = (
        i_whisper.load_model_quantized(
            args.model,
            device=device,
            download_root=(
                args.download_root
            ),
            quantize=False,
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Smooth calibration dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading LibriSpeech test-other "
        "dataset for smoothing calibration..."
    )

    ds = load_dataset(
        "LibriSpeech.test-other/test_other"
    )

    calibration_indices = (
        run_smoothing_calibration(
            model=model,
            dataset=ds,
            device=device,
            num_samples=(
                args.smooth_samples
            ),
            seed=args.seed,
        )
    )

    # --------------------------------------------------------
    # Hook encoder GeLU
    # --------------------------------------------------------

    collector = ActivationCollector()

    gelu_layers = (
        get_encoder_gelu_layers(
            model
        )
    )

    handles = []

    for name, module in (
        gelu_layers.items()
    ):
        handles.append(
            module.register_forward_hook(
                collector.hook(name)
            )
        )

    try:

        # ====================================================
        # BEFORE smoothing
        # ====================================================

        print()
        print(
            "Collecting BEFORE smoothing..."
        )

        before = collect_regression10(
            model=model,
            audio_paths=audio_paths,
            collector=collector,
            gelu_layers=gelu_layers,
            device=device,
        )

        save_activation_run(
            output_dir=(
                args.output_root
                / "before"
            ),
            per_layer=before,
            condition="before_smoothing",
            model_name=args.model,
            audio_paths=audio_paths,
            extra_metadata={
                "smooth_alpha": alpha,
                "smooth_samples": (
                    args.smooth_samples
                ),
                "smooth_seed": (
                    args.seed
                ),
                "smooth_calibration_indices": (
                    calibration_indices
                ),
            },
        )

        # Free BEFORE arrays before collecting AFTER.
        del before

        # ====================================================
        # Apply smoothing
        # ====================================================

        print()
        print(
            "Applying GeLU smoothing..."
        )

        smoothing_vectors = (
            apply_encoder_gelu_smoothing(
                model=model,
                alpha=alpha,
            )
        )

        # ====================================================
        # AFTER smoothing
        # ====================================================

        print()
        print(
            "Collecting AFTER smoothing..."
        )

        after = collect_regression10(
            model=model,
            audio_paths=audio_paths,
            collector=collector,
            gelu_layers=gelu_layers,
            device=device,
        )

        save_activation_run(
            output_dir=(
                args.output_root
                / "after"
            ),
            per_layer=after,
            condition="after_smoothing",
            model_name=args.model,
            audio_paths=audio_paths,
            extra_metadata={
                "smooth_alpha": alpha,
                "smooth_samples": (
                    args.smooth_samples
                ),
                "smooth_seed": (
                    args.seed
                ),
                "smooth_calibration_indices": (
                    calibration_indices
                ),
            },
        )

        # ----------------------------------------------------
        # Save smoothing vectors separately
        # ----------------------------------------------------

        smooth_dir = (
            args.output_root
            / "smoothing_scales"
        )

        smooth_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for layer_name, scales in (
            smoothing_vectors.items()
        ):
            np.save(
                smooth_dir
                / f"{layer_name}.npy",
                scales,
                allow_pickle=False,
            )

    finally:
        for handle in handles:
            handle.remove()

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"Before : "
        f"{args.output_root / 'before'}"
    )
    print(
        f"After  : "
        f"{args.output_root / 'after'}"
    )
    print(
        f"Scales : "
        f"{args.output_root / 'smoothing_scales'}"
    )


if __name__ == "__main__":
    main()