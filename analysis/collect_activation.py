#!/usr/bin/env python3

import argparse
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


class ActivationCollector:
    """
    Collect raw post-GeLU activations from selected encoder MLP layers.
    """

    def __init__(self):
        self.current = {}

    def hook(self, name):
        def fn(module, inputs, output):
            x = output.detach().float().cpu()

            # Remove batch dimension:
            # [1, T, C] -> [T, C]
            if x.ndim == 3 and x.shape[0] == 1:
                x = x.squeeze(0)

            self.current[name] = x.numpy()

        return fn

    def clear(self):
        self.current.clear()


def get_encoder_gelu_layers(model):
    """
    Return the GeLU module of every Transformer MLP in the encoder.

    Expected names:
        encoder.blocks.0.mlp.1
        encoder.blocks.1.mlp.1
        ...
    """
    layers = {}

    for i, block in enumerate(model.encoder.blocks):
        name = f"encoder.blocks.{i}.mlp.1"
        layers[name] = block.mlp[1]

    return layers


def validate_audio_files(paths):
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        message = "\n".join(
            f"  {path}" for path in missing
        )

        raise FileNotFoundError(
            "Missing audio files:\n"
            + message
        )


def prepare_mel(audio_path, model, device):
    """
    Use the same preprocessing path as I-Whisper:
        load_audio
        -> pad_or_trim
        -> log_mel_spectrogram
    """
    audio = i_whisper.load_audio(str(audio_path))
    audio = i_whisper.pad_or_trim(audio)

    mel = i_whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels,
    )

    # AudioEncoder expects:
    #   [batch, n_mels, n_ctx]
    mel = mel.unsqueeze(0).to(device)

    return mel


def collect_one_audio(
    model,
    collector,
    audio_path,
    device,
):
    collector.clear()

    mel = prepare_mel(
        audio_path,
        model,
        device,
    )

    with torch.inference_mode():
        _ = model.encoder(mel)

    return collector.current


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect FP32 Whisper encoder post-GeLU activations."
        )
    )

    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name. Default: base",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "PyTorch device, e.g. cuda or cpu. "
            "Default: cuda if available, otherwise cpu."
        ),
    )

    parser.add_argument(
        "--download-root",
        default="CKPT",
        help="Whisper checkpoint directory.",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help=(
            "Repository/root directory relative to which "
            "Regression-10 paths are resolved."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ACTIVATION_DUMP/fp32_base_regression10"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    if args.device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(f"Device : {device}")
    print(f"Model  : {args.model}")

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    audio_paths = [
        args.dataset_root / path
        for path in REGRESSION10
    ]

    validate_audio_files(audio_paths)

    print()
    print("Regression-10:")

    for path in audio_paths:
        print(f"  {path}")

    # ------------------------------------------------------------
    # Load original FP32 Whisper
    # ------------------------------------------------------------

    print()
    print("Loading original FP32 Whisper...")

    model = i_whisper.load_model(
        args.model,
        device=device,
        download_root=args.download_root,
    )

    model.eval()

    # ------------------------------------------------------------
    # Register hooks
    # ------------------------------------------------------------

    collector = ActivationCollector()

    gelu_layers = get_encoder_gelu_layers(model)

    print()
    print("Hooked layers:")

    handles = []

    for name, module in gelu_layers.items():
        print(f"  {name}")

        handle = module.register_forward_hook(
            collector.hook(name)
        )

        handles.append(handle)

    # ------------------------------------------------------------
    # We keep one list per layer.
    #
    # Regression-10 is small enough for this first experiment.
    # Each audio contributes roughly [1500, 2048] values per
    # layer for Whisper-base.
    # ------------------------------------------------------------

    per_layer = {
        name: []
        for name in gelu_layers
    }

    # ------------------------------------------------------------
    # Collect
    # ------------------------------------------------------------

    try:
        loop = tqdm(
            audio_paths,
            desc="Collecting post-GeLU activations",
        )

        for audio_path in loop:
            loop.set_postfix(
                file=audio_path.name
            )

            activations = collect_one_audio(
                model,
                collector,
                audio_path,
                device,
            )

            expected = set(gelu_layers.keys())
            observed = set(activations.keys())

            if expected != observed:
                missing = expected - observed
                extra = observed - expected

                raise RuntimeError(
                    "Hook output mismatch.\n"
                    f"Missing: {sorted(missing)}\n"
                    f"Extra  : {sorted(extra)}"
                )

            for name in gelu_layers:
                per_layer[name].append(
                    activations[name]
                )

    finally:
        for handle in handles:
            handle.remove()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    layer_metadata = {}

    print()
    print("Saving raw activations...")

    for name, arrays in per_layer.items():

        # Every Regression-10 audio is pad_or_trim'ed to the
        # same Whisper input length, so encoder activation
        # shapes should match.
        values = np.stack(
            arrays,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        # Expected for Whisper-base:
        #
        #   [10, 1500, 2048]
        #
        # Keep the original dimensions instead of flattening here.
        # plot_activation.py is responsible for deciding how the
        # population is interpreted.
        output_path = (
            args.output
            / f"{name}.npy"
        )

        np.save(
            output_path,
            values,
            allow_pickle=False,
        )

        abs_values = np.abs(
            values.reshape(-1)
        )

        p99 = float(
            np.percentile(abs_values, 99.0)
        )

        p999 = float(
            np.percentile(abs_values, 99.9)
        )

        max_value = float(
            np.max(abs_values)
        )

        layer_metadata[name] = {
            "file": output_path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "num_values": int(values.size),
            "p99_abs": p99,
            "p99.9_abs": p999,
            "max_abs": max_value,
        }

        print(
            f"{name}: "
            f"shape={values.shape}, "
            f"P99={p99:.6f}, "
            f"P99.9={p999:.6f}, "
            f"Max={max_value:.6f}"
        )

        # Release this layer before moving to the next one.
        del values
        del abs_values

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    metadata = {
        "experiment": "fp32_post_gelu_activation",
        "model": args.model,
        "condition": "fp32",
        "region": "encoder",
        "activation": "post_gelu",
        "statistic_domain": "absolute_activation",
        "dataset": "LibriSpeech test-other Regression-10",
        "num_audio": len(audio_paths),
        "audio_files": [
            str(path)
            for path in audio_paths
        ],
        "layers": layer_metadata,
    }

    metadata_path = (
        args.output / "metadata.json"
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
    print(f"Output   : {args.output}")
    print(f"Metadata : {metadata_path}")


if __name__ == "__main__":
    main()