#!/usr/bin/env python3
"""Standalone integer-only core audit for a serialized I-Whisper model.

Scope: Encoder + Decoder neural-network forward only. Audio preprocessing,
input quantization, and Whisper's token-decoding logic are outside this audit.
The final decoder dequantization is intentionally disabled during the check.
This tests PyTorch-dispatched tensor operations; inspect C++ kernels separately.
"""

import argparse
import logging

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import i_whisper
from i_whisper.model_int import (
    set_attention_backend,
    set_audio_encoder_backend,
    set_conv1d_backend,
    set_gelu_backend,
    set_layernorm_backend,
    set_residual_backend,
    set_text_decoder_backend,
)
from i_whisper.modules.linear import set_linear8_backend, set_linear32_backend


class IntegerOnlyAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operator_count = 0

    @staticmethod
    def _floating_tensors(value):
        flattened, _ = tree_flatten(value)
        return [
            (tuple(t.shape), str(t.dtype))
            for t in flattened
            if isinstance(t, torch.Tensor)
            and (t.is_floating_point() or t.is_complex())
        ]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        floating_inputs = self._floating_tensors((args, kwargs))
        if floating_inputs:
            raise RuntimeError(
                f"Floating input to {func}: {floating_inputs}"
            )

        output = func(*args, **kwargs)
        self.operator_count += 1

        floating_outputs = self._floating_tensors(output)
        if floating_outputs:
            raise RuntimeError(
                f"Floating output from {func}: {floating_outputs}"
            )
        return output


def select_simd_backends(model):
    # The production CPU inference backend, including INT32 LayerNorm.
    logger = logging.getLogger("iwhisper.integer_audit")
    for setter in (
        set_linear8_backend,
        set_linear32_backend,
        set_residual_backend,
        set_layernorm_backend,
        set_gelu_backend,
        set_attention_backend,
        set_conv1d_backend,
        set_audio_encoder_backend,
        set_text_decoder_backend,
    ):
        setter(model, backend="simd", logger=logger)


def prepare_input(model, audio_file):
    if audio_file is None:
        # Deterministic smoke input, without dependency on copyrighted audio.
        # Whisper's conv2 halves the number of time frames.
        frames = model.dims.n_audio_ctx * 2
        mel_int8 = torch.zeros(
            (1, model.dims.n_mels, frames), dtype=torch.int8
        )
        return mel_int8, "synthetic all-zero INT8 Mel"

    # Original Whisper floating-point frontend is intentionally outside audit.
    audio = i_whisper.load_audio(audio_file)
    audio = i_whisper.pad_or_trim(audio)
    mel = i_whisper.log_mel_spectrogram(
        audio, n_mels=model.dims.n_mels
    )
    mel_int8 = (
        (mel * model.encoder.input_alpha)
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
        .unsqueeze(0)
    )
    return mel_int8, audio_file


def main():
    parser = argparse.ArgumentParser(
        description="Check serialized I-Whisper Encoder/Decoder for floating tensor operations"
    )
    parser.add_argument(
        "--model", default="base",
        help="Model alias, model directory, or model.safetensors path",
    )
    parser.add_argument(
        "--audio", default=None,
        help="Optional real audio; otherwise use a deterministic INT8 Mel input",
    )
    args = parser.parse_args()

    model = i_whisper.load_int_model(args.model, device="cpu")
    select_simd_backends(model)
    mel_int8, source = prepare_input(model, args.audio)
    tokens = torch.tensor([[50258, 50259, 50359]], dtype=torch.long)

    previous_dequant_setting = model.decoder.dequant_need
    model.decoder.dequant_need = False
    try:
        with torch.no_grad(), IntegerOnlyAudit() as audit:
            features = model.encoder(mel_int8)
            logits = model.decoder(tokens, features)
    finally:
        model.decoder.dequant_need = previous_dequant_setting

    if features.dtype != torch.int8:
        raise AssertionError(f"Encoder dtype should be int8: {features.dtype}")
    if logits.dtype != torch.int8:
        raise AssertionError(f"Decoder dtype should be int8: {logits.dtype}")

    print(f"Model: {args.model}")
    print(f"Input: {source}, {tuple(mel_int8.shape)}, {mel_int8.dtype}")
    print(f"Encoder output: {tuple(features.shape)}, {features.dtype}")
    print(f"Decoder logits: {tuple(logits.shape)}, {logits.dtype}")
    print(f"PyTorch-dispatched operations inspected: {audit.operator_count}")
    print("Integer-only core audit: PASS")


if __name__ == "__main__":
    main()
