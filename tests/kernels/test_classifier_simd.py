import pytest
import torch

from i_whisper.kernels import _C


def _u32_scalar(value: int):
    return torch.tensor([value], dtype=torch.uint32)


def _i8_scalar(value: int):
    return torch.tensor([value], dtype=torch.int8)


def _accum_scalar(x, weight):
    return _C.classifier_accum_scalar(
        x.contiguous(),
        weight.contiguous(),
    )


def _accum_simd(x, weight):
    return _C.classifier_accum_simd(
        x.contiguous(),
        weight.contiguous(),
    )


def _classifier_scalar(x, weight, multiplier, shift):
    return _C.classifier_scalar(
        x.contiguous(),
        weight.contiguous(),
        _u32_scalar(multiplier),
        _i8_scalar(shift),
    )


def _classifier_simd(x, weight, multiplier, shift):
    return _C.classifier_simd(
        x.contiguous(),
        weight.contiguous(),
        _u32_scalar(multiplier),
        _i8_scalar(shift),
    )


@pytest.mark.parametrize(
    "shape",
    [
        # B,T,K,N
        (1, 1, 1, 1),
        (1, 1, 7, 3),
        (1, 1, 15, 4),
        (1, 1, 16, 5),
        (1, 2, 17, 7),
        (1, 3, 31, 9),
        (2, 5, 32, 17),
        (1, 4, 33, 33),

        # Whisper-like hidden sizes.
        (1, 1, 384, 257),
        (1, 4, 512, 257),
    ],
)
def test_classifier_accum_simd_matches_scalar(shape):
    B, T, K, N = shape

    torch.manual_seed(
        1000 + B * 11 + T * 13 + K * 17 + N * 19
    )

    x = torch.randint(
        -128, 128, (B, T, K), dtype=torch.int8
    )

    weight = torch.randint(
        -128, 128, (N, K), dtype=torch.int8
    )

    ref = _accum_scalar(x, weight)
    out = _accum_simd(x, weight)

    assert out.dtype == torch.int32
    assert out.shape == ref.shape
    assert torch.equal(out, ref), "classifier SIMD accumulation mismatch"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 7, 3),
        (1, 1, 15, 4),
        (1, 1, 16, 5),
        (1, 2, 17, 7),
        (1, 3, 31, 9),
        (2, 5, 32, 17),
        (1, 4, 33, 33),
        (1, 1, 384, 257),
        (1, 4, 512, 257),
    ],
)
@pytest.mark.parametrize(
    "multiplier,shift",
    [
        (0x00000001, 0),
        (0x40000000, 0),
        (0x7FFFFFFF, 1),
        (0x80000000, 7),
        (0xFFFFFFFF, 15),
        (0xFFFFFFFF, 31),
        (0xFFFFFFFF, 32),
        (0xFFFFFFFF, 64),
        (0x60000000, -3),
    ],
)
def test_classifier_simd_matches_scalar(
    shape,
    multiplier,
    shift,
):
    B, T, K, N = shape

    torch.manual_seed(
        3000
        + B * 23
        + T * 29
        + K * 31
        + N * 37
        + multiplier % 997
        + shift
    )

    x = torch.randint(
        -128, 128, (B, T, K), dtype=torch.int8
    )

    weight = torch.randint(
        -128, 128, (N, K), dtype=torch.int8
    )

    ref = _classifier_scalar(
        x,
        weight,
        multiplier,
        shift,
    )

    out = _classifier_simd(
        x,
        weight,
        multiplier,
        shift,
    )

    assert out.dtype == torch.int8
    assert out.shape == ref.shape
    assert torch.equal(out, ref), "classifier fused SIMD mismatch"


def test_classifier_simd_signed_extremes_and_n_tail():
    x = torch.tensor(
        [[[
            -128, -127, -1, 0,
            1, 126, 127, 127,
            -64, 64, -32, 32,
            -7, 7, -2, 2,
            1,
        ]]],
        dtype=torch.int8,
    )

    torch.manual_seed(4001)
    weight = torch.randint(
        -128,
        128,
        (7, 17),
        dtype=torch.int8,
    )

    assert torch.equal(
        _accum_simd(x, weight),
        _accum_scalar(x, weight),
    )

    assert torch.equal(
        _classifier_simd(x, weight, 0xFFFFFFFF, 0),
        _classifier_scalar(x, weight, 0xFFFFFFFF, 0),
    )


def test_classifier_accum_simd_int32_wrap():
    # 127*127*K exceeds INT32_MAX.
    K = 140_000

    x = torch.full(
        (1, 1, K),
        127,
        dtype=torch.int8,
    )

    weight = torch.full(
        (5, K),
        127,
        dtype=torch.int8,
    )

    ref = _accum_scalar(x, weight)
    out = _accum_simd(x, weight)

    assert torch.all(ref < 0)
    assert torch.equal(out, ref)


def test_classifier_simd_clamp():
    K = 512

    x = torch.full(
        (1, 1, K),
        127,
        dtype=torch.int8,
    )

    weight = torch.stack(
        [
            torch.full((K,), 127, dtype=torch.int8),
            torch.full((K,), -128, dtype=torch.int8),
            torch.full((K,), 127, dtype=torch.int8),
            torch.full((K,), -128, dtype=torch.int8),
            torch.full((K,), 127, dtype=torch.int8),
        ],
        dim=0,
    )

    ref = _classifier_scalar(
        x,
        weight,
        0xFFFFFFFF,
        0,
    )

    out = _classifier_simd(
        x,
        weight,
        0xFFFFFFFF,
        0,
    )

    assert torch.equal(out, ref)


def test_classifier_simd_does_not_modify_inputs():
    torch.manual_seed(6001)

    x = torch.randint(
        -128, 128, (2, 3, 31), dtype=torch.int8
    )

    weight = torch.randint(
        -128, 128, (33, 31), dtype=torch.int8
    )

    x_ref = x.clone()
    w_ref = weight.clone()

    _classifier_simd(
        x,
        weight,
        0x60000000,
        7,
    )

    assert torch.equal(x, x_ref)
    assert torch.equal(weight, w_ref)
