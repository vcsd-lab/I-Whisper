import pytest
import torch

from i_whisper.kernels import _C


def _u32_scalar(value: int):
    return torch.tensor(
        [value],
        dtype=torch.uint32,
    )


def _i8_scalar(value: int):
    return torch.tensor(
        [value],
        dtype=torch.int8,
    )


def _accum_scalar(x, w, b, stride, padding):
    return _C.conv1d_accum_scalar(
        x.contiguous(),
        w.contiguous(),
        None if b is None else b.contiguous(),
        stride,
        padding,
    )


def _accum_simd(x, w, b, stride, padding):
    return _C.conv1d_accum_simd(
        x.contiguous(),
        w.contiguous(),
        None if b is None else b.contiguous(),
        stride,
        padding,
    )


def _conv_scalar(
    x,
    w,
    b,
    multiplier,
    shift,
    stride,
    padding,
):
    return _C.conv1d_scalar(
        x.contiguous(),
        w.contiguous(),
        None if b is None else b.contiguous(),
        _u32_scalar(multiplier),
        _i8_scalar(shift),
        stride,
        padding,
    )


def _conv_simd(
    x,
    w,
    b,
    multiplier,
    shift,
    stride,
    padding,
):
    return _C.conv1d_simd(
        x.contiguous(),
        w.contiguous(),
        None if b is None else b.contiguous(),
        _u32_scalar(multiplier),
        _i8_scalar(shift),
        stride,
        padding,
    )


@pytest.mark.parametrize(
    "shape,stride,padding",
    [
        # B,Cin,Tin,Cout,K

        # Vector boundary / edge-padding cases.
        ((1, 1, 7, 2, 3), 1, 1),
        ((1, 2, 8, 3, 3), 1, 1),
        ((1, 3, 9, 4, 3), 1, 1),
        ((1, 4, 15, 5, 3), 1, 1),
        ((1, 7, 16, 9, 3), 1, 1),
        ((1, 7, 17, 9, 3), 1, 1),

        # stride-2 direct-deinterleave path.
        ((1, 2, 15, 3, 3), 2, 1),
        ((1, 3, 16, 4, 3), 2, 1),
        ((1, 7, 31, 9, 3), 2, 1),
        ((1, 7, 32, 9, 3), 2, 1),
        ((1, 7, 33, 9, 3), 2, 1),

        # Whisper-like channels with shorter T for unit-test cost.
        ((1, 80, 33, 64, 3), 1, 1),
        ((1, 384, 33, 64, 3), 2, 1),
    ],
)
@pytest.mark.parametrize(
    "with_bias",
    [False, True],
)
def test_conv1d_accum_simd_matches_scalar(
    shape,
    stride,
    padding,
    with_bias,
):
    B, Cin, Tin, Cout, K = shape

    torch.manual_seed(
        1000
        + B * 11
        + Cin * 13
        + Tin * 17
        + Cout * 19
        + K * 23
        + stride * 29
        + int(with_bias) * 31
    )

    x = torch.randint(
        -128,
        128,
        (B, Cin, Tin),
        dtype=torch.int8,
    )

    w = torch.randint(
        -128,
        128,
        (Cout, Cin * K),
        dtype=torch.int8,
    )

    b = None

    if with_bias:
        b = torch.randint(
            -(1 << 20),
            1 << 20,
            (Cout,),
            dtype=torch.int32,
        )

    ref = _accum_scalar(
        x,
        w,
        b,
        stride,
        padding,
    )

    out = _accum_simd(
        x,
        w,
        b,
        stride,
        padding,
    )

    assert out.dtype == torch.int32
    assert out.shape == ref.shape
    assert torch.equal(
        out,
        ref,
    ), "Conv1d SIMD accumulation mismatch"


@pytest.mark.parametrize(
    "shape,stride,padding",
    [
        ((1, 2, 17, 3, 3), 1, 1),
        ((1, 7, 33, 9, 3), 1, 1),
        ((1, 2, 31, 3, 3), 2, 1),
        ((1, 7, 33, 9, 3), 2, 1),
        ((1, 80, 33, 64, 3), 1, 1),
        ((1, 384, 33, 64, 3), 2, 1),
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
def test_conv1d_simd_matches_scalar(
    shape,
    stride,
    padding,
    multiplier,
    shift,
):
    B, Cin, Tin, Cout, K = shape

    torch.manual_seed(
        3000
        + B * 37
        + Cin * 41
        + Tin * 43
        + Cout * 47
        + multiplier % 997
        + shift
    )

    x = torch.randint(
        -128,
        128,
        (B, Cin, Tin),
        dtype=torch.int8,
    )

    w = torch.randint(
        -128,
        128,
        (Cout, Cin * K),
        dtype=torch.int8,
    )

    b = torch.randint(
        -(1 << 22),
        1 << 22,
        (Cout,),
        dtype=torch.int32,
    )

    ref = _conv_scalar(
        x,
        w,
        b,
        multiplier,
        shift,
        stride,
        padding,
    )

    out = _conv_simd(
        x,
        w,
        b,
        multiplier,
        shift,
        stride,
        padding,
    )

    assert out.dtype == torch.int8
    assert out.shape == ref.shape

    assert torch.equal(
        out,
        ref,
    ), "Conv1d fused SIMD mismatch"


def test_conv1d_simd_padding_boundaries_stride1():
    torch.manual_seed(5001)

    x = torch.randint(
        -128,
        128,
        (1, 3, 18),
        dtype=torch.int8,
    )

    w = torch.randint(
        -128,
        128,
        (5, 3 * 3),
        dtype=torch.int8,
    )

    b = torch.randint(
        -1000,
        1001,
        (5,),
        dtype=torch.int32,
    )

    assert torch.equal(
        _accum_simd(x, w, b, 1, 1),
        _accum_scalar(x, w, b, 1, 1),
    )


def test_conv1d_simd_padding_boundaries_stride2():
    torch.manual_seed(5002)

    x = torch.randint(
        -128,
        128,
        (1, 3, 35),
        dtype=torch.int8,
    )

    w = torch.randint(
        -128,
        128,
        (5, 3 * 3),
        dtype=torch.int8,
    )

    b = torch.randint(
        -1000,
        1001,
        (5,),
        dtype=torch.int32,
    )

    assert torch.equal(
        _accum_simd(x, w, b, 2, 1),
        _accum_scalar(x, w, b, 2, 1),
    )


def test_conv1d_simd_int32_wrap():
    # Force enough accumulation to exercise modulo-2^32 behavior.
    Cin = 70_000

    x = torch.full(
        (1, Cin, 8),
        127,
        dtype=torch.int8,
    )

    w = torch.full(
        (1, Cin),
        127,
        dtype=torch.int8,
    )

    b = torch.tensor(
        [1_500_000_000],
        dtype=torch.int32,
    )

    ref = _accum_scalar(
        x,
        w,
        b,
        1,
        0,
    )

    out = _accum_simd(
        x,
        w,
        b,
        1,
        0,
    )

    assert torch.equal(
        out,
        ref,
    )


def test_conv1d_simd_does_not_modify_inputs():
    torch.manual_seed(6001)

    x = torch.randint(
        -128,
        128,
        (1, 7, 33),
        dtype=torch.int8,
    )

    w = torch.randint(
        -128,
        128,
        (9, 7 * 3),
        dtype=torch.int8,
    )

    b = torch.randint(
        -10000,
        10001,
        (9,),
        dtype=torch.int32,
    )

    x_ref = x.clone()
    w_ref = w.clone()
    b_ref = b.clone()

    _conv_simd(
        x,
        w,
        b,
        0x60000000,
        7,
        1,
        1,
    )

    assert torch.equal(x, x_ref)
    assert torch.equal(w, w_ref)
    assert torch.equal(b, b_ref)
