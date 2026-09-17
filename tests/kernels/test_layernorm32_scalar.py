import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import (
    Rshift,
    Fast_Inverse_Square_Root,
    int_bit_length,
)


INT15_MIN = -16383
INT15_MAX = 16383


def _norm_params(hidden_size):
    norm_multiplier = {
        1280: [True, False, False],
        1024: [False, False, False],
        768: [False, True, True],
        512: [False, False, False],
        384: [False, True, True],
    }

    norm_shiftmount = {
        1280: 11,
        1024: 10,
        768: 10,
        512: 9,
        384: 9,
    }

    return (
        norm_multiplier[hidden_size],
        norm_shiftmount[hidden_size],
    )


def _mul_by_shift_multiplier(x, multiplier):
    out = x.clone()

    if multiplier[0]:
        out += x >> 1

    if multiplier[1]:
        out += x >> 2

    if multiplier[2]:
        out += x >> 3

    return out


def layernorm32_ref(
    x,
    weight,
    bias,
    ln_multiplier,
    ln_shift,
    out_multiplier,
    out_shift,
):
    hidden_size = x.shape[-1]
    norm_mul, norm_shift = _norm_params(hidden_size)

    mean = torch.sum(
        x.int(),
        dim=-1,
        keepdim=True,
    ).int()

    mean = (
        _mul_by_shift_multiplier(mean, norm_mul)
        >> norm_shift
    )

    sub_square = (x - mean).abs() ** 2

    max_bit_length = int_bit_length(
        sub_square.max(
            dim=-1,
            keepdim=True,
        ).values
    )

    renorm_shift = (
        max_bit_length - 16
    ).clamp(min=0)

    renorm_shift = (
        renorm_shift + 1
    ) & ~1

    var = torch.sum(
        Rshift(
            sub_square,
            renorm_shift,
        ),
        dim=-1,
        keepdim=True,
    ).int()

    var = (
        _mul_by_shift_multiplier(
            var,
            norm_mul,
        )
        >> norm_shift
    ).int()

    invstd = Fast_Inverse_Square_Root(
        var,
        offset=13,
        Newton_iter=2,
    )

    out = (x - mean) * invstd

    out = Rshift(
        out * ln_multiplier,
        ln_shift
        + (renorm_shift >> 1),
    )

    out = out.clamp(
        INT15_MIN,
        INT15_MAX,
    )

    out = out * weight.int()

    if bias is not None:
        out = out + bias

    out = (
        out.long()
        * out_multiplier.long()
        >> 32
    ).int()

    out = Rshift(
        out,
        out_shift,
    )

    return out.clamp(
        -127,
        127,
    ).to(torch.int8)


def _assert_equal(
    x,
    weight,
    bias,
    ln_multiplier,
    ln_shift,
    out_multiplier,
    out_shift,
):
    expected = layernorm32_ref(
        x,
        weight,
        bias,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )

    actual = _C.layernorm32_scalar(
        x,
        weight,
        bias,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )

    assert actual.dtype == torch.int8
    assert actual.shape == x.shape
    assert torch.equal(actual, expected), f"expected: {expected}, actual: {actual}"


@pytest.mark.parametrize(
    "hidden_size",
    [384, 512, 768, 1024, 1280],
)
def test_layernorm32_scalar_random(hidden_size):
    torch.manual_seed(4000 + hidden_size)

    x = torch.randint(
        -16383,
        16384,
        (3, hidden_size),
        dtype=torch.int16,
    )

    weight = torch.randint(
        -127,
        128,
        (hidden_size,),
        dtype=torch.int8,
    )

    bias = torch.randint(
        -20000,
        20001,
        (hidden_size,),
        dtype=torch.int32,
    )

    _assert_equal(
        x,
        weight,
        bias,
        torch.tensor([73], dtype=torch.uint8),
        torch.tensor([5], dtype=torch.int8),
        torch.tensor([0x6FFFFFFF], dtype=torch.uint32),
        torch.tensor([3], dtype=torch.int8),
    )


def test_layernorm32_scalar_zero_input():
    hidden_size = 512

    x = torch.zeros(
        (2, hidden_size),
        dtype=torch.int16,
    )

    weight = torch.ones(
        hidden_size,
        dtype=torch.int8,
    )

    bias = torch.zeros(
        hidden_size,
        dtype=torch.int32,
    )

    _assert_equal(
        x,
        weight,
        bias,
        torch.tensor([1], dtype=torch.uint8),
        torch.tensor([0], dtype=torch.int8),
        torch.tensor([0x7FFFFFFF], dtype=torch.uint32),
        torch.tensor([0], dtype=torch.int8),
    )


def test_layernorm32_scalar_extreme_pattern():
    hidden_size = 512

    x = torch.empty(
        (2, hidden_size),
        dtype=torch.int16,
    )
    x[:, 0::2] = 16383
    x[:, 1::2] = -16383

    weight = torch.full(
        (hidden_size,),
        127,
        dtype=torch.int8,
    )

    bias = torch.zeros(
        hidden_size,
        dtype=torch.int32,
    )

    _assert_equal(
        x,
        weight,
        bias,
        torch.tensor([255], dtype=torch.uint8),
        torch.tensor([7], dtype=torch.int8),
        torch.tensor([0xFFFFFFFF], dtype=torch.uint32),
        torch.tensor([4], dtype=torch.int8),
    )


def test_layernorm32_scalar_no_bias():
    torch.manual_seed(9001)

    hidden_size = 384

    x = torch.randint(
        -12000,
        12001,
        (2, 5, hidden_size),
        dtype=torch.int16,
    )

    weight = torch.randint(
        -127,
        128,
        (hidden_size,),
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        weight,
        None,
        torch.tensor([31], dtype=torch.uint8),
        torch.tensor([3], dtype=torch.int8),
        torch.tensor([0x50000000], dtype=torch.uint32),
        torch.tensor([2], dtype=torch.int8),
    )
