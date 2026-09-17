import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import Rshift, int_bit_length


def _norm_params(hidden_size):
    norm_multiplier = {
        1280: [True, False, False],
        1024: [False, False, False],
        768:  [False, True, True],
        512:  [False, False, False],
        384:  [False, True, True],
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


def layernorm32_stats_ref(x):
    assert x.dtype == torch.int16

    hidden_size = x.shape[-1]
    norm_mul, norm_shift = _norm_params(
        hidden_size
    )

    mean = torch.sum(
        x.int(),
        dim=-1,
        keepdim=True,
    ).int()

    mean = (
        _mul_by_shift_multiplier(
            mean,
            norm_mul,
        )
        >> norm_shift
    )

    sub_square = (
        (x - mean).abs() ** 2
    )

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

    variance = torch.sum(
        Rshift(
            sub_square,
            renorm_shift,
        ),
        dim=-1,
        keepdim=True,
    ).int()

    variance = (
        _mul_by_shift_multiplier(
            variance,
            norm_mul,
        )
        >> norm_shift
    ).int()

    return (
        mean.reshape(-1),
        variance.reshape(-1),
        renorm_shift.reshape(-1),
    )


def _assert_stats_equal(x):
    ref_mean, ref_var, ref_shift = (
        layernorm32_stats_ref(x)
    )

    mean, var, shift = (
        _C.layernorm32_stats_scalar(x)
    )

    assert mean.dtype == torch.int32
    assert var.dtype == torch.int32
    assert shift.dtype == torch.int32

    assert mean.shape == ref_mean.shape
    assert var.shape == ref_var.shape
    assert shift.shape == ref_shift.shape

    assert torch.equal(mean, ref_mean)
    assert torch.equal(var, ref_var)
    assert torch.equal(shift, ref_shift)


@pytest.mark.parametrize(
    "hidden_size",
    [384, 512, 768, 1024, 1280],
)
def test_layernorm32_stats_scalar_zero(
    hidden_size,
):
    x = torch.zeros(
        (2, hidden_size),
        dtype=torch.int16,
    )
    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    [384, 512, 768, 1024, 1280],
)
def test_layernorm32_stats_scalar_constant(
    hidden_size,
):
    x = torch.full(
        (3, hidden_size),
        1234,
        dtype=torch.int16,
    )
    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    [384, 512, 768, 1024, 1280],
)
def test_layernorm32_stats_scalar_extreme_pattern(
    hidden_size,
):
    x = torch.empty(
        (2, hidden_size),
        dtype=torch.int16,
    )
    x[:, 0::2] = 16383
    x[:, 1::2] = -16383

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    [384, 512, 768, 1024, 1280],
)
def test_layernorm32_stats_scalar_random(
    hidden_size,
):
    torch.manual_seed(
        1000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (4, hidden_size),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


def test_layernorm32_stats_scalar_3d():
    torch.manual_seed(5678)

    x = torch.randint(
        -16383,
        16384,
        (2, 7, 512),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


def test_layernorm32_stats_scalar_shift_zero_semantics():
    x = torch.zeros(
        (1, 512),
        dtype=torch.int16,
    )

    ref_mean, ref_var, ref_shift = (
        layernorm32_stats_ref(x)
    )

    mean, var, shift = (
        _C.layernorm32_stats_scalar(x)
    )

    assert ref_shift.item() == 0
    assert torch.equal(mean, ref_mean)
    assert torch.equal(var, ref_var)
    assert torch.equal(shift, ref_shift)


@pytest.mark.parametrize(
    "hidden_size",
    [1, 383, 513, 2048],
)
def test_layernorm32_stats_scalar_rejects_unsupported_hidden_size(
    hidden_size,
):
    x = torch.zeros(
        (2, hidden_size),
        dtype=torch.int16,
    )

    with pytest.raises(RuntimeError):
        _C.layernorm32_stats_scalar(x)


def test_layernorm32_stats_scalar_rejects_wrong_dtype():
    x = torch.zeros(
        (2, 512),
        dtype=torch.int32,
    )

    with pytest.raises(RuntimeError):
        _C.layernorm32_stats_scalar(x)



@pytest.mark.parametrize(
    "shift",
    [1, 2, 7, 8, 9, 10, 12, 15, 31],
)
def test_rounding_rshift_int32_boundaries(shift):
    x = torch.tensor(
        [
            -2147483648,
            -2147483647,
            -2147483000,
            -1000000,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            1000000,
            2147483000,
            2147483646,
            2147483647,
        ],
        dtype=torch.int32,
    )

    ref = Rshift(x, shift)
    out = _C.rounding_rshift_debug(
        x,
        torch.tensor([shift], dtype=torch.uint32)
    )

    assert torch.equal(out, ref)