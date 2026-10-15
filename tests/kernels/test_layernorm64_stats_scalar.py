import pytest
import torch

from i_whisper.kernels import _C


HIDDEN_SIZES = [
    384,
    512,
    768,
    1024,
    1280,
]


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


def _mul_by_shift_multiplier(
    x,
    multiplier,
):
    out = x.clone()

    if multiplier[0]:
        out += x >> 1

    if multiplier[1]:
        out += x >> 2

    if multiplier[2]:
        out += x >> 3

    return out


def layernorm64_stats_ref(x):
    assert x.dtype == torch.int16

    hidden_size = x.shape[-1]

    norm_mul, norm_shift = (
        _norm_params(hidden_size)
    )

    # Keep mean semantics identical to LayerNorm32.
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

    # LayerNorm64-specific path begins here.
    centered64 = (
        x.to(torch.int64)
        - mean.to(torch.int64)
    )

    square64 = (
        centered64
        * centered64
    )

    var_sum64 = torch.sum(
        square64,
        dim=-1,
        keepdim=True,
    )

    variance64 = (
        _mul_by_shift_multiplier(
            var_sum64,
            norm_mul,
        )
        >> norm_shift
    )

    return (
        mean.reshape(-1),
        variance64.reshape(-1),
    )


def _assert_stats_equal(x):
    ref_mean, ref_var = (
        layernorm64_stats_ref(x)
    )

    mean, var = (
        _C.layernorm64_stats_scalar(
            x.contiguous()
        )
    )

    assert mean.dtype == torch.int32
    assert var.dtype == torch.int64

    assert mean.shape == ref_mean.shape
    assert var.shape == ref_var.shape

    assert torch.equal(
        mean,
        ref_mean,
    )

    assert torch.equal(
        var,
        ref_var,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_scalar_zero(
    hidden_size,
):
    x = torch.zeros(
        (2, hidden_size),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_scalar_constant(
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
    HIDDEN_SIZES,
)
def test_layernorm64_stats_scalar_extreme_pattern(
    hidden_size,
):
    x = torch.empty(
        (2, hidden_size),
        dtype=torch.int16,
    )

    x[:, 0::2] = 16383
    x[:, 1::2] = -16383

    # Explicitly prove this test exercises a sum-of-squares
    # that cannot be represented by a signed INT32 accumulator.
    ref_mean, _ = layernorm64_stats_ref(x)

    mean = ref_mean.reshape(
        x.shape[:-1] + (1,)
    )

    centered64 = (
        x.to(torch.int64)
        - mean.to(torch.int64)
    )

    raw_sum64 = torch.sum(
        centered64 * centered64,
        dim=-1,
    )

    assert torch.all(
        raw_sum64
        > torch.iinfo(torch.int32).max
    )

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_scalar_random(
    hidden_size,
):
    torch.manual_seed(
        14000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (7, hidden_size),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_scalar_small_dynamic_range(
    hidden_size,
):
    torch.manual_seed(
        15000 + hidden_size
    )

    x = torch.randint(
        -32,
        33,
        (9, hidden_size),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


def test_layernorm64_stats_scalar_3d():
    torch.manual_seed(16000)

    x = torch.randint(
        -16383,
        16384,
        (2, 7, 512),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    [1, 383, 513, 2048],
)
def test_layernorm64_stats_scalar_rejects_unsupported_hidden_size(
    hidden_size,
):
    x = torch.zeros(
        (2, hidden_size),
        dtype=torch.int16,
    )

    with pytest.raises(RuntimeError):
        _C.layernorm64_stats_scalar(x)


def test_layernorm64_stats_scalar_rejects_wrong_dtype():
    x = torch.zeros(
        (2, 512),
        dtype=torch.int32,
    )

    with pytest.raises(RuntimeError):
        _C.layernorm64_stats_scalar(x)