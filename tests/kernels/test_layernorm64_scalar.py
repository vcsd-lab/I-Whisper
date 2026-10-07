import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import (
    Rshift,
    Fast_Inverse_Square_Root,
)


INT15_MIN = -16383
INT15_MAX = 16383

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


def layernorm64_ref(
    x,
    weight,
    bias,
    ln_multiplier,
    ln_shift,
    out_multiplier,
    out_shift,
):
    hidden_size = x.shape[-1]

    norm_mul, norm_shift = (
        _norm_params(hidden_size)
    )

    # ------------------------------------------------------------
    # Mean:
    # identical to LayerNorm32.
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # LayerNorm64 variance path.
    # ------------------------------------------------------------

    centered64 = (
        x.to(torch.int64)
        - mean.to(torch.int64)
    )

    square64 = (
        centered64
        * centered64
    )

    var64 = torch.sum(
        square64,
        dim=-1,
        keepdim=True,
    )

    var64 = (
        _mul_by_shift_multiplier(
            var64,
            norm_mul,
        )
        >> norm_shift
    )

    assert torch.all(
        var64 >= 0
    )

    assert torch.all(
        var64
        <= torch.iinfo(
            torch.int32
        ).max
    )

    var = var64.to(
        torch.int32
    )

    # ------------------------------------------------------------
    # Same integer FISR algorithm, but offset 15.
    # ------------------------------------------------------------

    additional_offset = 2
    rsqrt_offset = 15

    invstd = (
        Fast_Inverse_Square_Root(
            var,
            rsqrt_offset,
            2,
        )
    )

    # ------------------------------------------------------------
    # Return to current INT32 arithmetic semantics.
    # ------------------------------------------------------------

    out = (
        (x - mean)
        * invstd
    )

    out = Rshift(
        out
        * ln_multiplier,
        ln_shift
        + additional_offset,
    )

    out = out.clamp(
        INT15_MIN,
        INT15_MAX,
    )

    out = (
        out
        * weight.int()
    )

    if bias is not None:
        out = (
            out
            + bias
        )

    # Same signed-int32 × uint32 MULH semantics used by
    # the current Torch reference.
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
    expected = layernorm64_ref(
        x,
        weight,
        bias,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )

    actual = (
        _C.layernorm64_scalar(
            x.contiguous(),
            weight.contiguous(),
            None
            if bias is None
            else bias.contiguous(),
            ln_multiplier,
            ln_shift,
            out_multiplier,
            out_shift,
        )
    )

    assert actual.dtype == torch.int8
    assert actual.shape == x.shape

    if not torch.equal(
        actual,
        expected,
    ):
        mismatch = (
            actual != expected
        ).nonzero(
            as_tuple=False
        )

        idx = tuple(
            mismatch[0].tolist()
        )

        raise AssertionError(
            "LayerNorm64 scalar mismatch\n"
            f"shape: {tuple(x.shape)}\n"
            f"first mismatch: {idx}\n"
            f"expected: "
            f"{expected[idx].item()}\n"
            f"actual:   "
            f"{actual[idx].item()}\n"
            f"input:    "
            f"{x[idx].item()}"
        )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_scalar_random(
    hidden_size,
):
    torch.manual_seed(
        17000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (5, hidden_size),
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
        torch.tensor(
            [73],
            dtype=torch.uint8,
        ),
        torch.tensor(
            [5],
            dtype=torch.int8,
        ),
        torch.tensor(
            [0x6FFFFFFF],
            dtype=torch.uint32,
        ),
        torch.tensor(
            [3],
            dtype=torch.int8,
        ),
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_scalar_extreme_pattern(
    hidden_size,
):
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
        torch.tensor(
            [255],
            dtype=torch.uint8,
        ),
        torch.tensor(
            [7],
            dtype=torch.int8,
        ),
        torch.tensor(
            [0xFFFFFFFF],
            dtype=torch.uint32,
        ),
        torch.tensor(
            [4],
            dtype=torch.int8,
        ),
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_scalar_zero(
    hidden_size,
):
    x = torch.zeros(
        (3, hidden_size),
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
        torch.tensor(
            [1],
            dtype=torch.uint8,
        ),
        torch.tensor(
            [0],
            dtype=torch.int8,
        ),
        torch.tensor(
            [0x7FFFFFFF],
            dtype=torch.uint32,
        ),
        torch.tensor(
            [0],
            dtype=torch.int8,
        ),
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_scalar_no_bias(
    hidden_size,
):
    torch.manual_seed(
        18000 + hidden_size
    )

    x = torch.randint(
        -12000,
        12001,
        (2, 3, hidden_size),
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
        torch.tensor(
            [31],
            dtype=torch.uint8,
        ),
        torch.tensor(
            [3],
            dtype=torch.int8,
        ),
        torch.tensor(
            [0x50000000],
            dtype=torch.uint32,
        ),
        torch.tensor(
            [2],
            dtype=torch.int8,
        ),
    )