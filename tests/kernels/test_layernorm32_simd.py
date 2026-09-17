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


def _make_params(
    hidden_size,
    *,
    seed=1234,
    bias=True,
    out_multiplier=1409291776,
):
    torch.manual_seed(seed)

    weight = torch.randint(
        -127,
        128,
        (hidden_size,),
        dtype=torch.int8,
    )

    if bias:
        bias_tensor = torch.randint(
            -20000,
            20001,
            (hidden_size,),
            dtype=torch.int32,
        )
    else:
        bias_tensor = None

    # Representative LayerNorm requant parameters.
    #
    # Exact numerical values do not need to correspond to
    # a calibrated layer for SIMD-vs-scalar differential
    # testing; both kernels must implement identical integer
    # semantics.
    ln_multiplier = torch.tensor(
        [96],
        dtype=torch.uint8,
    )

    ln_shift = torch.tensor(
        [9],
        dtype=torch.int8,
    )

    out_multiplier = torch.tensor(
        [out_multiplier],
        dtype=torch.uint32,
    )

    out_shift = torch.tensor(
        [10],
        dtype=torch.int8,
    )

    return (
        weight,
        bias_tensor,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )


def _run_scalar(
    x,
    weight,
    bias,
    ln_multiplier,
    ln_shift,
    out_multiplier,
    out_shift,
):
    return _C.layernorm32_scalar(
        x.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )


def _run_simd(
    x,
    weight,
    bias,
    ln_multiplier,
    ln_shift,
    out_multiplier,
    out_shift,
):
    return _C.layernorm32_simd(
        x.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )


def _assert_layernorm_equal(
    x,
    weight,
    bias,
    ln_multiplier,
    ln_shift,
    out_multiplier,
    out_shift,
):
    ref = _run_scalar(
        x,
        weight,
        bias,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )

    out = _run_simd(
        x,
        weight,
        bias,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
    )

    assert out.dtype == torch.int8
    assert out.shape == x.shape

    if not torch.equal(
        out,
        ref,
    ):
        mismatch = (
            out != ref
        ).nonzero(
            as_tuple=False
        )

        idx = tuple(
            mismatch[0].tolist()
        )

        raise AssertionError(
            "LayerNorm32 SIMD mismatch\n"
            f"shape: {tuple(x.shape)}\n"
            f"first mismatch: {idx}\n"
            f"scalar: {ref[idx].item()}\n"
            f"simd:   {out[idx].item()}\n"
            f"input:  {x[idx].item()}\n"
            f"ln_multiplier: "
            f"{ln_multiplier.item()}\n"
            f"ln_shift: "
            f"{ln_shift.item()}\n"
            f"out_multiplier: "
            f"{out_multiplier.item()}\n"
            f"out_shift: "
            f"{out_shift.item()}"
        )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_simd_random(
    hidden_size,
):
    torch.manual_seed(
        1000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (7, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=2000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_simd_zero(
    hidden_size,
):
    x = torch.zeros(
        (4, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=3000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_simd_small_dynamic_range(
    hidden_size,
):
    """
    Exercise rows whose renorm_shift is expected to be zero.
    """
    torch.manual_seed(
        4000 + hidden_size
    )

    x = torch.randint(
        -32,
        33,
        (9, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=5000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_simd_extreme_pattern(
    hidden_size,
):
    x = torch.empty(
        (4, hidden_size),
        dtype=torch.int16,
    )

    # Alternating extrema.
    x[0, 0::2] = -16383
    x[0, 1::2] = 16383

    # Split extrema.
    half = hidden_size // 2

    x[1, :half] = -16383
    x[1, half:] = 16383

    # Single negative outlier.
    x[2].fill_(16383)
    x[2, 0] = -16383

    # Single positive outlier.
    x[3].fill_(-16383)
    x[3, -1] = 16383

    params = _make_params(
        hidden_size,
        seed=6000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_simd_no_bias(
    hidden_size,
):
    torch.manual_seed(
        7000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (5, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=8000 + hidden_size,
        bias=False,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_simd_3d(
    hidden_size,
):
    torch.manual_seed(
        9000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (2, 5, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=10000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
@pytest.mark.parametrize(
    "out_multiplier",
    [
        1,
        0x40000000,
        0x7FFFFFFF,
        0x80000000,
        0xF1234567,
        0xFFFFFFFF,
    ],
)
def test_layernorm32_simd_unsigned_mulh(
    hidden_size,
    out_multiplier,
):
    """
    Explicitly exercise signed activation × uint32 multiplier MULH.

    Values >= 2^31 are important because treating the multiplier
    as signed would produce the wrong result.
    """
    torch.manual_seed(
        11000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (3, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=12000 + hidden_size,
        out_multiplier=out_multiplier,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )