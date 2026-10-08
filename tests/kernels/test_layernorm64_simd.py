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
    return _C.layernorm64_scalar(
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


def _run_simd(
    x,
    weight,
    bias,
    ln_multiplier,
    ln_shift,
    out_multiplier,
    out_shift,
):
    return _C.layernorm64_simd(
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
            "LayerNorm64 SIMD mismatch\n"
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
def test_layernorm64_simd_random(
    hidden_size,
):
    torch.manual_seed(
        23000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (7, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=24000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_simd_zero(
    hidden_size,
):
    x = torch.zeros(
        (4, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=25000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_simd_small_dynamic_range(
    hidden_size,
):
    torch.manual_seed(
        26000 + hidden_size
    )

    x = torch.randint(
        -32,
        33,
        (9, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=27000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_simd_extreme_pattern(
    hidden_size,
):
    x = torch.empty(
        (4, hidden_size),
        dtype=torch.int16,
    )

    x[0, 0::2] = -16383
    x[0, 1::2] = 16383

    half = hidden_size // 2

    x[1, :half] = -16383
    x[1, half:] = 16383

    x[2].fill_(16383)
    x[2, 0] = -16383

    x[3].fill_(-16383)
    x[3, -1] = 16383

    params = _make_params(
        hidden_size,
        seed=28000 + hidden_size,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_simd_no_bias(
    hidden_size,
):
    torch.manual_seed(
        29000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (5, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=30000 + hidden_size,
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
def test_layernorm64_simd_3d(
    hidden_size,
):
    torch.manual_seed(
        31000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (2, 5, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=32000 + hidden_size,
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
def test_layernorm64_simd_unsigned_mulh(
    hidden_size,
    out_multiplier,
):
    torch.manual_seed(
        33000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (3, hidden_size),
        dtype=torch.int16,
    )

    params = _make_params(
        hidden_size,
        seed=34000 + hidden_size,
        out_multiplier=out_multiplier,
    )

    _assert_layernorm_equal(
        x,
        *params,
    )