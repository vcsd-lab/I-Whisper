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


def _run_scalar(x):
    return _C.layernorm64_stats_scalar(
        x.contiguous()
    )


def _run_simd(x):
    return _C.layernorm64_stats_simd(
        x.contiguous()
    )


def _assert_stats_equal(x):
    mean_scalar, var_scalar = (
        _run_scalar(x)
    )

    mean_simd, var_simd = (
        _run_simd(x)
    )

    assert mean_simd.dtype == torch.int32
    assert var_simd.dtype == torch.int64

    assert torch.equal(
        mean_simd,
        mean_scalar,
    ), (
        "LayerNorm64 mean mismatch\n"
        f"shape: {tuple(x.shape)}\n"
        f"scalar: {mean_scalar}\n"
        f"simd:   {mean_simd}"
    )

    if not torch.equal(
        var_simd,
        var_scalar,
    ):
        mismatch = (
            var_simd != var_scalar
        ).nonzero(
            as_tuple=False
        )

        idx = tuple(
            mismatch[0].tolist()
        )

        raise AssertionError(
            "LayerNorm64 variance mismatch\n"
            f"shape: {tuple(x.shape)}\n"
            f"first mismatch: {idx}\n"
            f"scalar: "
            f"{var_scalar[idx].item()}\n"
            f"simd:   "
            f"{var_simd[idx].item()}\n"
            f"mean scalar: "
            f"{mean_scalar[idx].item()}\n"
            f"mean simd:   "
            f"{mean_simd[idx].item()}"
        )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_simd_random(
    hidden_size,
):
    torch.manual_seed(
        19000 + hidden_size
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
def test_layernorm64_stats_simd_zero(
    hidden_size,
):
    x = torch.zeros(
        (4, hidden_size),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_simd_extreme_pattern(
    hidden_size,
):
    x = torch.empty(
        (4, hidden_size),
        dtype=torch.int16,
    )

    # Alternating extrema.
    x[0, 0::2] = -16383
    x[0, 1::2] = 16383

    half = hidden_size // 2

    # Half negative / half positive.
    x[1, :half] = -16383
    x[1, half:] = 16383

    # Single negative outlier.
    x[2].fill_(16383)
    x[2, 0] = -16383

    # Single positive outlier.
    x[3].fill_(-16383)
    x[3, -1] = 16383

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_simd_small_dynamic_range(
    hidden_size,
):
    torch.manual_seed(
        20000 + hidden_size
    )

    x = torch.randint(
        -32,
        33,
        (11, hidden_size),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_simd_multiple_ranges(
    hidden_size,
):
    torch.manual_seed(
        21000 + hidden_size
    )

    x = torch.empty(
        (5, hidden_size),
        dtype=torch.int16,
    )

    x[0] = torch.randint(
        -8,
        9,
        (hidden_size,),
        dtype=torch.int16,
    )

    x[1] = torch.randint(
        -64,
        65,
        (hidden_size,),
        dtype=torch.int16,
    )

    x[2] = torch.randint(
        -512,
        513,
        (hidden_size,),
        dtype=torch.int16,
    )

    x[3] = torch.randint(
        -4096,
        4097,
        (hidden_size,),
        dtype=torch.int16,
    )

    x[4] = torch.randint(
        -16383,
        16384,
        (hidden_size,),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm64_stats_simd_3d(
    hidden_size,
):
    torch.manual_seed(
        22000 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (2, 5, hidden_size),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)