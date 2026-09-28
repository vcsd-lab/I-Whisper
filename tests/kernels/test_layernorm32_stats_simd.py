import pytest
import torch

# Keep the same _C import used by test_layernorm32_stats_scalar.py
from i_whisper.kernels import _C


HIDDEN_SIZES = [384, 512, 768, 1024, 1280]


def _run_scalar(x: torch.Tensor):
    return _C.layernorm32_stats_scalar(
        x.contiguous()
    )


def _run_simd(x: torch.Tensor):
    return _C.layernorm32_stats_simd(
        x.contiguous()
    )


def _assert_stats_equal(
    x: torch.Tensor,
):
    mean_scalar, var_scalar, shift_scalar = (
        _run_scalar(x)
    )

    mean_simd, var_simd, shift_simd = (
        _run_simd(x)
    )

    assert torch.equal(
        mean_simd,
        mean_scalar,
    ), (
        "LayerNorm mean mismatch\n"
        f"shape: {tuple(x.shape)}\n"
        f"scalar: {mean_scalar}\n"
        f"simd:   {mean_simd}"
    )

    assert torch.equal(
        shift_simd,
        shift_scalar,
    ), (
        "LayerNorm renorm_shift mismatch\n"
        f"shape: {tuple(x.shape)}\n"
        f"scalar: {shift_scalar}\n"
        f"simd:   {shift_simd}"
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

        idx = mismatch[0]

        idx_tuple = tuple(
            idx.tolist()
        )

        raise AssertionError(
            "LayerNorm variance mismatch\n"
            f"shape: {tuple(x.shape)}\n"
            f"first mismatch index: {idx_tuple}\n"
            f"scalar: "
            f"{var_scalar[idx_tuple].item()}\n"
            f"simd:   "
            f"{var_simd[idx_tuple].item()}\n"
            f"mean scalar: "
            f"{mean_scalar[idx_tuple].item()}\n"
            f"mean simd:   "
            f"{mean_simd[idx_tuple].item()}\n"
            f"shift scalar: "
            f"{shift_scalar[idx_tuple].item()}\n"
            f"shift simd:   "
            f"{shift_simd[idx_tuple].item()}"
        )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_stats_simd_random(
    hidden_size,
):
    torch.manual_seed(
        1234 + hidden_size
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
def test_layernorm32_stats_simd_zero(
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
@pytest.mark.parametrize(
    "value",
    [
        -16383,
        -1000,
        -1,
        0,
        1,
        1000,
        16383,
    ],
)
def test_layernorm32_stats_simd_constant(
    hidden_size,
    value,
):
    x = torch.full(
        (3, hidden_size),
        value,
        dtype=torch.int16,
    )

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_stats_simd_extreme_pattern(
    hidden_size,
):
    x = torch.empty(
        (4, hidden_size),
        dtype=torch.int16,
    )

    # Alternating extremes.
    x[0, 0::2] = -16383
    x[0, 1::2] = 16383

    # Half negative / half positive.
    half = hidden_size // 2

    x[1, :half] = -16383
    x[1, half:] = 16383

    # Mostly positive with one negative outlier.
    x[2].fill_(16383)
    x[2, 0] = -16383

    # Mostly negative with one positive outlier.
    x[3].fill_(-16383)
    x[3, -1] = 16383

    _assert_stats_equal(x)


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_stats_simd_small_dynamic_range(
    hidden_size,
):
    """
    Exercise renorm_shift == 0 heavily.

    This is especially important after fixing the
    Rshift(x, 0) semantics.
    """
    torch.manual_seed(
        5678 + hidden_size
    )

    x = torch.randint(
        -32,
        33,
        (11, hidden_size),
        dtype=torch.int16,
    )

    mean_scalar, var_scalar, shift_scalar = (
        _run_scalar(x)
    )

    mean_simd, var_simd, shift_simd = (
        _run_simd(x)
    )

    assert torch.equal(
        mean_simd,
        mean_scalar,
    )

    assert torch.equal(
        var_simd,
        var_scalar,
    )

    assert torch.equal(
        shift_simd,
        shift_scalar,
    )

    # This test is intended to actually exercise shift == 0.
    assert torch.any(
        shift_scalar == 0
    )


@pytest.mark.parametrize(
    "hidden_size",
    HIDDEN_SIZES,
)
def test_layernorm32_stats_simd_multiple_renorm_ranges(
    hidden_size,
):
    """
    Construct rows with different dynamic ranges so one
    invocation exercises different row-wise renorm shifts.
    """
    torch.manual_seed(
        9012 + hidden_size
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
def test_layernorm32_stats_simd_3d(
    hidden_size,
):
    torch.manual_seed(
        1357 + hidden_size
    )

    x = torch.randint(
        -16383,
        16384,
        (2, 5, hidden_size),
        dtype=torch.int16,
    )

    _assert_stats_equal(x)