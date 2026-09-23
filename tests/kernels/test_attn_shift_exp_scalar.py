import pytest
import torch

from i_whisper.kernels import _C


# ============================================================
# Torch golden
# ============================================================

def _torch_shift_exp(
    x: torch.Tensor,
    mask=None,
    frac_bits: int = 7,
):
    assert x.dtype == torch.int16
    assert x.dim() == 4

    # The production Torch implementation modifies x in-place.
    # Clone here so the test reference never mutates the input
    # later passed into C++.
    x = x.clone()

    if mask is None:
        expanded_mask = torch.zeros_like(
            x,
            dtype=torch.bool,
        )

    else:
        assert mask.dtype == torch.bool
        assert mask.dim() == 2

        expanded_mask = (
            mask
            .unsqueeze(0)
            .unsqueeze(0)
        )

        expanded_mask = expanded_mask.repeat(
            x.shape[0]
            // expanded_mask.shape[0],

            x.shape[1]
            // expanded_mask.shape[1],

            x.shape[2]
            // expanded_mask.shape[2],

            x.shape[3]
            // expanded_mask.shape[3],
        )

    x[expanded_mask] = torch.iinfo(
        torch.int16
    ).min


    x_max = (
        x.max(
            dim=-1,
            keepdim=True,
        ).values
        + 1
    )


    x_sub = (
        x
        - x_max
    )


    x_mul_fix = (
        x_sub
        + (x_sub >> 1)
        - (x_sub >> 4)
    )


    x_mul_int = (
        x_mul_fix
        >> frac_bits
    )


    x_mul_frac = (
        x_mul_fix
        - (
            x_mul_int
            << frac_bits
        )
    )


    x_exp = (
        (
            (1 << frac_bits)
            + (x_mul_frac >> 1)
        )
        >> (-x_mul_int)
    )


    x_exp[expanded_mask] = 0

    x_exp[
        x_sub > 0
    ] = 0


    if frac_bits > 7:
        x_exp = (
            x_exp.int()
            >> (frac_bits - 7)
        )


    return x_exp.to(
        torch.uint8
    )


# ============================================================
# C++ backend
# ============================================================

def _run_scalar(
    x: torch.Tensor,
    mask=None,
    frac_bits: int = 7,
):
    return _C.shift_exp_scalar(
        x.contiguous(),
        None
        if mask is None
        else mask.contiguous(),
        torch.tensor(frac_bits, dtype=torch.int32),
    )


# ============================================================
# Differential helper
# ============================================================

def _assert_equal(
    x: torch.Tensor,
    mask=None,
    frac_bits: int = 7,
):
    original = x.clone()


    ref = _torch_shift_exp(
        x,
        mask,
        frac_bits,
    )


    out = _run_scalar(
        x,
        mask,
        frac_bits,
    )


    assert out.dtype == torch.uint8

    assert out.shape == x.shape


    # C++ kernel must not modify its input.
    assert torch.equal(
        x,
        original,
    )


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


        b, h, qi, ki = idx


        raise AssertionError(
            "shift_exp scalar mismatch\n"
            f"shape: {tuple(x.shape)}\n"
            f"frac_bits: {frac_bits}\n"
            f"first mismatch: {idx}\n"
            f"input value: "
            f"{x[b, h, qi, ki].item()}\n"
            f"input row: "
            f"{x[b, h, qi].tolist()}\n"
            f"mask row: "
            f"{None if mask is None else mask[qi].tolist()}\n"
            f"reference: {ref[idx].item()}\n"
            f"actual:    {out[idx].item()}"
        )


# ============================================================
# Basic no-mask
# ============================================================

def test_attn_shift_exp_scalar_basic():
    x = torch.tensor(
        [[[
            [
                -128,
                -64,
                -32,
                -16,
                -8,
                -4,
                -2,
                -1,
            ],
        ]]],
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# All equal
# ============================================================

def test_attn_shift_exp_scalar_constant_row():
    """
    row_max = value

    x_max = value + 1

    therefore every ordinary element starts from x_sub = -1.
    """

    x = torch.full(
        (1, 2, 5, 17),
        123,
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# Zero input
# ============================================================

def test_attn_shift_exp_scalar_zero():
    x = torch.zeros(
        (2, 4, 7, 19),
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# Production-like random QK range
# ============================================================

@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 3, 5),
        (1, 8, 17, 23),
        (2, 4, 13, 29),
        (1, 8, 31, 37),
    ],
)
def test_attn_shift_exp_scalar_random_production_range(
    shape,
):
    torch.manual_seed(1234)


    x = torch.randint(
        -4096,
        4097,
        shape,
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# Full INT16 random stress
# ============================================================

def test_attn_shift_exp_scalar_random_full_int16():
    """
    Deliberately goes far beyond normal calibrated QK scores.

    This test determines whether the simplified C++ datapath
    still reproduces Torch INT16 wrap semantics.
    """

    torch.manual_seed(2001)


    x = torch.randint(
        -32768,
        32768,
        (2, 3, 11, 37),
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# Causal mask
# ============================================================

def test_attn_shift_exp_scalar_causal_mask():
    torch.manual_seed(3001)


    T = 17


    x = torch.randint(
        -2048,
        2049,
        (1, 8, T, T),
        dtype=torch.int16,
    )


    mask = torch.triu(
        torch.ones(
            (T, T),
            dtype=torch.bool,
        ),
        diagonal=1,
    )


    _assert_equal(
        x,
        mask,
    )


# ============================================================
# Mask broadcast across batch and heads
# ============================================================

def test_attn_shift_exp_scalar_mask_broadcast():
    torch.manual_seed(4001)


    B = 3
    H = 5
    Tq = 11
    Tk = 19


    x = torch.randint(
        -3000,
        3001,
        (B, H, Tq, Tk),
        dtype=torch.int16,
    )


    mask = torch.zeros(
        (Tq, Tk),
        dtype=torch.bool,
    )


    mask[:, -3:] = True
    mask[::2, 1::3] = True


    _assert_equal(
        x,
        mask,
    )


# ============================================================
# Masked maximum must not participate in row max
# ============================================================

def test_attn_shift_exp_scalar_masked_maximum():
    x = torch.tensor(
        [[[
            [
                100,
                200,
                30000,
                50,
                -100,
            ],
        ]]],
        dtype=torch.int16,
    )


    mask = torch.tensor(
        [[
            False,
            False,
            True,
            False,
            False,
        ]],
        dtype=torch.bool,
    )


    _assert_equal(
        x,
        mask,
    )


# ============================================================
# Entire row masked
# ============================================================

def test_attn_shift_exp_scalar_all_masked():
    torch.manual_seed(5001)


    Tq = 7
    Tk = 13


    x = torch.randint(
        -32768,
        32768,
        (2, 4, Tq, Tk),
        dtype=torch.int16,
    )


    mask = torch.ones(
        (Tq, Tk),
        dtype=torch.bool,
    )


    _assert_equal(
        x,
        mask,
    )


# ============================================================
# INT16 minimum
# ============================================================

def test_attn_shift_exp_scalar_int16_min():
    x = torch.full(
        (1, 2, 5, 17),
        torch.iinfo(torch.int16).min,
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# INT16 maximum
# ============================================================

def test_attn_shift_exp_scalar_int16_max():
    """
    Explicitly exercises:

        row_max = 32767
        x_max   = 32767 + 1

    under Torch INT16 wrap semantics.
    """

    x = torch.full(
        (1, 2, 5, 17),
        torch.iinfo(torch.int16).max,
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# row_max == INT16_MAX with mixed values
# ============================================================

def test_attn_shift_exp_scalar_row_max_wrap():
    """
    More useful than the all-32767 case because x_max wraps
    while other values in the row are different.
    """

    x = torch.tensor(
        [[[
            [
                32767,
                30000,
                20000,
                10000,
                0,
                -1,
                -10000,
                -20000,
                -30000,
                -32768,
            ],
        ]]],
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# Extreme row spread
# ============================================================

def test_attn_shift_exp_scalar_extreme_spread():
    """
    Exercises INT16 wrap in:

        x_sub
        x_mul_fix

    rather than only normal production ranges.
    """

    row = torch.tensor(
        [
            -32768,
            -32000,
            -30000,
            -24576,
            -20000,
            -16384,
            -8192,
            -4096,
            -1024,
            -1,
            0,
            1,
            1024,
            4096,
            8192,
            16384,
            20000,
            24576,
            30000,
            32000,
            32767,
        ],
        dtype=torch.int16,
    )


    x = row.reshape(
        1,
        1,
        1,
        -1,
    )


    _assert_equal(
        x,
    )


# ============================================================
# Explicit x_mul_fix overflow region
# ============================================================

def test_attn_shift_exp_scalar_mul_fix_wrap():
    """
    x_mul_fix approximates:

        1.4375 * x_sub

    so sufficiently negative x_sub values cross the INT16
    lower boundary and wrap.

    This is specifically designed to verify that behavior.
    """

    x = torch.tensor(
        [[[
            [
                0,
                -16000,
                -20000,
                -24000,
                -28000,
                -30000,
                -32000,
                -32768,
            ],
        ]]],
        dtype=torch.int16,
    )


    _assert_equal(
        x,
    )


# ============================================================
# Multiple independent rows
# ============================================================

def test_attn_shift_exp_scalar_row_independence():
    """
    Each row has a different maximum.

    Catches accidental reuse of row_max between rows/heads.
    """

    B = 2
    H = 3
    Tq = 5
    Tk = 17


    x = torch.empty(
        (B, H, Tq, Tk),
        dtype=torch.int16,
    )


    base = torch.arange(
        Tk,
        dtype=torch.int32,
    )


    for b in range(B):
        for h in range(H):
            for q in range(Tq):
                row = (
                    base * (q + 1)
                    - 300
                    + b * 700
                    - h * 150
                )


                x[b, h, q] = (
                    row
                    .clamp(
                        -32768,
                        32767,
                    )
                    .to(torch.int16)
                )


    _assert_equal(
        x,
    )


# ============================================================
# qk_frac_bits production configuration
# ============================================================

def test_attn_shift_exp_scalar_frac_bits_7():
    torch.manual_seed(6001)


    x = torch.randint(
        -4096,
        4097,
        (1, 8, 17, 23),
        dtype=torch.int16,
    )


    _assert_equal(
        x,
        frac_bits=7,
    )


# ============================================================
# Exercise >7 output conversion branch
# ============================================================

@pytest.mark.parametrize(
    "frac_bits",
    [
        8,
        10,
        12,
    ],
)
def test_attn_shift_exp_scalar_higher_frac_bits(
    frac_bits,
):
    """
    Production currently uses Q1.7.

    These cases only verify the generic branch retained from
    the Torch golden:

        if qk_frac_bits > 7:
            x_exp >>= qk_frac_bits - 7
    """

    torch.manual_seed(
        7000 + frac_bits
    )


    x = torch.randint(
        -4096,
        4097,
        (1, 3, 11, 19),
        dtype=torch.int16,
    )


    _assert_equal(
        x,
        frac_bits=frac_bits,
    )


# ============================================================
# Input immutability with mask
# ============================================================

def test_attn_shift_exp_scalar_does_not_modify_input():
    """
    Torch golden internally overwrites masked locations with
    INT16_MIN.

    The C++ kernel intentionally treats input as const.
    """

    torch.manual_seed(8001)


    x = torch.randint(
        -4096,
        4097,
        (1, 4, 9, 13),
        dtype=torch.int16,
    )


    original = x.clone()


    mask = torch.triu(
        torch.ones(
            (9, 13),
            dtype=torch.bool,
        ),
        diagonal=1,
    )


    _run_scalar(
        x,
        mask,
        7,
    )


    assert torch.equal(
        x,
        original,
    )