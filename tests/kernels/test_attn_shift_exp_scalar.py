import pytest
import torch

from i_whisper.kernels import _C


INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max


def _frac_bits(value: int) -> torch.Tensor:
    return torch.tensor(
        value,
        dtype=torch.int32,
    )


def _expand_mask(
    mask: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Expand the shared [Tq, Tk] attention mask to the
    [B, H, Tq, Tk] shape used by the Torch golden.
    """
    assert mask.dtype == torch.bool
    assert mask.dim() == 2

    return (
        mask
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(
            x.shape[0],
            x.shape[1],
            1,
            1,
        )
    )


def _torch_shift_exp_golden(
    x: torch.Tensor,
    row_max_plus1: torch.Tensor,
    mask: torch.Tensor | None,
    frac_bits: torch.Tensor,
) -> torch.Tensor:
    """
    Golden copied from the current model_int.py single-pass
    shift_exp dataflow.

    Inputs
    ------
    x:
        INT16 [B, H, Tq, Tk]

    row_max_plus1:
        INT16 [B, H, Tq, 1]
        Stores max(valid qk) + 1 using INT16 wrap semantics.

    mask:
        BOOL [Tq, Tk] shared across batch/head, or None.

    frac_bits:
        INT32 scalar tensor.

    Output
    ------
    UINT8 [B, H, Tq, Tk]
    """
    assert x.dtype == torch.int16
    assert row_max_plus1.dtype == torch.int16
    assert frac_bits.dtype == torch.int32
    assert frac_bits.numel() == 1

    # Keep the golden local and preserve caller-owned inputs.
    x_sub = x - row_max_plus1

    # Current model_int.py semantics:
    # pathological positive values are replaced by INT16_MIN
    # before the exponential approximation.
    x_sub = x_sub.clone()
    x_sub[x_sub > 0] = INT16_MIN

    if mask is not None:
        torch_mask = _expand_mask(
            mask,
            x_sub,
        )
    else:
        torch_mask = None

    offset = frac_bits

    x_mul_fix = (
        x_sub
        + (x_sub >> 1)
        - (x_sub >> 4)
    )

    x_mul_int = (
        x_mul_fix
        >> offset
    )

    x_mul_frac = (
        x_mul_fix
        - (x_mul_int << offset)
    )

    x_exp = (
        (
            (1 << offset)
            + (x_mul_frac >> 1)
        )
        >> (-x_mul_int)
    )

    if torch_mask is not None:
        x_exp[torch_mask] = 0

    # Retained exactly from current model_int.py.
    # Since positive x_sub lanes were replaced above, this is
    # normally a no-op, but keeping it here prevents the test
    # golden from silently diverging from the model implementation.
    x_exp[x_sub > 0] = INT16_MIN

    frac_bits_value = int(
        frac_bits.item()
    )

    if frac_bits_value > 7:
        x_exp = (
            x_exp
            >> (frac_bits_value - 7)
        )

    qk_exp = x_exp.to(torch.uint8)

    qk_exp_sum = (
        qk_exp
        .sum(dim=-1, keepdim=True)
        .int()
    )

    return qk_exp, qk_exp_sum


def _row_max_plus1(
    x: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a shift-exp input consistent with the output contract of
    qk_requant_rowmax_scalar:

      - masked QK lanes are already INT16_MIN
      - row_max_plus1 is max(valid qk) + 1 in INT16

    Returns
    -------
    qk:
        INT16 [B, H, Tq, Tk]

    row_max_plus1:
        INT16 [B, H, Tq, 1]
    """
    assert x.dtype == torch.int16

    qk = x.clone()

    if mask is not None:
        qk[
            _expand_mask(
                mask,
                qk,
            )
        ] = INT16_MIN

    row_max = (
        qk
        .max(
            dim=-1,
            keepdim=True,
        )
        .values
        .to(torch.int16)
    )

    row_max_plus1 = (
        row_max
        + 1
    )

    assert row_max_plus1.dtype == torch.int16

    return (
        qk,
        row_max_plus1,
    )


def _cached_causal_mask(
    q_length: int,
    k_length: int,
) -> torch.Tensor:
    """
    Build the rectangular causal mask used during cached decoding.

    Example:
        Tq=1, Tk=4 -> absolute query row 3 -> all four keys valid.
    """
    assert q_length <= k_length

    full_mask = torch.triu(
        torch.ones(
            (k_length, k_length),
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    q_offset = (
        k_length
        - q_length
    )

    return (
        full_mask[
            q_offset:q_offset + q_length,
            :k_length,
        ]
        .contiguous()
    )


def _run_scalar(
    x: torch.Tensor,
    row_max_plus1: torch.Tensor,
    mask: torch.Tensor | None,
    frac_bits: torch.Tensor,
) -> torch.Tensor:
    return _C.shift_exp_scalar(
        x.contiguous(),
        row_max_plus1.contiguous(),
        None if mask is None else mask.contiguous(),
        frac_bits.contiguous(),
    )


def _assert_equal(
    x: torch.Tensor,
    row_max_plus1: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    frac_bits: int,
) -> None:
    frac_bits_t = _frac_bits(
        frac_bits
    )
    
    ref_exp, ref_sum = _torch_shift_exp_golden(
        x,
        row_max_plus1,
        mask,
        frac_bits_t,
    )
    cpp_exp, cpp_sum = _run_scalar(
        x,
        row_max_plus1,
        mask,
        frac_bits_t,
    )

    assert cpp_exp.dtype == torch.uint8
    assert cpp_sum.dtype == torch.int32

    assert cpp_exp.shape == ref_exp.shape
    assert cpp_sum.shape == ref_sum.shape

    assert cpp_sum.shape == (
        x.shape[0],
        x.shape[1],
        x.shape[2],
        1,
    )

    assert torch.equal(
        cpp_exp,
        ref_exp,
    ), "shift-exp mismatch"

    assert torch.equal(
        cpp_sum,
        ref_sum,
    ), "shift-exp row-sum mismatch"


# ============================================================
# Random differential tests
# ============================================================

@pytest.mark.parametrize(
    "shape",
    [
        # B, H, Tq, Tk
        (1, 1, 3, 3),
        (1, 4, 9, 13),
        (2, 3, 5, 7),
        # Cached autoregressive decode geometry.
        (1, 8, 1, 4),
    ],
)
@pytest.mark.parametrize(
    "frac_bits",
    [
        1,
        7,
        8,
        12,
        13,
    ],
)
def test_attn_shift_exp_scalar_random_unmasked(
    shape,
    frac_bits,
):
    B, H, Tq, Tk = shape

    torch.manual_seed(
        1000
        + B * 101
        + H * 103
        + Tq * 107
        + Tk * 109
        + frac_bits * 113
    )

    # Keep ordinary random values away from the full INT16 boundary.
    # Boundary behavior is tested explicitly below.
    raw = torch.randint(
        -16000,
        16001,
        (B, H, Tq, Tk),
        dtype=torch.int16,
    )

    qk, row_max = _row_max_plus1(
        raw,
        None,
    )

    _assert_equal(
        qk,
        row_max,
        mask=None,
        frac_bits=frac_bits,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 3, 3),
        (1, 4, 5, 11),
        (2, 3, 7, 13),
        (1, 8, 1, 4),
    ],
)
@pytest.mark.parametrize(
    "frac_bits",
    [
        7,
        8,
        12,
    ],
)
def test_attn_shift_exp_scalar_random_causal_mask(
    shape,
    frac_bits,
):
    B, H, Tq, Tk = shape

    torch.manual_seed(
        3000
        + B * 11
        + H * 13
        + Tq * 17
        + Tk * 19
        + frac_bits * 23
    )

    raw = torch.randint(
        -16000,
        16001,
        (B, H, Tq, Tk),
        dtype=torch.int16,
    )

    mask = _cached_causal_mask(
        Tq,
        Tk,
    )

    qk, row_max = _row_max_plus1(
        raw,
        mask,
    )

    _assert_equal(
        qk,
        row_max,
        mask=mask,
        frac_bits=frac_bits,
    )


# ============================================================
# Explicit semantic / overflow corner cases
# ============================================================

def test_attn_shift_exp_scalar_row_max_32767_wrap():
    """
    If row max is 32767:

        row_max_plus1 = -32768

    The subtraction must still use INT16 wrap semantics.

    This row also contains -1, which produces a positive wrapped
    subtraction and therefore exercises the x_sub > 0 guard.
    """
    qk = torch.tensor(
        [[[[
            32767,
            -1,
            -32768,
            0,
        ]]]],
        dtype=torch.int16,
    )

    row_max = torch.tensor(
        [[[[-32768]]]],
        dtype=torch.int16,
    )

    # Confirm this is the exact INT16 max+1 wrap case.
    assert (
        torch.tensor(
            32767,
            dtype=torch.int16,
        )
        + 1
    ).item() == -32768

    # The second lane wraps positive:
    #   -1 - (-32768) = 32767
    x_sub = (
        qk
        - row_max
    )

    assert x_sub[0, 0, 0, 1].item() > 0

    _assert_equal(
        qk,
        row_max,
        mask=None,
        frac_bits=7,
    )


def test_attn_shift_exp_scalar_masked_lane_is_zero():
    """
    Masked lanes must be zero regardless of their QK value.
    """
    qk = torch.tensor(
        [[[[
            1200,
            32767,
            -500,
            700,
        ]]]],
        dtype=torch.int16,
    )

    mask = torch.tensor(
        [[False, True, False, False]],
        dtype=torch.bool,
    )

    # Match qk_requant_rowmax_scalar output semantics:
    # masked QK is already INT16_MIN.
    qk[
        _expand_mask(
            mask,
            qk,
        )
    ] = INT16_MIN

    row_max = (
        qk.max(
            dim=-1,
            keepdim=True,
        ).values
        + 1
    )

    out, row_sum = _run_scalar(
        qk,
        row_max,
        mask,
        _frac_bits(7),
    )

    assert out[0, 0, 0, 1].item() == 0

    ref, ref_row_sum = _torch_shift_exp_golden(
        qk,
        row_max,
        mask,
        _frac_bits(7),
    )

    assert torch.equal(
        out,
        ref,
    )


def test_attn_shift_exp_scalar_all_masked_row():
    """
    Previous kernel semantics for an all-masked row:

        qk[:] = INT16_MIN
        row_max_plus1 = INT16_MIN + 1 = -32767

    Every shift-exp output must still be zero because mask wins.
    """
    B, H, Tq, Tk = (
        1,
        3,
        2,
        5,
    )

    qk = torch.full(
        (B, H, Tq, Tk),
        INT16_MIN,
        dtype=torch.int16,
    )

    row_max = torch.full(
        (B, H, Tq, 1),
        INT16_MIN + 1,
        dtype=torch.int16,
    )

    mask = torch.ones(
        (Tq, Tk),
        dtype=torch.bool,
    )

    out, row_sum = _run_scalar(
        qk,
        row_max,
        mask,
        _frac_bits(7),
    )

    assert torch.count_nonzero(out).item() == 0

    ref, ref_row_sum = _torch_shift_exp_golden(
        qk,
        row_max,
        mask,
        _frac_bits(7),
    )

    assert torch.equal(
        out,
        ref,
    )


@pytest.mark.parametrize(
    "frac_bits",
    [
        8,
        10,
        13,
    ],
)
def test_attn_shift_exp_scalar_q_to_q17_downshift(
    frac_bits,
):
    """
    Exercise the final:
        x_exp >> (frac_bits - 7)

    conversion back to Q1.7.
    """
    qk = torch.tensor(
        [[[[
            1000,
            900,
            700,
            0,
            -1000,
        ]]]],
        dtype=torch.int16,
    )

    row_max = (
        qk.max(
            dim=-1,
            keepdim=True,
        ).values
        + 1
    )

    _assert_equal(
        qk,
        row_max,
        mask=None,
        frac_bits=frac_bits,
    )


# ============================================================
# Input immutability and wrapper contract
# ============================================================

def test_attn_shift_exp_scalar_does_not_modify_inputs():
    torch.manual_seed(6001)

    raw = torch.randint(
        -12000,
        12001,
        (1, 4, 3, 7),
        dtype=torch.int16,
    )

    mask = _cached_causal_mask(
        3,
        7,
    )

    qk, row_max = _row_max_plus1(
        raw,
        mask,
    )

    qk_original = qk.clone()
    row_max_original = row_max.clone()
    mask_original = mask.clone()

    _run_scalar(
        qk,
        row_max,
        mask,
        _frac_bits(7),
    )

    assert torch.equal(
        qk,
        qk_original,
    )

    assert torch.equal(
        row_max,
        row_max_original,
    )

    assert torch.equal(
        mask,
        mask_original,
    )


@pytest.mark.parametrize(
    "bad_shape",
    [
        (1, 2, 3),
        (1, 2, 3, 2),
        (2, 2, 3, 1),
        (1, 3, 3, 1),
        (1, 2, 4, 1),
    ],
)
def test_attn_shift_exp_scalar_rejects_bad_row_max_shape(
    bad_shape,
):
    x = torch.zeros(
        (1, 2, 3, 5),
        dtype=torch.int16,
    )

    row_max = torch.zeros(
        bad_shape,
        dtype=torch.int16,
    )

    with pytest.raises(
        RuntimeError,
    ):
        _C.shift_exp_scalar(
            x,
            row_max,
            None,
            _frac_bits(7),
        )


def test_attn_shift_exp_scalar_rejects_4d_mask():
    x = torch.zeros(
        (1, 2, 3, 5),
        dtype=torch.int16,
    )

    row_max = torch.zeros(
        (1, 2, 3, 1),
        dtype=torch.int16,
    )

    bad_mask = torch.zeros(
        (1, 2, 3, 5),
        dtype=torch.bool,
    )

    with pytest.raises(
        RuntimeError,
        match=r"mask must be \[Tq,Tk\]",
    ):
        _C.shift_exp_scalar(
            x,
            row_max,
            bad_mask,
            _frac_bits(7),
        )


@pytest.mark.parametrize(
    "frac_bits",
    [
        0,
        14,
        15,
    ],
)
def test_attn_shift_exp_scalar_rejects_invalid_frac_bits(
    frac_bits,
):
    x = torch.zeros(
        (1, 1, 1, 1),
        dtype=torch.int16,
    )

    row_max = torch.zeros(
        (1, 1, 1, 1),
        dtype=torch.int16,
    )

    with pytest.raises(
        RuntimeError,
        match=r"frac_bits must be in \[1, 14\)",
    ):
        _C.shift_exp_scalar(
            x,
            row_max,
            None,
            _frac_bits(frac_bits),
        )
