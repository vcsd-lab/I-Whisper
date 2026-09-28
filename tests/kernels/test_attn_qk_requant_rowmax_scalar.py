import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import (
    Batch_Head_s8s8s32_GeMM,
    Rshift,
)


INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max


def _scalar_u8(value: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.uint8)


def _scalar_i8(value: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int8)


def _expand_mask(
    mask: torch.Tensor,
    qk: torch.Tensor,
) -> torch.Tensor:
    """
    Expand the shared [Tq, Tk] mask to the Torch golden's
    [B, H, Tq, Tk] indexing shape.
    """
    assert mask.dtype == torch.bool
    assert mask.dim() == 2

    return (
        mask
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(
            qk.shape[0],
            qk.shape[1],
            qk.shape[2],
            qk.shape[3],
        )
    )


def _torch_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Golden copied from the current model_int.py attention dataflow:

        qk = Q @ K^T                                  # INT32
        qk = Rshift(qk * multiplier, shift)           # INT32
        qk = clamp(qk, INT16_MIN, INT16_MAX).int16()
        qk[mask] = INT16_MIN
        row_max_plus1 = qk.max(-1, keepdim=True) + 1  # INT16 wrap

    Returns
    -------
    qk:
        INT16 [B, H, Tq, Tk]

    row_max_plus1:
        INT16 [B, H, Tq, 1]
    """
    assert q.dtype == torch.int8
    assert k.dtype == torch.int8
    assert multiplier.dtype == torch.uint8
    assert shift.dtype == torch.int8
    assert multiplier.numel() == 1
    assert shift.numel() == 1

    qk = Batch_Head_s8s8s32_GeMM(
        q,
        k,
    )

    assert qk.dtype == torch.int32

    qk = Rshift(
        qk.mul(multiplier.int()),
        shift,
    )

    assert qk.dtype == torch.int32

    qk = (
        qk
        .clamp(
            INT16_MIN,
            INT16_MAX,
        )
        .to(torch.int16)
    )

    if mask is not None:
        expanded_mask = _expand_mask(
            mask,
            qk,
        )

        # Keep the golden local: do not mutate caller-owned tensors.
        qk = qk.clone()
        qk[expanded_mask] = INT16_MIN

    row_max_plus1 = (
        qk
        .max(
            dim=-1,
            keepdim=True,
        )
        .values
        .to(torch.int16)
        + 1
    )

    assert qk.dtype == torch.int16
    assert row_max_plus1.dtype == torch.int16

    return (
        qk,
        row_max_plus1,
    )


def _run_scalar(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the fused scalar C++ kernel.

    The wrapper deliberately requires contiguous Q/K and a shared
    2-D [Tq, Tk] mask.
    """
    mask_arg = (
        None
        if mask is None
        else mask.contiguous()
    )

    return _C.qk_requant_rowmax_scalar(
        q.contiguous(),
        k.contiguous(),
        mask_arg,
        multiplier.contiguous(),
        shift.contiguous(),
    )


def _assert_equal(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    multiplier: int,
    shift: int,
) -> None:
    multiplier_t = _scalar_u8(multiplier)
    shift_t = _scalar_i8(shift)

    qk_ref, row_max_ref = _torch_golden(
        q,
        k,
        mask,
        multiplier_t,
        shift_t,
    )

    qk_cpp, row_max_cpp = _run_scalar(
        q,
        k,
        mask,
        multiplier_t,
        shift_t,
    )

    assert qk_cpp.dtype == torch.int16
    assert row_max_cpp.dtype == torch.int16

    assert qk_cpp.shape == qk_ref.shape
    assert row_max_cpp.shape == row_max_ref.shape

    assert torch.equal(
        qk_cpp,
        qk_ref,
    ), "QK requant mismatch"

    assert torch.equal(
        row_max_cpp,
        row_max_ref,
    ), "QK row-max(+1) mismatch"


def _cached_causal_mask(
    q_length: int,
    k_length: int,
) -> torch.Tensor:
    """
    Build the rectangular causal mask seen during cached decoding.

    Example:
        Tq = 1, Tk = 4
        -> take absolute query row 3 from a 4x4 causal mask
        -> [False, False, False, False]
    """
    assert q_length <= k_length

    full_mask = torch.triu(
        torch.ones(
            (k_length, k_length),
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    q_offset = k_length - q_length

    return (
        full_mask[
            q_offset:q_offset + q_length,
            :k_length,
        ]
        .contiguous()
    )


# ============================================================
# Random differential tests
# ============================================================

@pytest.mark.parametrize(
    "shape",
    [
        # B, H, Tq, Tk, D
        (1, 1, 3, 3, 8),
        (1, 4, 9, 13, 16),
        (2, 3, 5, 7, 11),
        # Typical cached-decode geometry:
        # one current query against cached + current keys.
        (1, 8, 1, 4, 64),
    ],
)
@pytest.mark.parametrize(
    "multiplier,shift",
    [
        (1, 0),
        (3, 1),
        (17, 7),
        (255, 15),
        (255, 31),
        # Torch Rshift semantics:
        # shift >= 32 -> zero.
        (255, 32),
        # Torch Rshift clamps negative shift counts to zero.
        (13, -3),
    ],
)
def test_attn_qk_requant_rowmax_scalar_random_unmasked(
    shape,
    multiplier,
    shift,
):
    B, H, Tq, Tk, D = shape

    seed = (
        1000
        + B * 101
        + H * 103
        + Tq * 107
        + Tk * 109
        + D * 113
        + multiplier * 127
        + (shift + 8) * 131
    )

    torch.manual_seed(seed)

    q = torch.randint(
        -128,
        128,
        (B, H, Tq, D),
        dtype=torch.int8,
    )

    k = torch.randint(
        -128,
        128,
        (B, H, Tk, D),
        dtype=torch.int8,
    )

    _assert_equal(
        q,
        k,
        mask=None,
        multiplier=multiplier,
        shift=shift,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 3, 3, 8),
        (1, 4, 5, 11, 16),
        (2, 3, 7, 13, 11),
        (1, 8, 1, 4, 64),
    ],
)
@pytest.mark.parametrize(
    "multiplier,shift",
    [
        (1, 0),
        (7, 3),
        (31, 8),
        (255, 15),
    ],
)
def test_attn_qk_requant_rowmax_scalar_random_causal_mask(
    shape,
    multiplier,
    shift,
):
    B, H, Tq, Tk, D = shape

    torch.manual_seed(
        3000
        + B
        + H * 11
        + Tq * 17
        + Tk * 19
        + D * 23
        + multiplier * 29
        + shift * 31
    )

    q = torch.randint(
        -128,
        128,
        (B, H, Tq, D),
        dtype=torch.int8,
    )

    k = torch.randint(
        -128,
        128,
        (B, H, Tk, D),
        dtype=torch.int8,
    )

    mask = _cached_causal_mask(
        Tq,
        Tk,
    )

    _assert_equal(
        q,
        k,
        mask=mask,
        multiplier=multiplier,
        shift=shift,
    )


# ============================================================
# Explicit arithmetic corner cases
# ============================================================

def test_attn_qk_requant_rowmax_scalar_int16_saturation_and_max_wrap():
    """
    Force one positive score above INT16_MAX and one negative score
    below INT16_MIN.

        +: 3 * 127 * 127  =  48387 ->  32767
        -: 3 * 127 * -128 = -48768 -> -32768

    row max is therefore 32767, and the Torch INT16 "+ 1"
    must wrap to -32768.
    """
    q = torch.tensor(
        [[[[127, 127, 127]]]],
        dtype=torch.int8,
    )

    k = torch.tensor(
        [[[
            [127, 127, 127],
            [-128, -128, -128],
        ]]],
        dtype=torch.int8,
    )

    multiplier = _scalar_u8(1)
    shift = _scalar_i8(0)

    qk_ref, row_max_ref = _torch_golden(
        q,
        k,
        None,
        multiplier,
        shift,
    )

    assert torch.equal(
        qk_ref,
        torch.tensor(
            [[[[INT16_MAX, INT16_MIN]]]],
            dtype=torch.int16,
        ),
    )

    assert row_max_ref.item() == INT16_MIN

    qk_cpp, row_max_cpp = _run_scalar(
        q,
        k,
        None,
        multiplier,
        shift,
    )

    assert torch.equal(qk_cpp, qk_ref)
    assert torch.equal(row_max_cpp, row_max_ref)


def test_attn_qk_requant_rowmax_scalar_mask_excludes_large_score():
    """
    A masked score is deliberately much larger than the valid score.
    The row maximum must be computed only from the valid lane.
    """
    q = torch.tensor(
        [[[[127, 127, 127]]]],
        dtype=torch.int8,
    )

    k = torch.tensor(
        [[[
            [127, 127, 127],  # 48387 -> 32767, but masked
            [1, 1, 1],        # 381, valid
        ]]],
        dtype=torch.int8,
    )

    mask = torch.tensor(
        [[True, False]],
        dtype=torch.bool,
    )

    qk_ref, row_max_ref = _torch_golden(
        q,
        k,
        mask,
        _scalar_u8(1),
        _scalar_i8(0),
    )

    assert qk_ref[0, 0, 0, 0].item() == INT16_MIN
    assert qk_ref[0, 0, 0, 1].item() == 381
    assert row_max_ref.item() == 382

    qk_cpp, row_max_cpp = _run_scalar(
        q,
        k,
        mask,
        _scalar_u8(1),
        _scalar_i8(0),
    )

    assert torch.equal(qk_cpp, qk_ref)
    assert torch.equal(row_max_cpp, row_max_ref)


def test_attn_qk_requant_rowmax_scalar_all_masked_row():
    """
    Torch semantics for an all-masked row:

        qk[:] = INT16_MIN
        max   = INT16_MIN
        max+1 = -32767
    """
    torch.manual_seed(5001)

    q = torch.randint(
        -128,
        128,
        (1, 3, 2, 16),
        dtype=torch.int8,
    )

    k = torch.randint(
        -128,
        128,
        (1, 3, 5, 16),
        dtype=torch.int8,
    )

    mask = torch.ones(
        (2, 5),
        dtype=torch.bool,
    )

    qk_cpp, row_max_cpp = _run_scalar(
        q,
        k,
        mask,
        _scalar_u8(17),
        _scalar_i8(4),
    )

    assert torch.all(
        qk_cpp == INT16_MIN
    )

    assert torch.all(
        row_max_cpp == (INT16_MIN + 1)
    )

    qk_ref, row_max_ref = _torch_golden(
        q,
        k,
        mask,
        _scalar_u8(17),
        _scalar_i8(4),
    )

    assert torch.equal(qk_cpp, qk_ref)
    assert torch.equal(row_max_cpp, row_max_ref)


def test_attn_qk_requant_rowmax_scalar_int32_multiplier_wrap():
    """
    Exercise INT32 wrap in:

        qk_int32 * uint8_multiplier

    without overflowing the QK accumulator itself.

        raw QK = 600 * 127 * 127 = 9,677,400
        raw QK * 255 > INT32_MAX

    The C++ kernel must reproduce Torch INT32 wrap semantics before
    the rounded shift/clamp stage.
    """
    D = 600

    q = torch.full(
        (1, 1, 1, D),
        127,
        dtype=torch.int8,
    )

    k = torch.full(
        (1, 1, 1, D),
        127,
        dtype=torch.int8,
    )

    qk_raw = Batch_Head_s8s8s32_GeMM(
        q,
        k,
    )

    mathematical_product = (
        qk_raw.to(torch.int64)
        * 255
    )

    assert mathematical_product.item() > torch.iinfo(torch.int32).max

    wrapped_product = (
        qk_raw
        * torch.tensor(
            [255],
            dtype=torch.int32,
        )
    )

    assert wrapped_product.dtype == torch.int32
    assert wrapped_product.item() < 0

    _assert_equal(
        q,
        k,
        mask=None,
        multiplier=255,
        shift=0,
    )


# ============================================================
# Shape / immutability contract
# ============================================================

def test_attn_qk_requant_rowmax_scalar_does_not_modify_inputs():
    torch.manual_seed(6001)

    q = torch.randint(
        -128,
        128,
        (1, 4, 3, 16),
        dtype=torch.int8,
    )

    k = torch.randint(
        -128,
        128,
        (1, 4, 7, 16),
        dtype=torch.int8,
    )

    mask = _cached_causal_mask(
        3,
        7,
    )

    q_original = q.clone()
    k_original = k.clone()
    mask_original = mask.clone()

    _run_scalar(
        q,
        k,
        mask,
        _scalar_u8(31),
        _scalar_i8(5),
    )

    assert torch.equal(q, q_original)
    assert torch.equal(k, k_original)
    assert torch.equal(mask, mask_original)


def test_attn_qk_requant_rowmax_scalar_rejects_4d_mask():
    """
    The fused kernel intentionally accepts a shared [Tq, Tk] mask.
    It must not silently accept the expanded [B, H, Tq, Tk] Torch
    indexing mask.
    """
    q = torch.zeros(
        (1, 2, 3, 8),
        dtype=torch.int8,
    )

    k = torch.zeros(
        (1, 2, 5, 8),
        dtype=torch.int8,
    )

    bad_mask = torch.zeros(
        (1, 2, 3, 5),
        dtype=torch.bool,
    )

    with pytest.raises(
        RuntimeError,
        match=r"mask must be \[Tq,Tk\]",
    ):
        _C.qk_requant_rowmax_scalar(
            q,
            k,
            bad_mask,
            _scalar_u8(1),
            _scalar_i8(0),
        )
