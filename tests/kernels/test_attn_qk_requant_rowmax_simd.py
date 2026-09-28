import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import (
    Batch_Head_s8s8s32_GeMM,
    Rshift,
)


INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max


def _u8_scalar(value: int) -> torch.Tensor:
    return torch.tensor(
        [value],
        dtype=torch.uint8,
    )


def _i8_scalar(value: int) -> torch.Tensor:
    return torch.tensor(
        [value],
        dtype=torch.int8,
    )


def _expand_mask(
    mask: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
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


def _torch_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
):
    qk = Batch_Head_s8s8s32_GeMM(
        q,
        k,
    )

    qk = Rshift(
        qk.mul(
            multiplier.int()
        ),
        shift,
    )

    qk = (
        qk
        .clamp(
            INT16_MIN,
            INT16_MAX,
        )
        .to(torch.int16)
    )

    if mask is not None:
        qk[
            _expand_mask(
                mask,
                qk,
            )
        ] = INT16_MIN

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

    return (
        qk,
        row_max_plus1,
    )


def _run_scalar(
    q,
    k,
    mask,
    multiplier,
    shift,
):
    return _C.qk_requant_rowmax_scalar(
        q.contiguous(),
        k.contiguous(),
        None if mask is None else mask.contiguous(),
        multiplier.contiguous(),
        shift.contiguous(),
    )


def _run_simd(
    q,
    k,
    mask,
    multiplier,
    shift,
):
    return _C.qk_requant_rowmax_simd(
        q.contiguous(),
        k.contiguous(),
        None if mask is None else mask.contiguous(),
        multiplier.contiguous(),
        shift.contiguous(),
    )


def _assert_equal(
    q,
    k,
    mask,
    *,
    multiplier,
    shift,
):
    multiplier_t = _u8_scalar(
        multiplier
    )

    shift_t = _i8_scalar(
        shift
    )

    ref_qk, ref_max = _torch_golden(
        q,
        k,
        mask,
        multiplier_t,
        shift_t,
    )

    scalar_qk, scalar_max = _run_scalar(
        q,
        k,
        mask,
        multiplier_t,
        shift_t,
    )

    simd_qk, simd_max = _run_simd(
        q,
        k,
        mask,
        multiplier_t,
        shift_t,
    )

    assert simd_qk.dtype == torch.int16
    assert simd_max.dtype == torch.int16

    assert simd_qk.shape == ref_qk.shape
    assert simd_max.shape == ref_max.shape

    # SIMD must independently match the Torch golden.
    assert torch.equal(
        simd_qk,
        ref_qk,
    ), "SIMD QK mismatch vs Torch"

    assert torch.equal(
        simd_max,
        ref_max,
    ), "SIMD row-max mismatch vs Torch"

    # And match the already-sealed scalar kernel.
    assert torch.equal(
        simd_qk,
        scalar_qk,
    ), "SIMD QK mismatch vs scalar"

    assert torch.equal(
        simd_max,
        scalar_max,
    ), "SIMD row-max mismatch vs scalar"


@pytest.mark.parametrize(
    "head_dim",
    [
        1,
        7,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
    ],
)
@pytest.mark.parametrize(
    "shift",
    [
        -3,
        0,
        1,
        7,
        31,
        32,
    ],
)
def test_attn_qk_requant_rowmax_simd_vector_boundaries(
    head_dim,
    shift,
):
    torch.manual_seed(
        1000
        + head_dim * 101
        + shift * 103
    )

    q = torch.randint(
        -128,
        128,
        (1, 3, 4, head_dim),
        dtype=torch.int8,
    )

    k = torch.randint(
        -128,
        128,
        (1, 3, 6, head_dim),
        dtype=torch.int8,
    )

    _assert_equal(
        q,
        k,
        None,
        multiplier=173,
        shift=shift,
    )


@pytest.mark.parametrize(
    "shape",
    [
        # B,H,Tq,Tk,D
        (1, 1, 3, 3, 64),
        (1, 4, 9, 13, 64),
        (2, 3, 5, 7, 17),

        # cached decode
        (1, 8, 1, 4, 64),
        (1, 8, 1, 31, 64),
    ],
)
def test_attn_qk_requant_rowmax_simd_random_masked(
    shape,
):
    B, H, Tq, Tk, D = shape

    torch.manual_seed(
        3000
        + B * 11
        + H * 13
        + Tq * 17
        + Tk * 19
        + D * 23
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

    # Shared 2-D mask, matching production contract.
    mask = (
        torch.rand(
            (Tq, Tk)
        )
        < 0.30
    )

    # Keep at least one valid key in every row.
    mask[:, 0] = False

    _assert_equal(
        q,
        k,
        mask,
        multiplier=211,
        shift=8,
    )


def test_attn_qk_requant_rowmax_simd_all_masked():
    torch.manual_seed(4001)

    q = torch.randint(
        -128,
        128,
        (1, 4, 3, 64),
        dtype=torch.int8,
    )

    k = torch.randint(
        -128,
        128,
        (1, 4, 5, 64),
        dtype=torch.int8,
    )

    mask = torch.ones(
        (3, 5),
        dtype=torch.bool,
    )

    simd_qk, simd_max = _run_simd(
        q,
        k,
        mask,
        _u8_scalar(255),
        _i8_scalar(0),
    )

    assert torch.all(
        simd_qk == INT16_MIN
    )

    assert torch.all(
        simd_max
        == INT16_MIN + 1
    )

    _assert_equal(
        q,
        k,
        mask,
        multiplier=255,
        shift=0,
    )


def test_attn_qk_requant_rowmax_simd_saturation():
    # Positive and negative saturation in the same row.
    D = 64

    q = torch.full(
        (1, 1, 1, D),
        127,
        dtype=torch.int8,
    )

    k = torch.empty(
        (1, 1, 2, D),
        dtype=torch.int8,
    )

    k[0, 0, 0].fill_(127)
    k[0, 0, 1].fill_(-128)

    _assert_equal(
        q,
        k,
        None,
        multiplier=255,
        shift=0,
    )


def test_attn_qk_requant_rowmax_simd_rowmax_plus1_wrap():
    """
    Construct a lane whose requantized value saturates to 32767.
    row_max_plus1 must wrap INT16:
        32767 + 1 -> -32768.
    """
    D = 64

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

    simd_qk, simd_max = _run_simd(
        q,
        k,
        None,
        _u8_scalar(255),
        _i8_scalar(0),
    )

    assert simd_qk.item() == 32767
    assert simd_max.item() == -32768

    _assert_equal(
        q,
        k,
        None,
        multiplier=255,
        shift=0,
    )


def test_attn_qk_requant_rowmax_simd_mask_excludes_large_score():
    D = 64

    q = torch.full(
        (1, 1, 1, D),
        127,
        dtype=torch.int8,
    )

    k = torch.zeros(
        (1, 1, 2, D),
        dtype=torch.int8,
    )

    # Lane 0 would be the huge maximum, but it is masked.
    k[0, 0, 0].fill_(127)
    k[0, 0, 1].fill_(1)

    mask = torch.tensor(
        [[True, False]],
        dtype=torch.bool,
    )

    _assert_equal(
        q,
        k,
        mask,
        multiplier=1,
        shift=0,
    )


def test_attn_qk_requant_rowmax_simd_does_not_modify_inputs():
    torch.manual_seed(5001)

    q = torch.randint(
        -128,
        128,
        (1, 4, 3, 64),
        dtype=torch.int8,
    )

    k = torch.randint(
        -128,
        128,
        (1, 4, 7, 64),
        dtype=torch.int8,
    )

    mask = torch.zeros(
        (3, 7),
        dtype=torch.bool,
    )

    q_ref = q.clone()
    k_ref = k.clone()
    mask_ref = mask.clone()

    _run_simd(
        q,
        k,
        mask,
        _u8_scalar(113),
        _i8_scalar(7),
    )

    assert torch.equal(q, q_ref)
    assert torch.equal(k, k_ref)
    assert torch.equal(mask, mask_ref)
