import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import (
    Fast_Inverse_Square_Root,
    Rshift,
)


ACT_MIN = -127
ACT_MAX = 127


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


def _torch_golden(
    p: torch.Tensor,
    v: torch.Tensor,
    row_sum: torch.Tensor,
    av_multiplier: torch.Tensor,
    av_shift: torch.Tensor,
    rsqrt_offset: int,
) -> torch.Tensor:
    """
    Exact current attention post-softmax golden.

    p:
        UINT8 [B,H,Tq,Tk]

    v:
        INT8 [B,H,D,Tk]

    row_sum:
        INT32 [B,H,Tq,1]

    output:
        INT8 [B,Tq,H*D]
    """
    out = (
        p.to(torch.int32)
        @ v.to(torch.int32).transpose(-1, -2)
    )

    inv = Fast_Inverse_Square_Root(
        row_sum,
        rsqrt_offset,
        Newton_iter=2,
    )

    out = Rshift(
        out.mul(inv).int(),
        rsqrt_offset,
    )

    out = Rshift(
        out.mul(inv).int(),
        rsqrt_offset,
    )

    out = Rshift(
        out.mul(
            av_multiplier.int()
        ).int(),
        av_shift,
    )

    out = (
        out
        .clamp(
            min=ACT_MIN,
            max=ACT_MAX,
        )
        .to(torch.int8)
    )

    return (
        out
        .permute(0, 2, 1, 3)
        .flatten(start_dim=2)
        .contiguous()
    )


def _run_scalar(
    p: torch.Tensor,
    v: torch.Tensor,
    row_sum: torch.Tensor,
    av_multiplier: torch.Tensor,
    av_shift: torch.Tensor,
    rsqrt_offset: int,
) -> torch.Tensor:
    return _C.pv_norm_requant_scalar(
        p.contiguous(),
        v.contiguous(),
        row_sum.contiguous(),
        av_multiplier.contiguous(),
        av_shift.contiguous(),
        rsqrt_offset,
    )


def _assert_equal(
    p: torch.Tensor,
    v: torch.Tensor,
    row_sum: torch.Tensor,
    *,
    av_multiplier: int,
    av_shift: int,
    rsqrt_offset: int,
) -> None:
    multiplier_t = _u8_scalar(
        av_multiplier
    )

    shift_t = _i8_scalar(
        av_shift
    )

    ref = _torch_golden(
        p,
        v,
        row_sum,
        multiplier_t,
        shift_t,
        rsqrt_offset,
    )

    cpp = _run_scalar(
        p,
        v,
        row_sum,
        multiplier_t,
        shift_t,
        rsqrt_offset,
    )

    assert cpp.dtype == torch.int8
    assert cpp.shape == ref.shape

    assert torch.equal(
        cpp,
        ref,
    ), "fused PV/normalize/requant mismatch"


@pytest.mark.parametrize(
    "shape",
    [
        # B, H, Tq, Tk, D
        (1, 1, 3, 5, 4),
        (1, 4, 9, 13, 16),
        (2, 3, 5, 7, 11),
        (1, 8, 7, 17, 64),

        # Cached decode.
        (1, 8, 1, 4, 64),
        (1, 8, 1, 31, 64),
    ],
)
@pytest.mark.parametrize(
    "av_multiplier,av_shift",
    [
        (1, 0),
        (3, 1),
        (17, 7),
        (255, 15),
        (255, 31),
        (255, 32),
        (13, -3),
    ],
)
def test_attn_pv_norm_requant_scalar_random(
    shape,
    av_multiplier,
    av_shift,
):
    B, H, Tq, Tk, D = shape

    torch.manual_seed(
        1000
        + B * 101
        + H * 103
        + Tq * 107
        + Tk * 109
        + D * 113
        + av_multiplier * 127
        + av_shift * 131
    )

    p = torch.randint(
        0,
        256,
        (B, H, Tq, Tk),
        dtype=torch.uint8,
    )

    v = torch.randint(
        -128,
        128,
        (B, H, D, Tk),
        dtype=torch.int8,
    )

    row_sum = (
        p.sum(
            dim=-1,
            keepdim=True,
        )
        .int()
    )

    _assert_equal(
        p,
        v,
        row_sum,
        av_multiplier=av_multiplier,
        av_shift=av_shift,
        rsqrt_offset=16,
    )


@pytest.mark.parametrize(
    "rsqrt_offset",
    [13, 16],
)
def test_attn_pv_norm_requant_scalar_rsqrt_offsets(
    rsqrt_offset,
):
    torch.manual_seed(
        5000 + rsqrt_offset
    )

    p = torch.randint(
        0,
        256,
        (1, 3, 4, 19),
        dtype=torch.uint8,
    )

    v = torch.randint(
        -128,
        128,
        (1, 3, 8, 19),
        dtype=torch.int8,
    )

    row_sum = (
        p.sum(
            dim=-1,
            keepdim=True,
        )
        .int()
    )

    _assert_equal(
        p,
        v,
        row_sum,
        av_multiplier=23,
        av_shift=8,
        rsqrt_offset=rsqrt_offset,
    )


def test_attn_pv_norm_requant_scalar_zero_probability_row():
    """
    All-masked/zero-exp row:
        P == 0
        row_sum == 0
        PV == 0

    Final output must remain zero regardless of the rsqrt seed.
    """
    B, H, Tq, Tk, D = (
        1,
        3,
        2,
        7,
        5,
    )

    p = torch.zeros(
        (B, H, Tq, Tk),
        dtype=torch.uint8,
    )

    torch.manual_seed(6001)

    v = torch.randint(
        -128,
        128,
        (B, H, D, Tk),
        dtype=torch.int8,
    )

    row_sum = torch.zeros(
        (B, H, Tq, 1),
        dtype=torch.int32,
    )

    cpp = _run_scalar(
        p,
        v,
        row_sum,
        _u8_scalar(255),
        _i8_scalar(0),
        16,
    )

    assert torch.count_nonzero(
        cpp
    ).item() == 0

    ref = _torch_golden(
        p,
        v,
        row_sum,
        _u8_scalar(255),
        _i8_scalar(0),
        16,
    )

    assert torch.equal(
        cpp,
        ref,
    )


def test_attn_pv_norm_requant_scalar_uses_supplied_row_sum():
    """
    row_sum is an explicit input from shift_exp_scalar.
    Ensure the fused kernel does not silently recompute it from P.
    """
    p = torch.tensor(
        [[[
            [10, 20, 30, 40],
            [4, 3, 2, 1],
        ]]],
        dtype=torch.uint8,
    )

    v = torch.tensor(
        [[[
            [1, 2, 3, 4],
            [-4, -3, -2, -1],
            [7, -5, 3, -1],
        ]]],
        dtype=torch.int8,
    )

    # Deliberately different from p.sum(...).
    row_sum = torch.tensor(
        [[[
            [123],
            [37],
        ]]],
        dtype=torch.int32,
    )

    _assert_equal(
        p,
        v,
        row_sum,
        av_multiplier=19,
        av_shift=4,
        rsqrt_offset=16,
    )


def test_attn_pv_norm_requant_scalar_int32_pv_wrap():
    """
    Force P@V above INT32_MAX before normalization.

        255 * 127 * 70000
        = 2,266,950,000
        > INT32_MAX

    This validates that fusion preserves the already-tested
    modulo-2^32 P@V accumulation semantics.
    """
    Tk = 70_000

    p = torch.full(
        (1, 1, 1, Tk),
        255,
        dtype=torch.uint8,
    )

    v = torch.full(
        (1, 1, 1, Tk),
        127,
        dtype=torch.int8,
    )

    row_sum = (
        p.sum(
            dim=-1,
            keepdim=True,
        )
        .int()
    )

    raw_pv = (
        p.to(torch.int32)
        @ v.to(torch.int32).transpose(-1, -2)
    )

    assert raw_pv.item() < 0

    _assert_equal(
        p,
        v,
        row_sum,
        av_multiplier=1,
        av_shift=0,
        rsqrt_offset=16,
    )


def test_attn_pv_norm_requant_scalar_final_layout():
    """
    Explicitly verify [B,H,Tq,D] -> [B,Tq,H*D] store order.
    """
    torch.manual_seed(7001)

    p = torch.randint(
        1,
        32,
        (1, 2, 3, 5),
        dtype=torch.uint8,
    )

    v = torch.randint(
        -16,
        17,
        (1, 2, 4, 5),
        dtype=torch.int8,
    )

    row_sum = (
        p.sum(
            dim=-1,
            keepdim=True,
        )
        .int()
    )

    ref = _torch_golden(
        p,
        v,
        row_sum,
        _u8_scalar(31),
        _i8_scalar(3),
        16,
    )

    cpp = _run_scalar(
        p,
        v,
        row_sum,
        _u8_scalar(31),
        _i8_scalar(3),
        16,
    )

    assert cpp.shape == (
        1,
        3,
        8,
    )

    assert torch.equal(
        cpp,
        ref,
    )


def test_attn_pv_norm_requant_scalar_does_not_modify_inputs():
    torch.manual_seed(8001)

    p = torch.randint(
        0,
        256,
        (1, 4, 3, 11),
        dtype=torch.uint8,
    )

    v = torch.randint(
        -128,
        128,
        (1, 4, 16, 11),
        dtype=torch.int8,
    )

    row_sum = (
        p.sum(
            dim=-1,
            keepdim=True,
        )
        .int()
    )

    p_ref = p.clone()
    v_ref = v.clone()
    sum_ref = row_sum.clone()

    _run_scalar(
        p,
        v,
        row_sum,
        _u8_scalar(17),
        _i8_scalar(7),
        16,
    )

    assert torch.equal(p, p_ref)
    assert torch.equal(v, v_ref)
    assert torch.equal(row_sum, sum_ref)


def test_attn_pv_norm_requant_scalar_rejects_negative_row_sum():
    p = torch.ones(
        (1, 1, 1, 4),
        dtype=torch.uint8,
    )

    v = torch.ones(
        (1, 1, 2, 4),
        dtype=torch.int8,
    )

    row_sum = torch.tensor(
        [[[[-1]]]],
        dtype=torch.int32,
    )

    with pytest.raises(
        (RuntimeError, ValueError),
    ):
        _run_scalar(
            p,
            v,
            row_sum,
            _u8_scalar(1),
            _i8_scalar(0),
            16,
        )


def test_attn_pv_norm_requant_scalar_rejects_bad_row_sum_shape():
    p = torch.ones(
        (1, 2, 3, 5),
        dtype=torch.uint8,
    )

    v = torch.ones(
        (1, 2, 4, 5),
        dtype=torch.int8,
    )

    bad_row_sum = torch.ones(
        (1, 2, 3, 2),
        dtype=torch.int32,
    )

    with pytest.raises(
        RuntimeError,
        match="last dimension 1",
    ):
        _C.pv_norm_requant_scalar(
            p,
            v,
            bad_row_sum,
            _u8_scalar(1),
            _i8_scalar(0),
            16,
        )
