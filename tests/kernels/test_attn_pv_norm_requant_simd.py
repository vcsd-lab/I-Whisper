import pytest
import torch

from i_whisper.kernels import _C


def _u8_scalar(value: int):
    return torch.tensor(
        [value],
        dtype=torch.uint8,
    )


def _i8_scalar(value: int):
    return torch.tensor(
        [value],
        dtype=torch.int8,
    )


def _pv_scalar(p, v):
    return _C.pv_scalar(
        p.contiguous(),
        v.contiguous(),
    )


def _pv_simd(p, v):
    return _C.pv_simd(
        p.contiguous(),
        v.contiguous(),
    )


def _fused_scalar(
    p,
    v,
    row_sum,
    multiplier,
    shift,
    rsqrt_offset=16,
):
    return _C.pv_norm_requant_scalar(
        p.contiguous(),
        v.contiguous(),
        row_sum.contiguous(),
        _u8_scalar(multiplier),
        _i8_scalar(shift),
        rsqrt_offset,
    )


def _fused_simd(
    p,
    v,
    row_sum,
    multiplier,
    shift,
    rsqrt_offset=16,
):
    return _C.pv_norm_requant_simd(
        p.contiguous(),
        v.contiguous(),
        row_sum.contiguous(),
        _u8_scalar(multiplier),
        _i8_scalar(shift),
        rsqrt_offset,
    )


@pytest.mark.parametrize(
    "shape",
    [
        # B,H,Tq,Tk,D
        (1, 1, 3, 7, 4),
        (1, 2, 3, 8, 5),
        (1, 3, 4, 15, 8),
        (1, 4, 5, 16, 9),
        (1, 4, 5, 17, 16),
        (1, 4, 3, 31, 17),
        (1, 8, 1, 32, 64),   # decode-like
        (1, 8, 1, 33, 65),
        (2, 3, 5, 64, 64),   # encoder-like
    ],
)
def test_attn_pv_simd_matches_scalar(
    shape,
):
    B, H, Tq, Tk, D = shape

    torch.manual_seed(
        1000
        + B * 11
        + H * 13
        + Tq * 17
        + Tk * 19
        + D * 23
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

    ref = _pv_scalar(
        p,
        v,
    )

    out = _pv_simd(
        p,
        v,
    )

    assert out.dtype == torch.int32
    assert out.shape == ref.shape

    assert torch.equal(
        out,
        ref,
    ), "raw PV SIMD mismatch vs sealed scalar"


def test_attn_pv_simd_unsigned_extremes():
    p = torch.tensor(
        [[[
            [0, 1, 127, 128, 129, 254, 255, 255,
             128, 0, 255, 17, 31, 63, 127, 255, 1]
        ]]],
        dtype=torch.uint8,
    )

    v = torch.tensor(
        [[[
            [
                -128, -127, -1, 0, 1, 126, 127, -128,
                127, -1, 1, -64, 64, -32, 32, 7, -7
            ],
            [
                127, 126, 1, 0, -1, -127, -128, 127,
                -128, 1, -1, 64, -64, 32, -32, -7, 7
            ],
            [1] * 17,
            [-1] * 17,
        ]]],
        dtype=torch.int8,
    )

    assert torch.equal(
        _pv_simd(p, v),
        _pv_scalar(p, v),
    )


def test_attn_pv_simd_int32_wrap():
    # Same overflow stress used for the sealed scalar PV kernel.
    Tk = 70_000

    p = torch.full(
        (1, 1, 1, Tk),
        255,
        dtype=torch.uint8,
    )

    v = torch.full(
        (1, 1, 4, Tk),
        127,
        dtype=torch.int8,
    )

    ref = _pv_scalar(p, v)
    out = _pv_simd(p, v)

    assert torch.all(ref < 0)
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "shape",
    [
        # Explicit D vector/tail boundaries.
        (1, 2, 3, 7, 3),
        (1, 2, 3, 8, 4),
        (1, 2, 3, 9, 5),
        (1, 4, 5, 15, 7),
        (1, 4, 5, 16, 8),
        (1, 4, 5, 17, 9),
        (1, 8, 1, 31, 63),
        (1, 8, 1, 32, 64),
        (1, 8, 1, 33, 65),
        (1, 8, 12, 64, 64),
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
        (255, 32),
        (13, -3),
    ],
)
def test_attn_pv_norm_requant_simd_matches_scalar(
    shape,
    multiplier,
    shift,
):
    B, H, Tq, Tk, D = shape

    torch.manual_seed(
        3000
        + B * 29
        + H * 31
        + Tq * 37
        + Tk * 41
        + D * 43
        + multiplier
        + shift
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

    ref = _fused_scalar(
        p,
        v,
        row_sum,
        multiplier,
        shift,
    )

    out = _fused_simd(
        p,
        v,
        row_sum,
        multiplier,
        shift,
    )

    assert out.dtype == torch.int8
    assert out.shape == ref.shape

    assert torch.equal(
        out,
        ref,
    ), "fused PV SIMD mismatch vs sealed scalar"


def test_attn_pv_norm_requant_simd_zero_probability_row():
    p = torch.zeros(
        (1, 4, 3, 17),
        dtype=torch.uint8,
    )

    torch.manual_seed(5001)

    v = torch.randint(
        -128,
        128,
        (1, 4, 64, 17),
        dtype=torch.int8,
    )

    row_sum = torch.zeros(
        (1, 4, 3, 1),
        dtype=torch.int32,
    )

    out = _fused_simd(
        p,
        v,
        row_sum,
        255,
        0,
    )

    assert torch.count_nonzero(
        out
    ).item() == 0

    assert torch.equal(
        out,
        _fused_scalar(
            p,
            v,
            row_sum,
            255,
            0,
        ),
    )


def test_attn_pv_norm_requant_simd_uses_supplied_row_sum():
    torch.manual_seed(6001)

    p = torch.randint(
        1,
        32,
        (1, 2, 3, 16),
        dtype=torch.uint8,
    )

    v = torch.randint(
        -32,
        33,
        (1, 2, 8, 16),
        dtype=torch.int8,
    )

    # Deliberately not equal to p.sum().
    row_sum = torch.tensor(
        [
            [
                [[123], [257], [511]],
                [[91], [333], [701]],
            ]
        ],
        dtype=torch.int32,
    )

    assert torch.equal(
        _fused_simd(
            p,
            v,
            row_sum,
            19,
            4,
        ),
        _fused_scalar(
            p,
            v,
            row_sum,
            19,
            4,
        ),
    )


def test_attn_pv_norm_requant_simd_final_layout():
    # Different head values make head ordering observable.
    B, H, Tq, Tk, D = (
        1, 3, 2, 16, 4
    )

    p = torch.zeros(
        (B, H, Tq, Tk),
        dtype=torch.uint8,
    )

    p[:, 0].fill_(1)
    p[:, 1].fill_(2)
    p[:, 2].fill_(3)

    v = torch.zeros(
        (B, H, D, Tk),
        dtype=torch.int8,
    )

    for h in range(H):
        for d in range(D):
            v[:, h, d].fill_(
                h * 10 + d + 1
            )

    row_sum = (
        p.sum(
            dim=-1,
            keepdim=True,
        )
        .int()
    )

    ref = _fused_scalar(
        p,
        v,
        row_sum,
        31,
        4,
    )

    out = _fused_simd(
        p,
        v,
        row_sum,
        31,
        4,
    )

    assert out.shape == (
        B,
        Tq,
        H * D,
    )

    assert torch.equal(
        out,
        ref,
    )


def test_attn_pv_norm_requant_simd_does_not_modify_inputs():
    torch.manual_seed(7001)

    p = torch.randint(
        0,
        256,
        (1, 4, 3, 33),
        dtype=torch.uint8,
    )

    v = torch.randint(
        -128,
        128,
        (1, 4, 64, 33),
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

    _fused_simd(
        p,
        v,
        row_sum,
        17,
        7,
    )

    assert torch.equal(p, p_ref)
    assert torch.equal(v, v_ref)
    assert torch.equal(row_sum, sum_ref)
