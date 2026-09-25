import pytest
import torch

from i_whisper.kernels import _C


def _torch_golden(
    p: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    p:
        UINT8 [B,H,Tq,Tk]

    v:
        INT8 [B,H,D,Tk]

    return:
        INT32 [B,H,Tq,D]
    """
    assert p.dtype == torch.uint8
    assert v.dtype == torch.int8

    return (
        p.to(torch.int32)
        @ v.to(torch.int32).transpose(-1, -2)
    )


def _run_scalar(
    p: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return _C.pv_scalar(
        p.contiguous(),
        v.contiguous(),
    )


def _assert_equal(
    p: torch.Tensor,
    v: torch.Tensor,
) -> None:
    ref = _torch_golden(
        p,
        v,
    )

    cpp = _run_scalar(
        p,
        v,
    )

    assert cpp.dtype == torch.int32

    assert cpp.shape == ref.shape

    assert torch.equal(
        cpp,
        ref,
    ), "P@V scalar accumulation mismatch"


@pytest.mark.parametrize(
    "shape",
    [
        # B, H, Tq, Tk, D
        (1, 1, 3, 5, 4),
        (1, 4, 9, 13, 16),
        (2, 3, 5, 7, 11),

        # Typical head dimension.
        (1, 8, 7, 17, 64),

        # Cached decode:
        # one query attends over a growing KV cache.
        (1, 8, 1, 4, 64),
        (1, 8, 1, 31, 64),
    ],
)
def test_attn_pv_scalar_random(
    shape,
):
    B, H, Tq, Tk, D = shape

    torch.manual_seed(
        1000
        + B * 101
        + H * 103
        + Tq * 107
        + Tk * 109
        + D * 113
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

    _assert_equal(
        p,
        v,
    )


def test_attn_pv_scalar_explicit_signed_unsigned_products():
    """
    Exercise the full UINT8 x INT8 product range, including:

        255 *  127
        255 * -128
          0 * -128
          1 *   -1
    """
    p = torch.tensor(
        [[[[
            255,
            255,
            0,
            1,
        ]]]],
        dtype=torch.uint8,
    )

    v = torch.tensor(
        [[[
            [
                127,
                -128,
                -128,
                -1,
            ],
            [
                -128,
                -128,
                127,
                127,
            ],
        ]]],
        dtype=torch.int8,
    )

    _assert_equal(
        p,
        v,
    )


def test_attn_pv_scalar_int32_accumulation_wrap():
    """
    Torch INT32 matmul wraps modulo 2^32.

    Force the mathematical result above INT32_MAX:

        255 * 127 * 70000
        = 2,266,950,000
        > 2,147,483,647

    The scalar kernel must reproduce the same INT32 bit pattern.
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

    mathematical = (
        255
        * 127
        * Tk
    )

    assert mathematical > torch.iinfo(
        torch.int32
    ).max

    ref = _torch_golden(
        p,
        v,
    )

    # Confirm that this test actually hit signed INT32 wrap.
    assert ref.item() < 0

    cpp = _run_scalar(
        p,
        v,
    )

    assert torch.equal(
        cpp,
        ref,
    )


def test_attn_pv_scalar_zero_probability_row():
    """
    A zero P row must produce an all-zero output vector.
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

    torch.manual_seed(5001)

    v = torch.randint(
        -128,
        128,
        (B, H, D, Tk),
        dtype=torch.int8,
    )

    cpp = _run_scalar(
        p,
        v,
    )

    assert torch.count_nonzero(
        cpp
    ).item() == 0


def test_attn_pv_scalar_does_not_modify_inputs():
    torch.manual_seed(6001)

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

    p_original = p.clone()
    v_original = v.clone()

    _run_scalar(
        p,
        v,
    )

    assert torch.equal(
        p,
        p_original,
    )

    assert torch.equal(
        v,
        v_original,
    )


def test_attn_pv_scalar_rejects_key_length_mismatch():
    p = torch.zeros(
        (1, 2, 3, 5),
        dtype=torch.uint8,
    )

    v = torch.zeros(
        (1, 2, 4, 6),
        dtype=torch.int8,
    )

    with pytest.raises(
        RuntimeError,
        match="key length mismatch",
    ):
        _C.pv_scalar(
            p,
            v,
        )


def test_attn_pv_scalar_rejects_head_mismatch():
    p = torch.zeros(
        (1, 2, 3, 5),
        dtype=torch.uint8,
    )

    v = torch.zeros(
        (1, 3, 4, 5),
        dtype=torch.int8,
    )

    with pytest.raises(
        RuntimeError,
        match="head count mismatch",
    ):
        _C.pv_scalar(
            p,
            v,
        )
