import pytest
import torch

from i_whisper.kernels import _C


def _torch_golden(
    q: torch.Tensor,
    k: torch.Tensor,
):
    assert q.dtype == torch.int8
    assert k.dtype == torch.int8

    return torch.einsum(
        "bhtd,bhsd->bhts",
        q.int(),
        k.int(),
    )


def _run_scalar(
    q: torch.Tensor,
    k: torch.Tensor,
):
    return _C.qk_scalar(
        q.contiguous(),
        k.contiguous(),
    )


def _assert_equal(
    q,
    k,
):
    ref = _torch_golden(
        q,
        k,
    )

    out = _run_scalar(
        q,
        k,
    )

    assert out.dtype == torch.int32
    assert out.shape == ref.shape

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
            "QK scalar mismatch\n"
            f"q shape: {tuple(q.shape)}\n"
            f"k shape: {tuple(k.shape)}\n"
            f"first mismatch: {idx}\n"
            f"reference: {ref[idx].item()}\n"
            f"actual:    {out[idx].item()}"
        )


def test_qk_scalar_tiny():
    q = torch.tensor(
        [[[[
            1, 2, 3, 4,
            -1, -2, -3, -4,
        ]]]],
        dtype=torch.int8,
    )

    k = torch.tensor(
        [[[
            [
                1, 1, 1, 1,
                1, 1, 1, 1,
            ],
            [
                -1, -1, -1, -1,
                -1, -1, -1, -1,
            ],
        ]]],
        dtype=torch.int8,
    )

    _assert_equal(
        q,
        k,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 3, 5, 8),
        (1, 2, 7, 11, 16),
        (2, 4, 17, 23, 32),
        (1, 8, 31, 37, 64),
    ],
)
def test_qk_scalar_random(
    shape,
):
    torch.manual_seed(1234)

    B, H, Tq, Tk, D = shape

    q = torch.randint(
        -127,
        128,
        (B, H, Tq, D),
        dtype=torch.int8,
    )

    k = torch.randint(
        -127,
        128,
        (B, H, Tk, D),
        dtype=torch.int8,
    )

    _assert_equal(
        q,
        k,
    )


def test_qk_scalar_extreme_positive():
    B = 1
    H = 2
    Tq = 5
    Tk = 7
    D = 64

    q = torch.full(
        (B, H, Tq, D),
        127,
        dtype=torch.int8,
    )

    k = torch.full(
        (B, H, Tk, D),
        127,
        dtype=torch.int8,
    )

    _assert_equal(
        q,
        k,
    )


def test_qk_scalar_extreme_negative():
    B = 1
    H = 2
    Tq = 5
    Tk = 7
    D = 64

    q = torch.full(
        (B, H, Tq, D),
        -127,
        dtype=torch.int8,
    )

    k = torch.full(
        (B, H, Tk, D),
        -127,
        dtype=torch.int8,
    )

    _assert_equal(
        q,
        k,
    )


def test_qk_scalar_mixed_sign():
    D = 64

    q_pattern = torch.tensor(
        [
            127 if i % 2 == 0
            else -127
            for i in range(D)
        ],
        dtype=torch.int8,
    )

    k_pattern = -q_pattern

    q = q_pattern.reshape(
        1,
        1,
        1,
        D,
    )

    k = k_pattern.reshape(
        1,
        1,
        1,
        D,
    )

    _assert_equal(
        q,
        k,
    )


def test_qk_scalar_cross_attention():
    """
    Explicitly test Tq != Tk.
    """

    torch.manual_seed(5678)

    q = torch.randint(
        -127,
        128,
        (1, 8, 13, 64),
        dtype=torch.int8,
    )

    k = torch.randint(
        -127,
        128,
        (1, 8, 29, 64),
        dtype=torch.int8,
    )

    _assert_equal(
        q,
        k,
    )