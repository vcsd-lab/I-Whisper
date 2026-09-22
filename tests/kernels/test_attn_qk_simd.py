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


def _run_simd(
    q: torch.Tensor,
    k: torch.Tensor,
):
    return _C.qk_simd(
        q.contiguous(),
        k.contiguous(),
    )


def _raise_mismatch(
    name,
    actual,
    ref,
    q,
    k,
):
    mismatch = (
        actual != ref
    ).nonzero(
        as_tuple=False
    )

    idx = tuple(
        mismatch[0].tolist()
    )

    b, h, qi, ki = idx

    raise AssertionError(
        f"{name} mismatch\n"
        f"q shape: {tuple(q.shape)}\n"
        f"k shape: {tuple(k.shape)}\n"
        f"first mismatch: {idx}\n"
        f"q vector: {q[b, h, qi].tolist()}\n"
        f"k vector: {k[b, h, ki].tolist()}\n"
        f"reference: {ref[idx].item()}\n"
        f"actual:    {actual[idx].item()}"
    )


def _assert_all_equal(
    q: torch.Tensor,
    k: torch.Tensor,
):
    ref = _torch_golden(
        q,
        k,
    )

    scalar = _run_scalar(
        q,
        k,
    )

    simd = _run_simd(
        q,
        k,
    )

    assert ref.dtype == torch.int32
    assert scalar.dtype == torch.int32
    assert simd.dtype == torch.int32

    expected_shape = (
        q.shape[0],
        q.shape[1],
        q.shape[2],
        k.shape[2],
    )

    assert ref.shape == expected_shape
    assert scalar.shape == expected_shape
    assert simd.shape == expected_shape

    if not torch.equal(
        scalar,
        ref,
    ):
        _raise_mismatch(
            "scalar vs Torch",
            scalar,
            ref,
            q,
            k,
        )

    if not torch.equal(
        simd,
        ref,
    ):
        _raise_mismatch(
            "SIMD vs Torch",
            simd,
            ref,
            q,
            k,
        )

    if not torch.equal(
        simd,
        scalar,
    ):
        _raise_mismatch(
            "SIMD vs scalar",
            simd,
            scalar,
            q,
            k,
        )


# ============================================================
# Basic correctness
# ============================================================

def test_attn_qk_simd_tiny():
    q = torch.tensor(
        [[[
            [
                1, 2, 3, 4,
                -1, -2, -3, -4,
            ],
        ]]],
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

    _assert_all_equal(
        q,
        k,
    )


# ============================================================
# Random shapes
# ============================================================

@pytest.mark.parametrize(
    "shape",
    [
        # B, H, Tq, Tk, D

        # Tiny
        (1, 1, 3, 5, 8),

        # Multiple heads
        (1, 2, 7, 11, 16),

        # Multiple batches + heads
        (2, 4, 17, 23, 32),

        # Whisper-like head_dim
        (1, 8, 31, 37, 64),
    ],
)
def test_attn_qk_simd_random(
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

    _assert_all_equal(
        q,
        k,
    )


# ============================================================
# SIMD boundary / tail
# ============================================================

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
def test_attn_qk_simd_head_dim_tail(
    head_dim,
):
    """
    Exercise all SIMD/tail boundaries.

        D < 16
            scalar tail only

        D == 16
            one complete SIMD block

        D == 16n
            SIMD only

        D == 16n +/- 1
            SIMD + scalar tail
    """
    torch.manual_seed(
        2000 + head_dim
    )

    q = torch.randint(
        -127,
        128,
        (1, 3, 7, head_dim),
        dtype=torch.int8,
    )

    k = torch.randint(
        -127,
        128,
        (1, 3, 9, head_dim),
        dtype=torch.int8,
    )

    _assert_all_equal(
        q,
        k,
    )


# ============================================================
# Production head dimension
# ============================================================

def test_attn_qk_simd_whisper_head_dim():
    """
    Whisper production-relevant head dimension.
    """
    torch.manual_seed(3001)

    q = torch.randint(
        -127,
        128,
        (1, 8, 37, 64),
        dtype=torch.int8,
    )

    k = torch.randint(
        -127,
        128,
        (1, 8, 41, 64),
        dtype=torch.int8,
    )

    _assert_all_equal(
        q,
        k,
    )


# ============================================================
# Cross attention
# ============================================================

def test_attn_qk_simd_cross_attention():
    """
    Explicit Tq != Tk case.

    Important because decoder cross-attention naturally has
    different query and key sequence lengths.
    """
    torch.manual_seed(4001)

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

    _assert_all_equal(
        q,
        k,
    )


# ============================================================
# Extreme value cases
# ============================================================

def test_attn_qk_simd_extreme_positive():
    q = torch.full(
        (1, 2, 5, 64),
        127,
        dtype=torch.int8,
    )

    k = torch.full(
        (1, 2, 7, 64),
        127,
        dtype=torch.int8,
    )

    _assert_all_equal(
        q,
        k,
    )


def test_attn_qk_simd_extreme_negative():
    q = torch.full(
        (1, 2, 5, 64),
        -127,
        dtype=torch.int8,
    )

    k = torch.full(
        (1, 2, 7, 64),
        -127,
        dtype=torch.int8,
    )

    _assert_all_equal(
        q,
        k,
    )


def test_attn_qk_simd_opposite_extremes():
    """
    Maximum negative accumulation.
    """
    q = torch.full(
        (1, 2, 5, 64),
        127,
        dtype=torch.int8,
    )

    k = torch.full(
        (1, 2, 7, 64),
        -127,
        dtype=torch.int8,
    )

    _assert_all_equal(
        q,
        k,
    )


# ============================================================
# Cancellation / lane pattern
# ============================================================

def test_attn_qk_simd_alternating_sign():
    """
    Exercise lane-wise sign handling and cancellation.
    """
    D = 64

    q_pattern = torch.tensor(
        [
            127 if i % 2 == 0
            else -127
            for i in range(D)
        ],
        dtype=torch.int8,
    )

    k_same = q_pattern.clone()
    k_opposite = -q_pattern

    q = q_pattern.reshape(
        1,
        1,
        1,
        D,
    )

    k = torch.stack(
        [
            k_same,
            k_opposite,
        ],
        dim=0,
    ).reshape(
        1,
        1,
        2,
        D,
    )

    _assert_all_equal(
        q,
        k,
    )


# ============================================================
# Zero
# ============================================================

def test_attn_qk_simd_zero():
    q = torch.zeros(
        (2, 4, 7, 64),
        dtype=torch.int8,
    )

    k = torch.randint(
        -127,
        128,
        (2, 4, 11, 64),
        dtype=torch.int8,
    )

    _assert_all_equal(
        q,
        k,
    )


# ============================================================
# Head independence
# ============================================================

def test_attn_qk_simd_multiple_heads_independent():
    """
    Give every head a visibly different activation pattern.

    This catches incorrect B/H base-address calculations.
    """
    B = 2
    H = 8
    Tq = 5
    Tk = 7
    D = 64

    q = torch.empty(
        (B, H, Tq, D),
        dtype=torch.int8,
    )

    k = torch.empty(
        (B, H, Tk, D),
        dtype=torch.int8,
    )

    for b in range(B):
        for h in range(H):
            q_value = (
                (b * H + h) * 11
            ) % 127 + 1

            k_value = (
                (b * H + h) * 17
            ) % 127 + 1

            if h % 2:
                q_value = -q_value

            if b % 2:
                k_value = -k_value

            q[b, h].fill_(
                q_value
            )

            k[b, h].fill_(
                k_value
            )

    _assert_all_equal(
        q,
        k,
    )