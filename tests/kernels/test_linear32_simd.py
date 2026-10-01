import pytest
import torch

from i_whisper.kernels import _C


@pytest.mark.parametrize(
    "M,K,N",
    [
        (1, 1, 1),
        (7, 11, 13),

        # SIMD width boundaries
        (3, 15, 5),
        (3, 16, 5),
        (3, 17, 5),

        (3, 31, 5),
        (3, 32, 5),
        (3, 33, 5),
    ],
)
@pytest.mark.parametrize(
    "use_bias,use_zero",
    [
        (False, False),
        (True,  False),
        (False, True),
        (True,  True),
    ],
)
def test_linear32_accum_simd_matches_scalar(
    M,
    K,
    N,
    use_bias,
    use_zero,
):
    torch.manual_seed(1234)

    x = torch.randint(
        -128,
        128,
        (M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (N, K),
        dtype=torch.int8,
    )

    bias = (
        torch.randint(
            -100_000,
            100_001,
            (N,),
            dtype=torch.int32,
        )
        if use_bias
        else None
    )

    zero = (
        torch.randint(
            -16,
            17,
            (N,),
            dtype=torch.int8,
        )
        if use_zero
        else None
    )

    scalar = _C.linear32_accum_scalar_debug(
        x,
        weight,
        bias,
        zero,
    )

    simd = _C.linear32_accum_simd_debug(
        x,
        weight,
        bias,
        zero,
    )

    assert torch.equal(
        simd,
        scalar,
    )


@pytest.mark.parametrize(
    "M,K,N",
    [
        (1, 1, 1),
        (7, 11, 13),

        # SIMD width boundaries
        (3, 15, 5),
        (3, 16, 5),
        (3, 17, 5),

        (3, 31, 5),
        (3, 32, 5),
        (3, 33, 5),
        # Large-M tiled paths
        (31, 512, 8),   # 2x4 fallback
        (32, 512, 8),   # 4x4 path
        (36, 512, 8),   # multiple 4x4 M tiles
        (32, 513, 8),   # K tail
        (32, 512, 7),   # N tail
    ],
)
@pytest.mark.parametrize(
    "use_bias,use_zero",
    [
        (False, False),
        (True,  False),
        (False, True),
        (True,  True),
    ],
)
def test_linear32_simd_matches_scalar(
    M,
    K,
    N,
    use_bias,
    use_zero,
):
    torch.manual_seed(1234)

    x = torch.randint(
        -128,
        128,
        (M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (N, K),
        dtype=torch.int8,
    )

    bias = (
        torch.randint(
            -100_000,
            100_001,
            (N,),
            dtype=torch.int32,
        )
        if use_bias
        else None
    )

    zero = (
        torch.randint(
            -16,
            17,
            (N,),
            dtype=torch.int8,
        )
        if use_zero
        else None
    )

    multiplier = torch.randint(
        1,
        2**32,
        (N,),
        dtype=torch.uint32,
    )

    shift = torch.randint(
        0,
        8,
        (N,),
        dtype=torch.int8,
    )

    scalar = _C.linear32_scalar(
        x,
        weight,
        bias,
        zero,
        multiplier,
        shift,
    )

    simd = _C.linear32_simd(
        x,
        weight,
        bias,
        zero,
        multiplier,
        shift,
    )

    assert torch.equal(
        simd,
        scalar,
    )


def test_linear32_accum_simd_3d_matches_scalar():
    torch.manual_seed(2234)

    B = 2
    M = 5
    K = 17
    N = 7

    x = torch.randint(
        -128,
        128,
        (B, M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (N, K),
        dtype=torch.int8,
    )

    bias = torch.randint(
        -100_000,
        100_001,
        (N,),
        dtype=torch.int32,
    )

    zero = torch.randint(
        -16,
        17,
        (N,),
        dtype=torch.int8,
    )

    scalar = _C.linear32_accum_scalar_debug(
        x,
        weight,
        bias,
        zero,
    )

    simd = _C.linear32_accum_simd_debug(
        x,
        weight,
        bias,
        zero,
    )

    assert scalar.shape == (B, M, N)
    assert simd.shape == (B, M, N)
    assert torch.equal(simd, scalar)


def test_linear32_simd_3d_matches_scalar():
    torch.manual_seed(3234)

    B = 2
    M = 5
    K = 17
    N = 7

    x = torch.randint(
        -128,
        128,
        (B, M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (N, K),
        dtype=torch.int8,
    )

    bias = torch.randint(
        -100_000,
        100_001,
        (N,),
        dtype=torch.int32,
    )

    zero = torch.randint(
        -16,
        17,
        (N,),
        dtype=torch.int8,
    )

    multiplier = torch.randint(
        1,
        2**32,
        (N,),
        dtype=torch.uint32,
    )

    shift = torch.randint(
        0,
        8,
        (N,),
        dtype=torch.int8,
    )

    scalar = _C.linear32_scalar(
        x,
        weight,
        bias,
        zero,
        multiplier,
        shift,
    )

    simd = _C.linear32_simd(
        x,
        weight,
        bias,
        zero,
        multiplier,
        shift,
    )

    assert scalar.shape == (B, M, N)
    assert simd.shape == (B, M, N)
    assert torch.equal(simd, scalar)


def test_linear32_simd_shift_boundaries_match_scalar():
    torch.manual_seed(4234)

    M = 3
    K = 17
    N = 6

    x = torch.randint(
        -32,
        33,
        (M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -32,
        33,
        (N, K),
        dtype=torch.int8,
    )

    bias = torch.randint(
        -10_000,
        10_001,
        (N,),
        dtype=torch.int32,
    )

    zero = torch.randint(
        -8,
        9,
        (N,),
        dtype=torch.int8,
    )

    multiplier = torch.tensor(
        [
            0x20000000,
            0x40000000,
            0x60000000,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFF,
        ],
        dtype=torch.uint32,
    )

    shift = torch.tensor(
        [0, 1, 8, 16, 31, 32],
        dtype=torch.int8,
    )

    scalar = _C.linear32_scalar(
        x,
        weight,
        bias,
        zero,
        multiplier,
        shift,
    )

    simd = _C.linear32_simd(
        x,
        weight,
        bias,
        zero,
        multiplier,
        shift,
    )

    assert torch.equal(simd, scalar)
