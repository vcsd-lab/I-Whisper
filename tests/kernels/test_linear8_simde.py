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
def test_linear8_accum_simde_matches_scalar(
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

    scalar = _C.linear8_accum_scalar_debug(
        x,
        weight,
        bias,
        zero,
    )

    simde = _C.linear8_accum_simde_debug(
        x,
        weight,
        bias,
        zero,
    )

    assert torch.equal(
        simde,
        scalar,
    )