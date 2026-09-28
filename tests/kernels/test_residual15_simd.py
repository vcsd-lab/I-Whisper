import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import Rshift


INT15_MIN = -16383
INT15_MAX = 16383


def residual15_ref(
    branch: torch.Tensor,
    residual: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    residual_scaled = (
        residual.to(torch.int32)
        * multiplier.to(torch.int32)
    )

    out = Rshift(
        branch + residual_scaled,
        15,
    )

    return out.clamp(
        INT15_MIN,
        INT15_MAX,
    ).to(torch.int16)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (7,),
        (8,),
        (9,),
        (15,),
        (16,),
        (17,),
        (3, 5),
        (2, 3, 7),
    ],
)
@pytest.mark.parametrize(
    "multiplier_value",
    [
        1,
        12345,
        32768,
        65535,
    ],
)
def test_residual15_simd_matches_scalar(
    shape,
    multiplier_value,
):
    torch.manual_seed(
        1000
        + multiplier_value
        + sum(shape)
    )

    branch = torch.randint(
        -500_000_000,
        500_000_001,
        shape,
        dtype=torch.int32,
    )

    residual = torch.randint(
        -10_000,
        10_001,
        shape,
        dtype=torch.int16,
    )

    multiplier = torch.tensor(
        [multiplier_value],
        dtype=torch.uint16,
    )

    scalar = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    simd = _C.residual15_simd(
        branch,
        residual,
        multiplier,
    )

    assert simd.dtype == torch.int16
    assert simd.shape == branch.shape
    assert torch.equal(simd, scalar)


def test_residual15_simd_rounding_boundaries():
    branch = torch.tensor(
        [
            -49153,
            -49152,
            -49151,
            -16385,
            -16384,
            -16383,
            -1,
            0,
            1,
            16383,
            16384,
            16385,
            49151,
            49152,
            49153,
            81919,
            -81919,
        ],
        dtype=torch.int32,
    )

    residual = torch.zeros(
        branch.shape,
        dtype=torch.int16,
    )

    multiplier = torch.tensor(
        [1],
        dtype=torch.uint16,
    )

    expected = residual15_ref(
        branch,
        residual,
        multiplier,
    )

    simd = _C.residual15_simd(
        branch,
        residual,
        multiplier,
    )

    assert torch.equal(simd, expected)


def test_residual15_simd_saturation():
    branch = torch.tensor(
        [
            16382 << 15,
            16383 << 15,
            16384 << 15,
            20000 << 15,
            -(16382 << 15),
            -(16383 << 15),
            -(16384 << 15),
            -(20000 << 15),
            0,
        ],
        dtype=torch.int32,
    )

    residual = torch.zeros(
        branch.shape,
        dtype=torch.int16,
    )

    multiplier = torch.tensor(
        [1],
        dtype=torch.uint16,
    )

    scalar = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    simd = _C.residual15_simd(
        branch,
        residual,
        multiplier,
    )

    assert torch.equal(simd, scalar)
    assert simd.max().item() == INT15_MAX
    assert simd.min().item() == INT15_MIN


def test_residual15_simd_realistic_int15_range():
    torch.manual_seed(5678)

    shape = (1, 1500, 512)

    branch = torch.randint(
        -300_000_000,
        300_000_001,
        shape,
        dtype=torch.int32,
    )

    residual = torch.randint(
        INT15_MIN,
        INT15_MAX + 1,
        shape,
        dtype=torch.int16,
    )

    multiplier = torch.tensor(
        [32768],
        dtype=torch.uint16,
    )

    scalar = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    simd = _C.residual15_simd(
        branch,
        residual,
        multiplier,
    )

    assert torch.equal(simd, scalar)
