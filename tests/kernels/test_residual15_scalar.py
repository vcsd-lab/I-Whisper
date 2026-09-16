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
    """
    Frozen Torch golden from ResidualAttentionBlock_int:

        x = residual.int().mul(multiplier.int())
        x = Rshift(branch + x, 15)
        x = x.clamp(-16383, 16383).to(torch.int16)
    """
    assert branch.dtype == torch.int32
    assert residual.dtype == torch.int16
    assert multiplier.dtype == torch.uint16
    assert multiplier.numel() == 1
    assert branch.shape == residual.shape

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


def test_residual15_scalar_basic():
    branch = torch.tensor(
        [
            [0, 32768, -32768, 100000],
            [200000, -200000, 123456, -123456],
        ],
        dtype=torch.int32,
    )

    residual = torch.tensor(
        [
            [0, 1, -1, 10],
            [-10, 100, -100, 123],
        ],
        dtype=torch.int16,
    )

    multiplier = torch.tensor(
        [32768],
        dtype=torch.uint16,
    )

    expected = residual15_ref(
        branch,
        residual,
        multiplier,
    )

    actual = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    assert actual.dtype == torch.int16
    assert actual.shape == branch.shape
    assert torch.equal(actual, expected)


def test_residual15_scalar_rounding_boundaries():
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
        ],
        dtype=torch.int32,
    )

    residual = torch.zeros_like(
        branch,
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

    actual = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    assert torch.equal(actual, expected)

    half = torch.tensor(
        [16384, -16384],
        dtype=torch.int32,
    )
    half_residual = torch.zeros(
        2,
        dtype=torch.int16,
    )

    half_actual = _C.residual15_scalar(
        half,
        half_residual,
        multiplier,
    )

    assert torch.equal(
        half_actual,
        torch.tensor([1, -1], dtype=torch.int16),
    )


def test_residual15_scalar_multiplier_path():
    branch = torch.tensor(
        [
            0,
            1000,
            -1000,
            500000,
            -500000,
        ],
        dtype=torch.int32,
    )

    residual = torch.tensor(
        [
            1,
            -1,
            127,
            -127,
            16383,
        ],
        dtype=torch.int16,
    )

    multiplier = torch.tensor(
        [24576],
        dtype=torch.uint16,
    )

    expected = residual15_ref(
        branch,
        residual,
        multiplier,
    )

    actual = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    assert torch.equal(actual, expected)


def test_residual15_scalar_int15_saturation():
    branch = torch.tensor(
        [
            16383 << 15,
            16384 << 15,
            20000 << 15,
            -(16383 << 15),
            -(16384 << 15),
            -(20000 << 15),
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

    actual = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    assert torch.equal(actual, expected)
    assert actual.max().item() == INT15_MAX
    assert actual.min().item() == INT15_MIN


@pytest.mark.parametrize(
    "multiplier_value",
    [
        1,
        12345,
        32768,
        65535,
    ],
)
def test_residual15_scalar_random(multiplier_value):
    torch.manual_seed(1234 + multiplier_value)

    shape = (7, 11)

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

    expected = residual15_ref(
        branch,
        residual,
        multiplier,
    )

    actual = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    assert torch.equal(actual, expected)


def test_residual15_scalar_3d():
    torch.manual_seed(5678)

    shape = (2, 5, 17)

    branch = torch.randint(
        -300_000_000,
        300_000_001,
        shape,
        dtype=torch.int32,
    )

    residual = torch.randint(
        -8_000,
        8_001,
        shape,
        dtype=torch.int16,
    )

    multiplier = torch.tensor(
        [32768],
        dtype=torch.uint16,
    )

    expected = residual15_ref(
        branch,
        residual,
        multiplier,
    )

    actual = _C.residual15_scalar(
        branch,
        residual,
        multiplier,
    )

    assert actual.dtype == torch.int16
    assert actual.shape == shape
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "bad_branch,bad_residual,bad_multiplier",
    [
        (
            torch.zeros(8, dtype=torch.int16),
            torch.zeros(8, dtype=torch.int16),
            torch.ones(1, dtype=torch.uint16),
        ),
        (
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int32),
            torch.ones(1, dtype=torch.uint16),
        ),
        (
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int16),
            torch.ones(1, dtype=torch.int32),
        ),
    ],
)
def test_residual15_scalar_rejects_wrong_dtype(
    bad_branch,
    bad_residual,
    bad_multiplier,
):
    with pytest.raises(RuntimeError):
        _C.residual15_scalar(
            bad_branch,
            bad_residual,
            bad_multiplier,
        )


def test_residual15_scalar_rejects_non_scalar_multiplier():
    branch = torch.zeros(
        8,
        dtype=torch.int32,
    )

    residual = torch.zeros(
        8,
        dtype=torch.int16,
    )

    multiplier = torch.ones(
        2,
        dtype=torch.uint16,
    )

    with pytest.raises(RuntimeError):
        _C.residual15_scalar(
            branch,
            residual,
            multiplier,
        )


def test_residual15_scalar_rejects_shape_mismatch():
    branch = torch.zeros(
        (2, 4),
        dtype=torch.int32,
    )

    residual = torch.zeros(
        (4, 2),
        dtype=torch.int16,
    )

    multiplier = torch.ones(
        1,
        dtype=torch.uint16,
    )

    with pytest.raises(RuntimeError):
        _C.residual15_scalar(
            branch,
            residual,
            multiplier,
        )
