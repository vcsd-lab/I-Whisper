import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import Fast_Inverse_Square_Root


def rsqrt_ref(var):
    assert var.dtype == torch.int32
    assert torch.all(var >= 0)

    return Fast_Inverse_Square_Root(
        var,
        offset=13,
        Newton_iter=2,
    )


def _assert_equal(var):
    expected = rsqrt_ref(var)
    actual = _C.layernorm32_rsqrt_scalar(var)

    assert actual.dtype == torch.int32
    assert actual.shape == var.shape
    assert torch.equal(actual, expected)


def test_layernorm32_rsqrt_scalar_zero():
    _assert_equal(
        torch.zeros(16, dtype=torch.int32)
    )


def test_layernorm32_rsqrt_scalar_small_values():
    var = torch.tensor(
        [
            0, 1, 2, 3, 4, 5, 7, 8,
            15, 16, 31, 32, 63, 64,
            127, 128, 255, 256, 511, 512,
            1023, 1024,
        ],
        dtype=torch.int32,
    )

    _assert_equal(var)


def test_layernorm32_rsqrt_scalar_power_boundaries():
    values = {0}

    for bit in range(0, 30):
        p = 1 << bit

        for delta in (-1, 0, 1):
            value = p + delta

            if 0 <= value <= torch.iinfo(torch.int32).max:
                values.add(value)

    _assert_equal(
        torch.tensor(
            sorted(values),
            dtype=torch.int32,
        )
    )


@pytest.mark.parametrize(
    "seed",
    [1, 17, 1234, 5678],
)
def test_layernorm32_rsqrt_scalar_random(seed):
    torch.manual_seed(seed)

    var = torch.randint(
        0,
        1_000_000_001,
        (257,),
        dtype=torch.int32,
    )

    _assert_equal(var)


def test_layernorm32_rsqrt_scalar_3d():
    torch.manual_seed(8888)

    var = torch.randint(
        0,
        1_000_000_001,
        (2, 7, 13),
        dtype=torch.int32,
    )

    _assert_equal(var)


def test_layernorm32_rsqrt_scalar_rejects_negative():
    var = torch.tensor(
        [0, 1, -1, 4],
        dtype=torch.int32,
    )

    with pytest.raises(RuntimeError):
        _C.layernorm32_rsqrt_scalar(var)


def test_layernorm32_rsqrt_scalar_rejects_wrong_dtype():
    var = torch.ones(
        8,
        dtype=torch.int64,
    )

    with pytest.raises(RuntimeError):
        _C.layernorm32_rsqrt_scalar(var)
