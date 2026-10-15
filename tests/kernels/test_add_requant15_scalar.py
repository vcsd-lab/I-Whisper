import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import Rshift

INT15_MIN = -16383
INT15_MAX = 16383


def _torch_golden(x, addend, multiplier, shift):
    out = x.int() + addend.int()
    out = out.mul(multiplier.int())
    out = Rshift(out, shift)
    return out.clamp(INT15_MIN, INT15_MAX).to(torch.int16)


def _run(x, addend, multiplier, shift):
    return _C.add_requant15_scalar(
        x.contiguous(),
        addend.contiguous(),
        multiplier.contiguous(),
        shift.contiguous(),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (1, 3, 7),
        (2, 5, 16),
        (1, 17, 31),
        (2, 9, 64),
        (1, 13, 384),
        (1, 13, 512),
    ],
)
@pytest.mark.parametrize("per_channel", [False, True])
def test_add_requant15_scalar_random(shape, per_channel):
    B, T, C = shape

    torch.manual_seed(
        1000 + B * 11 + T * 13 + C * 17 + int(per_channel) * 19
    )

    x = torch.randint(
        -128, 128, (B, T, C), dtype=torch.int8
    )
    addend = torch.randint(
        -128, 128, (T, C), dtype=torch.int8
    )

    count = C if per_channel else 1

    multiplier = torch.randint(
        0, 65536, (count,), dtype=torch.int32
    ).to(torch.uint16)

    shift = torch.randint(
        -3, 33, (count,), dtype=torch.int8
    )

    ref = _torch_golden(x, addend, multiplier, shift)
    out = _run(x, addend, multiplier, shift)

    assert out.dtype == torch.int16
    assert out.shape == ref.shape
    assert torch.equal(out.to(torch.int64), ref.to(torch.int64))


@pytest.mark.parametrize("shift_value", [-8, -1, 0, 1, 7, 31, 32, 64])
def test_add_requant15_scalar_shift_edges(shift_value):
    x = torch.tensor(
        [[[-128, -127, -1, 0, 1, 126, 127, 127]]],
        dtype=torch.int8,
    )
    pos = torch.tensor(
        [[-128, 127, -1, 1, 127, -128, 126, -127]],
        dtype=torch.int8,
    )
    multiplier = torch.tensor([65535], dtype=torch.uint16)
    shift = torch.tensor([shift_value], dtype=torch.int8)

    assert torch.equal(
        _run(x, pos, multiplier, shift),
        _torch_golden(x, pos, multiplier, shift),
    )


def test_add_requant15_scalar_per_channel_parameters():
    x = torch.tensor(
        [[[100, -100, 127, -128]]],
        dtype=torch.int8,
    )
    pos = torch.tensor(
        [[27, -28, -127, 127]],
        dtype=torch.int8,
    )
    multiplier = torch.tensor(
        [1, 257, 32768, 65535],
        dtype=torch.uint16,
    )
    shift = torch.tensor(
        [0, 3, 8, 15],
        dtype=torch.int8,
    )

    assert torch.equal(
        _run(x, pos, multiplier, shift),
        _torch_golden(x, pos, multiplier, shift),
    )


def test_add_requant15_scalar_broadcast_batch():
    x = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=torch.int8,
    )
    pos = torch.tensor(
        [[10, 20, 30], [-1, -2, -3]],
        dtype=torch.int8,
    )
    multiplier = torch.tensor(
        [1, 2, 3],
        dtype=torch.uint16,
    )
    shift = torch.tensor(
        [0, 0, 0],
        dtype=torch.int8,
    )

    assert torch.equal(
        _run(x, pos, multiplier, shift),
        _torch_golden(x, pos, multiplier, shift),
    )


def test_add_requant15_scalar_clamp():
    x = torch.full(
        (1, 2, 8), 127, dtype=torch.int8
    )
    pos = torch.full(
        (2, 8), 127, dtype=torch.int8
    )
    multiplier = torch.full(
        (8,), 65535, dtype=torch.uint16
    )
    shift = torch.zeros(
        8, dtype=torch.int8
    )

    out = _run(x, pos, multiplier, shift)
    ref = _torch_golden(x, pos, multiplier, shift)

    assert torch.equal(out, ref)
    assert torch.all(out == INT15_MAX)


def test_add_requant15_scalar_does_not_modify_inputs():
    torch.manual_seed(6001)

    x = torch.randint(
        -128, 128, (2, 7, 31), dtype=torch.int8
    )
    pos = torch.randint(
        -128, 128, (7, 31), dtype=torch.int8
    )
    multiplier = torch.randint(
        0, 65536, (31,), dtype=torch.int32
    ).to(torch.uint16)
    shift = torch.randint(
        0, 16, (31,), dtype=torch.int8
    )

    refs = (x.clone(), pos.clone(), multiplier.clone(), shift.clone())

    _run(x, pos, multiplier, shift)

    assert x.dtype == refs[0].dtype
    assert torch.equal(
        x.to(torch.int64),
        refs[0].to(torch.int64),
    )
    assert pos.dtype == refs[1].dtype
    assert torch.equal(
        pos.to(torch.int64),
        refs[1].to(torch.int64),
    )
    assert multiplier.dtype == refs[2].dtype
    assert torch.equal(
        multiplier.to(torch.int64),
        refs[2].to(torch.int64),
    )
    assert shift.dtype == refs[3].dtype
    assert torch.equal(
        shift.to(torch.int64),
        refs[3].to(torch.int64),
    )


def test_add_requant15_scalar_rejects_bad_addend_shape():
    x = torch.zeros((1, 3, 4), dtype=torch.int8)
    bad_addend = torch.zeros((2, 4), dtype=torch.int8)
    multiplier = torch.ones(4, dtype=torch.uint16)
    shift = torch.zeros(4, dtype=torch.int8)

    with pytest.raises(RuntimeError, match="addend"):
        _run(x, bad_addend, multiplier, shift)


def test_add_requant15_scalar_rejects_bad_parameter_count():
    x = torch.zeros((1, 3, 4), dtype=torch.int8)
    addend = torch.zeros((3, 4), dtype=torch.int8)
    bad_multiplier = torch.ones(2, dtype=torch.uint16)
    shift = torch.zeros(4, dtype=torch.int8)

    with pytest.raises(RuntimeError, match="multiplier"):
        _run(x, addend, bad_multiplier, shift)
