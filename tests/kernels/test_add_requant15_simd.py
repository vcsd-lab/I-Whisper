import pytest
import torch

from i_whisper.kernels import _C


def _run_scalar(x, addend, multiplier, shift):
    return _C.add_requant15_scalar(
        x.contiguous(),
        addend.contiguous(),
        multiplier.contiguous(),
        shift.contiguous(),
    )


def _run_simd(x, addend, multiplier, shift):
    return _C.add_requant15_simd(
        x.contiguous(),
        addend.contiguous(),
        multiplier.contiguous(),
        shift.contiguous(),
    )


@pytest.mark.parametrize(
    "channels",
    [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 383, 384, 385, 511, 512, 513],
)
@pytest.mark.parametrize("per_channel", [False, True])
def test_add_requant15_simd_vector_boundaries(channels, per_channel):
    B, T, C = 2, 5, channels

    torch.manual_seed(1000 + C * 11 + int(per_channel) * 13)

    x = torch.randint(-128, 128, (B, T, C), dtype=torch.int8)
    addend = torch.randint(-128, 128, (T, C), dtype=torch.int8)

    count = C if per_channel else 1

    multiplier = torch.randint(
        0, 65536, (count,), dtype=torch.int32
    ).to(torch.uint16)

    shift = torch.randint(-4, 40, (count,), dtype=torch.int8)

    ref = _run_scalar(x, addend, multiplier, shift)
    out = _run_simd(x, addend, multiplier, shift)

    assert out.dtype == torch.int16
    assert out.shape == ref.shape
    assert torch.equal(out, ref), "add_requant15 SIMD mismatch"


@pytest.mark.parametrize(
    "shift_value",
    [-128, -8, -1, 0, 1, 7, 15, 31, 32, 33, 64, 127],
)
def test_add_requant15_simd_shift_edges(shift_value):
    torch.manual_seed(3000 + shift_value)

    x = torch.randint(-128, 128, (1, 3, 17), dtype=torch.int8)
    addend = torch.randint(-128, 128, (3, 17), dtype=torch.int8)

    multiplier = torch.full((17,), 65535, dtype=torch.uint16)
    shift = torch.full((17,), shift_value, dtype=torch.int8)

    assert torch.equal(
        _run_simd(x, addend, multiplier, shift),
        _run_scalar(x, addend, multiplier, shift),
    )


def test_add_requant15_simd_mixed_lane_shifts():
    x = torch.tensor(
        [[[
            -128, -100, -1, 0,
            1, 100, 126, 127,
            -127, -64, -2, 2,
            63, 64, 120, 127,
        ]]],
        dtype=torch.int8,
    )

    addend = torch.tensor(
        [[
            127, -28, -1, 1,
            127, -128, 126, -127,
            100, -64, 2, -2,
            -63, 64, -120, 0,
        ]],
        dtype=torch.int8,
    )

    multiplier = torch.tensor(
        [
            1, 3, 17, 255,
            1024, 4096, 32768, 65535,
            65535, 32768, 4096, 1024,
            255, 17, 3, 1,
        ],
        dtype=torch.uint16,
    )

    shift = torch.tensor(
        [
            -3, 0, 1, 2,
            3, 7, 15, 31,
            32, 33, 4, 8,
            12, 16, 20, 24,
        ],
        dtype=torch.int8,
    )

    assert torch.equal(
        _run_simd(x, addend, multiplier, shift),
        _run_scalar(x, addend, multiplier, shift),
    )


def test_add_requant15_simd_int15_clamp():
    x = torch.full((1, 2, 16), 127, dtype=torch.int8)
    addend = torch.full((2, 16), 127, dtype=torch.int8)
    multiplier = torch.full((16,), 65535, dtype=torch.uint16)
    shift = torch.zeros(16, dtype=torch.int8)

    out = _run_simd(x, addend, multiplier, shift)
    ref = _run_scalar(x, addend, multiplier, shift)

    assert torch.equal(out, ref)
    assert torch.all(out == 16383)


def test_add_requant15_simd_batch_broadcast():
    torch.manual_seed(5001)

    x = torch.randint(-128, 128, (4, 7, 32), dtype=torch.int8)
    addend = torch.randint(-128, 128, (7, 32), dtype=torch.int8)

    multiplier = torch.randint(
        0, 65536, (32,), dtype=torch.int32
    ).to(torch.uint16)

    shift = torch.randint(0, 20, (32,), dtype=torch.int8)

    assert torch.equal(
        _run_simd(x, addend, multiplier, shift),
        _run_scalar(x, addend, multiplier, shift),
    )


def test_add_requant15_simd_does_not_modify_inputs():
    torch.manual_seed(6001)

    x = torch.randint(-128, 128, (2, 7, 33), dtype=torch.int8)
    addend = torch.randint(-128, 128, (7, 33), dtype=torch.int8)

    multiplier = torch.randint(
        0, 65536, (33,), dtype=torch.int32
    ).to(torch.uint16)

    shift = torch.randint(0, 16, (33,), dtype=torch.int8)

    refs = (
        x.clone(),
        addend.clone(),
        multiplier.clone(),
        shift.clone(),
    )

    _run_simd(x, addend, multiplier, shift)

    assert x.dtype == refs[0].dtype
    assert torch.equal(
        x.to(torch.int64),
        refs[0].to(torch.int64),
    )
    assert addend.dtype == refs[1].dtype
    assert torch.equal(
        addend.to(torch.int64),
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
