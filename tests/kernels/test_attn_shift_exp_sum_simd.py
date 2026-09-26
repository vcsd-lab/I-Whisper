import pytest
import torch

from i_whisper.kernels import _C


INT16_MIN = torch.iinfo(torch.int16).min


def _frac_bits(value):
    return torch.tensor(value, dtype=torch.int32)


def _expand_mask(mask, x):
    return (
        mask.unsqueeze(0).unsqueeze(0)
        .repeat(x.shape[0], x.shape[1], 1, 1)
    )


def _row_max_plus1(x, mask):
    qk = x.clone()
    if mask is not None:
        qk[_expand_mask(mask, qk)] = INT16_MIN

    row_max = (
        qk.max(dim=-1, keepdim=True)
        .values.to(torch.int16)
    )
    return qk, row_max + 1


def _run_scalar(x, row_max, mask, frac_bits):
    return _C.shift_exp_scalar(
        x.contiguous(),
        row_max.contiguous(),
        None if mask is None else mask.contiguous(),
        _frac_bits(frac_bits),
    )


def _run_simd(x, row_max, mask, frac_bits):
    return _C.shift_exp_simd(
        x.contiguous(),
        row_max.contiguous(),
        None if mask is None else mask.contiguous(),
        _frac_bits(frac_bits),
    )


def _assert_equal(x, row_max, mask, frac_bits):
    scalar_exp, scalar_sum = _run_scalar(
        x, row_max, mask, frac_bits
    )
    simd_exp, simd_sum = _run_simd(
        x, row_max, mask, frac_bits
    )

    assert simd_exp.dtype == torch.uint8
    assert simd_sum.dtype == torch.int32
    assert torch.equal(simd_exp, scalar_exp)
    assert torch.equal(simd_sum, scalar_sum)

    expected_sum = (
        simd_exp.sum(dim=-1, keepdim=True).int()
    )
    assert torch.equal(simd_sum, expected_sum)


@pytest.mark.parametrize(
    "k_length",
    [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65],
)
@pytest.mark.parametrize(
    "frac_bits",
    [1, 7, 8, 12, 13],
)
def test_shift_exp_simd_vector_boundaries(k_length, frac_bits):
    torch.manual_seed(
        1000 + k_length * 101 + frac_bits * 103
    )
    raw = torch.randint(
        -16000, 16001,
        (1, 3, 4, k_length),
        dtype=torch.int16,
    )
    x, row_max = _row_max_plus1(raw, None)
    _assert_equal(
        x, row_max, None, frac_bits
    )


@pytest.mark.parametrize(
    "k_length",
    [7, 8, 9, 31, 32, 33, 64, 65],
)
def test_shift_exp_simd_random_mask(k_length):
    torch.manual_seed(3000 + k_length)

    raw = torch.randint(
        -16000, 16001,
        (2, 4, 3, k_length),
        dtype=torch.int16,
    )
    mask = torch.rand((3, k_length)) < 0.35
    mask[:, 0] = False

    x, row_max = _row_max_plus1(raw, mask)
    _assert_equal(
        x, row_max, mask, 7
    )


def test_shift_exp_simd_all_masked():
    B, H, Tq, Tk = 1, 4, 3, 17

    x = torch.full(
        (B, H, Tq, Tk),
        INT16_MIN,
        dtype=torch.int16,
    )
    row_max = torch.full(
        (B, H, Tq, 1),
        INT16_MIN + 1,
        dtype=torch.int16,
    )
    mask = torch.ones(
        (Tq, Tk),
        dtype=torch.bool,
    )

    exp, row_sum = _run_simd(
        x, row_max, mask, 7
    )

    assert torch.count_nonzero(exp).item() == 0
    assert torch.count_nonzero(row_sum).item() == 0
    _assert_equal(
        x, row_max, mask, 7
    )


def test_shift_exp_simd_rowmax_wrap():
    x = torch.tensor(
        [[[[
            32767, -1, -32768, 0,
            123, -456, 30000, -30000, 42,
        ]]]],
        dtype=torch.int16,
    )
    row_max = torch.tensor(
        [[[[-32768]]]],
        dtype=torch.int16,
    )
    _assert_equal(
        x, row_max, None, 7
    )


def test_shift_exp_simd_mixed_mask_vector_tail():
    Tk = 17
    raw = torch.arange(
        -8, 9,
        dtype=torch.int16,
    ).view(1, 1, 1, Tk)

    mask = torch.tensor(
        [[
            False, True, False, True,
            False, True, False, True,
            True, False, True, False,
            True, False, True, False,
            True,
        ]],
        dtype=torch.bool,
    )

    x, row_max = _row_max_plus1(raw, mask)
    _assert_equal(
        x, row_max, mask, 7
    )


def test_shift_exp_simd_does_not_modify_inputs():
    torch.manual_seed(5001)

    raw = torch.randint(
        -12000, 12001,
        (1, 4, 3, 33),
        dtype=torch.int16,
    )
    mask = torch.rand((3, 33)) < 0.25
    mask[:, 0] = False

    x, row_max = _row_max_plus1(raw, mask)

    x_ref = x.clone()
    max_ref = row_max.clone()
    mask_ref = mask.clone()

    _run_simd(
        x, row_max, mask, 7
    )

    assert torch.equal(x, x_ref)
    assert torch.equal(row_max, max_ref)
    assert torch.equal(mask, mask_ref)
