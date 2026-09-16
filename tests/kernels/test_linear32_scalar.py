from i_whisper.kernels import _C

import pytest
import torch


def linear32_accum_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    weight_zero=None,
):
    x32 = x.to(torch.int32)
    w32 = weight.to(torch.int32)

    # [*, K] @ [K, N] -> [*, N]
    out = torch.matmul(x32, w32.t())

    if bias is not None:
        out = out + bias.to(torch.int32)

    if weight_zero is not None:
        row_sum = x32.sum(dim=-1, keepdim=True)
        out = out - row_sum * weight_zero.to(torch.int32)

    return out.to(torch.int32)


def int32_wrap(x: int) -> int:
    x &= 0xFFFFFFFF
    if x >= 0x80000000:
        x -= 0x100000000
    return x


def mulh_s32_u32_ref(x: int, multiplier: int) -> int:
    """
    Signed int32 x unsigned uint32, return upper 32 bits
    of the signed 64-bit product.
    """
    product = int(x) * int(multiplier)
    return product >> 32


def linear32_requant_ref(
    x: torch.Tensor,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
):
    shape = x.shape
    N = shape[-1]

    x_flat = x.reshape(-1, N)
    out = torch.empty_like(x_flat, dtype=torch.int32)

    for m in range(x_flat.shape[0]):
        for n in range(N):
            scaled = mulh_s32_u32_ref(
                int(x_flat[m, n]),
                int(multiplier[n]),
            )

            shifted = scaled * (1 << int(shift[n]))

            out[m, n] = int32_wrap(shifted)

    return out.reshape(shape)


# ============================================================
# Accumulator
# ============================================================

def test_linear32_accum_scalar_basic():
    torch.manual_seed(0)

    M = 7
    K = 11
    N = 13

    x = torch.randint(
        -128, 128,
        (M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128, 128,
        (N, K),
        dtype=torch.int8,
    )

    y = _C.linear32_accum_scalar_debug(
        x,
        weight,
        None,
        None,
    )

    expected = linear32_accum_ref(
        x,
        weight,
    )

    assert y.dtype == torch.int32
    assert y.shape == (M, N)
    assert torch.equal(y, expected)


def test_linear32_accum_scalar_with_bias():
    torch.manual_seed(1)

    M = 7
    K = 11
    N = 13

    x = torch.randint(
        -128, 128,
        (M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128, 128,
        (N, K),
        dtype=torch.int8,
    )

    # Keep bias moderate so this test is about correctness,
    # not signed-overflow behavior.
    bias = torch.randint(
        -100000, 100000,
        (N,),
        dtype=torch.int32,
    )

    y = _C.linear32_accum_scalar_debug(
        x,
        weight,
        bias,
        None,
    )

    expected = linear32_accum_ref(
        x,
        weight,
        bias=bias,
    )

    assert torch.equal(y, expected)


def test_linear32_accum_scalar_asymmetric():
    torch.manual_seed(2)

    M = 7
    K = 11
    N = 13

    x = torch.randint(
        -128, 128,
        (M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128, 128,
        (N, K),
        dtype=torch.int8,
    )

    weight_zero = torch.randint(
        -128, 128,
        (N,),
        dtype=torch.int8,
    )

    y = _C.linear32_accum_scalar_debug(
        x,
        weight,
        None,
        weight_zero,
    )

    expected = linear32_accum_ref(
        x,
        weight,
        weight_zero=weight_zero,
    )

    assert torch.equal(y, expected)


def test_linear32_accum_scalar_bias_asymmetric():
    torch.manual_seed(3)

    M = 7
    K = 11
    N = 13

    x = torch.randint(
        -128, 128,
        (M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128, 128,
        (N, K),
        dtype=torch.int8,
    )

    bias = torch.randint(
        -100000, 100000,
        (N,),
        dtype=torch.int32,
    )

    weight_zero = torch.randint(
        -128, 128,
        (N,),
        dtype=torch.int8,
    )

    y = _C.linear32_accum_scalar_debug(
        x,
        weight,
        bias,
        weight_zero,
    )

    expected = linear32_accum_ref(
        x,
        weight,
        bias=bias,
        weight_zero=weight_zero,
    )

    assert torch.equal(y, expected)


def test_linear32_accum_scalar_3d():
    torch.manual_seed(4)

    B = 2
    M = 5
    K = 11
    N = 13

    x = torch.randint(
        -128, 128,
        (B, M, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128, 128,
        (N, K),
        dtype=torch.int8,
    )

    bias = torch.randint(
        -100000, 100000,
        (N,),
        dtype=torch.int32,
    )

    weight_zero = torch.randint(
        -128, 128,
        (N,),
        dtype=torch.int8,
    )

    y = _C.linear32_accum_scalar_debug(
        x,
        weight,
        bias,
        weight_zero,
    )

    expected = linear32_accum_ref(
        x,
        weight,
        bias=bias,
        weight_zero=weight_zero,
    )

    assert y.shape == (B, M, N)
    assert torch.equal(y, expected)


# ============================================================
# Requant
# ============================================================

def test_linear32_requant_scalar_basic():
    x = torch.tensor(
        [
            [100000, -100000, 123456, -654321],
            [200000, -200000, 345678, -765432],
        ],
        dtype=torch.int32,
    )

    multiplier = torch.tensor(
        [
            0x40000000,
            0x80000000,
            0x60000000,
            0x70000000,
        ],
        dtype=torch.uint32,
    )

    shift = torch.tensor(
        [1, 2, 3, 4],
        dtype=torch.int8,
    )

    y = _C.linear32_requant_scalar_debug(
        x,
        multiplier,
        shift,
    )

    expected = linear32_requant_ref(
        x,
        multiplier,
        shift,
    )

    assert y.dtype == torch.int32
    assert y.shape == x.shape
    assert torch.equal(y, expected)


def test_linear32_requant_scalar_3d():
    torch.manual_seed(5)

    B = 2
    M = 7
    N = 13

    x = torch.randint(
        -1_000_000,
        1_000_000,
        (B, M, N),
        dtype=torch.int32,
    )

    # Torch randint 對 uint32 不太方便，先用 int64 再轉。
    multiplier = torch.randint(
        1 << 30,
        1 << 32,
        (N,),
        dtype=torch.int64,
    ).to(torch.uint32)

    shift = torch.randint(
        0,
        8,
        (N,),
        dtype=torch.int8,
    )

    y = _C.linear32_requant_scalar_debug(
        x,
        multiplier,
        shift,
    )

    expected = linear32_requant_ref(
        x,
        multiplier,
        shift,
    )

    assert y.shape == x.shape
    assert torch.equal(y, expected)



@pytest.mark.parametrize(
    "has_bias,has_zero",
    [
        (False, False),
        (True,  False),
        (False, True),
        (True,  True),
    ],
)
def test_linear32_scalar_complete_random(
    has_bias,
    has_zero,
):
    torch.manual_seed(10)

    M = 7
    K = 11
    N = 13

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
            -100000,
            100000,
            (N,),
            dtype=torch.int32,
        )
        if has_bias
        else None
    )

    weight_zero = (
        torch.randint(
            -128,
            128,
            (N,),
            dtype=torch.int8,
        )
        if has_zero
        else None
    )

    multiplier = torch.randint(
        1 << 30,
        1 << 32,
        (N,),
        dtype=torch.int64,
    ).to(torch.uint32)

    shift = torch.randint(
        0,
        8,
        (N,),
        dtype=torch.int8,
    )

    y = _C.linear32_scalar(
        x,
        weight,
        bias,
        weight_zero,
        multiplier,
        shift,
    )

    accum_expected = linear32_accum_ref(
        x,
        weight,
        bias=bias,
        weight_zero=weight_zero,
    )

    expected = linear32_requant_ref(
        accum_expected,
        multiplier,
        shift,
    )

    assert y.dtype == torch.int32
    assert y.shape == (M, N)
    assert torch.equal(y, expected)