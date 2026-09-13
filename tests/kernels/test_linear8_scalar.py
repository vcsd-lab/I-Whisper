import pytest
import torch

from i_whisper.kernels import _C

def linear8_accum_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    weight_zero=None,
):
    """
    Reference for Linear8 accumulator stage.

    x:           [*, K] int8
    weight:      [N, K] int8
    bias:        [N] int32 or None
    weight_zero: [N] int8 or None

    return:      [*, N] int32
    """

    out = torch.matmul(
        x.to(torch.int32),
        weight.to(torch.int32).t(),
    )

    if bias is not None:
        out = out + bias

    if weight_zero is not None:
        x_sum = torch.sum(
            x.to(torch.int32),
            dim=-1,
            keepdim=True,
        )

        out = (
            out
            - x_sum * weight_zero.to(torch.int32)
        )

    return out




def test_linear8_accum_rejects_wrong_zero_dtype():
    x = torch.ones(
        3,
        11,
        dtype=torch.int8,
    )

    weight = torch.ones(
        13,
        11,
        dtype=torch.int8,
    )

    zero = torch.ones(
        13,
        dtype=torch.int32,  # wrong
    )

    with pytest.raises(RuntimeError):
        _C.linear8_accum_scalar_debug(
            x,
            weight,
            None,
            zero,
        )


def test_linear8_accum_rejects_wrong_zero_shape():
    x = torch.ones(
        3,
        11,
        dtype=torch.int8,
    )

    weight = torch.ones(
        13,
        11,
        dtype=torch.int8,
    )

    zero = torch.ones(
        12,              # should be N=13
        dtype=torch.int8,
    )

    with pytest.raises(RuntimeError):
        _C.linear8_accum_scalar_debug(
            x,
            weight,
            None,
            zero,
        )



"""
Debugging test
"""
def test_linear8_accum_handcrafted_gemm_only():
    x = torch.tensor(
        [
            [1,  2,  3],
            [-1, 4,  2],
        ],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [
            [1,  0, -1],
            [2,  1,  3],
        ],
        dtype=torch.int8,
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        None,
        None,
    )

    # row 0:
    # [1,2,3] dot [1,0,-1] = -2
    # [1,2,3] dot [2,1,3]  = 13
    #
    # row 1:
    # [-1,4,2] dot [1,0,-1] = -3
    # [-1,4,2] dot [2,1,3]  = 8
    expected = torch.tensor(
        [
            [-2, 13],
            [-3,  8],
        ],
        dtype=torch.int32,
    )

    assert torch.equal(actual, expected), f"Expected {expected}, but got {actual}"


def test_linear8_accum_handcrafted_bias():
    x = torch.tensor(
        [
            [1,  2,  3],
            [-1, 4,  2],
        ],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [
            [1,  0, -1],
            [2,  1,  3],
        ],
        dtype=torch.int8,
    )

    bias = torch.tensor(
        [10, -5],
        dtype=torch.int32,
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        bias,
        None,
    )

    # GEMM:
    # [-2, 13]
    # [-3,  8]
    #
    # + bias [10, -5]
    expected = torch.tensor(
        [
            [8, 8],
            [7, 3],
        ],
        dtype=torch.int32,
    )

    assert torch.equal(actual, expected), f"Expected {expected}, but got {actual}"


def test_linear8_accum_handcrafted_zero():
    x = torch.tensor(
        [
            [1,  2,  3],   # sum = 6
            [-1, 4,  2],   # sum = 5
        ],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [
            [1,  0, -1],
            [2,  1,  3],
        ],
        dtype=torch.int8,
    )

    weight_zero = torch.tensor(
        [2, -1],
        dtype=torch.int8,
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        None,
        weight_zero,
    )

    # GEMM:
    # [-2, 13]
    # [-3,  8]
    #
    # correction = sum(x) * zero
    #
    # row0:
    # [6*2, 6*(-1)] = [12, -6]
    # [-2,13] - [12,-6] = [-14,19]
    #
    # row1:
    # [5*2, 5*(-1)] = [10,-5]
    # [-3,8] - [10,-5] = [-13,13]
    expected = torch.tensor(
        [
            [-14, 19],
            [-13, 13],
        ],
        dtype=torch.int32,
    )

    assert torch.equal(actual, expected), f"Expected {expected}, but got {actual}"


def test_linear8_accum_handcrafted_zero_only():
    x = torch.tensor(
        [
            [1, 2, 3],   # sum = 6
            [-1, 4, 2],  # sum = 5
        ],
        dtype=torch.int8,
    )

    weight = torch.zeros(
        (2, 3),
        dtype=torch.int8,
    )

    weight_zero = torch.tensor(
        [2, -1],
        dtype=torch.int8,
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        None,
        weight_zero,
    )

    expected = torch.tensor(
        [
            [-12,  6],
            [-10,  5],
        ],
        dtype=torch.int32,
    )

    assert torch.equal(actual, expected)


def test_linear8_accum_handcrafted_bias():
    x = torch.tensor(
        [
            [1,  2,  3],
            [-1, 4,  2],
        ],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [
            [1,  0, -1],
            [2,  1,  3],
        ],
        dtype=torch.int8,
    )

    bias = torch.tensor(
        [10, -5],
        dtype=torch.int32,
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        bias,
        None,
    )

    # GEMM:
    # [-2, 13]
    # [-3,  8]
    #
    # + bias [10, -5]
    expected = torch.tensor(
        [
            [8, 8],
            [7, 3],
        ],
        dtype=torch.int32,
    )

    assert torch.equal(actual, expected), f"Expected {expected}, but got {actual}"


def test_linear8_accum_handcrafted_bias_and_zero():
    x = torch.tensor(
        [
            [1,  2,  3],
            [-1, 4,  2],
        ],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [
            [1,  0, -1],
            [2,  1,  3],
        ],
        dtype=torch.int8,
    )

    bias = torch.tensor(
        [10, -5],
        dtype=torch.int32,
    )

    weight_zero = torch.tensor(
        [2, -1],
        dtype=torch.int8,
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        bias,
        weight_zero,
    )

    # GEMM + bias:
    # [8,8]
    # [7,3]
    #
    # zero correction:
    # row0 [12,-6]
    # row1 [10,-5]
    expected = torch.tensor(
        [
            [-4, 14],
            [-3,  8],
        ],
        dtype=torch.int32,
    )

    assert torch.equal(actual, expected), f"Expected {expected}, but got {actual}"



"""
Randomized tests
"""
def test_linear8_accum_scalar_bias_no_zero():
    M = 7
    K = 11
    N = 13

    torch.manual_seed(0)
    
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

    # 故意不要使用整個 int32 range，
    # 避免 accumulator signed overflow 干擾目前測試。
    bias = torch.randint(
        -100_000,
        100_001,
        (N,),
        dtype=torch.int32,
    )

    weight_zero = None

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        bias,
        weight_zero,
    )

    expected = linear8_accum_reference(
        x,
        weight,
        bias,
        weight_zero,
    )

    assert actual.dtype == torch.int32
    assert actual.shape == (M, N)
    assert torch.equal(actual, expected)


def test_linear8_accum_scalar_bias_with_zero():
    M = 7
    K = 11
    N = 13

    torch.manual_seed(0)
    
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

    bias = torch.randint(
        -100_000,
        100_001,
        (N,),
        dtype=torch.int32,
    )

    weight_zero = torch.randint(
        -128,
        128,
        (N,),
        dtype=torch.int8,
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        bias,
        weight_zero,
    )

    expected = linear8_accum_reference(
        x,
        weight,
        bias,
        weight_zero,
    )

    assert actual.dtype == torch.int32
    assert actual.shape == (M, N)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "use_bias,use_zero",
    [
        (False, False),
        (True,  False),
        (False, True),
        (True,  True),
    ],
)
def test_linear8_accum_scalar_optional_paths(
    use_bias,
    use_zero,
):
    M = 7
    K = 11
    N = 13

    torch.manual_seed(0)

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

    weight_zero = (
        torch.randint(
            -128,
            128,
            (N,),
            dtype=torch.int8,
        )
        if use_zero
        else None
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        bias,
        weight_zero,
    )

    expected = linear8_accum_reference(
        x,
        weight,
        bias,
        weight_zero,
    )

    assert torch.equal(actual, expected)


def test_linear8_accum_scalar_3d_input():
    B = 2
    M = 7
    K = 11
    N = 13

    torch.manual_seed(1)

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

    weight_zero = torch.randint(
        -128,
        128,
        (N,),
        dtype=torch.int8,
    )

    actual = _C.linear8_accum_scalar_debug(
        x,
        weight,
        bias,
        weight_zero,
    )

    expected = linear8_accum_reference(
        x,
        weight,
        bias,
        weight_zero,
    )

    assert actual.shape == (B, M, N)
    assert torch.equal(actual, expected)



    

def mulh_reference(
    x,
    multiplier
):
    out = (
        x.to(torch.int64)
        * multiplier.to(torch.int64)
    ) >> 32

    return out.to(torch.int32)


def rounding_rshift_reference(
    x: torch.Tensor,
    shift: torch.Tensor,
):
    """
    x:     [*, N] int32/int64
    shift: [N] int8
    """

    x = x.to(torch.int64)
    shift = shift.to(torch.int64)

    # broadcast [N] over all leading dimensions
    shift_view = shift.view(
        *([1] * (x.dim() - 1)),
        -1,
    )

    result = x.clone()

    mask = shift_view > 0

    # PyTorch broadcast this to x's shape
    rounding = (
        (1 << torch.clamp(shift_view - 1, min=0))
        - (x < 0).to(torch.int64)
    )

    shifted = (x + rounding) >> shift_view

    result = torch.where(
        mask,
        shifted,
        result,
    )

    return result


def requant_reference(
    accum: torch.Tensor,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
):
    multiplier_view = multiplier.to(torch.int64).view(
        *([1] * (accum.dim() - 1)),
        -1,
    )

    mulh = (
        accum.to(torch.int64)
        * multiplier_view
    ) >> 32

    return rounding_rshift_reference(
        mulh,
        shift,
    ).to(torch.int32)


def test_linear8_requant_handcrafted_mulh_only():
    accum = torch.tensor(
        [
            [
                100,
                -100,
                1234567,
                -1234567,
            ]
        ],
        dtype=torch.int32,
    )

    multiplier = torch.tensor(
        [
            0x80000000,
            0x80000000,
            0x40000000,
            0x40000000,
        ],
        dtype=torch.uint32,
    )

    shift = torch.zeros(
        4,
        dtype=torch.int8,
    )

    actual = _C.linear8_requant_scalar_debug(
        accum,
        multiplier,
        shift,
    )

    expected = (
        accum.to(torch.int64)
        * multiplier.to(torch.int64)
    ) >> 32

    expected = expected.to(torch.int32)

    assert torch.equal(
        actual,
        expected,
    )


def test_linear8_requant_rounding_boundaries():
    accum = torch.tensor(
        [[
             2,
             3,
            -2,
            -3,
             6,
             7,
            -6,
            -7,
        ]],
        dtype=torch.int32,
    )

    multiplier = torch.full(
        (8,),
        0x80000000,
        dtype=torch.uint32,
    )

    shift = torch.tensor(
        [1, 1, 1, 1, 2, 2, 2, 2],
        dtype=torch.int8,
    )

    actual = _C.linear8_requant_scalar_debug(
        accum,
        multiplier,
        shift,
    )

    expected = requant_reference(
        accum,
        multiplier,
        shift,
    )

    assert torch.equal(actual, expected), (
        f"\nexpected:\n{expected}"
        f"\nactual:\n{actual}"
    )


def test_linear8_requant_shift_zero():
    accum = torch.tensor(
        [[
            0,
            1,
            -1,
            1234567,
            -1234567,
        ]],
        dtype=torch.int32,
    )

    multiplier = torch.tensor(
        [
            0x80000000,
            0x40000000,
            0xFFFFFFFF,
            0x80000000,
            0x80000000,
        ],
        dtype=torch.uint32,
    )

    shift = torch.zeros(
        5,
        dtype=torch.int8,
    )

    actual = _C.linear8_requant_scalar_debug(
        accum,
        multiplier,
        shift,
    )

    expected = requant_reference(
        accum,
        multiplier,
        shift,
    )

    assert torch.equal(actual, expected)



def test_linear8_requant_per_channel_parameters():
    accum = torch.tensor(
        [
            [1000, 1000, 1000, 1000],
            [-1000, -1000, -1000, -1000],
        ],
        dtype=torch.int32,
    )

    multiplier = torch.tensor(
        [
            0x80000000,
            0x40000000,
            0x20000000,
            0x7FFFFFFF,
        ],
        dtype=torch.uint32,
    )

    shift = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.int8,
    )

    actual = _C.linear8_requant_scalar_debug(
        accum,
        multiplier,
        shift,
    )

    expected = requant_reference(
        accum,
        multiplier,
        shift,
    )

    assert torch.equal(actual, expected)


def test_linear8_requant_3d():
    torch.manual_seed(0)

    B = 2
    T = 7
    N = 13

    accum = torch.randint(
        -1_000_000,
        1_000_001,
        (B, T, N),
        dtype=torch.int32,
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

    actual = _C.linear8_requant_scalar_debug(
        accum,
        multiplier,
        shift,
    )

    expected = requant_reference(
        accum,
        multiplier,
        shift,
    )

    assert actual.shape == (B, T, N)
    assert torch.equal(actual, expected)


def test_linear8_requant_rejects_wrong_shift_shape():
    accum = torch.ones(
        (7, 13),
        dtype=torch.int32,
    )

    multiplier = torch.ones(
        13,
        dtype=torch.uint32,
    )

    shift = torch.ones(
        12,  # wrong
        dtype=torch.int8,
    )

    with pytest.raises(RuntimeError):
        _C.linear8_requant_scalar_debug(
            accum,
            multiplier,
            shift,
        )