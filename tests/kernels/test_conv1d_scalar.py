import pytest
import torch
import torch.nn.functional as F

from i_whisper.kernels import _C
from i_whisper.modules.functional import Rshift


def _u32_scalar(value: int) -> torch.Tensor:
    return torch.tensor(
        [value],
        dtype=torch.uint32,
    )


def _i8_scalar(value: int) -> torch.Tensor:
    return torch.tensor(
        [value],
        dtype=torch.int8,
    )


def _torch_unfold_accum(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    Exact shape/order used by current Conv1d_int._conv_forward.

    x:
        [B,Cin,Tin] INT8

    weight:
        [Cout,Cin*K] INT8

    output:
        [B,Cout,Tout] INT32
    """
    B, Cin, _ = x.shape
    Cout = weight.shape[0]

    assert weight.shape[1] % Cin == 0

    K = weight.shape[1] // Cin

    padded = F.pad(
        x,
        (padding, padding),
    )

    patches = padded.unfold(
        dimension=2,
        size=K,
        step=stride,
    )

    # [B,Cin,Tout,K]
    # ->
    # [B,Tout,Cin,K]
    # ->
    # [B,Tout,Cin*K]
    patches = (
        patches
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(
            B,
            -1,
            Cin * K,
        )
    )

    out = (
        patches.int()
        @ weight.int().t()
    )

    if bias is not None:
        out = out + bias

    return out.permute(
        0,
        2,
        1,
    ).contiguous()


def _torch_golden(
    x,
    weight,
    bias,
    multiplier,
    shift,
    stride,
    padding,
):
    out = _torch_unfold_accum(
        x,
        weight,
        bias,
        stride,
        padding,
    )

    out = (
        out.long()
        .mul(
            multiplier.long()
        )
        >> 32
    ).int()

    out = Rshift(
        out,
        shift,
    )

    return (
        out
        .clamp(-127, 127)
        .to(torch.int8)
    )


def _run_accum(
    x,
    weight,
    bias,
    stride,
    padding,
):
    return _C.conv1d_accum_scalar(
        x.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
        stride,
        padding,
    )


def _run_scalar(
    x,
    weight,
    bias,
    multiplier,
    shift,
    stride,
    padding,
):
    return _C.conv1d_scalar(
        x.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
        multiplier.contiguous(),
        shift.contiguous(),
        stride,
        padding,
    )


@pytest.mark.parametrize(
    "shape,stride,padding",
    [
        # B, Cin, Tin, Cout, K

        # Small indexing cases.
        ((1, 1, 5, 2, 1), 1, 0),
        ((1, 2, 7, 3, 3), 1, 1),
        ((2, 3, 8, 4, 3), 2, 1),

        # Non-multiple dimensions for future SIMD tails.
        ((1, 7, 17, 5, 3), 1, 1),
        ((1, 15, 19, 9, 3), 2, 1),

        # Whisper-like channel counts, short temporal length.
        ((1, 80, 21, 384, 3), 1, 1),
        ((1, 384, 21, 384, 3), 2, 1),
    ],
)
@pytest.mark.parametrize(
    "with_bias",
    [False, True],
)
def test_conv1d_accum_scalar_matches_unfold(
    shape,
    stride,
    padding,
    with_bias,
):
    B, Cin, Tin, Cout, K = shape

    torch.manual_seed(
        1000
        + B * 11
        + Cin * 13
        + Tin * 17
        + Cout * 19
        + K * 23
        + stride * 29
        + padding * 31
        + int(with_bias) * 37
    )

    x = torch.randint(
        -128,
        128,
        (B, Cin, Tin),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (Cout, Cin * K),
        dtype=torch.int8,
    )

    bias = None

    if with_bias:
        bias = torch.randint(
            -(1 << 20),
            1 << 20,
            (Cout,),
            dtype=torch.int32,
        )

    ref = _torch_unfold_accum(
        x,
        weight,
        bias,
        stride,
        padding,
    )

    out = _run_accum(
        x,
        weight,
        bias,
        stride,
        padding,
    )

    assert out.dtype == torch.int32
    assert out.shape == ref.shape

    assert torch.equal(
        out,
        ref,
    ), "Conv1d raw accumulation mismatch"


@pytest.mark.parametrize(
    "shape,stride,padding",
    [
        ((1, 2, 7, 3, 3), 1, 1),
        ((2, 3, 8, 4, 3), 2, 1),
        ((1, 7, 17, 5, 3), 1, 1),
        ((1, 80, 21, 384, 3), 1, 1),
        ((1, 384, 21, 384, 3), 2, 1),
    ],
)
@pytest.mark.parametrize(
    "multiplier,shift",
    [
        (0x00000001, 0),
        (0x40000000, 0),
        (0x7FFFFFFF, 1),
        (0x80000000, 7),
        (0xFFFFFFFF, 15),
        (0xFFFFFFFF, 31),
        (0xFFFFFFFF, 32),
        (0xFFFFFFFF, 64),
        (0x60000000, -3),
    ],
)
def test_conv1d_scalar_matches_model_golden(
    shape,
    stride,
    padding,
    multiplier,
    shift,
):
    B, Cin, Tin, Cout, K = shape

    torch.manual_seed(
        3000
        + B * 41
        + Cin * 43
        + Tin * 47
        + Cout * 53
        + multiplier % 997
        + shift
    )

    x = torch.randint(
        -128,
        128,
        (B, Cin, Tin),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (Cout, Cin * K),
        dtype=torch.int8,
    )

    bias = torch.randint(
        -(1 << 22),
        1 << 22,
        (Cout,),
        dtype=torch.int32,
    )

    multiplier_t = _u32_scalar(
        multiplier
    )

    shift_t = _i8_scalar(
        shift
    )

    ref = _torch_golden(
        x,
        weight,
        bias,
        multiplier_t,
        shift_t,
        stride,
        padding,
    )

    out = _run_scalar(
        x,
        weight,
        bias,
        multiplier_t,
        shift_t,
        stride,
        padding,
    )

    assert out.dtype == torch.int8
    assert out.shape == ref.shape

    assert torch.equal(
        out,
        ref,
    ), "Conv1d fused scalar mismatch"


def test_conv1d_scalar_padding_edges():
    """
    Make the zero-padding contribution observable by hand.
    """
    x = torch.tensor(
        [[[1, 2, 3, 4]]],
        dtype=torch.int8,
    )

    # One output channel, one input channel, K=3.
    # Flattened [Cin,K] order.
    weight = torch.tensor(
        [[10, 20, 30]],
        dtype=torch.int8,
    )

    ref = _torch_unfold_accum(
        x,
        weight,
        None,
        stride=1,
        padding=1,
    )

    # Expected:
    # t0: 0*10 + 1*20 + 2*30 = 80
    # t1: 1*10 + 2*20 + 3*30 = 140
    # t2: 2*10 + 3*20 + 4*30 = 200
    # t3: 3*10 + 4*20 + 0*30 = 110
    expected = torch.tensor(
        [[[80, 140, 200, 110]]],
        dtype=torch.int32,
    )

    assert torch.equal(
        ref,
        expected,
    )

    out = _run_accum(
        x,
        weight,
        None,
        1,
        1,
    )

    assert torch.equal(
        out,
        expected,
    )


def test_conv1d_scalar_stride2():
    x = torch.tensor(
        [[[1, 2, 3, 4, 5]]],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [[1, 1, 1]],
        dtype=torch.int8,
    )

    out = _run_accum(
        x,
        weight,
        None,
        2,
        1,
    )

    # padded [0,1,2,3,4,5,0]
    # windows:
    # [0,1,2] -> 3
    # [2,3,4] -> 9
    # [4,5,0] -> 9
    expected = torch.tensor(
        [[[3, 9, 9]]],
        dtype=torch.int32,
    )

    assert torch.equal(
        out,
        expected,
    )


def test_conv1d_accum_scalar_int32_wrap_with_bias():
    """
    Raw conv accumulation and bias add must retain INT32
    modulo-2^32 semantics.

    A large K is used only as a stress test.
    """
    Cin = 70_000
    K = 1

    x = torch.full(
        (1, Cin, 1),
        127,
        dtype=torch.int8,
    )

    weight = torch.full(
        (1, Cin * K),
        127,
        dtype=torch.int8,
    )

    # Push the mathematical sum farther beyond INT32_MAX.
    bias = torch.tensor(
        [1_500_000_000],
        dtype=torch.int32,
    )

    ref = _torch_unfold_accum(
        x,
        weight,
        bias,
        1,
        0,
    )

    out = _run_accum(
        x,
        weight,
        bias,
        1,
        0,
    )

    assert torch.equal(
        out,
        ref,
    )


def test_conv1d_scalar_does_not_modify_inputs():
    torch.manual_seed(7001)

    x = torch.randint(
        -128,
        128,
        (1, 7, 17),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (9, 7 * 3),
        dtype=torch.int8,
    )

    bias = torch.randint(
        -10000,
        10001,
        (9,),
        dtype=torch.int32,
    )

    x_ref = x.clone()
    w_ref = weight.clone()
    b_ref = bias.clone()

    _run_scalar(
        x,
        weight,
        bias,
        _u32_scalar(0x60000000),
        _i8_scalar(7),
        1,
        1,
    )

    assert torch.equal(x, x_ref)
    assert torch.equal(weight, w_ref)
    assert torch.equal(bias, b_ref)


def test_conv1d_scalar_rejects_bad_weight_shape():
    x = torch.zeros(
        (1, 3, 7),
        dtype=torch.int8,
    )

    # 10 is not divisible by Cin=3.
    weight = torch.zeros(
        (4, 10),
        dtype=torch.int8,
    )

    with pytest.raises(
        RuntimeError,
        match="divisible",
    ):
        _C.conv1d_accum_scalar(
            x,
            weight,
            None,
            1,
            1,
        )
