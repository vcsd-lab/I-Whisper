import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import Rshift


def _u32_scalar(value: int):
    return torch.tensor(
        [value],
        dtype=torch.uint32,
    )


def _i8_scalar(value: int):
    return torch.tensor(
        [value],
        dtype=torch.int8,
    )


def _torch_accum(
    x: torch.Tensor,
    weight: torch.Tensor,
):
    return torch.matmul(
        x.int(),
        weight.int().t(),
    )


def _torch_golden(
    x,
    weight,
    multiplier,
    shift,
):
    out = _torch_accum(
        x,
        weight,
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
):
    return _C.classifier_accum_scalar(
        x.contiguous(),
        weight.contiguous(),
    )


def _run_scalar(
    x,
    weight,
    multiplier,
    shift,
):
    return _C.classifier_scalar(
        x.contiguous(),
        weight.contiguous(),
        multiplier.contiguous(),
        shift.contiguous(),
    )


@pytest.mark.parametrize(
    "shape",
    [
        # B,T,K,N
        (1, 1, 1, 1),
        (1, 1, 7, 13),
        (1, 3, 16, 17),
        (2, 5, 31, 33),
        (1, 7, 64, 129),

        # Whisper-like hidden sizes with reduced vocab.
        (1, 1, 384, 257),
        (1, 4, 512, 257),
    ],
)
def test_classifier_accum_scalar_matches_torch(
    shape,
):
    B, T, K, N = shape

    torch.manual_seed(
        1000
        + B * 11
        + T * 13
        + K * 17
        + N * 19
    )

    x = torch.randint(
        -128,
        128,
        (B, T, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (N, K),
        dtype=torch.int8,
    )

    ref = _torch_accum(
        x,
        weight,
    )

    out = _run_accum(
        x,
        weight,
    )

    assert out.dtype == torch.int32
    assert out.shape == ref.shape

    assert torch.equal(
        out.to(torch.int64),
        ref.to(torch.int64),
    ), "classifier raw accumulation mismatch"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 7, 13),
        (1, 3, 16, 17),
        (2, 5, 31, 33),
        (1, 7, 64, 129),
        (1, 1, 384, 257),
        (1, 4, 512, 257),
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
def test_classifier_scalar_matches_model_golden(
    shape,
    multiplier,
    shift,
):
    B, T, K, N = shape

    torch.manual_seed(
        3000
        + B * 23
        + T * 29
        + K * 31
        + N * 37
        + multiplier % 997
        + shift
    )

    x = torch.randint(
        -128,
        128,
        (B, T, K),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (N, K),
        dtype=torch.int8,
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
        multiplier_t,
        shift_t,
    )

    out = _run_scalar(
        x,
        weight,
        multiplier_t,
        shift_t,
    )

    assert out.dtype == torch.int8
    assert out.shape == ref.shape

    assert torch.equal(
        out,
        ref,
    ), "classifier fused scalar mismatch"


def test_classifier_scalar_signed_extremes():
    x = torch.tensor(
        [[[
            -128, -127, -1, 0,
            1, 126, 127, 127,
        ]]],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [
            [-128, -127, -1, 0, 1, 126, 127, 127],
            [127, 126, 1, 0, -1, -127, -128, 127],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int8,
    )

    multiplier = _u32_scalar(
        0xFFFFFFFF
    )

    shift = _i8_scalar(0)

    assert torch.equal(
        _run_scalar(
            x,
            weight,
            multiplier,
            shift,
        ),
        _torch_golden(
            x,
            weight,
            multiplier,
            shift,
        ),
    )


def test_classifier_accum_scalar_int32_wrap():
    # 127*127*K exceeds INT32_MAX.
    K = 140_000

    x = torch.full(
        (1, 1, K),
        127,
        dtype=torch.int8,
    )

    weight = torch.full(
        (2, K),
        127,
        dtype=torch.int8,
    )

    ref = _torch_accum(
        x,
        weight,
    )

    out = _run_accum(
        x,
        weight,
    )

    assert torch.all(ref < 0)

    assert torch.equal(
        out,
        ref,
    )


def test_classifier_scalar_clamp():
    K = 512

    x = torch.full(
        (1, 2, K),
        127,
        dtype=torch.int8,
    )

    weight = torch.stack(
        [
            torch.full(
                (K,),
                127,
                dtype=torch.int8,
            ),
            torch.full(
                (K,),
                -128,
                dtype=torch.int8,
            ),
        ],
        dim=0,
    )

    multiplier = _u32_scalar(
        0xFFFFFFFF
    )

    shift = _i8_scalar(0)

    out = _run_scalar(
        x,
        weight,
        multiplier,
        shift,
    )

    ref = _torch_golden(
        x,
        weight,
        multiplier,
        shift,
    )

    assert torch.equal(
        out,
        ref,
    )

    assert torch.all(
        out[..., 0] == 127
    )

    assert torch.all(
        out[..., 1] == -127
    )


def test_classifier_scalar_does_not_modify_inputs():
    torch.manual_seed(6001)

    x = torch.randint(
        -128,
        128,
        (2, 3, 31),
        dtype=torch.int8,
    )

    weight = torch.randint(
        -128,
        128,
        (33, 31),
        dtype=torch.int8,
    )

    multiplier = _u32_scalar(
        0x60000000
    )

    shift = _i8_scalar(7)

    x_ref = x.clone()
    w_ref = weight.clone()
    m_ref = multiplier.clone()
    s_ref = shift.clone()

    _run_scalar(
        x,
        weight,
        multiplier,
        shift,
    )

    assert x.dtype == x_ref.dtype
    assert torch.equal(
        x.to(torch.int64),
        x_ref.to(torch.int64),
    )
    assert weight.dtype == w_ref.dtype
    assert torch.equal(
        weight.to(torch.int64),
        w_ref.to(torch.int64),
    )
    assert multiplier.dtype == m_ref.dtype
    assert torch.equal(
        multiplier.to(torch.int64),
        m_ref.to(torch.int64),
    )
    assert shift.dtype == s_ref.dtype
    assert torch.equal(
        shift.to(torch.int64),
        s_ref.to(torch.int64),
    )


def test_classifier_scalar_rejects_hidden_mismatch():
    x = torch.zeros(
        (1, 2, 7),
        dtype=torch.int8,
    )

    weight = torch.zeros(
        (13, 8),
        dtype=torch.int8,
    )

    with pytest.raises(
        RuntimeError,
        match="hidden",
    ):
        _run_accum(
            x,
            weight,
        )
