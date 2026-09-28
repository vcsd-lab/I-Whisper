import pytest
import torch

from i_whisper.kernels import _C
from i_whisper.modules.functional import Rshift


def _torch_golden(
    x,
    lut,
    luti_multiplier,
    luti_shift,
    luto_multiplier,
    luto_shift,
):
    """
    Frozen integer GeLU Torch golden.

    LUT input:
        MULH -> rounded Rshift -> +8 -> clamp [0, 15]

    LUT output:
        x * LUT -> normal multiply -> rounded Rshift
        -> clamp [-127, 127]
    """

    index = (
        x.long()
        .mul(luti_multiplier.long())
        >> 32
    ).int()

    index = Rshift(
        index,
        luti_shift,
    )

    index = (
        index + 8
    ).clamp(
        0,
        15,
    ).long()

    gate = lut[index]

    out = (
        x.int()
        * gate.int()
    )

    out = (
        out
        * luto_multiplier.int()
    )

    out = Rshift(
        out,
        luto_shift,
    )

    out = out.clamp(
        -127,
        127,
    ).to(torch.int8)

    return out


def _run_cpp(
    x,
    lut,
    luti_multiplier,
    luti_shift,
    luto_multiplier,
    luto_shift,
):
    return _C.gelu_scalar(
        x.contiguous(),
        lut.contiguous(),
        luti_multiplier.contiguous(),
        luti_shift.contiguous(),
        luto_multiplier.contiguous(),
        luto_shift.contiguous(),
    )


def _assert_equal(
    x,
    lut,
    luti_multiplier,
    luti_shift,
    luto_multiplier,
    luto_shift,
):
    ref = _torch_golden(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )

    out = _run_cpp(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )

    assert out.dtype == torch.int8
    assert out.shape == x.shape

    if not torch.equal(out, ref):
        mismatch = (
            out != ref
        ).nonzero(
            as_tuple=False
        )

        idx = tuple(
            mismatch[0].tolist()
        )

        raise AssertionError(
            "GeLU scalar mismatch\n"
            f"shape: {tuple(x.shape)}\n"
            f"first mismatch: {idx}\n"
            f"input: {x[idx].item()}\n"
            f"torch: {ref[idx].item()}\n"
            f"cpp:   {out[idx].item()}"
        )


def _default_lut():
    """
    Use a monotonic uint8 LUT with the same 16-entry contract
    as the production integer GeLU.
    """
    return torch.tensor(
        [
            0,
            3,
            7,
            15,
            29,
            49,
            75,
            108,
            147,
            180,
            206,
            226,
            240,
            248,
            252,
            255,
        ],
        dtype=torch.uint8,
    )


def test_gelu_scalar_exhaustive_int8():
    """
    Exhaust every valid symmetric INT8 activation:
        -127 ... +127

    This is much stronger than random-only testing.
    """

    x = torch.arange(
        -127,
        128,
        dtype=torch.int16,
    ).to(torch.int8)

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [0x68000000],
        dtype=torch.uint32,
    )

    luti_shift = torch.tensor(
        [4],
        dtype=torch.int8,
    )

    luto_multiplier = torch.tensor(
        [173],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [8],
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


@pytest.mark.parametrize(
    "luti_shift_value",
    [
        0,
        1,
        2,
        7,
        15,
        31,
    ],
)
def test_gelu_scalar_luti_shift(
    luti_shift_value,
):
    """
    Explicitly cover new Rshift semantics,
    especially shift == 0 -> identity.
    """

    x = torch.arange(
        -127,
        128,
        dtype=torch.int16,
    ).to(torch.int8)

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [0x70000000],
        dtype=torch.uint32,
    )

    luti_shift = torch.tensor(
        [luti_shift_value],
        dtype=torch.int8,
    )

    luto_multiplier = torch.tensor(
        [127],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [7],
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


@pytest.mark.parametrize(
    "luto_shift_value",
    [
        0,
        1,
        2,
        7,
        15,
        23,
    ],
)
def test_gelu_scalar_luto_shift(
    luto_shift_value,
):
    """
    Exercise positive and negative rounding in the
    LUT-output requant path.
    """

    x = torch.arange(
        -127,
        128,
        dtype=torch.int16,
    ).to(torch.int8)

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [0x60000000],
        dtype=torch.uint32,
    )

    luti_shift = torch.tensor(
        [3],
        dtype=torch.int8,
    )

    luto_multiplier = torch.tensor(
        [255],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [luto_shift_value],
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


@pytest.mark.parametrize(
    "multiplier",
    [
        1,
        0x40000000,
        0x7FFFFFFF,
        0x80000000,
        0xF1234567,
        0xFFFFFFFF,
    ],
)
def test_gelu_scalar_luti_unsigned_mulh(
    multiplier,
):
    """
    luti multiplier is uint32.

    Values above INT32_MAX are important because an
    accidental signed interpretation would be incorrect.
    """

    x = torch.arange(
        -127,
        128,
        dtype=torch.int16,
    ).to(torch.int8)

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [multiplier],
        dtype=torch.uint32,
    )

    luti_shift = torch.tensor(
        [1],
        dtype=torch.int8,
    )

    luto_multiplier = torch.tensor(
        [128],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [7],
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_scalar_per_channel():
    """
    Production MLP case:
        input [..., C]
        parameters [C]
    """

    torch.manual_seed(1234)

    channels = 256

    x = torch.randint(
        -127,
        128,
        (2, 7, channels),
        dtype=torch.int8,
    )

    lut = _default_lut()

    luti_multiplier = torch.randint(
        1,
        2**32,
        (channels,),
        dtype=torch.uint32,
    )

    luti_shift = torch.randint(
        0,
        16,
        (channels,),
        dtype=torch.int8,
    )

    luto_multiplier = torch.randint(
        1,
        256,
        (channels,),
        dtype=torch.uint8,
    )

    luto_shift = torch.randint(
        0,
        16,
        (channels,),
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_scalar_scalar_broadcast():
    """
    Production encoder gelu1-style case:
    scalar quantization parameters broadcast over the tensor.
    """

    torch.manual_seed(5678)

    x = torch.randint(
        -127,
        128,
        (2, 512, 750),
        dtype=torch.int8,
    )

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [0x71000000],
        dtype=torch.uint32,
    )

    luti_shift = torch.tensor(
        [8],
        dtype=torch.int8,
    )

    luto_multiplier = torch.tensor(
        [183],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [8],
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_scalar_lut_boundaries():
    """
    Force LUT indices to saturate at both endpoints:
        index <= 0
        index >= 15
    """

    x = torch.tensor(
        [
            -127,
            -126,
            -64,
            -1,
            0,
            1,
            64,
            126,
            127,
        ],
        dtype=torch.int8,
    )

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [0xFFFFFFFF],
        dtype=torch.uint32,
    )

    luti_shift = torch.tensor(
        [0],
        dtype=torch.int8,
    )

    luto_multiplier = torch.tensor(
        [255],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [8],
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_scalar_custom_lut():
    """
    Ensure the C++ kernel actually consumes the provided LUT
    rather than assuming a hard-coded table.
    """

    x = torch.arange(
        -127,
        128,
        dtype=torch.int16,
    ).to(torch.int8)

    lut = torch.tensor(
        [
            0,
            1,
            5,
            9,
            17,
            31,
            55,
            91,
            139,
            177,
            207,
            227,
            241,
            249,
            253,
            255,
        ],
        dtype=torch.uint8,
    )

    luti_multiplier = torch.tensor(
        [0x65000000],
        dtype=torch.uint32,
    )

    luti_shift = torch.tensor(
        [5],
        dtype=torch.int8,
    )

    luto_multiplier = torch.tensor(
        [211],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [8],
        dtype=torch.int8,
    )

    _assert_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )