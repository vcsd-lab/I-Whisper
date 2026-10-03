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
        normal multiply -> rounded Rshift -> +8 -> clamp [0, 15]

    LUT output:
        x * LUT -> normal multiply -> rounded Rshift
        -> clamp [-127, 127]
    """
    index = (
        x.int()
        .mul(luti_multiplier.int())
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

    # x * sigmoid LUT
    out = (
        x.int()
        * gate.int()
    )

    # LUT-output requant:
    # normal multiply, NOT MULH
    out = (
        out
        * luto_multiplier.int()
    )

    out = Rshift(
        out,
        luto_shift,
    )

    return out.clamp(
        -127,
        127,
    ).to(torch.int8)


# ============================================================
# Backends
# ============================================================

def _run_scalar(
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


def _run_simd(
    x,
    lut,
    luti_multiplier,
    luti_shift,
    luto_multiplier,
    luto_shift,
):
    return _C.gelu_simd(
        x.contiguous(),
        lut.contiguous(),
        luti_multiplier.contiguous(),
        luti_shift.contiguous(),
        luto_multiplier.contiguous(),
        luto_shift.contiguous(),
    )


# ============================================================
# Helpers
# ============================================================

def _default_lut():
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


def _raise_mismatch(
    name,
    actual,
    ref,
    x,
):
    mismatch = (
        actual != ref
    ).nonzero(
        as_tuple=False
    )

    idx = tuple(
        mismatch[0].tolist()
    )

    raise AssertionError(
        f"{name} mismatch\n"
        f"shape: {tuple(x.shape)}\n"
        f"first mismatch: {idx}\n"
        f"input: {x[idx].item()}\n"
        f"reference: {ref[idx].item()}\n"
        f"actual:    {actual[idx].item()}"
    )


def _assert_all_equal(
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

    scalar = _run_scalar(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )

    simd = _run_simd(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )

    assert scalar.dtype == torch.int8
    assert simd.dtype == torch.int8

    assert scalar.shape == x.shape
    assert simd.shape == x.shape

    if not torch.equal(
        scalar,
        ref,
    ):
        _raise_mismatch(
            "scalar vs Torch",
            scalar,
            ref,
            x,
        )

    if not torch.equal(
        simd,
        ref,
    ):
        _raise_mismatch(
            "SIMD vs Torch",
            simd,
            ref,
            x,
        )

    if not torch.equal(
        simd,
        scalar,
    ):
        _raise_mismatch(
            "SIMD vs scalar",
            simd,
            scalar,
            x,
        )


# ============================================================
# Tests
# ============================================================

def test_gelu_simd_exhaustive_int8():
    """
    Exhaust all valid symmetric INT8 activation values.

    255 elements also exercises:
        15 full SIMD blocks
        +
        15-element scalar tail
    """
    x = torch.arange(
        -127,
        128,
        dtype=torch.int16,
    ).to(torch.int8)

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [104],
        dtype=torch.uint8,
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

    _assert_all_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_simd_per_channel_random():
    """
    257 channels deliberately gives:
        16 * 16 + 1

    so both SIMD and per-channel scalar tail are exercised.
    """
    torch.manual_seed(1001)

    channels = 257

    x = torch.randint(
        -127,
        128,
        (3, 5, channels),
        dtype=torch.int8,
    )

    lut = _default_lut()

    luti_multiplier = torch.randint(
        64,
        129,
        (channels,),
        dtype=torch.uint8,
    )

    luti_shift = torch.randint(
        0,
        33,
        (channels,),
        dtype=torch.int8,
    )

    luto_multiplier = torch.randint(
        0,
        256,
        (channels,),
        dtype=torch.uint8,
    )

    luto_shift = torch.randint(
        0,
        33,
        (channels,),
        dtype=torch.int8,
    )

    _assert_all_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_simd_variable_shift_boundaries():
    """
    Directly stress per-lane variable Rshift.

    Every shift from 0 through 32 is present.
    """
    channels = 99

    x = torch.arange(
        channels,
        dtype=torch.int16,
    )

    x = (
        (x * 37) % 255
        - 127
    ).to(torch.int8)

    x = x.reshape(
        1,
        1,
        channels,
    )

    lut = _default_lut()

    luti_multiplier = torch.full(
        (channels,),
        104,
        dtype=torch.uint8,
    )

    luti_shift = (
        torch.arange(
            channels,
            dtype=torch.int64,
        )
        % 33
    ).to(torch.int8)

    luto_multiplier = (
        torch.arange(
            channels,
            dtype=torch.int64,
        )
        % 256
    ).to(torch.uint8)

    luto_shift = (
        torch.arange(
            channels - 1,
            -1,
            -1,
            dtype=torch.int64,
        )
        % 33
    ).to(torch.int8)

    _assert_all_equal(
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
        0,
        1,
        63,
        64,
        104,
        127,
        128,
        255,
    ],
)
def test_gelu_simd_unsigned_mulh(
    multiplier,
):
    """
    Catch signed/unsigned MULH mistakes.
    """
    x = torch.arange(
        -127,
        128,
        dtype=torch.int16,
    ).to(torch.int8)

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [multiplier],
        dtype=torch.uint8,
    )

    luti_shift = torch.tensor(
        [1],
        dtype=torch.int8,
    )

    luto_multiplier = torch.tensor(
        [137],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [7],
        dtype=torch.int8,
    )

    _assert_all_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_simd_lut_saturation():
    """
    Drive LUT indices toward both clamp boundaries.
    """
    pattern = torch.tensor(
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

    # > 16 so actual SIMD table lookup is exercised.
    x = pattern.repeat(5)

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [104],
        dtype=torch.uint8,
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

    _assert_all_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_simd_custom_lut():
    """
    Ensure SIMD vqtbl lookup consumes the supplied LUT.
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
        [101],
        dtype=torch.uint8,
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

    _assert_all_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_simd_encoder_scalar_broadcast():
    """
    Representative encoder gelu1 layout:

        [B, C, T]

    Quantization parameters are shared scalars.

    T=750 is deliberately not divisible by 16.
    """
    torch.manual_seed(2002)

    x = torch.randint(
        -127,
        128,
        (1, 512, 750),
        dtype=torch.int8,
    )

    lut = _default_lut()

    luti_multiplier = torch.tensor(
        [104],
        dtype=torch.uint8,
    )

    luti_shift = torch.tensor(
        [7],
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

    _assert_all_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_simd_mlp_per_channel():
    """
    Representative base-model MLP layout:

        [..., 2048]

    with per-channel requant parameters.
    """
    torch.manual_seed(3003)

    channels = 2048

    x = torch.randint(
        -127,
        128,
        (2, 8, channels),
        dtype=torch.int8,
    )

    lut = _default_lut()

    luti_multiplier = torch.randint(
        64,
        129,
        (channels,),
        dtype=torch.uint8,
    )

    luti_shift = torch.randint(
        0,
        16,
        (channels,),
        dtype=torch.int8,
    )

    luto_multiplier = torch.randint(
        0,
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

    _assert_all_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )


def test_gelu_simd_luti_uint8_widening():
    """
    Verify uint8 LUT-input multipliers are widened without
    signed interpretation.

    Especially important for values >= 128:
        uint8 128..255
    must remain positive after SIMD widening.

    Exactly 16 channels forces one full SIMD block.
    """

    channels = 16

    # Two SIMD rows:
    # one positive, one negative.
    x = torch.stack(
        [
            torch.full(
                (channels,),
                127,
                dtype=torch.int8,
            ),
            torch.full(
                (channels,),
                -127,
                dtype=torch.int8,
            ),
        ]
    )

    lut = _default_lut()

    # Deliberately cross the int8 signed boundary.
    luti_multiplier = torch.tensor(
        [
            0,
            1,
            63,
            64,
            104,
            127,
            128,
            129,
            160,
            191,
            192,
            200,
            224,
            240,
            254,
            255,
        ],
        dtype=torch.uint8,
    )

    # Keep the resulting LUT indices mostly away from
    # immediate saturation, so multiplier errors remain visible.
    luti_shift = torch.full(
        (channels,),
        12,
        dtype=torch.int8,
    )

    # Keep LUT-output requant simple so differences in
    # LUT selection remain observable.
    luto_multiplier = torch.tensor(
        [1],
        dtype=torch.uint8,
    )

    luto_shift = torch.tensor(
        [8],
        dtype=torch.int8,
    )

    _assert_all_equal(
        x,
        lut,
        luti_multiplier,
        luti_shift,
        luto_multiplier,
        luto_shift,
    )