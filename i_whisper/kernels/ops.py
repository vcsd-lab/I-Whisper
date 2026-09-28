# i_whisper/kernels/ops.py

from . import _C


def linear8_scalar(
    x,
    weight,
    bias,
    weight_zero,
    multiplier,
    shift,
    asym=True,
):
    return _C.linear8_scalar(
        x,
        weight,
        bias,
        weight_zero,
        multiplier,
        shift,
        bias is not None,
        asym,
    )