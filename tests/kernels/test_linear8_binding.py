from i_whisper.kernels import _C
import torch
import pytest


def test_cpp_extension_import():
    assert _C is not None


def test_cpp_extension_hello():
    result = _C.hello()

    assert result == "Hello from I-Whisper C++ extension!"

def test_linear8_scalar_basic():
    x = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=torch.int8,
    )

    y = _C.linear8_scalar(x, weight)

    expected = torch.tensor(
        [
            [1, 2],
            [4, 5],
        ],
        dtype=torch.int8,
    )

    assert torch.equal(y, expected)


def test_linear8_scalar_3d_input():
    x = torch.tensor(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ],
        dtype=torch.int8,
    )

    weight = torch.tensor(
        [
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=torch.int8,
    )

    y = _C.linear8_scalar(x, weight)

    expected = torch.tensor(
        [
            [
                [1, 3],
                [4, 6],
            ]
        ],
        dtype=torch.int8,
    )

    assert y.shape == (1, 2, 2)
    assert torch.equal(y, expected)


def test_linear8_scalar_rejects_float_input():
    x = torch.randn(2, 3)
    weight = torch.ones(
        4,
        3,
        dtype=torch.int8,
    )

    with pytest.raises(RuntimeError):
        _C.linear8_scalar(x, weight)