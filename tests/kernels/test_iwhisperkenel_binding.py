from i_whisper.kernels import _C
import torch
import pytest


def test_cpp_extension_import():
    assert _C is not None


def test_cpp_extension_hello():
    result = _C.hello()

    assert result == "Hello from I-Whisper C++ extension!"