#include <torch/extension.h>

#include <string>

#include "linear8/linear8_torch.h"

std::string hello()
{
    return "Hello from I-Whisper C++ extension!";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "hello",
        &hello,
        "I-Whisper C++ extension smoke test");
    m.def(
        "linear8_scalar",
        &linear8_scalar_torch,
        "I-Whisper complete Linear8 scalar kernel",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("weight_zero"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"));
    m.def(
        "linear8_simde",
        &linear8_simde_torch,
        "I-Whisper complete Linear8 SIMDe kernel",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("weight_zero"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"));
    m.def(
        "linear8_accum_scalar_debug",
        &linear8_accum_scalar_debug_torch);
    m.def(
        "linear8_requant_scalar_debug",
        &linear8_requant_scalar_debug_torch);
    m.def(
        "linear8_accum_simde_debug",
        &linear8_accum_simde_debug_torch,
        "Linear8 SIMDe accumulator debug");
}