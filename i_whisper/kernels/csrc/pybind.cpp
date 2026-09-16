#include <torch/extension.h>

#include <string>

#include "linear8/linear8_torch.h"
#include "linear32/linear32_torch.h"

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
        "linear8_simd",
        &linear8_simd_torch,
        "I-Whisper complete Linear8 SIMD kernel",
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
        "linear8_accum_simd_debug",
        &linear8_accum_simd_debug_torch,
        "Linear8 simd accumulator debug");

    m.def(
        "linear32_scalar",
        &linear32_scalar_torch,
        "I-Whisper complete Linear32 scalar kernel",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("weight_zero"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"));
    m.def(
        "linear32_simd",
        &linear32_simd_torch,
        "I-Whisper complete Linear32 SIMD kernel",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("weight_zero"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"));
    m.def(
        "linear32_accum_scalar_debug",
        &linear32_accum_scalar_debug_torch,
        "Linear32 scalar accumulator debug");
    m.def(
        "linear32_requant_scalar_debug",
        &linear32_requant_scalar_debug_torch,
        "Linear32 requantization debug");
    m.def(
        "linear32_accum_simd_debug",
        &linear32_accum_simd_debug_torch,
        "Linear32 simd accumulator debug");
}