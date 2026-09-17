#include <torch/extension.h>

#include <string>

#include "linear8/linear8_torch.h"
#include "linear32/linear32_torch.h"
#include "residual15/residual15_torch.h"
#include "layernorm/layernorm_torch.h"
#include "common/fixedpoint.h"
#include "common/checktensor.h"

std::string hello()
{
    return "Hello from I-Whisper C++ extension!";
}

torch::Tensor rounding_rshift_debug(
    const torch::Tensor &x,
    const torch::Tensor &shift)
{
    iwhisper::checktensor::CHECK_CPU_INT32(x, "x");
    iwhisper::checktensor::CHECK_CPU_UINT32(shift, "shift");

    auto output = torch::empty_like(x);

    const int64_t K = x.numel();
    const int32_t *x_ptr = x.data_ptr<int32_t>();
    const uint32_t shamt = shift.data_ptr<uint32_t>()[0];
    int32_t *output_ptr = output.data_ptr<int32_t>();

    for (int64_t j = 0; j < K; ++j)
    {
        const int64_t idx = 0 * K + j;
        output_ptr[idx] = iwhisper::fixedpoint::rounding_rshift(x_ptr[idx], shamt);
    }

    return output;
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

    m.def(
        "residual15_scalar",
        &residual15_scalar_torch,
        "I-Whisper complete Residual15 scalar kernel",
        pybind11::arg("branch"),
        pybind11::arg("residual"),
        pybind11::arg("multiplier"));
    m.def(
        "residual15_simd",
        &residual15_simd_torch,
        "I-Whisper complete Residual15 SIMD kernel",
        pybind11::arg("branch"),
        pybind11::arg("residual"),
        pybind11::arg("multiplier"));

    m.def(
        "layernorm32_scalar",
        &layernorm32_scalar_torch,
        "I-Whisper complete LayerNorm32 scalar kernel",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("ln_multiplier"),
        pybind11::arg("ln_shift"),
        pybind11::arg("out_multiplier"),
        pybind11::arg("out_shift"));
    m.def(
        "layernorm32_simd",
        &layernorm32_simd_torch,
        "I-Whisper complete LayerNorm32 SIMD kernel",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("ln_multiplier"),
        pybind11::arg("ln_shift"),
        pybind11::arg("out_multiplier"),
        pybind11::arg("out_shift"));
    m.def(
        "layernorm32_stats_scalar",
        &layernorm32_stats_scalar_torch,
        "I-Whisper complete LayerNorm32 stats scalar kernel",
        pybind11::arg("input"));
    m.def(
        "layernorm32_stats_simd",
        &layernorm32_stats_simd_torch,
        "I-Whisper complete LayerNorm32 stats SIMD kernel",
        pybind11::arg("input"));

    m.def(
        "layernorm32_rsqrt_scalar",
        &layernorm32_rsqrt_scalar_torch,
        "I-Whisper complete LayerNorm32 rsqrt scalar kernel",
        pybind11::arg("input"));
    m.def(
        "rounding_rshift_debug",
        &rounding_rshift_debug,
        "I-Whisper complete rounding rshift debug kernel",
        pybind11::arg("x"),
        pybind11::arg("shift"));
}