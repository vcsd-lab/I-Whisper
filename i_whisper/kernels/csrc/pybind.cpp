#include <torch/extension.h>

#include <string>

#include "linear8/linear8_torch.h"
#include "linear32/linear32_torch.h"
#include "residual15/residual15_torch.h"
#include "layernorm/layernorm_torch.h"
#include "gelu/gelu_torch.h"
#include "attention/attention_torch.h"
#include "conv1d/conv1d_torch.h"
#include "add_requant15/add_requant15_torch.h"
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

    m.def(
        "gelu_scalar",
        &gelu_scalar_torch,
        "I-Whisper integer GeLU scalar kernel",
        pybind11::arg("input"),
        pybind11::arg("lut"),
        pybind11::arg("luti_multiplier"),
        pybind11::arg("luti_shift"),
        pybind11::arg("luto_multiplier"),
        pybind11::arg("luto_shift"));
    m.def(
        "gelu_simd",
        &gelu_simd_torch,
        "I-Whisper integer GeLU SIMD kernel",
        pybind11::arg("input"),
        pybind11::arg("lut"),
        pybind11::arg("luti_multiplier"),
        pybind11::arg("luti_shift"),
        pybind11::arg("luto_multiplier"),
        pybind11::arg("luto_shift"));

    m.def(
        "qk_scalar",
        &qk_scalar_torch,
        "I-Whisper QK^T scalar kernel",
        pybind11::arg("q"),
        pybind11::arg("k"));

    m.def(
        "qk_simd",
        &qk_simd_torch,
        "I-Whisper QK^T SIMD kernel",
        pybind11::arg("q"),
        pybind11::arg("k"));

    m.def(
        "qk_requant_rowmax_scalar",
        &qk_requant_rowmax_scalar_torch,
        "I-Whisper fused QK requantization and row-max scalar kernel",
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("mask"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"));

    m.def(
        "qk_requant_rowmax_simd",
        &qk_requant_rowmax_simd_torch,
        "I-Whisper fused QK requant row-max SIMD kernel",
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("mask"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"));

    m.def(
        "shift_exp_scalar",
        &shift_exp_scalar_torch,
        "I-Whisper shift-exp + row-sum scalar kernel",
        pybind11::arg("input"),
        pybind11::arg("row_max_plus1"),
        pybind11::arg("mask"),
        pybind11::arg("frac_bits"));

    m.def(
        "shift_exp_simd",
        &shift_exp_simd_torch,
        "I-Whisper shift-exp + row-sum SIMD kernel",
        pybind11::arg("input"),
        pybind11::arg("row_max_plus1"),
        pybind11::arg("mask"),
        pybind11::arg("frac_bits"));

    m.def(
        "pv_scalar",
        &pv_scalar_torch,
        "I-Whisper UINT8 x INT8 P@V scalar accumulation kernel",
        pybind11::arg("p"),
        pybind11::arg("v"));

    m.def(
        "pv_simd",
        &pv_simd_torch,
        "I-Whisper UINT8 x INT8 P@V SIMD accumulation kernel",
        pybind11::arg("p"),
        pybind11::arg("v"));

    m.def(
        "pv_norm_requant_scalar",
        &pv_norm_requant_scalar_torch,
        "I-Whisper fused P@V + normalize + AV requant scalar kernel",
        pybind11::arg("p"),
        pybind11::arg("v"),
        pybind11::arg("row_sum"),
        pybind11::arg("av_multiplier"),
        pybind11::arg("av_shift"),
        pybind11::arg("rsqrt_offset"));

    m.def(
        "pv_norm_requant_simd",
        &pv_norm_requant_simd_torch,
        "I-Whisper fused P@V + normalize + AV requant SIMD kernel",
        pybind11::arg("p"),
        pybind11::arg("v"),
        pybind11::arg("row_sum"),
        pybind11::arg("av_multiplier"),
        pybind11::arg("av_shift"),
        pybind11::arg("rsqrt_offset"));

    m.def(
        "conv1d_accum_scalar",
        &conv1d_accum_scalar_torch,
        "I-Whisper raw INT8 Conv1d scalar accumulation",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"));

    m.def(
        "conv1d_accum_simd",
        &conv1d_accum_simd_torch,
        "I-Whisper raw INT8 Conv1d SIMD accumulation",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"));

    m.def(
        "conv1d_scalar",
        &conv1d_scalar_torch,
        "I-Whisper fused INT8 Conv1d scalar kernel",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"),
        pybind11::arg("stride"),
        pybind11::arg("padding"));

    m.def(
        "conv1d_simd",
        &conv1d_simd_torch,
        "I-Whisper fused direct Conv1d SIMD kernel",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"),
        pybind11::arg("stride"),
        pybind11::arg("padding"));

    m.def(
        "add_requant15_scalar",
        &add_requant15_scalar_torch,
        "I-Whisper INT8 add + requant-to-INT15 scalar kernel",
        pybind11::arg("input"),
        pybind11::arg("addend"),
        pybind11::arg("multiplier"),
        pybind11::arg("shift"));
}