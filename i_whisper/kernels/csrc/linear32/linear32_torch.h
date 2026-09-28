// linear32_torch.h

#pragma once
#include <torch/extension.h>

torch::Tensor linear32_scalar_torch(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& weight_zero,
    const torch::Tensor& multiplier,
    const torch::Tensor& shift
);

torch::Tensor linear32_accum_scalar_debug_torch(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& weight_zero
);

torch::Tensor linear32_requant_scalar_debug_torch(
    const torch::Tensor& input,
    const torch::Tensor& multiplier,
    const torch::Tensor& shift
);

torch::Tensor linear32_simd_torch(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& weight_zero,
    const torch::Tensor& multiplier,
    const torch::Tensor& shift
);

torch::Tensor linear32_accum_simd_debug_torch(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& weight_zero
);

torch::Tensor linear32_requant_simd_debug_torch(
    const torch::Tensor& input,
    const torch::Tensor& multiplier,
    const torch::Tensor& shift
);