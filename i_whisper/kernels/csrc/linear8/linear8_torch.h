// linear8_torch.h

#pragma once
#include <torch/extension.h>



// torch::Tensor linear8_scalar_torch(
//     const torch::Tensor& x,
//     const torch::Tensor& weight,
//     const torch::Tensor& bias,
//     const torch::Tensor& weight_zero,
//     const torch::Tensor& multiplier,
//     const torch::Tensor& shift,
//     bool has_bias,
//     bool asymmetric
// );



torch::Tensor linear8_scalar_torch(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& weight_zero
);

torch::Tensor linear8_accum_scalar_debug_torch(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& weight_zero
);

torch::Tensor linear8_requant_scalar_debug_torch(
    const torch::Tensor& input,
    const torch::Tensor& multiplier,
    const torch::Tensor& shift
);