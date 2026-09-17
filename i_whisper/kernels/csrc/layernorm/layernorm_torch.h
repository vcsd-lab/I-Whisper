#pragma once

#include <torch/extension.h>
#include <tuple>

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
layernorm32_stats_scalar_torch(
    const torch::Tensor &input);


torch::Tensor layernorm32_rsqrt_scalar_torch(
    const torch::Tensor &input);

torch::Tensor layernorm32_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias,
    const torch::Tensor &ln_multiplier,
    const torch::Tensor &ln_shift,
    const torch::Tensor &out_multiplier,
    const torch::Tensor &out_shift);