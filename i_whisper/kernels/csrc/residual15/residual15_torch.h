#pragma once

#include <torch/extension.h>

torch::Tensor residual15_scalar_torch(
    const torch::Tensor &branch,
    const torch::Tensor &residual,
    const torch::Tensor &multiplier);

torch::Tensor residual15_simd_torch(
    const torch::Tensor &branch,
    const torch::Tensor &residual,
    const torch::Tensor &multiplier);