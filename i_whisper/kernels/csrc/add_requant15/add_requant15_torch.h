#pragma once
#include <torch/extension.h>

torch::Tensor add_requant15_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &addend,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift);
