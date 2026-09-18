#pragma once

#include <torch/extension.h>

torch::Tensor gelu_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &luti_multiplier,
    const torch::Tensor &luti_shift,
    const torch::Tensor &luto_multiplier,
    const torch::Tensor &luto_shift);