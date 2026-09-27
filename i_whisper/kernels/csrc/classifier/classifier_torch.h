#pragma once

#include <torch/extension.h>

torch::Tensor classifier_accum_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight);

torch::Tensor classifier_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift);
