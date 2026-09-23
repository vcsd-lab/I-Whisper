#pragma once

#include <torch/extension.h>



torch::Tensor qk_scalar_torch(
    const torch::Tensor &q,
    const torch::Tensor &k);

torch::Tensor qk_simd_torch(
    const torch::Tensor &q,
    const torch::Tensor &k);



torch::Tensor shift_exp_scalar_torch(
    const torch::Tensor &input,
    const std::optional<torch::Tensor> &mask,
    const torch::Tensor &frac_bits);