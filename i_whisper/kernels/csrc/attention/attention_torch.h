#pragma once

#include <torch/extension.h>

#include <tuple>



torch::Tensor qk_scalar_torch(
    const torch::Tensor &q,
    const torch::Tensor &k);

torch::Tensor qk_simd_torch(
    const torch::Tensor &q,
    const torch::Tensor &k);

std::tuple<torch::Tensor, torch::Tensor>
qk_requant_rowmax_scalar_torch(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const std::optional<torch::Tensor> &mask,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift);

std::tuple<torch::Tensor, torch::Tensor>
shift_exp_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &row_max_plus1,
    const std::optional<torch::Tensor> &mask,
    const torch::Tensor &frac_bits);



torch::Tensor pv_scalar_torch(
    const torch::Tensor &p,
    const torch::Tensor &v);




torch::Tensor pv_norm_requant_scalar_torch(
    const torch::Tensor &p,
    const torch::Tensor &v,
    const torch::Tensor &row_sum,
    const torch::Tensor &av_multiplier,
    const torch::Tensor &av_shift,
    int64_t rsqrt_offset);




