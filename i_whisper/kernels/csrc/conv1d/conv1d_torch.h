#pragma once

#include <torch/extension.h>

#include <optional>

torch::Tensor conv1d_accum_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    int64_t stride,
    int64_t padding);

torch::Tensor conv1d_accum_simd_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    int64_t stride,
    int64_t padding);

torch::Tensor conv1d_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift,
    int64_t stride,
    int64_t padding);

torch::Tensor conv1d_simd_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift,
    int64_t stride,
    int64_t padding);
