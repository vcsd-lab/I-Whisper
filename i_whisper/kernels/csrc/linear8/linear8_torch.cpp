// linear8_torch.cpp

#pragma once
#include <torch/extension.h>

#include "linear8_torch.h"
#include "linear8.h"

#include <vector>

static inline void CHECK_INPUT_WEIGHT(
    const torch::Tensor &x,
    const torch::Tensor &weight
)
{
    TORCH_CHECK(
        x.device().is_cpu(),
        "x must be a CPU tensor");

    TORCH_CHECK(
        weight.device().is_cpu(),
        "weight must be a CPU tensor");

    TORCH_CHECK(
        x.scalar_type() == torch::kInt8,
        "x must be int8");

    TORCH_CHECK(
        weight.scalar_type() == torch::kInt8,
        "weight must be int8");

    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous");

    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous");
}

static inline void CHECK_BIAS(
    const torch::Tensor &bias,
    int64_t N)
{
    TORCH_CHECK(
        bias.device().is_cpu(),
        "bias must be a CPU tensor");

    TORCH_CHECK(
        bias.scalar_type() == torch::kInt32,
        "bias must be int32");

    TORCH_CHECK(
        bias.dim() == 1,
        "bias must have shape [N]");

    TORCH_CHECK(
        bias.size(0) == N,
        "bias.size(0) must equal weight.size(0)");
}

static inline void CHECK_WEIGHT_ZERO(
    const torch::Tensor &weight_zero,
    int64_t N)
{
    TORCH_CHECK(
        weight_zero.device().is_cpu(),
        "weight_zero must be a CPU tensor");

    TORCH_CHECK(
        weight_zero.scalar_type() == torch::kInt8,
        "weight_zero must be int8");

    TORCH_CHECK(
        weight_zero.dim() == 1,
        "weight_zero must have shape [N]");

    TORCH_CHECK(
        weight_zero.size(0) == N,
        "weight_zero.size(0) must equal weight.size(0)");
}

torch::Tensor linear8_scalar_torch(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const std::optional<torch::Tensor> &weight_zero
)
{
    CHECK_INPUT_WEIGHT(x, weight);

    const int64_t K = x.size(-1);
    const int64_t N = weight.size(0);
    const int64_t M = x.numel() / K;

    /**
     * x:      [*, K]
     * weight: [N, K]
     * bias:   [N] or None
     * output: [*, N]
     */
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
    TORCH_CHECK(weight.size(1) == K, "weight.size(1) must equal x.size(-1)");

    // --------------------------------
    // Allocate output (int8_t)
    // --------------------------------

    auto output_shape = x.sizes().vec();
    output_shape.back() = N;

    auto output = torch::empty(
        output_shape,
        x.options().dtype(torch::kInt8));

    // --------------------------------
    // Raw pointers
    // --------------------------------

    const int8_t *x_ptr = x.data_ptr<int8_t>();

    const int8_t *weight_ptr = weight.data_ptr<int8_t>();

    int8_t *output_ptr = output.data_ptr<int8_t>();

    // --------------------------------
    // Optional bias
    // --------------------------------

    const bool has_bias = bias.has_value();

    const int32_t *bias_ptr = nullptr;

    if (has_bias)
    {
        const torch::Tensor &b = bias.value();

        CHECK_BIAS(b, N);

        bias_ptr = b.data_ptr<int32_t>();
    }

    // --------------------------------
    // Optional weight_zero
    // --------------------------------

    const bool has_zero = weight_zero.has_value();

    const int8_t *weight_zero_ptr = nullptr;

    if (has_zero)
    {
        const torch::Tensor &zero = weight_zero.value();

        CHECK_WEIGHT_ZERO(zero, N);

        weight_zero_ptr = zero.data_ptr<int8_t>();
    }

    iwhisper::linear8_scalar(
        x_ptr,
        weight_ptr,
        bias_ptr,
        weight_zero_ptr,
        nullptr,
        nullptr,
        output_ptr,
        M,
        K,
        N,
        has_bias,
        has_zero);

    return output;
}

/**
 * A debug version that only accumulates the results into int32_t output, without quantization.
 * This is useful for testing and debugging the accumulation logic.
 */
torch::Tensor linear8_accum_scalar_debug_torch(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const std::optional<torch::Tensor> &weight_zero
)
{
    CHECK_INPUT_WEIGHT(x, weight);

    const int64_t K = x.size(-1);
    const int64_t N = weight.size(0);
    const int64_t M = x.numel() / K;

    /**
     * x:      [*, K]
     * weight: [N, K]
     * bias:   [N] or None
     * output: [*, N]
     */
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
    TORCH_CHECK(weight.size(1) == K, "weight.size(1) must equal x.size(-1)");

    // --------------------------------
    // Allocate output (int32_t)
    // --------------------------------

    auto output_shape = x.sizes().vec();
    output_shape.back() = N;

    auto output = torch::empty(
        output_shape,
        x.options().dtype(torch::kInt32));

    // --------------------------------
    // Raw pointers
    // --------------------------------

    const int8_t *x_ptr = x.data_ptr<int8_t>();

    const int8_t *weight_ptr = weight.data_ptr<int8_t>();

    int32_t *output_ptr = output.data_ptr<int32_t>();

    // --------------------------------
    // Optional bias
    // --------------------------------

    const bool has_bias = bias.has_value();

    const int32_t *bias_ptr = nullptr;

    if (has_bias)
    {
        const torch::Tensor &b = bias.value();

        CHECK_BIAS(b, N);

        bias_ptr = b.data_ptr<int32_t>();
    }

    // --------------------------------
    // Optional weight_zero
    // --------------------------------

    const bool asymmetric = weight_zero.has_value();

    const int8_t *weight_zero_ptr = nullptr;

    if (asymmetric)
    {
        const torch::Tensor &zero = weight_zero.value();

        CHECK_WEIGHT_ZERO(zero, N);

        weight_zero_ptr = zero.data_ptr<int8_t>();
    }

    iwhisper::linear8_accum_scalar(
        x_ptr,
        weight_ptr,
        bias_ptr,
        weight_zero_ptr,
        output_ptr,
        M,
        K,
        N,
        has_bias,
        asymmetric);

    return output;
}