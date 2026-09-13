// linear8_torch.cpp

#pragma once
#include <torch/extension.h>

#include "linear8_torch.h"
#include "linear8.h"

#include <vector>
#include <string>

static inline void CHECK_CPU_INT8(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name + " must be a CPU tensor");

    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt8,
        name + " must be int8");

    TORCH_CHECK(
        tensor.is_contiguous(),
        name + " must be contiguous");
}

static inline void CHECK_CPU_UINT8(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name + " must be a CPU tensor");

    TORCH_CHECK(
        tensor.scalar_type() == torch::kUInt8,
        name + " must be uint8");

    TORCH_CHECK(
        tensor.is_contiguous(),
        name + " must be contiguous");
}

static inline void CHECK_CPU_INT32(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name + " must be a CPU tensor");

    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt32,
        name + " must be int32");

    TORCH_CHECK(
        tensor.is_contiguous(),
        name + " must be contiguous");
}

static inline void CHECK_CPU_UINT32(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name + " must be a CPU tensor");

    TORCH_CHECK(
        tensor.scalar_type() == torch::kUInt32,
        name + " must be uint32");

    TORCH_CHECK(
        tensor.is_contiguous(),
        name + " must be contiguous");
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
    const std::optional<torch::Tensor> &weight_zero)
{
    CHECK_CPU_INT8(x, "x");
    CHECK_CPU_INT8(weight, "weight");

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

        CHECK_CPU_INT32(b, "bias");
        TORCH_CHECK(b.dim() == 1, "bias must have shape [N]");
        TORCH_CHECK(b.size(0) == N, "bias.size(0) must equal weight.size(0)");

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

        TORCH_CHECK(zero.dim() == 1, "weight_zero must have shape [N]");
        TORCH_CHECK(zero.size(0) == N, "weight_zero.size(0) must equal weight.size(0)");
        CHECK_CPU_INT8(zero, "weight_zero");

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
    const std::optional<torch::Tensor> &weight_zero)
{
    CHECK_CPU_INT8(x, "x");
    CHECK_CPU_INT8(weight, "weight");

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

        CHECK_CPU_INT32(b, "bias");
        TORCH_CHECK(b.dim() == 1, "bias must have shape [N]");
        TORCH_CHECK(b.size(0) == N, "bias.size(0) must equal weight.size(0)");

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

        CHECK_CPU_INT8(zero, "weight_zero");
        TORCH_CHECK(zero.dim() == 1, "weight_zero must have shape [N]");
        TORCH_CHECK(zero.size(0) == N, "weight_zero.size(0) must equal weight.size(0)");

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

torch::Tensor linear8_requant_scalar_debug_torch(
    const torch::Tensor &input,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift)
{
    // --------------------------------
    // CHECK: Input types requirements
    // --------------------------------
    CHECK_CPU_INT32(input, "input");
    CHECK_CPU_UINT32(multiplier, "multiplier");
    CHECK_CPU_INT8(shift, "shift");

    // --------------------------------
    // CHECK: Basic shape requirements
    // --------------------------------
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(multiplier.dim() == 1, "multiplier should be a 1D tensor");
    TORCH_CHECK(shift.dim() == 1, "shift should be a 1D tensor");

    // --------------------------------
    // CHECK: Shape compatibility
    // --------------------------------
    TORCH_CHECK(input.size(-1) == multiplier.numel(), "input.size(-1) must equal multiplier's number of elements");
    TORCH_CHECK(multiplier.numel() == shift.numel(), "multiplier's number of elements must equal shift's number of elements");
    const int64_t N = input.size(-1);
    const int64_t M = input.numel() / N;

    // --------------------------------
    // Allocate output (int32_t)
    // --------------------------------
    auto output = torch::empty_like(
        input,
        input.options().dtype(torch::kInt32)
    );

    // --------------------------------
    // Raw pointers
    // --------------------------------
    const int32_t *input_ptr = input.data_ptr<int32_t>();
    const uint32_t *multiplier_ptr = multiplier.data_ptr<uint32_t>();
    const int8_t *shift_ptr = shift.data_ptr<int8_t>();
    int32_t* output_ptr = output.data_ptr<int32_t>();

    iwhisper::linear8_requant_scalar(
        input_ptr,
        multiplier_ptr,
        shift_ptr,
        output_ptr,
        M,
        N
    );

    return output;
}