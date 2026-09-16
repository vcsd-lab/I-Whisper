// linear32_torch.cpp

#include <torch/extension.h>

#include "../common/checktensor.h"
#include "../common/backend.h"
#include "linear32_torch.h"
#include "linear32.h"

static torch::Tensor linear32_torch_impl(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const std::optional<torch::Tensor> &weight_zero,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift,
    iwhisper::backend::ArmBackend backend)
{
    // --------------------------------
    // CHECK: Input types requirements
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_INT8(x, "x");
    iwhisper::checktensor::CHECK_CPU_INT8(weight, "weight");
    iwhisper::checktensor::CHECK_CPU_UINT32(multiplier, "multiplier");
    iwhisper::checktensor::CHECK_CPU_INT8(shift, "shift");

    // --------------------------------
    // CHECK: Basic shape requirements
    // --------------------------------
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    TORCH_CHECK(weight.dim() == 2, "weight must have 2 dimensions, got " + std::to_string(weight.dim()));
    TORCH_CHECK(multiplier.dim() == 1, "multiplier should be a 1D tensor, got " + std::to_string(multiplier.dim()));
    TORCH_CHECK(shift.dim() == 1, "shift should be a 1D tensor, got " + std::to_string(shift.dim()));
    const int64_t K = x.size(-1);
    TORCH_CHECK(K > 0, "x.size(-1) must be greater than 0");
    const int64_t N = weight.size(0);
    const int64_t M = x.numel() / K;

    // --------------------------------
    // CHECK: Shape compatibility
    // --------------------------------
    TORCH_CHECK(weight.size(1) == K, "weight.size(1) must equal x.size(-1), expected " + std::to_string(K) + ", got " + std::to_string(weight.size(1)));
    TORCH_CHECK(N == multiplier.numel(), "weight.size(0) must equal multiplier's number of elements, expected " + std::to_string(N) + ", got " + std::to_string(multiplier.numel()));
    TORCH_CHECK(multiplier.numel() == shift.numel(), "multiplier's number of elements must equal shift's number of elements, expected " + std::to_string(multiplier.numel()) + ", got " + std::to_string(shift.numel()));

    // --------------------------------
    // Allocate output (int8_t)
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

    const uint32_t *multiplier_ptr = multiplier.data_ptr<uint32_t>();

    const int8_t *shift_ptr = shift.data_ptr<int8_t>();

    int32_t *output_ptr = output.data_ptr<int32_t>();

    // --------------------------------
    // Optional bias
    // --------------------------------

    const bool has_bias = bias.has_value();

    const int32_t *bias_ptr = nullptr;

    if (has_bias)
    {
        const torch::Tensor &b = bias.value();

        iwhisper::checktensor::CHECK_CPU_INT32(b, "bias"); // dtype & device check
        TORCH_CHECK(b.dim() == 1, "bias must be a 1D tensor, got " + std::to_string(b.dim()));
        TORCH_CHECK(b.size(0) == N, "bias.size(0) must equal weight.size(0), got " + std::to_string(b.size(0)) + " expected " + std::to_string(N));

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

        iwhisper::checktensor::CHECK_CPU_INT8(zero, "weight_zero"); // dtype & device check
        TORCH_CHECK(zero.dim() == 1, "weight_zero must be a 1D tensor, got " + std::to_string(zero.dim()));
        TORCH_CHECK(zero.size(0) == N, "weight_zero.size(0) must equal weight.size(0), got " + std::to_string(zero.size(0)) + " expected " + std::to_string(N));

        weight_zero_ptr = zero.data_ptr<int8_t>();
    }

    // --------------------------------
    // Call the C++ kernel
    // --------------------------------
    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::linear32_scalar(
            x_ptr,
            weight_ptr,
            bias_ptr,
            weight_zero_ptr,
            multiplier_ptr,
            shift_ptr,
            output_ptr,
            M,
            K,
            N,
            has_bias,
            has_zero);
        break;

        // case iwhisper::backend::ArmBackend::SIMD:
        //     iwhisper::linear32_simd(
        //         x_ptr,
        //         weight_ptr,
        //         bias_ptr,
        //         weight_zero_ptr,
        //         multiplier_ptr,
        //         shift_ptr,
        //         output_ptr,
        //         M,
        //         K,
        //         N,
        //         has_bias,
        //         has_zero);
        //     break;
    }

    return output;
}

torch::Tensor linear32_scalar_torch(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const std::optional<torch::Tensor> &weight_zero,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift)
{
    return linear32_torch_impl(
        x,
        weight,
        bias,
        weight_zero,
        multiplier,
        shift,
        iwhisper::backend::ArmBackend::Scalar);
}

static torch::Tensor linear32_accum_debug_torch_impl(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const std::optional<torch::Tensor> &weight_zero,
    iwhisper::backend::ArmBackend backend)
{
    iwhisper::checktensor::CHECK_CPU_INT8(x, "x");
    iwhisper::checktensor::CHECK_CPU_INT8(weight, "weight");

    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    const int64_t K = x.size(-1);
    TORCH_CHECK(K > 0, "x.size(-1) must be greater than 0, got " + std::to_string(K));

    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K], got " + std::to_string(weight.dim()));
    TORCH_CHECK(weight.size(1) == K, "weight.size(1) must equal x.size(-1), got " + std::to_string(weight.size(1)) + " expected " + std::to_string(K));
    const int64_t N = weight.size(0);
    const int64_t M = x.numel() / K;

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

        iwhisper::checktensor::CHECK_CPU_INT32(b, "bias"); // dtype & device check
        TORCH_CHECK(b.dim() == 1, "bias must be a 1D tensor, got " + std::to_string(b.dim()));
        TORCH_CHECK(b.size(0) == N, "bias.size(0) must equal weight.size(0), got " + std::to_string(b.size(0)) + " expected " + std::to_string(N));

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

        iwhisper::checktensor::CHECK_CPU_INT8(zero, "weight_zero"); // dtype & device check
        TORCH_CHECK(zero.dim() == 1, "weight_zero must be a 1D tensor, got " + std::to_string(zero.dim()));
        TORCH_CHECK(zero.size(0) == N, "weight_zero.size(0) must equal weight.size(0), got " + std::to_string(zero.size(0)) + " expected " + std::to_string(N));

        weight_zero_ptr = zero.data_ptr<int8_t>();
    }

    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::linear32_accum_scalar(
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
        break;

        // case iwhisper::backend::ArmBackend::SIMD:
        //     iwhisper::linear32_accum_simd(
        //         x_ptr,
        //         weight_ptr,
        //         bias_ptr,
        //         weight_zero_ptr,
        //         output_ptr,
        //         M,
        //         K,
        //         N,
        //         has_bias,
        //         asymmetric);
        //     break;
    }

    return output;
}

torch::Tensor linear32_accum_scalar_debug_torch(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const std::optional<torch::Tensor> &weight_zero)
{
    return linear32_accum_debug_torch_impl(
        x,
        weight,
        bias,
        weight_zero,
        iwhisper::backend::ArmBackend::Scalar);
}

torch::Tensor linear32_requant_scalar_debug_torch(
    const torch::Tensor &input,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift)
{
    // --------------------------------
    // CHECK: Input types requirements
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_INT32(input, "input");
    iwhisper::checktensor::CHECK_CPU_UINT32(multiplier, "multiplier");
    iwhisper::checktensor::CHECK_CPU_INT8(shift, "shift");

    // --------------------------------
    // CHECK: Basic shape requirements
    // --------------------------------
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension, got " + std::to_string(input.dim()));
    TORCH_CHECK(multiplier.dim() == 1, "multiplier should be a 1D tensor, got " + std::to_string(multiplier.dim()));
    TORCH_CHECK(shift.dim() == 1, "shift should be a 1D tensor, got " + std::to_string(shift.dim()));

    // --------------------------------
    // CHECK: Shape compatibility
    // --------------------------------
    TORCH_CHECK(input.size(-1) == multiplier.numel(), "input.size(-1) must equal multiplier's number of elements, got " + std::to_string(input.size(-1)) + " expected " + std::to_string(multiplier.numel()));
    TORCH_CHECK(multiplier.numel() == shift.numel(), "multiplier's number of elements must equal shift's number of elements, got " + std::to_string(multiplier.numel()) + " expected " + std::to_string(shift.numel()));
    const int64_t N = input.size(-1);
    const int64_t M = input.numel() / N;

    // --------------------------------
    // Allocate output (int32_t)
    // --------------------------------
    auto output = torch::empty_like(
        input,
        input.options().dtype(torch::kInt32));

    // --------------------------------
    // Raw pointers
    // --------------------------------
    const int32_t *input_ptr = input.data_ptr<int32_t>();
    const uint32_t *multiplier_ptr = multiplier.data_ptr<uint32_t>();
    const int8_t *shift_ptr = shift.data_ptr<int8_t>();
    int32_t *output_ptr = output.data_ptr<int32_t>();

    iwhisper::linear32_requant_scalar(
        input_ptr,
        multiplier_ptr,
        shift_ptr,
        output_ptr,
        M,
        N);

    return output;
}