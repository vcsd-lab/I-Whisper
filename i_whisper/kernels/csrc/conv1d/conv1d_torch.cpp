#include "conv1d_torch.h"
#include "conv1d.h"

#include "../common/checktensor.h"
#include "../common/backend.h"

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace
{

    struct Conv1dShape
    {
        int64_t batch;
        int64_t in_channels;
        int64_t input_length;
        int64_t out_channels;
        int64_t kernel_size;
        int64_t output_length;
    };

    Conv1dShape check_conv1d_common(
        const torch::Tensor &input,
        const torch::Tensor &weight,
        const std::optional<torch::Tensor> &bias,
        int64_t stride,
        int64_t padding)
    {
        iwhisper::checktensor::CHECK_CPU_INT8(
            input,
            "input");

        iwhisper::checktensor::CHECK_CPU_INT8(
            weight,
            "weight");

        TORCH_CHECK(
            input.dim() == 3,
            "input must be [B,Cin,Tin]");

        TORCH_CHECK(
            weight.dim() == 2,
            "weight must be flattened [Cout,Cin*K]");

        TORCH_CHECK(
            stride > 0,
            "stride must be positive");

        TORCH_CHECK(
            padding >= 0,
            "padding must be non-negative");

        const int64_t B =
            input.size(0);

        const int64_t Cin =
            input.size(1);

        const int64_t Tin =
            input.size(2);

        const int64_t Cout =
            weight.size(0);

        TORCH_CHECK(
            Cin > 0,
            "input channel count must be positive");

        TORCH_CHECK(
            Tin > 0,
            "input length must be positive");

        TORCH_CHECK(
            Cout > 0,
            "output channel count must be positive");

        TORCH_CHECK(
            weight.size(1) % Cin == 0,
            "weight.size(1) must be divisible by input channels");

        const int64_t K =
            weight.size(1) / Cin;

        TORCH_CHECK(
            K > 0,
            "kernel size must be positive");

        const int64_t numerator =
            Tin + 2 * padding - K;

        TORCH_CHECK(
            numerator >= 0,
            "invalid Conv1d geometry: output length would be non-positive");

        const int64_t Tout =
            numerator / stride + 1;

        TORCH_CHECK(
            Tout > 0,
            "output length must be positive");

        if (bias.has_value())
        {
            const torch::Tensor &b =
                bias.value();

            iwhisper::checktensor::CHECK_CPU_INT32(
                b,
                "bias");

            TORCH_CHECK(
                b.dim() == 1,
                "bias must be [Cout]");

            TORCH_CHECK(
                b.numel() == Cout,
                "bias.numel() must equal output channels");
        }

        return {
            B,
            Cin,
            Tin,
            Cout,
            K,
            Tout,
        };
    }

} // namespace

torch::Tensor conv1d_accum_torch_impl(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    int64_t stride,
    int64_t padding,
    iwhisper::backend::ArmBackend backend)
{
    const Conv1dShape s =
        check_conv1d_common(
            input,
            weight,
            bias,
            stride,
            padding);

    auto output =
        torch::empty(
            {
                s.batch,
                s.out_channels,
                s.output_length,
            },
            input.options().dtype(
                torch::kInt32));

    const bool has_bias =
        bias.has_value();

    const int32_t *bias_ptr =
        has_bias
            ? bias.value().data_ptr<int32_t>()
            : nullptr;

    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::conv1d_accum_scalar(
            input.data_ptr<int8_t>(),
            weight.data_ptr<int8_t>(),
            bias_ptr,
            output.data_ptr<int32_t>(),
            s.batch,
            s.in_channels,
            s.input_length,
            s.out_channels,
            s.kernel_size,
            stride,
            padding,
            has_bias);
        break;
    case iwhisper::backend::ArmBackend::SIMD:
        iwhisper::conv1d_accum_simd(
            input.data_ptr<int8_t>(),
            weight.data_ptr<int8_t>(),
            bias_ptr,
            output.data_ptr<int32_t>(),
            s.batch,
            s.in_channels,
            s.input_length,
            s.out_channels,
            s.kernel_size,
            stride,
            padding,
            has_bias);
        break;
    default:
        TORCH_CHECK(
            false,
            "unsupported backend");
    }

    return output;
}

torch::Tensor conv1d_accum_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    int64_t stride,
    int64_t padding)
{
    return conv1d_accum_torch_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        iwhisper::backend::ArmBackend::Scalar);
}

torch::Tensor conv1d_accum_simd_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    int64_t stride,
    int64_t padding)
{
    return conv1d_accum_torch_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        iwhisper::backend::ArmBackend::SIMD);
}

torch::Tensor conv1d_torch_impl(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift,
    int64_t stride,
    int64_t padding,
    iwhisper::backend::ArmBackend backend)
{
    const Conv1dShape s =
        check_conv1d_common(
            input,
            weight,
            bias,
            stride,
            padding);

    iwhisper::checktensor::CHECK_CPU_UINT32(
        multiplier,
        "multiplier");

    iwhisper::checktensor::CHECK_CPU_INT8(
        shift,
        "shift");

    TORCH_CHECK(
        multiplier.numel() == 1,
        "multiplier must contain exactly one element");

    TORCH_CHECK(
        shift.numel() == 1,
        "shift must contain exactly one element");

    auto output =
        torch::empty(
            {
                s.batch,
                s.out_channels,
                s.output_length,
            },
            input.options().dtype(
                torch::kInt8));

    const bool has_bias =
        bias.has_value();

    const int32_t *bias_ptr =
        has_bias
            ? bias.value().data_ptr<int32_t>()
            : nullptr;

    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::conv1d_scalar(
            input.data_ptr<int8_t>(),
            weight.data_ptr<int8_t>(),
            bias_ptr,
            multiplier.data_ptr<uint32_t>()[0],
            shift.data_ptr<int8_t>()[0],
            output.data_ptr<int8_t>(),
            s.batch,
            s.in_channels,
            s.input_length,
            s.out_channels,
            s.kernel_size,
            stride,
            padding,
            has_bias);
        break;

    case iwhisper::backend::ArmBackend::SIMD:
        iwhisper::conv1d_simd(
            input.data_ptr<int8_t>(),
            weight.data_ptr<int8_t>(),
            bias_ptr,
            multiplier.data_ptr<uint32_t>()[0],
            shift.data_ptr<int8_t>()[0],
            output.data_ptr<int8_t>(),
            s.batch,
            s.in_channels,
            s.input_length,
            s.out_channels,
            s.kernel_size,
            stride,
            padding,
            has_bias);
        break;

    default:
        TORCH_CHECK(
            false,
            "unsupported backend");
    }

    return output;
}

torch::Tensor conv1d_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift,
    int64_t stride,
    int64_t padding)
{
    return conv1d_torch_impl(
        input,
        weight,
        bias,
        multiplier,
        shift,
        stride,
        padding,
        iwhisper::backend::ArmBackend::Scalar);
}

torch::Tensor conv1d_simd_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const std::optional<torch::Tensor> &bias,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift,
    int64_t stride,
    int64_t padding)
{
    return conv1d_torch_impl(
        input,
        weight,
        bias,
        multiplier,
        shift,
        stride,
        padding,
        iwhisper::backend::ArmBackend::SIMD);
}