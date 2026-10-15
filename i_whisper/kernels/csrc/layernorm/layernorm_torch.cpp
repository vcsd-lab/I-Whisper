#include "layernorm_torch.h"
#include "layernorm.h"

#include "../common/checktensor.h"
#include "../common/backend.h"

#include <torch/extension.h>
#include <chrono>

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
layernorm32_stats_impl(
    const torch::Tensor &input,
    iwhisper::backend::ArmBackend backend)
{
    iwhisper::checktensor::CHECK_CPU_INT16(
        input,
        "input");

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension");

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous");

    const int64_t hidden_size =
        input.size(-1);

    TORCH_CHECK(
        hidden_size == 384 ||
            hidden_size == 512 ||
            hidden_size == 768 ||
            hidden_size == 1024 ||
            hidden_size == 1280,
        "unsupported LayerNorm hidden size: ",
        hidden_size);

    const int64_t rows =
        input.numel() / hidden_size;

    auto int32_options =
        input.options().dtype(torch::kInt32);

    auto mean =
        torch::empty({rows}, int32_options);

    auto variance =
        torch::empty({rows}, int32_options);

    auto renorm_shift =
        torch::empty({rows}, int32_options);

    // --------------------------------
    // Call the appropriate backend implementation
    // --------------------------------
    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::layernorm32_stats_scalar(
            input.data_ptr<int16_t>(),
            mean.data_ptr<int32_t>(),
            variance.data_ptr<int32_t>(),
            renorm_shift.data_ptr<int32_t>(),
            rows,
            hidden_size);
        break;
    case iwhisper::backend::ArmBackend::SIMD:
        iwhisper::layernorm32_stats_simd(
            input.data_ptr<int16_t>(),
            mean.data_ptr<int32_t>(),
            variance.data_ptr<int32_t>(),
            renorm_shift.data_ptr<int32_t>(),
            rows,
            hidden_size);
        break;
    default:
        TORCH_CHECK(false, "Unsupported backend");
    }

    return {
        mean,
        variance,
        renorm_shift};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
layernorm32_stats_scalar_torch(
    const torch::Tensor &input)
{
    return layernorm32_stats_impl(
        input,
        iwhisper::backend::ArmBackend::Scalar);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
layernorm32_stats_simd_torch(
    const torch::Tensor &input)
{
    return layernorm32_stats_impl(
        input,
        iwhisper::backend::ArmBackend::SIMD);
}

torch::Tensor layernorm32_rsqrt_scalar_torch(const torch::Tensor &input)
{
    iwhisper::checktensor::CHECK_CPU_INT32(
        input,
        "input");

    TORCH_CHECK(
        torch::all(input >= 0).item<bool>(),
        "input must be a 1D tensor");

    const int64_t numel = input.numel();
    auto output = torch::empty_like(input);

    constexpr uint32_t kLayerNormRsqrtOffset = 13;
    iwhisper::layernorm32_rsqrt_scalar(
        input.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        numel,
        kLayerNormRsqrtOffset);

    return output;
}

std::tuple<
    torch::Tensor,
    torch::Tensor>
layernorm64_stats_scalar_torch(
    const torch::Tensor &input)
{
    iwhisper::checktensor::CHECK_CPU_INT16(
        input,
        "input");

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension");

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous");

    const int64_t hidden_size =
        input.size(-1);

    TORCH_CHECK(
        hidden_size == 384 ||
            hidden_size == 512 ||
            hidden_size == 768 ||
            hidden_size == 1024 ||
            hidden_size == 1280,
        "unsupported LayerNorm hidden size: ",
        hidden_size);

    const int64_t rows =
        input.numel() / hidden_size;

    auto int32_options =
        input.options().dtype(
            torch::kInt32);

    auto int64_options =
        input.options().dtype(
            torch::kInt64);

    auto mean =
        torch::empty(
            {rows},
            int32_options);

    auto variance =
        torch::empty(
            {rows},
            int64_options);

    iwhisper::layernorm64_stats_scalar(
        input.data_ptr<int16_t>(),
        mean.data_ptr<int32_t>(),
        variance.data_ptr<int64_t>(),
        rows,
        hidden_size);

    return {
        mean,
        variance};
}

torch::Tensor layernorm32_impl(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias,
    const torch::Tensor &ln_multiplier,
    const torch::Tensor &ln_shift,
    const torch::Tensor &out_multiplier,
    const torch::Tensor &out_shift,
    iwhisper::backend::ArmBackend backend)
{
    // --------------------------------
    // CHECK: Input types requirements
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_INT16(input, "input");
    iwhisper::checktensor::CHECK_CPU_INT8(weight, "weight");
    iwhisper::checktensor::CHECK_CPU_UINT8(ln_multiplier, "ln_multiplier");
    iwhisper::checktensor::CHECK_CPU_INT8(ln_shift, "ln_shift");
    iwhisper::checktensor::CHECK_CPU_UINT32(out_multiplier, "out_multiplier");
    iwhisper::checktensor::CHECK_CPU_INT8(out_shift, "out_shift");

    // --------------------------------
    // CHECK: Basic shape requirements
    // --------------------------------
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor");

    const int64_t hidden_size = input.size(-1);
    TORCH_CHECK(
        hidden_size == 384 ||
            hidden_size == 512 ||
            hidden_size == 768 ||
            hidden_size == 1024 ||
            hidden_size == 1280,
        "unsupported LayerNorm hidden size: ",
        hidden_size);
    const int64_t rows = input.numel() / hidden_size;

    // --------------------------------
    // CHECK: Shape compatibility
    // --------------------------------
    TORCH_CHECK(
        weight.numel() == hidden_size,
        "weight.numel() must equal input.size(-1), got " +
            std::to_string(weight.numel()) + " expected " +
            std::to_string(hidden_size));

    // --------------------------------
    // Optional bias
    // --------------------------------
    const bool has_bias = bias.has_value();
    if (has_bias)
    {
        iwhisper::checktensor::CHECK_CPU_INT32(bias.value(), "bias");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be a 1D tensor");
        TORCH_CHECK(
            bias.value().numel() == hidden_size,
            "bias.numel() must equal input.size(-1), got " +
                std::to_string(bias.value().numel()) + " expected " +
                std::to_string(hidden_size));
    }

    // --------------------------------
    // Allocate output (int8_t)
    // --------------------------------

    auto output = torch::empty_like(input, torch::kInt8);
    constexpr uint32_t kLayerNormRsqrtOffset = 13;

    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::layernorm32_scalar(
            input.data_ptr<int16_t>(),
            weight.data_ptr<int8_t>(),
            has_bias ? bias.value().data_ptr<int32_t>() : nullptr,
            ln_multiplier.data_ptr<uint8_t>()[0],
            ln_shift.data_ptr<int8_t>()[0],
            out_multiplier.data_ptr<uint32_t>()[0],
            out_shift.data_ptr<int8_t>()[0],
            kLayerNormRsqrtOffset,
            output.data_ptr<int8_t>(),
            rows,
            hidden_size,
            has_bias);
        break;
    case iwhisper::backend::ArmBackend::SIMD:
        iwhisper::layernorm32_simd(
            input.data_ptr<int16_t>(),
            weight.data_ptr<int8_t>(),
            has_bias ? bias.value().data_ptr<int32_t>() : nullptr,
            ln_multiplier.data_ptr<uint8_t>()[0],
            ln_shift.data_ptr<int8_t>()[0],
            out_multiplier.data_ptr<uint32_t>()[0],
            out_shift.data_ptr<int8_t>()[0],
            kLayerNormRsqrtOffset,
            output.data_ptr<int8_t>(),
            rows,
            hidden_size,
            has_bias);
        break;
    default:
        TORCH_CHECK(false, "Unsupported backend");
    }

    return output;
}

torch::Tensor layernorm32_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias,
    const torch::Tensor &ln_multiplier,
    const torch::Tensor &ln_shift,
    const torch::Tensor &out_multiplier,
    const torch::Tensor &out_shift)
{
    return layernorm32_impl(
        input,
        weight,
        bias,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
        iwhisper::backend::ArmBackend::Scalar);
}

torch::Tensor layernorm32_simd_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias,
    const torch::Tensor &ln_multiplier,
    const torch::Tensor &ln_shift,
    const torch::Tensor &out_multiplier,
    const torch::Tensor &out_shift)
{
    return layernorm32_impl(
        input,
        weight,
        bias,
        ln_multiplier,
        ln_shift,
        out_multiplier,
        out_shift,
        iwhisper::backend::ArmBackend::SIMD);
}

torch::Tensor layernorm64_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias,
    const torch::Tensor &ln_multiplier,
    const torch::Tensor &ln_shift,
    const torch::Tensor &out_multiplier,
    const torch::Tensor &out_shift)
{
    // --------------------------------
    // Input dtype checks
    // --------------------------------

    iwhisper::checktensor::CHECK_CPU_INT16(
        input,
        "input");

    iwhisper::checktensor::CHECK_CPU_INT8(
        weight,
        "weight");

    iwhisper::checktensor::CHECK_CPU_UINT8(
        ln_multiplier,
        "ln_multiplier");

    iwhisper::checktensor::CHECK_CPU_INT8(
        ln_shift,
        "ln_shift");

    iwhisper::checktensor::CHECK_CPU_UINT32(
        out_multiplier,
        "out_multiplier");

    iwhisper::checktensor::CHECK_CPU_INT8(
        out_shift,
        "out_shift");

    // --------------------------------
    // Shapes
    // --------------------------------

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension");

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous");

    TORCH_CHECK(
        weight.dim() == 1,
        "weight must be a 1D tensor");

    const int64_t hidden_size =
        input.size(-1);

    TORCH_CHECK(
        hidden_size == 384 ||
            hidden_size == 512 ||
            hidden_size == 768 ||
            hidden_size == 1024 ||
            hidden_size == 1280,
        "unsupported LayerNorm hidden size: ",
        hidden_size);

    TORCH_CHECK(
        weight.numel() == hidden_size,
        "weight.numel() must equal input.size(-1)");

    const bool has_bias =
        bias.has_value();

    if (has_bias)
    {
        iwhisper::checktensor::CHECK_CPU_INT32(
            bias.value(),
            "bias");

        TORCH_CHECK(
            bias.value().dim() == 1,
            "bias must be a 1D tensor");

        TORCH_CHECK(
            bias.value().numel() == hidden_size,
            "bias.numel() must equal input.size(-1)");
    }

    const int64_t rows =
        input.numel() /
        hidden_size;

    auto output =
        torch::empty_like(
            input,
            torch::kInt8);

    // LayerNorm32 uses offset = 13.
    // LayerNorm64 uses +2 fractional bits to preserve FISR
    // resolution for the full, unshifted variance.
    constexpr uint32_t kLayerNorm64RsqrtOffset = 15;
    constexpr int32_t kLayerNorm64AdditionalOffset = 2;

    iwhisper::layernorm64_scalar(
        input.data_ptr<int16_t>(),
        weight.data_ptr<int8_t>(),
        has_bias
            ? bias.value().data_ptr<int32_t>()
            : nullptr,
        ln_multiplier.data_ptr<uint8_t>()[0],
        ln_shift.data_ptr<int8_t>()[0],
        out_multiplier.data_ptr<uint32_t>()[0],
        out_shift.data_ptr<int8_t>()[0],
        kLayerNorm64RsqrtOffset,
        kLayerNorm64AdditionalOffset,
        output.data_ptr<int8_t>(),
        rows,
        hidden_size,
        has_bias);

    return output;
}

std::tuple<
    torch::Tensor,
    torch::Tensor>
layernorm64_stats_simd_torch(
    const torch::Tensor &input)
{
    iwhisper::checktensor::CHECK_CPU_INT16(
        input,
        "input");

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension");

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous");

    const int64_t hidden_size =
        input.size(-1);

    TORCH_CHECK(
        hidden_size == 384 ||
            hidden_size == 512 ||
            hidden_size == 768 ||
            hidden_size == 1024 ||
            hidden_size == 1280,
        "unsupported LayerNorm hidden size: ",
        hidden_size);

    const int64_t rows =
        input.numel() /
        hidden_size;

    auto mean =
        torch::empty(
            {rows},
            input.options().dtype(
                torch::kInt32));

    auto variance =
        torch::empty(
            {rows},
            input.options().dtype(
                torch::kInt64));

    iwhisper::layernorm64_stats_simd(
        input.data_ptr<int16_t>(),
        mean.data_ptr<int32_t>(),
        variance.data_ptr<int64_t>(),
        rows,
        hidden_size);

    return {
        mean,
        variance};
}

torch::Tensor layernorm64_simd_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias,
    const torch::Tensor &ln_multiplier,
    const torch::Tensor &ln_shift,
    const torch::Tensor &out_multiplier,
    const torch::Tensor &out_shift)
{
    iwhisper::checktensor::CHECK_CPU_INT16(
        input,
        "input");
    iwhisper::checktensor::CHECK_CPU_INT8(
        weight,
        "weight");
    iwhisper::checktensor::CHECK_CPU_UINT8(
        ln_multiplier,
        "ln_multiplier");
    iwhisper::checktensor::CHECK_CPU_INT8(
        ln_shift,
        "ln_shift");
    iwhisper::checktensor::CHECK_CPU_UINT32(
        out_multiplier,
        "out_multiplier");
    iwhisper::checktensor::CHECK_CPU_INT8(
        out_shift,
        "out_shift");

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension");

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous");

    TORCH_CHECK(
        weight.dim() == 1,
        "weight must be a 1D tensor");

    const int64_t hidden_size =
        input.size(-1);

    TORCH_CHECK(
        hidden_size == 384 ||
            hidden_size == 512 ||
            hidden_size == 768 ||
            hidden_size == 1024 ||
            hidden_size == 1280,
        "unsupported LayerNorm hidden size: ",
        hidden_size);

    TORCH_CHECK(
        weight.numel() == hidden_size,
        "weight.numel() must equal input.size(-1)");

    const bool has_bias =
        bias.has_value();

    if (has_bias)
    {
        iwhisper::checktensor::CHECK_CPU_INT32(
            bias.value(),
            "bias");

        TORCH_CHECK(
            bias.value().dim() == 1,
            "bias must be a 1D tensor");

        TORCH_CHECK(
            bias.value().numel() == hidden_size,
            "bias.numel() must equal input.size(-1)");
    }

    const int64_t rows =
        input.numel() /
        hidden_size;

    auto output =
        torch::empty_like(
            input,
            torch::kInt8);

    constexpr uint32_t
        kLayerNorm64RsqrtOffset = 15;

    constexpr int32_t
        kLayerNorm64AdditionalOffset = 2;

    iwhisper::layernorm64_simd(
        input.data_ptr<int16_t>(),
        weight.data_ptr<int8_t>(),
        has_bias
            ? bias.value().data_ptr<int32_t>()
            : nullptr,
        ln_multiplier.data_ptr<uint8_t>()[0],
        ln_shift.data_ptr<int8_t>()[0],
        out_multiplier.data_ptr<uint32_t>()[0],
        out_shift.data_ptr<int8_t>()[0],
        kLayerNorm64RsqrtOffset,
        kLayerNorm64AdditionalOffset,
        output.data_ptr<int8_t>(),
        rows,
        hidden_size,
        has_bias);

    return output;
}


std::tuple<double, double>
layernorm_stats_raw_benchmark_torch(
    const torch::Tensor &input,
    int64_t warmup,
    int64_t repeat)
{
    iwhisper::checktensor::CHECK_CPU_INT16(
        input,
        "input");

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension");

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous");

    TORCH_CHECK(
        warmup >= 0,
        "warmup must be non-negative");

    TORCH_CHECK(
        repeat > 0,
        "repeat must be positive");

    const int64_t hidden_size =
        input.size(-1);

    TORCH_CHECK(
        hidden_size == 384 ||
            hidden_size == 512 ||
            hidden_size == 768 ||
            hidden_size == 1024 ||
            hidden_size == 1280,
        "unsupported LayerNorm hidden size: ",
        hidden_size);

    const int64_t rows =
        input.numel() / hidden_size;

    // ------------------------------------------------------------
    // Allocate all outputs before timing.
    // ------------------------------------------------------------

    auto int32_options =
        input.options().dtype(torch::kInt32);

    auto int64_options =
        input.options().dtype(torch::kInt64);

    auto mean32 =
        torch::empty(
            {rows},
            int32_options);

    auto variance32 =
        torch::empty(
            {rows},
            int32_options);

    auto renorm32 =
        torch::empty(
            {rows},
            int32_options);

    auto mean64 =
        torch::empty(
            {rows},
            int32_options);

    auto variance64 =
        torch::empty(
            {rows},
            int64_options);

    const int16_t *input_ptr =
        input.data_ptr<int16_t>();

    int32_t *mean32_ptr =
        mean32.data_ptr<int32_t>();

    int32_t *variance32_ptr =
        variance32.data_ptr<int32_t>();

    int32_t *renorm32_ptr =
        renorm32.data_ptr<int32_t>();

    int32_t *mean64_ptr =
        mean64.data_ptr<int32_t>();

    int64_t *variance64_ptr =
        variance64.data_ptr<int64_t>();

    // ------------------------------------------------------------
    // Warmup.
    // ------------------------------------------------------------

    for (int64_t i = 0;
         i < warmup;
         ++i)
    {
        iwhisper::layernorm32_stats_simd(
            input_ptr,
            mean32_ptr,
            variance32_ptr,
            renorm32_ptr,
            rows,
            hidden_size);

        iwhisper::layernorm64_stats_simd(
            input_ptr,
            mean64_ptr,
            variance64_ptr,
            rows,
            hidden_size);
    }

    // ------------------------------------------------------------
    // Raw kernel timing.
    //
    // Alternate execution order to reduce thermal / DVFS /
    // cache-order bias.
    // ------------------------------------------------------------

    using Clock =
        std::chrono::steady_clock;

    int64_t total32_ns = 0;
    int64_t total64_ns = 0;

    for (int64_t i = 0;
         i < repeat;
         ++i)
    {
        if ((i & 1) == 0)
        {
            auto t0 = Clock::now();

            iwhisper::layernorm32_stats_simd(
                input_ptr,
                mean32_ptr,
                variance32_ptr,
                renorm32_ptr,
                rows,
                hidden_size);

            auto t1 = Clock::now();

            total32_ns +=
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    t1 - t0)
                    .count();

            t0 = Clock::now();

            iwhisper::layernorm64_stats_simd(
                input_ptr,
                mean64_ptr,
                variance64_ptr,
                rows,
                hidden_size);

            t1 = Clock::now();

            total64_ns +=
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    t1 - t0)
                    .count();
        }
        else
        {
            auto t0 = Clock::now();

            iwhisper::layernorm64_stats_simd(
                input_ptr,
                mean64_ptr,
                variance64_ptr,
                rows,
                hidden_size);

            auto t1 = Clock::now();

            total64_ns +=
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    t1 - t0)
                    .count();

            t0 = Clock::now();

            iwhisper::layernorm32_stats_simd(
                input_ptr,
                mean32_ptr,
                variance32_ptr,
                renorm32_ptr,
                rows,
                hidden_size);

            t1 = Clock::now();

            total32_ns +=
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    t1 - t0)
                    .count();
        }
    }

    const double mean32_ns =
        static_cast<double>(
            total32_ns) /
        static_cast<double>(
            repeat);

    const double mean64_ns =
        static_cast<double>(
            total64_ns) /
        static_cast<double>(
            repeat);

    return {
        mean32_ns,
        mean64_ns};
}