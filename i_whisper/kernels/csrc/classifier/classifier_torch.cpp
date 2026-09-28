#include "classifier_torch.h"
#include "classifier.h"

#include "../common/checktensor.h"
#include "../common/backend.h"

#include <torch/extension.h>

#include <cstdint>

namespace
{

    struct ClassifierShape
    {
        int64_t batch;
        int64_t tokens;
        int64_t hidden;
        int64_t vocab;
        int64_t rows;
    };

    ClassifierShape check_classifier_common(
        const torch::Tensor &input,
        const torch::Tensor &weight)
    {
        iwhisper::checktensor::CHECK_CPU_INT8(
            input,
            "input");

        iwhisper::checktensor::CHECK_CPU_INT8(
            weight,
            "weight");

        TORCH_CHECK(
            input.dim() == 3,
            "input must be [B,T,K]");

        TORCH_CHECK(
            weight.dim() == 2,
            "weight must be [N,K]");

        const int64_t B =
            input.size(0);

        const int64_t T =
            input.size(1);

        const int64_t K =
            input.size(2);

        const int64_t N =
            weight.size(0);

        TORCH_CHECK(
            weight.size(1) == K,
            "weight hidden size must match input");

        TORCH_CHECK(
            K > 0,
            "hidden size must be positive");

        TORCH_CHECK(
            N > 0,
            "vocabulary size must be positive");

        return {
            B,
            T,
            K,
            N,
            B * T,
        };
    }

} // namespace

torch::Tensor classifier_accum_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    iwhisper::backend::ArmBackend backend)
{
    const ClassifierShape s =
        check_classifier_common(
            input,
            weight);

    auto output =
        torch::empty(
            {
                s.batch,
                s.tokens,
                s.vocab,
            },
            input.options().dtype(
                torch::kInt32));

    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::classifier_accum_scalar(
            input.data_ptr<int8_t>(),
            weight.data_ptr<int8_t>(),
            output.data_ptr<int32_t>(),
            s.rows,
            s.hidden,
            s.vocab);
        break;
    case iwhisper::backend::ArmBackend::SIMD:
        iwhisper::classifier_accum_simd(
            input.data_ptr<int8_t>(),
            weight.data_ptr<int8_t>(),
            output.data_ptr<int32_t>(),
            s.rows,
            s.hidden,
            s.vocab);
        break;
    default:
        TORCH_CHECK(false, "Unsupported backend");
    }

    return output;
}

torch::Tensor classifier_accum_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight)
{
    return classifier_accum_scalar_torch(
        input,
        weight,
        iwhisper::backend::ArmBackend::Scalar);
}

torch::Tensor classifier_accum_simd_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight)
{
    return classifier_accum_scalar_torch(
        input,
        weight,
        iwhisper::backend::ArmBackend::SIMD);
}

torch::Tensor classifier_torch_impl(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift,
    iwhisper::backend::ArmBackend backend)
{
    const ClassifierShape s =
        check_classifier_common(
            input,
            weight);

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
                s.tokens,
                s.vocab,
            },
            input.options().dtype(
                torch::kInt8));

    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::classifier_scalar(
            input.data_ptr<int8_t>(),
            weight.data_ptr<int8_t>(),
            multiplier.data_ptr<uint32_t>()[0],
            shift.data_ptr<int8_t>()[0],
            output.data_ptr<int8_t>(),
            s.rows,
            s.hidden,
            s.vocab);
        break;
    case iwhisper::backend::ArmBackend::SIMD:
        iwhisper::classifier_simd(
            input.data_ptr<int8_t>(),
            weight.data_ptr<int8_t>(),
            multiplier.data_ptr<uint32_t>()[0],
            shift.data_ptr<int8_t>()[0],
            output.data_ptr<int8_t>(),
            s.rows,
            s.hidden,
            s.vocab);
        break;
    default:
        TORCH_CHECK(false, "Unsupported backend");
    }

    return output;
}

torch::Tensor classifier_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift)
{
    return classifier_torch_impl(
        input,
        weight,
        multiplier,
        shift,
        iwhisper::backend::ArmBackend::Scalar);
}

torch::Tensor classifier_simd_torch(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift)
{
    return classifier_torch_impl(
        input,
        weight,
        multiplier,
        shift,
        iwhisper::backend::ArmBackend::SIMD);
}