#include <torch/extension.h>
#include "attention_torch.h"
#include "attention.h"

#include "../common/checktensor.h"
#include "../common/backend.h"

torch::Tensor qk_torch_impl(
    const torch::Tensor &q,
    const torch::Tensor &k,
    iwhisper::backend::ArmBackend backend)
{
    iwhisper::checktensor::CHECK_CPU_INT8(q, "q");
    iwhisper::checktensor::CHECK_CPU_INT8(k, "k");

    TORCH_CHECK(
        q.dim() == 4,
        "q must be [B,H,Tq,D]");

    TORCH_CHECK(
        k.dim() == 4,
        "k must be [B,H,Tk,D]");

    TORCH_CHECK(
        q.size(0) == k.size(0),
        "batch mismatch");

    TORCH_CHECK(
        q.size(1) == k.size(1),
        "head count mismatch");

    TORCH_CHECK(
        q.size(3) == k.size(3),
        "head dimension mismatch");

    const int64_t B =
        q.size(0);

    const int64_t H =
        q.size(1);

    const int64_t Tq =
        q.size(2);

    const int64_t Tk =
        k.size(2);

    const int64_t D =
        q.size(3);

    auto output =
        torch::empty(
            {
                B,
                H,
                Tq,
                Tk,
            },
            torch::TensorOptions()
                .dtype(torch::kInt32)
                .device(torch::kCPU));

    switch (backend)
    {
    case iwhisper::backend::ArmBackend::Scalar:
        iwhisper::qk_scalar(
            q.data_ptr<int8_t>(),
            k.data_ptr<int8_t>(),
            output.data_ptr<int32_t>(),
            B,
            H,
            Tq,
            Tk,
            D);
        break;
    case iwhisper::backend::ArmBackend::SIMD:
        iwhisper::qk_simd(
            q.data_ptr<int8_t>(),
            k.data_ptr<int8_t>(),
            output.data_ptr<int32_t>(),
            B,
            H,
            Tq,
            Tk,
            D);
        break;
    default:
        TORCH_CHECK(false, "Unsupported backend");
    }

    return output;
}

torch::Tensor qk_scalar_torch(
    const torch::Tensor &q,
    const torch::Tensor &k)
{
    return qk_torch_impl(
        q,
        k,
        iwhisper::backend::ArmBackend::Scalar);
}

torch::Tensor qk_simd_torch(
    const torch::Tensor &q,
    const torch::Tensor &k)
{
    return qk_torch_impl(
        q,
        k,
        iwhisper::backend::ArmBackend::SIMD);
}