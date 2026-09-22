#include <torch/extension.h>
#include "attention_torch.h"
#include "attention.h"

#include "../common/checktensor.h"

torch::Tensor qk_scalar_torch(
    const torch::Tensor &q,
    const torch::Tensor &k)
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

    iwhisper::qk_scalar(
        q.data_ptr<int8_t>(),
        k.data_ptr<int8_t>(),
        output.data_ptr<int32_t>(),
        B,
        H,
        Tq,
        Tk,
        D);

    return output;
}