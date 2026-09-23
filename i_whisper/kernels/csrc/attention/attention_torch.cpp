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

torch::Tensor shift_exp_scalar_torch(
    const torch::Tensor &input,
    const std::optional<torch::Tensor> &mask,
    const torch::Tensor &frac_bits)
{
    // --------------------------------
    // CHECK: Input types requirements
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_INT16(input, "input");
    iwhisper::checktensor::CHECK_CPU_INT32(frac_bits, "frac_bits");

    // --------------------------------
    // CHECK: Input shapes requirements
    // --------------------------------
    TORCH_CHECK(input.dim() == 4, "input must be [B,H,Tq,Tk]");
    int64_t batch = input.size(0);
    int64_t heads = input.size(1);
    int64_t q_length = input.size(2);
    int64_t k_length = input.size(3);

    TORCH_CHECK(frac_bits.numel() == 1, "frac_bits must be a scalar");
    int32_t frac_bits_item = frac_bits.data_ptr<int32_t>()[0];
    TORCH_CHECK(frac_bits_item > 0 && frac_bits_item < 14, "frac_bits must be in [1, 14)");

    bool has_mask = mask.has_value();
    if (has_mask)
    {
        iwhisper::checktensor::CHECK_CPU_BOOL(mask.value(), "mask");
        TORCH_CHECK(mask.value().dim() == 2, "mask must be [Tq,Tk]");
        TORCH_CHECK(
            mask.value().size(0) == input.size(2),
            "mask.size(0) must equal input.size(2)");

        TORCH_CHECK(
            mask.value().size(1) == input.size(3),
            "mask.size(1) must equal input.size(3)");
    }

    // --------------------------------
    // Allocate output (int8_t)
    // --------------------------------
    auto output = torch::empty_like(
        input,
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCPU));

    // --------------------------------
    // Raw pointers
    // --------------------------------
    const int16_t *input_ptr = input.data_ptr<int16_t>();
    const bool *mask_ptr = has_mask ? mask.value().data_ptr<bool>() : nullptr;
    uint8_t *output_ptr = output.data_ptr<uint8_t>();

    iwhisper::shift_exp_scalar(
        input_ptr,
        mask_ptr,
        output_ptr,
        batch,
        heads,
        q_length,
        k_length,
        frac_bits_item,
        has_mask);

    return output;
}