#include <torch/extension.h>
#include "attention_torch.h"
#include "attention.h"

#include "../common/checktensor.h"
#include "../common/backend.h"

#include <tuple>

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

std::tuple<torch::Tensor, torch::Tensor>
qk_requant_rowmax_scalar_torch(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const std::optional<torch::Tensor> &mask,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift)
{
    // --------------------------------
    // CHECK: Input types requirements
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_INT8(q, "q");
    iwhisper::checktensor::CHECK_CPU_INT8(k, "k");
    iwhisper::checktensor::CHECK_CPU_UINT8(multiplier, "multiplier");
    iwhisper::checktensor::CHECK_CPU_INT8(shift, "shift");

    // --------------------------------
    // CHECK: Input shapes requirements
    // --------------------------------
    TORCH_CHECK(q.dim() == 4, "q must be [B,H,Tq,D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B,H,Tk,D]");

    TORCH_CHECK(q.size(0) == k.size(0), "batch mismatch");
    TORCH_CHECK(q.size(1) == k.size(1), "head count mismatch");
    TORCH_CHECK(q.size(3) == k.size(3), "head dimension mismatch");

    TORCH_CHECK(multiplier.numel() == 1, "multiplier must be a scalar tensor");
    TORCH_CHECK(shift.numel() == 1, "shift must be a scalar tensor");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t Tq = q.size(2);
    const int64_t Tk = k.size(2);
    const int64_t D = q.size(3);

    const bool has_mask = mask.has_value();
    const bool *mask_ptr = nullptr;

    if (has_mask)
    {
        const torch::Tensor &m = mask.value();

        iwhisper::checktensor::CHECK_CPU_BOOL(m, "mask");
        TORCH_CHECK(m.dim() == 2, "mask must be [Tq,Tk]");
        TORCH_CHECK(m.size(0) == Tq, "mask.size(0) must equal Tq");
        TORCH_CHECK(m.size(1) == Tk, "mask.size(1) must equal Tk");

        mask_ptr = m.data_ptr<bool>();
    }

    // --------------------------------
    // Allocate outputs
    // --------------------------------
    auto qk_output = torch::empty(
        {B, H, Tq, Tk},
        q.options().dtype(torch::kInt16));

    auto row_max_output = torch::empty(
        {B, H, Tq, 1},
        q.options().dtype(torch::kInt16));

    const uint8_t multiplier_value =
        multiplier.data_ptr<uint8_t>()[0];

    const int32_t shift_value =
        static_cast<int32_t>(
            shift.data_ptr<int8_t>()[0]);

    iwhisper::qk_requant_rowmax_scalar(
        q.data_ptr<int8_t>(),
        k.data_ptr<int8_t>(),
        mask_ptr,
        qk_output.data_ptr<int16_t>(),
        row_max_output.data_ptr<int16_t>(),
        B,
        H,
        Tq,
        Tk,
        D,
        multiplier_value,
        shift_value,
        has_mask);

    return std::make_tuple(
        qk_output,
        row_max_output);
}

std::tuple<torch::Tensor, torch::Tensor>
shift_exp_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &row_max_plus1,
    const std::optional<torch::Tensor> &mask,
    const torch::Tensor &frac_bits)
{
    // --------------------------------
    // CHECK: Input types requirements
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_INT16(input, "input");
    iwhisper::checktensor::CHECK_CPU_INT16(row_max_plus1, "row_max_plus1");
    iwhisper::checktensor::CHECK_CPU_INT32(frac_bits, "frac_bits");

    // --------------------------------
    // CHECK: Input shapes requirements
    // --------------------------------
    TORCH_CHECK(input.dim() == 4, "input must be [B,H,Tq,Tk]");
    TORCH_CHECK(
        row_max_plus1.dim() == 4,
        "row_max_plus1 must be [B,H,Tq,1]");

    const int64_t batch = input.size(0);
    const int64_t heads = input.size(1);
    const int64_t q_length = input.size(2);
    const int64_t k_length = input.size(3);

    TORCH_CHECK(
        row_max_plus1.size(0) == batch,
        "row_max_plus1.size(0) must equal input.size(0)");
    TORCH_CHECK(
        row_max_plus1.size(1) == heads,
        "row_max_plus1.size(1) must equal input.size(1)");
    TORCH_CHECK(
        row_max_plus1.size(2) == q_length,
        "row_max_plus1.size(2) must equal input.size(2)");
    TORCH_CHECK(
        row_max_plus1.size(3) == 1,
        "row_max_plus1.size(3) must be 1");

    TORCH_CHECK(
        frac_bits.numel() == 1,
        "frac_bits must be a scalar");

    const int32_t frac_bits_item =
        frac_bits.data_ptr<int32_t>()[0];

    TORCH_CHECK(
        frac_bits_item > 0 && frac_bits_item < 14,
        "frac_bits must be in [1, 14)");

    const bool has_mask = mask.has_value();
    const bool *mask_ptr = nullptr;

    if (has_mask)
    {
        const torch::Tensor &m = mask.value();

        iwhisper::checktensor::CHECK_CPU_BOOL(m, "mask");
        TORCH_CHECK(m.dim() == 2, "mask must be [Tq,Tk]");

        TORCH_CHECK(
            m.size(0) == q_length,
            "mask.size(0) must equal input.size(2)");

        TORCH_CHECK(
            m.size(1) == k_length,
            "mask.size(1) must equal input.size(3)");

        mask_ptr = m.data_ptr<bool>();
    }

    auto output = torch::empty(
        {batch, heads, q_length, k_length},
        input.options().dtype(torch::kUInt8));
    
    auto row_sum_output = torch::empty(
        {batch, heads, q_length, 1},
        input.options().dtype(torch::kInt32));

    iwhisper::shift_exp_scalar(
        input.data_ptr<int16_t>(),
        row_max_plus1.data_ptr<int16_t>(),
        mask_ptr,
        output.data_ptr<uint8_t>(),
        row_sum_output.data_ptr<int32_t>(),
        batch,
        heads,
        q_length,
        k_length,
        frac_bits_item,
        has_mask);

    return std::make_tuple(output, row_sum_output);
}





torch::Tensor pv_scalar_torch(
    const torch::Tensor &p,
    const torch::Tensor &v)
{
    // --------------------------------
    // CHECK: Input types / placement
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_UINT8(
        p,
        "p");

    iwhisper::checktensor::CHECK_CPU_INT8(
        v,
        "v");

    // --------------------------------
    // CHECK: Shapes
    //
    // p: [B,H,Tq,Tk]
    // v: [B,H,D,Tk]
    // --------------------------------
    TORCH_CHECK(
        p.dim() == 4,
        "p must be [B,H,Tq,Tk]");

    TORCH_CHECK(
        v.dim() == 4,
        "v must be [B,H,D,Tk]");

    TORCH_CHECK(
        p.size(0) == v.size(0),
        "batch mismatch");

    TORCH_CHECK(
        p.size(1) == v.size(1),
        "head count mismatch");

    TORCH_CHECK(
        p.size(3) == v.size(3),
        "key length mismatch");

    const int64_t B =
        p.size(0);

    const int64_t H =
        p.size(1);

    const int64_t Tq =
        p.size(2);

    const int64_t Tk =
        p.size(3);

    const int64_t D =
        v.size(2);

    auto output =
        torch::empty(
            {
                B,
                H,
                Tq,
                D,
            },
            p.options().dtype(
                torch::kInt32));

    iwhisper::pv_scalar(
        p.data_ptr<uint8_t>(),
        v.data_ptr<int8_t>(),
        output.data_ptr<int32_t>(),
        B,
        H,
        Tq,
        Tk,
        D);

    return output;
}




torch::Tensor pv_norm_requant_scalar_torch(
    const torch::Tensor &p,
    const torch::Tensor &v,
    const torch::Tensor &row_sum,
    const torch::Tensor &av_multiplier,
    const torch::Tensor &av_shift,
    int64_t rsqrt_offset)
{
    // --------------------------------
    // CHECK: dtype / device / contiguity
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_UINT8(
        p,
        "p");

    iwhisper::checktensor::CHECK_CPU_INT8(
        v,
        "v");

    iwhisper::checktensor::CHECK_CPU_INT32(
        row_sum,
        "row_sum");

    iwhisper::checktensor::CHECK_CPU_UINT8(
        av_multiplier,
        "av_multiplier");

    iwhisper::checktensor::CHECK_CPU_INT8(
        av_shift,
        "av_shift");

    // --------------------------------
    // CHECK: shapes
    // --------------------------------
    TORCH_CHECK(
        p.dim() == 4,
        "p must be [B,H,Tq,Tk]");

    TORCH_CHECK(
        v.dim() == 4,
        "v must be [B,H,D,Tk]");

    TORCH_CHECK(
        row_sum.dim() == 4,
        "row_sum must be [B,H,Tq,1]");

    TORCH_CHECK(
        p.size(0) == v.size(0)
        && p.size(0) == row_sum.size(0),
        "batch mismatch");

    TORCH_CHECK(
        p.size(1) == v.size(1)
        && p.size(1) == row_sum.size(1),
        "head count mismatch");

    TORCH_CHECK(
        p.size(2) == row_sum.size(2),
        "query length mismatch");

    TORCH_CHECK(
        row_sum.size(3) == 1,
        "row_sum must have last dimension 1");

    TORCH_CHECK(
        p.size(3) == v.size(3),
        "key length mismatch");

    TORCH_CHECK(
        av_multiplier.numel() == 1,
        "av_multiplier must be scalar");

    TORCH_CHECK(
        av_shift.numel() == 1,
        "av_shift must be scalar");

    TORCH_CHECK(
        rsqrt_offset >= 0
        && rsqrt_offset <= 29,
        "rsqrt_offset must be in [0,29]");

    const int64_t B =
        p.size(0);

    const int64_t H =
        p.size(1);

    const int64_t Tq =
        p.size(2);

    const int64_t Tk =
        p.size(3);

    const int64_t D =
        v.size(2);

    auto output =
        torch::empty(
            {
                B,
                Tq,
                H * D,
            },
            p.options().dtype(
                torch::kInt8));

    iwhisper::pv_norm_requant_scalar(
        p.data_ptr<uint8_t>(),
        v.data_ptr<int8_t>(),
        row_sum.data_ptr<int32_t>(),
        av_multiplier.data_ptr<uint8_t>()[0],
        av_shift.data_ptr<int8_t>()[0],
        static_cast<uint32_t>(
            rsqrt_offset),
        output.data_ptr<int8_t>(),
        B,
        H,
        Tq,
        Tk,
        D);

    return output;
}