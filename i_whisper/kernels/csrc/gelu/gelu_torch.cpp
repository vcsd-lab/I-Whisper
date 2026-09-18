#include "gelu_torch.h"
#include "gelu.h"

#include "../common/checktensor.h"

#include <torch/extension.h>


torch::Tensor gelu_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &luti_multiplier,
    const torch::Tensor &luti_shift,
    const torch::Tensor &luto_multiplier,
    const torch::Tensor &luto_shift)
{
    // ========================================================
    // dtype / device checks
    // ========================================================

    iwhisper::checktensor::CHECK_CPU_INT8(
        input,
        "input");

    iwhisper::checktensor::CHECK_CPU_UINT8(
        lut,
        "lut");

    iwhisper::checktensor::CHECK_CPU_UINT32(
        luti_multiplier,
        "luti_multiplier");

    iwhisper::checktensor::CHECK_CPU_INT8(
        luti_shift,
        "luti_shift");

    iwhisper::checktensor::CHECK_CPU_UINT8(
        luto_multiplier,
        "luto_multiplier");

    iwhisper::checktensor::CHECK_CPU_INT8(
        luto_shift,
        "luto_shift");


    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension");

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous");


    // ========================================================
    // LUT contract
    // ========================================================

    TORCH_CHECK(
        lut.dim() == 1 &&
        lut.numel() == 16,
        "lut must contain exactly 16 entries");


    // ========================================================
    // Parameter shapes
    // ========================================================

    TORCH_CHECK(
        luti_multiplier.dim() == 1,
        "luti_multiplier must be 1D");

    TORCH_CHECK(
        luti_shift.dim() == 1,
        "luti_shift must be 1D");

    TORCH_CHECK(
        luto_multiplier.dim() == 1,
        "luto_multiplier must be 1D");

    TORCH_CHECK(
        luto_shift.dim() == 1,
        "luto_shift must be 1D");


    TORCH_CHECK(
        luti_multiplier.numel()
            == luti_shift.numel(),
        "luti multiplier/shift size mismatch");

    TORCH_CHECK(
        luto_multiplier.numel()
            == luto_shift.numel(),
        "luto multiplier/shift size mismatch");


    const int64_t channels =
        input.size(-1);

    const int64_t luti_count =
        luti_multiplier.numel();

    const int64_t luto_count =
        luto_multiplier.numel();


    TORCH_CHECK(
        luti_count == 1 ||
        luti_count == channels,
        "luti parameters must be scalar or match "
        "input.size(-1)");

    TORCH_CHECK(
        luto_count == 1 ||
        luto_count == channels,
        "luto parameters must be scalar or match "
        "input.size(-1)");


    auto output =
        torch::empty_like(
            input,
            torch::kInt8);


    iwhisper::gelu_scalar(
        input.data_ptr<int8_t>(),
        lut.data_ptr<uint8_t>(),
        luti_multiplier.data_ptr<uint32_t>(),
        luti_shift.data_ptr<int8_t>(),
        luto_multiplier.data_ptr<uint8_t>(),
        luto_shift.data_ptr<int8_t>(),
        output.data_ptr<int8_t>(),
        input.numel(),
        channels,
        luti_count,
        luto_count);


    return output;
}