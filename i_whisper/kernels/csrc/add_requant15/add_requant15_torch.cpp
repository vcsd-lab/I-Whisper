#include "add_requant15_torch.h"
#include "add_requant15.h"
#include "../common/checktensor.h"

#include <torch/extension.h>

torch::Tensor add_requant15_scalar_torch(
    const torch::Tensor &input,
    const torch::Tensor &addend,
    const torch::Tensor &multiplier,
    const torch::Tensor &shift)
{
    iwhisper::checktensor::CHECK_CPU_INT8(input, "input");
    iwhisper::checktensor::CHECK_CPU_INT8(addend, "addend");
    iwhisper::checktensor::CHECK_CPU_UINT16(multiplier, "multiplier");
    iwhisper::checktensor::CHECK_CPU_INT8(shift, "shift");

    TORCH_CHECK(input.dim() == 3, "input must be [B,T,C]");
    TORCH_CHECK(addend.dim() == 2, "addend must be [T,C]");

    const int64_t B = input.size(0);
    const int64_t T = input.size(1);
    const int64_t C = input.size(2);

    TORCH_CHECK(
        addend.size(0) == T && addend.size(1) == C,
        "addend must match input [T,C]");

    TORCH_CHECK(multiplier.dim() == 1, "multiplier must be 1D");
    TORCH_CHECK(shift.dim() == 1, "shift must be 1D");

    const int64_t multiplier_count = multiplier.numel();
    const int64_t shift_count = shift.numel();

    TORCH_CHECK(
        multiplier_count == 1 || multiplier_count == C,
        "multiplier must contain 1 or C elements");
    TORCH_CHECK(
        shift_count == 1 || shift_count == C,
        "shift must contain 1 or C elements");

    auto output = torch::empty(
        {B, T, C},
        input.options().dtype(torch::kInt16));

    iwhisper::add_requant15_scalar(
        input.data_ptr<int8_t>(),
        addend.data_ptr<int8_t>(),
        multiplier.data_ptr<uint16_t>(),
        shift.data_ptr<int8_t>(),
        output.data_ptr<int16_t>(),
        B, T, C,
        multiplier_count,
        shift_count);

    return output;
}
