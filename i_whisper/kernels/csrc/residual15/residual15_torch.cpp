// residual15_torch.cpp

#include "residual15_torch.h"
#include "residual15.h"
#include "../common/backend.h"
#include "../common/checktensor.h"

#include <torch/extension.h>


torch::Tensor residual15_scalar_torch_impl(
    const torch::Tensor &branch,
    const torch::Tensor &residual,
    const torch::Tensor &multiplier,
    iwhisper::backend::ArmBackend backend
)
{
    // --------------------------------
    // CHECK: Input types requirements
    // --------------------------------
    iwhisper::checktensor::CHECK_CPU_INT32(branch, "branch");
    iwhisper::checktensor::CHECK_CPU_INT16(residual, "residual");
    iwhisper::checktensor::CHECK_CPU_UINT16(multiplier, "multiplier");

    // --------------------------------
    // CHECK: Basic shape requirements
    // --------------------------------
    TORCH_CHECK(
        multiplier.numel() == 1,
        "multiplier must be a scalar tensor"
    );

    // --------------------------------
    // CHECK: Shape compatibility
    // --------------------------------
    const int64_t numel = branch.numel();
    TORCH_CHECK(
        residual.sizes() == branch.sizes(),
        "branch and residual must have the same shape"
    );

    const uint16_t multiplier_value = multiplier.data_ptr<uint16_t>()[0];
    // --------------------------------
    // Allocate output (int32_t)
    // --------------------------------
    auto output = torch::empty_like(residual);

    switch(backend)
    {
        case iwhisper::backend::ArmBackend::Scalar:
            iwhisper::residual15_scalar(
                branch.data_ptr<int32_t>(),
                residual.data_ptr<int16_t>(),
                multiplier_value,
                output.data_ptr<int16_t>(),
                numel
            );
            break;

        // case iwhisper::backend::ArmBackend::SIMD:
        //     iwhisper::residual15_simd(
        //         branch.data_ptr<int32_t>(),
        //         residual.data_ptr<int16_t>(),
        //         multiplier.item<uint16_t>(),
        //         output.data_ptr<int16_t>(),
        //         numel
        //     );
        //     break;
    }

    return output;
}


torch::Tensor residual15_scalar_torch(
    const torch::Tensor &branch,
    const torch::Tensor &residual,
    const torch::Tensor &multiplier
)
{
    return residual15_scalar_torch_impl(
        branch,
        residual,
        multiplier,
        iwhisper::backend::ArmBackend::Scalar
    );
}