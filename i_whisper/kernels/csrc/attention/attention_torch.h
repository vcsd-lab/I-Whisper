#pragma once

#include <torch/extension.h>



torch::Tensor qk_scalar_torch(
    const torch::Tensor &q,
    const torch::Tensor &k);

