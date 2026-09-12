// linear8_torch.cpp

#pragma once
#include <torch/extension.h>

#include "linear8_torch.h"
#include "linear8.h"

#include <vector>




torch::Tensor linear8_scalar_torch(
    const torch::Tensor& x,
    const torch::Tensor& weight
)
{
    TORCH_CHECK(
        x.device().is_cpu(),
        "x must be a CPU tensor"
    );

    TORCH_CHECK(
        weight.device().is_cpu(),
        "weight must be a CPU tensor"
    );

    TORCH_CHECK(
        x.scalar_type() == torch::kInt8,
        "x must be int8"
    );

    TORCH_CHECK(
        weight.scalar_type() == torch::kInt8,
        "weight must be int8"
    );

    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous"
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous"
    );

    TORCH_CHECK(
        weight.dim() == 2,
        "weight must have shape [N, K]"
    );

    TORCH_CHECK(
        x.dim() >= 2,
        "x must have at least 2 dimensions"
    );


    const int64_t K = x.size(-1);
    const int64_t N = weight.size(0);

    TORCH_CHECK(
        weight.size(1) == K,
        "weight.size(1) must equal x.size(-1)"
    );

    const int64_t M = x.numel() / K;


    std::vector<int64_t> output_shape(
        x.sizes().begin(),
        x.sizes().end()
    );

    output_shape.back() = N;


    auto output = torch::empty(
        output_shape,
        x.options().dtype(torch::kInt8)
    );


    const int8_t* x_ptr =
        x.data_ptr<int8_t>();

    const int8_t* weight_ptr =
        weight.data_ptr<int8_t>();

    int8_t* output_ptr =
        output.data_ptr<int8_t>();


    iwhisper::linear8_scalar(
        x_ptr,
        weight_ptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        output_ptr,
        M,
        K,
        N,
        false,
        false
    );


    return output;
}