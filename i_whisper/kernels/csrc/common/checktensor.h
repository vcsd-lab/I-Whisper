#pragma once

#include <torch/extension.h>
#include <string>
#include <c10/core/ScalarType.h>


namespace iwhisper {
namespace checktensor {




static inline void CHECK_CPU_BOOL(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name,
        " must be a CPU tensor, got device ",
        tensor.device().str());

    TORCH_CHECK(
        tensor.scalar_type() == torch::kBool,
        name,
        " must be bool, got ",
        c10::toString(tensor.scalar_type()));

    TORCH_CHECK(
        tensor.is_contiguous(),
        name + " must be contiguous");
}



static inline void CHECK_CPU_INT8(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name,
        " must be a CPU tensor, got device ",
        tensor.device().str());

    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt8,
        name,
        " must be int8, got ",
        c10::toString(tensor.scalar_type()));

    TORCH_CHECK(
        tensor.is_contiguous(),
        name + " must be contiguous");
}

static inline void CHECK_CPU_UINT8(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name,
        " must be a CPU tensor, got device ",
        tensor.device().str());
    TORCH_CHECK(
        tensor.scalar_type() == torch::kUInt8,
        name,
        " must be uint8, got ",
        c10::toString(tensor.scalar_type()));

    TORCH_CHECK(
        tensor.is_contiguous(),
        name,
        " must be contiguous");
}


static inline void CHECK_CPU_INT16(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name,
        " must be a CPU tensor, got device ",
        tensor.device().str());

    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt16,
        name,
        " must be int16, got ",
        c10::toString(tensor.scalar_type()));

    TORCH_CHECK(
        tensor.is_contiguous(),
        name + " must be contiguous");
}


static inline void CHECK_CPU_UINT16(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name,
        " must be a CPU tensor, got device ",
        tensor.device().str());

    TORCH_CHECK(
        tensor.scalar_type() == torch::kUInt16,
        name,
        " must be uint16, got ",
        c10::toString(tensor.scalar_type()));

    TORCH_CHECK(
        tensor.is_contiguous(),
        name + " must be contiguous");
}


static inline void CHECK_CPU_INT32(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name,
        " must be a CPU tensor, got device ",
        tensor.device().str());

    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt32,
        name,
        " must be int32, got ",
        c10::toString(tensor.scalar_type()));

    TORCH_CHECK(
        tensor.is_contiguous(),
        name,
        " must be contiguous");
}

static inline void CHECK_CPU_UINT32(
    const torch::Tensor &tensor,
    const std::string &name)
{
    TORCH_CHECK(
        tensor.device().is_cpu(),
        name,
        " must be a CPU tensor, got device ",
        tensor.device().str());

    TORCH_CHECK(
        tensor.scalar_type() == torch::kUInt32,
        name,
        " must be uint32, got ",
        c10::toString(tensor.scalar_type()));

    TORCH_CHECK(
        tensor.is_contiguous(),
        name,
        " must be contiguous");
}



} // namespace checktensor
} // namespace iwhisper