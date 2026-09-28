#pragma once

#include <torch/extension.h>

namespace iwhisper {
namespace backend {


enum class ArmBackend
{
    Scalar,
    SIMD
};



} // namespace backend
} // namespace iwhisper