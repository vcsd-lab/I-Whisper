#pragma once

#include <cstdint>

namespace iwhisper
{

void residual15_scalar(
    const int32_t *branch,
    const int16_t *residual,
    uint16_t multiplier,
    int16_t *output,
    int64_t numel);

// void residual15_simd(
//     const int32_t *branch,
//     const int16_t *residual,
//     uint16_t multiplier,
//     int16_t *output,
//     int64_t numel);

} // namespace iwhisper