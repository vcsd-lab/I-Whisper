#pragma once

#include <cstdint>

namespace iwhisper
{

void gelu_scalar(
    const int8_t *input,
    const uint8_t *lut,
    const uint8_t *luti_multiplier,
    const int8_t *luti_shift,
    const uint8_t *luto_multiplier,
    const int8_t *luto_shift,
    int8_t *output,
    int64_t numel,
    int64_t channels,
    int64_t luti_param_count,
    int64_t luto_param_count);

void gelu_simd(
    const int8_t *input,
    const uint8_t *lut,
    const uint8_t *luti_multiplier,
    const int8_t *luti_shift,
    const uint8_t *luto_multiplier,
    const int8_t *luto_shift,
    int8_t *output,
    int64_t numel,
    int64_t channels,
    int64_t luti_param_count,
    int64_t luto_param_count);

} // namespace iwhisper