#pragma once
#include <cstdint>
namespace iwhisper
{
void add_requant15_scalar(
    const int8_t *input,
    const int8_t *addend,
    const uint16_t *multiplier,
    const int8_t *shift,
    int16_t *output,
    int64_t batch,
    int64_t positions,
    int64_t channels,
    int64_t multiplier_count,
    int64_t shift_count);
} // namespace iwhisper
