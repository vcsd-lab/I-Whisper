#pragma once

#include <cstdint>

namespace iwhisper
{

    void layernorm32_scalar(
        const int16_t *input,
        const int8_t *weight,
        const int32_t *bias,
        uint8_t ln_multiplier,
        int8_t ln_shift,
        uint32_t out_multiplier,
        int8_t out_shift,
        uint32_t rsqrt_offset,
        int8_t *output,
        int64_t rows,
        int64_t hidden_size,
        bool has_bias);

    void layernorm32_stats_scalar(
        const int16_t *input,
        int32_t *mean,
        int32_t *variance,
        int32_t *renorm_shift,
        int64_t rows,
        int64_t hidden_size);

    void layernorm32_rsqrt_scalar(
        const int32_t *variance,
        int32_t *invstd,
        int64_t numel,
        uint32_t offset);

    void layernorm32_simd(
        const int16_t *input,
        const int8_t *weight,
        const int32_t *bias,
        uint8_t ln_multiplier,
        int8_t ln_shift,
        uint32_t out_multiplier,
        int8_t out_shift,
        uint32_t rsqrt_offset,
        int8_t *output,
        int64_t rows,
        int64_t hidden_size,
        bool has_bias);

    void layernorm32_stats_simd(
        const int16_t *input,
        int32_t *mean,
        int32_t *variance,
        int32_t *renorm_shift,
        int64_t rows,
        int64_t hidden_size);

    void layernorm64_stats_scalar(
        const int16_t *input,
        int32_t *mean,
        int64_t *variance,
        int64_t rows,
        int64_t hidden_size);

    void layernorm64_scalar(
        const int16_t *input,
        const int8_t *weight,
        const int32_t *bias,
        uint8_t ln_multiplier,
        int8_t ln_shift,
        uint32_t out_multiplier,
        int8_t out_shift,
        uint32_t rsqrt_offset,
        int32_t additional_offset,
        int8_t *output,
        int64_t rows,
        int64_t hidden_size,
        bool has_bias);

    void layernorm64_stats_simd(
        const int16_t *input,
        int32_t *mean,
        int64_t *variance,
        int64_t rows,
        int64_t hidden_size);

    void layernorm64_simd(
        const int16_t *input,
        const int8_t *weight,
        const int32_t *bias,
        uint8_t ln_multiplier,
        int8_t ln_shift,
        uint32_t out_multiplier,
        int8_t out_shift,
        uint32_t rsqrt_offset,
        int32_t additional_offset,
        int8_t *output,
        int64_t rows,
        int64_t hidden_size,
        bool has_bias);

} // namespace iwhisper