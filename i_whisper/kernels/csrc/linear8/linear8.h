// linear8.h

#pragma once

#include <cstdint>

namespace iwhisper
{

    void linear8_scalar(
        const int8_t *x,
        const int8_t *weight,
        const int32_t *bias,
        const int8_t *weight_zero,
        const uint32_t *multiplier,
        const int8_t *shift,
        int8_t *output,

        int64_t M,
        int64_t K,
        int64_t N,

        bool has_bias,
        bool asymmetric);

    void linear8_simde(
        const int8_t *x,
        const int8_t *weight,
        const int32_t *bias,
        const int8_t *weight_zero,
        const uint32_t *multiplier,
        const int8_t *shift,
        int8_t *output,
        int64_t M,
        int64_t K,
        int64_t N,
        bool has_bias,
        bool asymmetric);

    void linear8_accum_scalar(
        const int8_t *x,
        const int8_t *weight,
        const int32_t *bias,
        const int8_t *weight_zero,
        int32_t *output,

        int64_t M,
        int64_t K,
        int64_t N,

        bool has_bias,
        bool asymmetric);

    void linear8_accum_simde(
        const int8_t *x,
        const int8_t *weight,
        const int32_t *bias,
        const int8_t *weight_zero,
        int32_t *output,

        int64_t M,
        int64_t K,
        int64_t N,

        bool has_bias,
        bool asymmetric);

    void linear8_requant_scalar(
        const int32_t *input,
        const uint32_t *multiplier,
        const int8_t *shift,
        int32_t *output,
        int64_t M,
        int64_t N);

} // namespace iwhisper