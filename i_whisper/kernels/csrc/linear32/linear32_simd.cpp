#include "linear32.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

#include <cstdint>

namespace iwhisper
{

static inline int32_t dot_product_simd(
    const int8_t *x,
    const int8_t *weight,
    int64_t K)
{
#ifdef IWHISPER_USE_DOTPROD

    // --------------------------------
    // Dot-product optimized path
    // --------------------------------

    int64_t k = 0;

    int32x4_t vacc =
        vdupq_n_s32(0);

    for (; k + 15 < K; k += 16)
    {
        const int8x16_t vx =
            vld1q_s8(x + k);

        const int8x16_t vw =
            vld1q_s8(weight + k);

        vacc =
            vdotq_s32(
                vacc,
                vx,
                vw);
    }

    int32_t sum =
        vaddvq_s32(vacc);

#else

    // --------------------------------
    // Baseline NEON path
    // widen -> mul -> int32 accumulate
    // --------------------------------

    int64_t k = 0;

    int32x4_t acc0 =
        vdupq_n_s32(0);

    int32x4_t acc1 =
        vdupq_n_s32(0);

    for (; k + 15 < K; k += 16)
    {
        const int8x16_t vx =
            vld1q_s8(x + k);

        const int8x16_t vw =
            vld1q_s8(weight + k);

        const int16x8_t x_lo =
            vmovl_s8(vget_low_s8(vx));

        const int16x8_t x_hi =
            vmovl_s8(vget_high_s8(vx));

        const int16x8_t w_lo =
            vmovl_s8(vget_low_s8(vw));

        const int16x8_t w_hi =
            vmovl_s8(vget_high_s8(vw));

        const int16x8_t prod_lo =
            vmulq_s16(x_lo, w_lo);

        const int16x8_t prod_hi =
            vmulq_s16(x_hi, w_hi);

        acc0 =
            vaddw_s16(
                acc0,
                vget_low_s16(prod_lo));

        acc1 =
            vaddw_s16(
                acc1,
                vget_high_s16(prod_lo));

        acc0 =
            vaddw_s16(
                acc0,
                vget_low_s16(prod_hi));

        acc1 =
            vaddw_s16(
                acc1,
                vget_high_s16(prod_hi));
    }

    int32_t sum =
        vaddvq_s32(acc0) +
        vaddvq_s32(acc1);

#endif

    // --------------------------------
    // Common scalar tail
    // --------------------------------

    for (; k < K; ++k)
    {
        sum +=
            static_cast<int32_t>(x[k]) *
            static_cast<int32_t>(weight[k]);
    }

    return sum;
}


static inline int32_t row_sum_simd(
    const int8_t *x,
    int64_t K)
{
    int64_t k = 0;
    int32_t sum = 0;

    for (; k + 15 < K; k += 16)
    {
        const int8x16_t vx =
            vld1q_s8(x + k);

        sum += static_cast<int32_t>(
            vaddlvq_s8(vx));
    }

    // scalar tail
    for (; k < K; ++k)
    {
        sum += static_cast<int32_t>(x[k]);
    }

    return sum;
}

static inline int32_t left_shift_wrap_s32(
    int32_t value,
    uint32_t shift)
{
    if (shift >= 32)
    {
        return 0;
    }

    const uint32_t bits =
        static_cast<uint32_t>(value);

    return static_cast<int32_t>(
        bits << shift);
}

static inline int32_t requant32(
    int32_t input,
    uint32_t multiplier,
    int8_t shift)
{
    const int32_t scaled =
        fixedpoint::mulh(
            input,
            multiplier);
    
    return left_shift_wrap_s32(
        scaled,
        static_cast<uint32_t>(shift));

    // const int64_t widened =
    //     static_cast<int64_t>(scaled);

    // const int64_t shifted =
    //     widened * (int64_t{1} << shift);

    // return static_cast<int32_t>(shifted);
}




void linear32_simd(
    const int8_t *x,
    const int8_t *weight,
    const int32_t *bias,
    const int8_t *weight_zero,
    const uint32_t *multiplier,
    const int8_t *shift,
    int32_t *output,
    int64_t M,
    int64_t K,
    int64_t N,
    bool has_bias,
    bool asymmetric)
{
    for (int64_t m = 0; m < M; ++m)
    {
        const int8_t *x_row =
            x + m * K;

        const int32_t x_sum =
            asymmetric
                ? row_sum_simd(x_row, K)
                : 0;

        for (int64_t n = 0; n < N; ++n)
        {
            const int8_t *w_row =
                weight + n * K;

            int32_t acc =
                dot_product_simd(
                    x_row,
                    w_row,
                    K);

            if (has_bias)
            {
                acc += bias[n];
            }

            if (asymmetric)
            {
                acc -=
                    x_sum *
                    static_cast<int32_t>(
                        weight_zero[n]);
            }

            // Full path is intentionally self-contained:
            // GEMM -> bias/zero correction -> requant -> output.
            output[m * N + n] =
                requant32(
                    acc,
                    multiplier[n],
                    shift[n]);
        }
    }
}


void linear32_accum_simd(
    const int8_t *x,
    const int8_t *weight,
    const int32_t *bias,
    const int8_t *weight_zero,
    int32_t *output,
    int64_t M,
    int64_t K,
    int64_t N,
    bool has_bias,
    bool asymmetric)
{
    for (int64_t m = 0; m < M; ++m)
    {
        const int8_t *x_row =
            x + m * K;

        const int32_t x_sum =
            asymmetric
                ? row_sum_simd(x_row, K)
                : 0;

        for (int64_t n = 0; n < N; ++n)
        {
            const int8_t *w_row =
                weight + n * K;

            int32_t acc =
                dot_product_simd(
                    x_row,
                    w_row,
                    K);

            if (has_bias)
            {
                acc += bias[n];
            }

            if (asymmetric)
            {
                acc -=
                    x_sum *
                    static_cast<int32_t>(
                        weight_zero[n]);
            }

            output[m * N + n] =
                acc;
        }
    }
}

} // namespace iwhisper
