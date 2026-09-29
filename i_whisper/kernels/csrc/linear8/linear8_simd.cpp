#include "linear8.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

#include <ATen/Parallel.h>

#include <cstdint>
#include <algorithm>

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

    // static inline int32_t row_sum_simd(
    //     const int8_t *x,
    //     int64_t K)
    // {
    //     int64_t k = 0;

    //     int32x4_t acc0 =
    //         vdupq_n_s32(0);

    //     int32x4_t acc1 =
    //         vdupq_n_s32(0);

    //     for (; k + 15 < K; k += 16)
    //     {
    //         const int8x16_t vx =
    //             vld1q_s8(x + k);

    //         const int16x8_t lo =
    //             vmovl_s8(
    //                 vget_low_s8(vx));

    //         const int16x8_t hi =
    //             vmovl_s8(
    //                 vget_high_s8(vx));

    //         acc0 = vaddw_s16(
    //             acc0,
    //             vget_low_s16(lo));

    //         acc1 = vaddw_s16(
    //             acc1,
    //             vget_high_s16(lo));

    //         acc0 = vaddw_s16(
    //             acc0,
    //             vget_low_s16(hi));

    //         acc1 = vaddw_s16(
    //             acc1,
    //             vget_high_s16(hi));
    //     }

    //     int32_t sum =
    //         vaddvq_s32(acc0) +
    //         vaddvq_s32(acc1);

    //     for (; k < K; ++k)
    //     {
    //         sum += static_cast<int32_t>(x[k]);
    //     }

    //     return sum;
    // }
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

        // tail
        for (; k < K; ++k)
        {
            sum += static_cast<int32_t>(x[k]);
        }

        return sum;
    }

    void linear8_simd(
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
        bool asymmetric)
    {
        at::parallel_for(
            0,
            M,
            1,
            [&](int64_t begin, int64_t end)
            {
                for (int64_t m = begin; m < end; ++m)
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

                        int32_t result =
                            fixedpoint::requant(
                                acc,
                                multiplier[n],
                                static_cast<uint32_t>(
                                    shift[n]));

                        result =
                            std::clamp(
                                result,
                                int32_t{-127},
                                int32_t{127});

                        output[m * N + n] =
                            static_cast<int8_t>(result);
                    }
                }
            });
    }

    void linear8_accum_simd(
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
                const int8_t *weight_row =
                    weight + n * K;

                int32_t acc =
                    dot_product_simd(
                        x_row,
                        weight_row,
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

}