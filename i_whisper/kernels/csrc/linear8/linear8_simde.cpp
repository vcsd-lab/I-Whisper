#include "linear8.h"
#include "../common/fixedpoint.h"

#include <simde/arm/neon.h>

#include <cstdint>
#include <algorithm>

namespace iwhisper
{

    // static inline int32_t dot_product_simde(
    //     const int8_t *x,
    //     const int8_t *weight,
    //     int64_t K)
    // {
    //     int64_t k = 0;

    //     simde_int32x4_t acc0 =
    //         simde_vdupq_n_s32(0);

    //     simde_int32x4_t acc1 =
    //         simde_vdupq_n_s32(0);

    //     for (; k + 15 < K; k += 16)
    //     {
    //         const simde_int8x16_t vx =
    //             simde_vld1q_s8(x + k);

    //         const simde_int8x16_t vw =
    //             simde_vld1q_s8(weight + k);

    //         // lower 8
    //         const simde_int16x8_t x_lo =
    //             simde_vmovl_s8(
    //                 simde_vget_low_s8(vx));

    //         const simde_int16x8_t w_lo =
    //             simde_vmovl_s8(
    //                 simde_vget_low_s8(vw));

    //         // upper 8
    //         const simde_int16x8_t x_hi =
    //             simde_vmovl_s8(
    //                 simde_vget_high_s8(vx));

    //         const simde_int16x8_t w_hi =
    //             simde_vmovl_s8(
    //                 simde_vget_high_s8(vw));

    //         const simde_int16x8_t prod_lo =
    //             simde_vmulq_s16(
    //                 x_lo,
    //                 w_lo);

    //         const simde_int16x8_t prod_hi =
    //             simde_vmulq_s16(
    //                 x_hi,
    //                 w_hi);

    //         acc0 = simde_vaddw_s16(
    //             acc0,
    //             simde_vget_low_s16(prod_lo));

    //         acc1 = simde_vaddw_s16(
    //             acc1,
    //             simde_vget_high_s16(prod_lo));

    //         acc0 = simde_vaddw_s16(
    //             acc0,
    //             simde_vget_low_s16(prod_hi));

    //         acc1 = simde_vaddw_s16(
    //             acc1,
    //             simde_vget_high_s16(prod_hi));

    //     }

    //     int32_t sum =
    //             simde_vaddvq_s32(acc0) +
    //             simde_vaddvq_s32(acc1);

    //     for (; k < K; ++k)
    //     {
    //         sum +=
    //             static_cast<int32_t>(x[k]) *
    //             static_cast<int32_t>(weight[k]);
    //     }

    //     return sum;
    // }
    static inline int32_t dot_product_simde(
        const int8_t *x,
        const int8_t *weight,
        int64_t K)
    {
        int64_t k = 0;

        simde_int32x4_t vacc =
            simde_vdupq_n_s32(0);

        for (; k + 15 < K; k += 16)
        {
            const simde_int8x16_t vx =
                simde_vld1q_s8(x + k);

            const simde_int8x16_t vw =
                simde_vld1q_s8(weight + k);

            vacc = simde_vdotq_s32(
                vacc,
                vx,
                vw);
        }

        int32_t sum =
            simde_vaddvq_s32(vacc);

        // tail
        for (; k < K; ++k)
        {
            sum +=
                static_cast<int32_t>(x[k]) *
                static_cast<int32_t>(weight[k]);
        }

        return sum;
    }

    // static inline int32_t row_sum_simde(
    //     const int8_t *x,
    //     int64_t K)
    // {
    //     int64_t k = 0;

    //     simde_int32x4_t acc0 =
    //         simde_vdupq_n_s32(0);

    //     simde_int32x4_t acc1 =
    //         simde_vdupq_n_s32(0);

    //     for (; k + 15 < K; k += 16)
    //     {
    //         const simde_int8x16_t vx =
    //             simde_vld1q_s8(x + k);

    //         const simde_int16x8_t lo =
    //             simde_vmovl_s8(
    //                 simde_vget_low_s8(vx));

    //         const simde_int16x8_t hi =
    //             simde_vmovl_s8(
    //                 simde_vget_high_s8(vx));

    //         acc0 = simde_vaddw_s16(
    //             acc0,
    //             simde_vget_low_s16(lo));

    //         acc1 = simde_vaddw_s16(
    //             acc1,
    //             simde_vget_high_s16(lo));

    //         acc0 = simde_vaddw_s16(
    //             acc0,
    //             simde_vget_low_s16(hi));

    //         acc1 = simde_vaddw_s16(
    //             acc1,
    //             simde_vget_high_s16(hi));
    //     }

    //     int32_t sum =
    //         simde_vaddvq_s32(acc0) +
    //         simde_vaddvq_s32(acc1);

    //     for (; k < K; ++k)
    //     {
    //         sum += static_cast<int32_t>(x[k]);
    //     }

    //     return sum;
    // }
    static inline int32_t row_sum_simde(
        const int8_t *x,
        int64_t K)
    {
        int64_t k = 0;
        int32_t sum = 0;

        for (; k + 15 < K; k += 16)
        {
            const simde_int8x16_t vx =
                simde_vld1q_s8(x + k);

            sum += static_cast<int32_t>(
                simde_vaddlvq_s8(vx));
        }

        // tail
        for (; k < K; ++k)
        {
            sum += static_cast<int32_t>(x[k]);
        }

        return sum;
    }

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
        bool asymmetric)
    {
        for (int64_t m = 0; m < M; ++m)
        {
            const int8_t *x_row =
                x + m * K;

            const int32_t x_sum =
                asymmetric
                    ? row_sum_simde(x_row, K)
                    : 0;

            for (int64_t n = 0; n < N; ++n)
            {
                const int8_t *w_row =
                    weight + n * K;

                int32_t acc =
                    dot_product_simde(
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
    }

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
        bool asymmetric)
    {
        for (int64_t m = 0; m < M; ++m)
        {
            const int8_t *x_row =
                x + m * K;

            const int32_t x_sum =
                asymmetric
                    ? row_sum_simde(x_row, K)
                    : 0;

            for (int64_t n = 0; n < N; ++n)
            {
                const int8_t *weight_row =
                    weight + n * K;

                int32_t acc =
                    dot_product_simde(
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