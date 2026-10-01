#include "linear32.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

#include <ATen/Parallel.h>
#include <cstdint>

namespace iwhisper
{
#ifdef IWHISPER_USE_DOTPROD

    static inline void dot_product4x4_simd(
        const int8_t *x0,
        const int8_t *x1,
        const int8_t *x2,
        const int8_t *x3,
        const int8_t *w0,
        const int8_t *w1,
        const int8_t *w2,
        const int8_t *w3,
        int64_t K,
        int32_t out0[4],
        int32_t out1[4],
        int32_t out2[4],
        int32_t out3[4])
    {
        int64_t k = 0;

        int32x4_t acc00 = vdupq_n_s32(0);
        int32x4_t acc01 = vdupq_n_s32(0);
        int32x4_t acc02 = vdupq_n_s32(0);
        int32x4_t acc03 = vdupq_n_s32(0);

        int32x4_t acc10 = vdupq_n_s32(0);
        int32x4_t acc11 = vdupq_n_s32(0);
        int32x4_t acc12 = vdupq_n_s32(0);
        int32x4_t acc13 = vdupq_n_s32(0);

        int32x4_t acc20 = vdupq_n_s32(0);
        int32x4_t acc21 = vdupq_n_s32(0);
        int32x4_t acc22 = vdupq_n_s32(0);
        int32x4_t acc23 = vdupq_n_s32(0);

        int32x4_t acc30 = vdupq_n_s32(0);
        int32x4_t acc31 = vdupq_n_s32(0);
        int32x4_t acc32 = vdupq_n_s32(0);
        int32x4_t acc33 = vdupq_n_s32(0);

        for (; k + 15 < K; k += 16)
        {
            const int8x16_t vx0 = vld1q_s8(x0 + k);
            const int8x16_t vx1 = vld1q_s8(x1 + k);
            const int8x16_t vx2 = vld1q_s8(x2 + k);
            const int8x16_t vx3 = vld1q_s8(x3 + k);

            const int8x16_t vw0 = vld1q_s8(w0 + k);
            const int8x16_t vw1 = vld1q_s8(w1 + k);
            const int8x16_t vw2 = vld1q_s8(w2 + k);
            const int8x16_t vw3 = vld1q_s8(w3 + k);

            acc00 = vdotq_s32(acc00, vx0, vw0);
            acc01 = vdotq_s32(acc01, vx0, vw1);
            acc02 = vdotq_s32(acc02, vx0, vw2);
            acc03 = vdotq_s32(acc03, vx0, vw3);

            acc10 = vdotq_s32(acc10, vx1, vw0);
            acc11 = vdotq_s32(acc11, vx1, vw1);
            acc12 = vdotq_s32(acc12, vx1, vw2);
            acc13 = vdotq_s32(acc13, vx1, vw3);

            acc20 = vdotq_s32(acc20, vx2, vw0);
            acc21 = vdotq_s32(acc21, vx2, vw1);
            acc22 = vdotq_s32(acc22, vx2, vw2);
            acc23 = vdotq_s32(acc23, vx2, vw3);

            acc30 = vdotq_s32(acc30, vx3, vw0);
            acc31 = vdotq_s32(acc31, vx3, vw1);
            acc32 = vdotq_s32(acc32, vx3, vw2);
            acc33 = vdotq_s32(acc33, vx3, vw3);
        }

        out0[0] = vaddvq_s32(acc00);
        out0[1] = vaddvq_s32(acc01);
        out0[2] = vaddvq_s32(acc02);
        out0[3] = vaddvq_s32(acc03);

        out1[0] = vaddvq_s32(acc10);
        out1[1] = vaddvq_s32(acc11);
        out1[2] = vaddvq_s32(acc12);
        out1[3] = vaddvq_s32(acc13);

        out2[0] = vaddvq_s32(acc20);
        out2[1] = vaddvq_s32(acc21);
        out2[2] = vaddvq_s32(acc22);
        out2[3] = vaddvq_s32(acc23);

        out3[0] = vaddvq_s32(acc30);
        out3[1] = vaddvq_s32(acc31);
        out3[2] = vaddvq_s32(acc32);
        out3[3] = vaddvq_s32(acc33);

        for (; k < K; ++k)
        {
            const int32_t xv0 =
                static_cast<int32_t>(x0[k]);

            const int32_t xv1 =
                static_cast<int32_t>(x1[k]);

            const int32_t xv2 =
                static_cast<int32_t>(x2[k]);

            const int32_t xv3 =
                static_cast<int32_t>(x3[k]);

            const int32_t wv0 =
                static_cast<int32_t>(w0[k]);

            const int32_t wv1 =
                static_cast<int32_t>(w1[k]);

            const int32_t wv2 =
                static_cast<int32_t>(w2[k]);

            const int32_t wv3 =
                static_cast<int32_t>(w3[k]);

            out0[0] += xv0 * wv0;
            out0[1] += xv0 * wv1;
            out0[2] += xv0 * wv2;
            out0[3] += xv0 * wv3;

            out1[0] += xv1 * wv0;
            out1[1] += xv1 * wv1;
            out1[2] += xv1 * wv2;
            out1[3] += xv1 * wv3;

            out2[0] += xv2 * wv0;
            out2[1] += xv2 * wv1;
            out2[2] += xv2 * wv2;
            out2[3] += xv2 * wv3;

            out3[0] += xv3 * wv0;
            out3[1] += xv3 * wv1;
            out3[2] += xv3 * wv2;
            out3[3] += xv3 * wv3;
        }
    }

#endif

#ifdef IWHISPER_USE_DOTPROD

    static inline void dot_product2x4_simd(
        const int8_t *x0,
        const int8_t *x1,
        const int8_t *w0,
        const int8_t *w1,
        const int8_t *w2,
        const int8_t *w3,
        int64_t K,
        int32_t out0[4],
        int32_t out1[4])
    {
        int64_t k = 0;

        int32x4_t acc00 = vdupq_n_s32(0);
        int32x4_t acc01 = vdupq_n_s32(0);
        int32x4_t acc02 = vdupq_n_s32(0);
        int32x4_t acc03 = vdupq_n_s32(0);

        int32x4_t acc10 = vdupq_n_s32(0);
        int32x4_t acc11 = vdupq_n_s32(0);
        int32x4_t acc12 = vdupq_n_s32(0);
        int32x4_t acc13 = vdupq_n_s32(0);

        for (; k + 15 < K; k += 16)
        {
            const int8x16_t vx0 = vld1q_s8(x0 + k);
            const int8x16_t vx1 = vld1q_s8(x1 + k);

            const int8x16_t vw0 = vld1q_s8(w0 + k);
            const int8x16_t vw1 = vld1q_s8(w1 + k);
            const int8x16_t vw2 = vld1q_s8(w2 + k);
            const int8x16_t vw3 = vld1q_s8(w3 + k);

            acc00 = vdotq_s32(acc00, vx0, vw0);
            acc01 = vdotq_s32(acc01, vx0, vw1);
            acc02 = vdotq_s32(acc02, vx0, vw2);
            acc03 = vdotq_s32(acc03, vx0, vw3);

            acc10 = vdotq_s32(acc10, vx1, vw0);
            acc11 = vdotq_s32(acc11, vx1, vw1);
            acc12 = vdotq_s32(acc12, vx1, vw2);
            acc13 = vdotq_s32(acc13, vx1, vw3);
        }

        out0[0] = vaddvq_s32(acc00);
        out0[1] = vaddvq_s32(acc01);
        out0[2] = vaddvq_s32(acc02);
        out0[3] = vaddvq_s32(acc03);

        out1[0] = vaddvq_s32(acc10);
        out1[1] = vaddvq_s32(acc11);
        out1[2] = vaddvq_s32(acc12);
        out1[3] = vaddvq_s32(acc13);

        for (; k < K; ++k)
        {
            const int32_t xv0 =
                static_cast<int32_t>(x0[k]);

            const int32_t xv1 =
                static_cast<int32_t>(x1[k]);

            out0[0] += xv0 * static_cast<int32_t>(w0[k]);
            out0[1] += xv0 * static_cast<int32_t>(w1[k]);
            out0[2] += xv0 * static_cast<int32_t>(w2[k]);
            out0[3] += xv0 * static_cast<int32_t>(w3[k]);

            out1[0] += xv1 * static_cast<int32_t>(w0[k]);
            out1[1] += xv1 * static_cast<int32_t>(w1[k]);
            out1[2] += xv1 * static_cast<int32_t>(w2[k]);
            out1[3] += xv1 * static_cast<int32_t>(w3[k]);
        }
    }

#endif

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
#ifdef IWHISPER_USE_DOTPROD

        // ========================================================
        // Large-M 4x4 tiled GEMM path
        // ========================================================
        if (M >= 32 &&
            (M % 4) == 0)
        {
            const int64_t m_tiles =
                M / 4;

            at::parallel_for(
                0,
                m_tiles,
                1,
                [&](int64_t begin, int64_t end)
                {
                    for (int64_t mt = begin;
                         mt < end;
                         ++mt)
                    {
                        const int64_t m0 =
                            mt * 4;

                        const int64_t m1 =
                            m0 + 1;

                        const int64_t m2 =
                            m0 + 2;

                        const int64_t m3 =
                            m0 + 3;

                        const int8_t *x0 =
                            x + m0 * K;

                        const int8_t *x1 =
                            x + m1 * K;

                        const int8_t *x2 =
                            x + m2 * K;

                        const int8_t *x3 =
                            x + m3 * K;

                        const int32_t x_sum0 =
                            asymmetric
                                ? row_sum_simd(x0, K)
                                : 0;

                        const int32_t x_sum1 =
                            asymmetric
                                ? row_sum_simd(x1, K)
                                : 0;

                        const int32_t x_sum2 =
                            asymmetric
                                ? row_sum_simd(x2, K)
                                : 0;

                        const int32_t x_sum3 =
                            asymmetric
                                ? row_sum_simd(x3, K)
                                : 0;

                        int64_t n = 0;

                        // ----------------------------------------
                        // 4 input rows x 4 output channels
                        // ----------------------------------------
                        for (; n + 3 < N; n += 4)
                        {
                            const int8_t *w0 =
                                weight + (n + 0) * K;

                            const int8_t *w1 =
                                weight + (n + 1) * K;

                            const int8_t *w2 =
                                weight + (n + 2) * K;

                            const int8_t *w3 =
                                weight + (n + 3) * K;

                            int32_t acc0[4];
                            int32_t acc1[4];
                            int32_t acc2[4];
                            int32_t acc3[4];

                            dot_product4x4_simd(
                                x0,
                                x1,
                                x2,
                                x3,
                                w0,
                                w1,
                                w2,
                                w3,
                                K,
                                acc0,
                                acc1,
                                acc2,
                                acc3);

                            for (int64_t j = 0;
                                 j < 4;
                                 ++j)
                            {
                                if (has_bias)
                                {
                                    acc0[j] +=
                                        bias[n + j];

                                    acc1[j] +=
                                        bias[n + j];

                                    acc2[j] +=
                                        bias[n + j];

                                    acc3[j] +=
                                        bias[n + j];
                                }

                                if (asymmetric)
                                {
                                    const int32_t wz =
                                        static_cast<int32_t>(
                                            weight_zero[n + j]);

                                    acc0[j] -=
                                        x_sum0 * wz;

                                    acc1[j] -=
                                        x_sum1 * wz;

                                    acc2[j] -=
                                        x_sum2 * wz;

                                    acc3[j] -=
                                        x_sum3 * wz;
                                }

                                output[m0 * N +
                                       n + j] =
                                    requant32(
                                        acc0[j],
                                        multiplier[n + j],
                                        shift[n + j]);

                                output[m1 * N +
                                       n + j] =
                                    requant32(
                                        acc1[j],
                                        multiplier[n + j],
                                        shift[n + j]);

                                output[m2 * N +
                                       n + j] =
                                    requant32(
                                        acc2[j],
                                        multiplier[n + j],
                                        shift[n + j]);

                                output[m3 * N +
                                       n + j] =
                                    requant32(
                                        acc3[j],
                                        multiplier[n + j],
                                        shift[n + j]);
                            }
                        }

                        // ----------------------------------------
                        // N tail
                        // ----------------------------------------
                        for (; n < N; ++n)
                        {
                            const int8_t *w_row =
                                weight + n * K;

                            int32_t acc0 =
                                dot_product_simd(
                                    x0,
                                    w_row,
                                    K);

                            int32_t acc1 =
                                dot_product_simd(
                                    x1,
                                    w_row,
                                    K);

                            int32_t acc2 =
                                dot_product_simd(
                                    x2,
                                    w_row,
                                    K);

                            int32_t acc3 =
                                dot_product_simd(
                                    x3,
                                    w_row,
                                    K);

                            if (has_bias)
                            {
                                acc0 += bias[n];
                                acc1 += bias[n];
                                acc2 += bias[n];
                                acc3 += bias[n];
                            }

                            if (asymmetric)
                            {
                                const int32_t wz =
                                    static_cast<int32_t>(
                                        weight_zero[n]);

                                acc0 -=
                                    x_sum0 * wz;

                                acc1 -=
                                    x_sum1 * wz;

                                acc2 -=
                                    x_sum2 * wz;

                                acc3 -=
                                    x_sum3 * wz;
                            }

                            output[m0 * N + n] =
                                requant32(
                                    acc0,
                                    multiplier[n],
                                    shift[n]);

                            output[m1 * N + n] =
                                requant32(
                                    acc1,
                                    multiplier[n],
                                    shift[n]);

                            output[m2 * N + n] =
                                requant32(
                                    acc2,
                                    multiplier[n],
                                    shift[n]);

                            output[m3 * N + n] =
                                requant32(
                                    acc3,
                                    multiplier[n],
                                    shift[n]);
                        }
                    }
                });

            return;
        }

        // ========================================================
        // Existing 2x4 path -- keep unchanged
        // ========================================================
        if (M >= 16)
        {
            const int64_t m_tiles =
                M / 2;

            at::parallel_for(
                0,
                m_tiles,
                1,
                [&](int64_t begin, int64_t end)
                {
                    for (int64_t mt = begin;
                         mt < end;
                         ++mt)
                    {
                        const int64_t m0 =
                            mt * 2;

                        const int64_t m1 =
                            m0 + 1;

                        const int8_t *x0 =
                            x + m0 * K;

                        const int8_t *x1 =
                            x + m1 * K;

                        const int32_t x_sum0 =
                            asymmetric
                                ? row_sum_simd(x0, K)
                                : 0;

                        const int32_t x_sum1 =
                            asymmetric
                                ? row_sum_simd(x1, K)
                                : 0;

                        int64_t n = 0;

                        for (; n + 3 < N; n += 4)
                        {
                            const int8_t *w0 =
                                weight + (n + 0) * K;

                            const int8_t *w1 =
                                weight + (n + 1) * K;

                            const int8_t *w2 =
                                weight + (n + 2) * K;

                            const int8_t *w3 =
                                weight + (n + 3) * K;

                            int32_t acc0[4];
                            int32_t acc1[4];

                            dot_product2x4_simd(
                                x0,
                                x1,
                                w0,
                                w1,
                                w2,
                                w3,
                                K,
                                acc0,
                                acc1);

                            for (int64_t j = 0;
                                 j < 4;
                                 ++j)
                            {
                                if (has_bias)
                                {
                                    acc0[j] +=
                                        bias[n + j];

                                    acc1[j] +=
                                        bias[n + j];
                                }

                                if (asymmetric)
                                {
                                    const int32_t wz =
                                        static_cast<int32_t>(
                                            weight_zero[n + j]);

                                    acc0[j] -=
                                        x_sum0 * wz;

                                    acc1[j] -=
                                        x_sum1 * wz;
                                }

                                output[m0 * N +
                                       n + j] =
                                    requant32(
                                        acc0[j],
                                        multiplier[n + j],
                                        shift[n + j]);

                                output[m1 * N +
                                       n + j] =
                                    requant32(
                                        acc1[j],
                                        multiplier[n + j],
                                        shift[n + j]);
                            }
                        }

                        // N tail
                        for (; n < N; ++n)
                        {
                            const int8_t *w_row =
                                weight + n * K;

                            int32_t acc0 =
                                dot_product_simd(
                                    x0,
                                    w_row,
                                    K);

                            int32_t acc1 =
                                dot_product_simd(
                                    x1,
                                    w_row,
                                    K);

                            if (has_bias)
                            {
                                acc0 += bias[n];
                                acc1 += bias[n];
                            }

                            if (asymmetric)
                            {
                                const int32_t wz =
                                    static_cast<int32_t>(
                                        weight_zero[n]);

                                acc0 -=
                                    x_sum0 * wz;

                                acc1 -=
                                    x_sum1 * wz;
                            }

                            output[m0 * N + n] =
                                requant32(
                                    acc0,
                                    multiplier[n],
                                    shift[n]);

                            output[m1 * N + n] =
                                requant32(
                                    acc1,
                                    multiplier[n],
                                    shift[n]);
                        }
                    }
                });

            // Odd-M tail
            if ((M & 1) != 0)
            {
                const int64_t m =
                    M - 1;

                const int8_t *x_row =
                    x + m * K;

                const int32_t x_sum =
                    asymmetric
                        ? row_sum_simd(x_row, K)
                        : 0;

                for (int64_t n = 0;
                     n < N;
                     ++n)
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
                        requant32(
                            acc,
                            multiplier[n],
                            shift[n]);
                }
            }

            return;
        }

#endif

        at::parallel_for(
            0,
            M,
            1,
            [&](int64_t begin, int64_t end)
            {
                for (int64_t m = begin;
                     m < end;
                     ++m)
                {
                    const int8_t *x_row =
                        x + m * K;

                    const int32_t x_sum =
                        asymmetric
                            ? row_sum_simd(
                                  x_row,
                                  K)
                            : 0;

                    for (int64_t n = 0;
                         n < N;
                         ++n)
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
                            requant32(
                                acc,
                                multiplier[n],
                                shift[n]);
                    }
                }
            });
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
