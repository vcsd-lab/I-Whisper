#pragma once

#include "simd.h"

#include <cstdint>

namespace iwhisper
{
namespace dotprod
{

#ifdef IWHISPER_USE_DOTPROD

static inline void dot_s8s8_4x4(
    const int8_t *a0,
    const int8_t *a1,
    const int8_t *a2,
    const int8_t *a3,
    const int8_t *b0,
    const int8_t *b1,
    const int8_t *b2,
    const int8_t *b3,
    int64_t length,
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

    for (; k + 15 < length; k += 16)
    {
        const int8x16_t va0 =
            vld1q_s8(a0 + k);

        const int8x16_t va1 =
            vld1q_s8(a1 + k);

        const int8x16_t va2 =
            vld1q_s8(a2 + k);

        const int8x16_t va3 =
            vld1q_s8(a3 + k);

        const int8x16_t vb0 =
            vld1q_s8(b0 + k);

        const int8x16_t vb1 =
            vld1q_s8(b1 + k);

        const int8x16_t vb2 =
            vld1q_s8(b2 + k);

        const int8x16_t vb3 =
            vld1q_s8(b3 + k);

        acc00 = vdotq_s32(acc00, va0, vb0);
        acc01 = vdotq_s32(acc01, va0, vb1);
        acc02 = vdotq_s32(acc02, va0, vb2);
        acc03 = vdotq_s32(acc03, va0, vb3);

        acc10 = vdotq_s32(acc10, va1, vb0);
        acc11 = vdotq_s32(acc11, va1, vb1);
        acc12 = vdotq_s32(acc12, va1, vb2);
        acc13 = vdotq_s32(acc13, va1, vb3);

        acc20 = vdotq_s32(acc20, va2, vb0);
        acc21 = vdotq_s32(acc21, va2, vb1);
        acc22 = vdotq_s32(acc22, va2, vb2);
        acc23 = vdotq_s32(acc23, va2, vb3);

        acc30 = vdotq_s32(acc30, va3, vb0);
        acc31 = vdotq_s32(acc31, va3, vb1);
        acc32 = vdotq_s32(acc32, va3, vb2);
        acc33 = vdotq_s32(acc33, va3, vb3);
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

    // Generic scalar reduction tail.
    for (; k < length; ++k)
    {
        const int32_t av0 =
            static_cast<int32_t>(a0[k]);

        const int32_t av1 =
            static_cast<int32_t>(a1[k]);

        const int32_t av2 =
            static_cast<int32_t>(a2[k]);

        const int32_t av3 =
            static_cast<int32_t>(a3[k]);

        const int32_t bv0 =
            static_cast<int32_t>(b0[k]);

        const int32_t bv1 =
            static_cast<int32_t>(b1[k]);

        const int32_t bv2 =
            static_cast<int32_t>(b2[k]);

        const int32_t bv3 =
            static_cast<int32_t>(b3[k]);

        out0[0] += av0 * bv0;
        out0[1] += av0 * bv1;
        out0[2] += av0 * bv2;
        out0[3] += av0 * bv3;

        out1[0] += av1 * bv0;
        out1[1] += av1 * bv1;
        out1[2] += av1 * bv2;
        out1[3] += av1 * bv3;

        out2[0] += av2 * bv0;
        out2[1] += av2 * bv1;
        out2[2] += av2 * bv2;
        out2[3] += av2 * bv3;

        out3[0] += av3 * bv0;
        out3[1] += av3 * bv1;
        out3[2] += av3 * bv2;
        out3[3] += av3 * bv3;
    }
}

#endif

} // namespace dotprod
} // namespace iwhisper