#include "classifier.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

#include <algorithm>
#include <cstdint>

namespace iwhisper
{
namespace
{

static inline int32_t s32_from_bits(uint32_t bits)
{
    const int64_t value =
        (bits & uint32_t{0x80000000u})
            ? static_cast<int64_t>(bits) - (int64_t{1} << 32)
            : static_cast<int64_t>(bits);
    return static_cast<int32_t>(value);
}


static inline int32x4_t mulhi_s32_u32x4(
    int32x4_t x,
    uint32_t multiplier)
{
    const uint32x4_t ux =
        vreinterpretq_u32_s32(x);

    const uint32x2_t m =
        vdup_n_u32(multiplier);

    const uint64x2_t product_lo =
        vmull_u32(
            vget_low_u32(ux),
            m);

    const uint64x2_t product_hi =
        vmull_u32(
            vget_high_u32(ux),
            m);

    uint32x4_t high =
        vcombine_u32(
            vshrn_n_u64(product_lo, 32),
            vshrn_n_u64(product_hi, 32));

    const uint32x4_t sign =
        vshrq_n_u32(ux, 31);

    const uint32x4_t correction =
        vmulq_u32(
            sign,
            vdupq_n_u32(multiplier));

    high =
        vsubq_u32(
            high,
            correction);

    return vreinterpretq_s32_u32(high);
}


static inline int32x4_t rounding_rshift_s32x4(
    int32x4_t x,
    uint32_t shift)
{
    if (shift == 0)
        return x;

    if (shift >= 32)
        return vdupq_n_s32(0);

    const uint32x4_t ux =
        vreinterpretq_u32_s32(x);

    const uint32x4_t sign =
        vshrq_n_u32(ux, 31);

    const uint32x4_t rounding =
        vsubq_u32(
            vdupq_n_u32(
                uint32_t{1} << (shift - 1)),
            sign);

    const uint32x4_t rounded =
        vaddq_u32(
            ux,
            rounding);

    const int32x4_t shift_vec =
        vdupq_n_s32(
            -static_cast<int32_t>(shift));

    return vshlq_s32(
        vreinterpretq_s32_u32(rounded),
        shift_vec);
}


// Four vocabulary outputs at once.
// x is shared across the four independent weight rows.
static inline void dot_s8_s8_4_simd(
    const int8_t *x,
    const int8_t *w0,
    const int8_t *w1,
    const int8_t *w2,
    const int8_t *w3,
    int64_t K,
    int32_t &out0,
    int32_t &out1,
    int32_t &out2,
    int32_t &out3)
{
    int64_t k = 0;

    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

#ifdef IWHISPER_USE_DOTPROD
    for (; k + 16 <= K; k += 16)
    {
        const int8x16_t vx =
            vld1q_s8(x + k);

        acc0 = vdotq_s32(
            acc0,
            vx,
            vld1q_s8(w0 + k));

        acc1 = vdotq_s32(
            acc1,
            vx,
            vld1q_s8(w1 + k));

        acc2 = vdotq_s32(
            acc2,
            vx,
            vld1q_s8(w2 + k));

        acc3 = vdotq_s32(
            acc3,
            vx,
            vld1q_s8(w3 + k));
    }
#else
    for (; k + 16 <= K; k += 16)
    {
        const int8x16_t vx =
            vld1q_s8(x + k);

        const int8x8_t x_lo =
            vget_low_s8(vx);

        const int8x8_t x_hi =
            vget_high_s8(vx);

        const int8x16_t vw0 =
            vld1q_s8(w0 + k);
        const int8x16_t vw1 =
            vld1q_s8(w1 + k);
        const int8x16_t vw2 =
            vld1q_s8(w2 + k);
        const int8x16_t vw3 =
            vld1q_s8(w3 + k);

#define IWHISPER_ACCUM_W(VAR, VW)                                           \
        do                                                                   \
        {                                                                    \
            const int16x8_t p_lo =                                           \
                vmull_s8(x_lo, vget_low_s8(VW));                             \
            const int16x8_t p_hi =                                           \
                vmull_s8(x_hi, vget_high_s8(VW));                            \
            VAR = vaddq_s32(VAR, vpaddlq_s16(p_lo));                         \
            VAR = vaddq_s32(VAR, vpaddlq_s16(p_hi));                         \
        } while (0)

        IWHISPER_ACCUM_W(acc0, vw0);
        IWHISPER_ACCUM_W(acc1, vw1);
        IWHISPER_ACCUM_W(acc2, vw2);
        IWHISPER_ACCUM_W(acc3, vw3);

#undef IWHISPER_ACCUM_W
    }
#endif

    uint32_t bits0 =
        static_cast<uint32_t>(
            vaddvq_s32(acc0));
    uint32_t bits1 =
        static_cast<uint32_t>(
            vaddvq_s32(acc1));
    uint32_t bits2 =
        static_cast<uint32_t>(
            vaddvq_s32(acc2));
    uint32_t bits3 =
        static_cast<uint32_t>(
            vaddvq_s32(acc3));

    for (; k < K; ++k)
    {
        const int32_t xv =
            static_cast<int32_t>(x[k]);

        bits0 += static_cast<uint32_t>(
            xv * static_cast<int32_t>(w0[k]));
        bits1 += static_cast<uint32_t>(
            xv * static_cast<int32_t>(w1[k]));
        bits2 += static_cast<uint32_t>(
            xv * static_cast<int32_t>(w2[k]));
        bits3 += static_cast<uint32_t>(
            xv * static_cast<int32_t>(w3[k]));
    }

    out0 = s32_from_bits(bits0);
    out1 = s32_from_bits(bits1);
    out2 = s32_from_bits(bits2);
    out3 = s32_from_bits(bits3);
}


static inline int32_t dot_s8_s8_single_simd(
    const int8_t *x,
    const int8_t *w,
    int64_t K)
{
    int64_t k = 0;
    int32x4_t acc = vdupq_n_s32(0);

#ifdef IWHISPER_USE_DOTPROD
    for (; k + 16 <= K; k += 16)
    {
        acc = vdotq_s32(
            acc,
            vld1q_s8(x + k),
            vld1q_s8(w + k));
    }
#else
    for (; k + 16 <= K; k += 16)
    {
        const int8x16_t vx = vld1q_s8(x + k);
        const int8x16_t vw = vld1q_s8(w + k);

        acc = vaddq_s32(
            acc,
            vpaddlq_s16(
                vmull_s8(
                    vget_low_s8(vx),
                    vget_low_s8(vw))));

        acc = vaddq_s32(
            acc,
            vpaddlq_s16(
                vmull_s8(
                    vget_high_s8(vx),
                    vget_high_s8(vw))));
    }
#endif

    uint32_t bits =
        static_cast<uint32_t>(
            vaddvq_s32(acc));

    for (; k < K; ++k)
    {
        bits += static_cast<uint32_t>(
            static_cast<int32_t>(x[k]) *
            static_cast<int32_t>(w[k]));
    }

    return s32_from_bits(bits);
}


static inline int8_t requant_one(
    int32_t value,
    uint32_t multiplier,
    uint32_t shift)
{
    value = iwhisper::fixedpoint::mulh(
        value,
        multiplier);

    value = iwhisper::fixedpoint::rounding_rshift(
        value,
        shift);

    value = std::clamp(
        value,
        int32_t{-127},
        int32_t{127});

    return static_cast<int8_t>(value);
}

} // namespace


void classifier_accum_simd(
    const int8_t *input,
    const int8_t *weight,
    int32_t *output,
    int64_t M,
    int64_t K,
    int64_t N)
{
    for (int64_t m = 0; m < M; ++m)
    {
        const int8_t *x_row =
            input + m * K;

        int32_t *out_row =
            output + m * N;

        int64_t n = 0;

        for (; n + 4 <= N; n += 4)
        {
            dot_s8_s8_4_simd(
                x_row,
                weight + (n + 0) * K,
                weight + (n + 1) * K,
                weight + (n + 2) * K,
                weight + (n + 3) * K,
                K,
                out_row[n + 0],
                out_row[n + 1],
                out_row[n + 2],
                out_row[n + 3]);
        }

        for (; n < N; ++n)
        {
            out_row[n] =
                dot_s8_s8_single_simd(
                    x_row,
                    weight + n * K,
                    K);
        }
    }
}


void classifier_simd(
    const int8_t *input,
    const int8_t *weight,
    uint32_t multiplier,
    int8_t shift,
    int8_t *output,
    int64_t M,
    int64_t K,
    int64_t N)
{
    const uint32_t requant_shift =
        static_cast<uint32_t>(
            std::max<int32_t>(
                static_cast<int32_t>(shift),
                0));

    const int32x4_t clamp_min =
        vdupq_n_s32(-127);

    const int32x4_t clamp_max =
        vdupq_n_s32(127);

    for (int64_t m = 0; m < M; ++m)
    {
        const int8_t *x_row =
            input + m * K;

        int8_t *out_row =
            output + m * N;

        int64_t n = 0;

        for (; n + 4 <= N; n += 4)
        {
            alignas(16) int32_t dots[4];

            dot_s8_s8_4_simd(
                x_row,
                weight + (n + 0) * K,
                weight + (n + 1) * K,
                weight + (n + 2) * K,
                weight + (n + 3) * K,
                K,
                dots[0],
                dots[1],
                dots[2],
                dots[3]);

            int32x4_t value =
                vld1q_s32(dots);

            value = mulhi_s32_u32x4(
                value,
                multiplier);

            value = rounding_rshift_s32x4(
                value,
                requant_shift);

            value = vminq_s32(
                vmaxq_s32(value, clamp_min),
                clamp_max);

            const int16x4_t value16 =
                vmovn_s32(value);

            const int16x8_t value16x8 =
                vcombine_s16(
                    value16,
                    vdup_n_s16(0));

            const int8x8_t value8 =
                vmovn_s16(value16x8);

            // Store exactly four vocab outputs.
            alignas(8) int8_t packed[8];
            vst1_s8(packed, value8);

            out_row[n + 0] = packed[0];
            out_row[n + 1] = packed[1];
            out_row[n + 2] = packed[2];
            out_row[n + 3] = packed[3];
        }

        for (; n < N; ++n)
        {
            out_row[n] = requant_one(
                dot_s8_s8_single_simd(
                    x_row,
                    weight + n * K,
                    K),
                multiplier,
                requant_shift);
        }
    }
}

} // namespace iwhisper
