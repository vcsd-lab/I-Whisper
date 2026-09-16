#include "residual15.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

#include <algorithm>
#include <cstdint>

namespace iwhisper
{

void residual15_simd(
    const int32_t *branch,
    const int16_t *residual,
    uint16_t multiplier,
    int16_t *output,
    int64_t numel)
{
    constexpr int32_t kInt15Min = -16383;
    constexpr int32_t kInt15Max =  16383;
    constexpr int32_t kRounding = 1 << 14;

    const int32_t multiplier_s32 =
        static_cast<int32_t>(multiplier);

    const int32x4_t rounding_base =
        vdupq_n_s32(kRounding);

    const int32x4_t clamp_min =
        vdupq_n_s32(kInt15Min);

    const int32x4_t clamp_max =
        vdupq_n_s32(kInt15Max);

    int64_t i = 0;

    for (; i + 7 < numel; i += 8)
    {
        const int16x8_t vres =
            vld1q_s16(residual + i);

        const int32x4_t vres_lo =
            vmovl_s16(vget_low_s16(vres));

        const int32x4_t vres_hi =
            vmovl_s16(vget_high_s16(vres));

        const int32x4_t vbranch_lo =
            vld1q_s32(branch + i);

        const int32x4_t vbranch_hi =
            vld1q_s32(branch + i + 4);

        const int32x4_t vsum_lo =
            vmlaq_n_s32(
                vbranch_lo,
                vres_lo,
                multiplier_s32);

        const int32x4_t vsum_hi =
            vmlaq_n_s32(
                vbranch_hi,
                vres_hi,
                multiplier_s32);

        const uint32x4_t sign_lo_u32 =
            vshrq_n_u32(
                vreinterpretq_u32_s32(vsum_lo),
                31);

        const uint32x4_t sign_hi_u32 =
            vshrq_n_u32(
                vreinterpretq_u32_s32(vsum_hi),
                31);

        const int32x4_t bias_lo =
            vsubq_s32(
                rounding_base,
                vreinterpretq_s32_u32(sign_lo_u32));

        const int32x4_t bias_hi =
            vsubq_s32(
                rounding_base,
                vreinterpretq_s32_u32(sign_hi_u32));

        int32x4_t vout_lo =
            vshrq_n_s32(
                vaddq_s32(vsum_lo, bias_lo),
                15);

        int32x4_t vout_hi =
            vshrq_n_s32(
                vaddq_s32(vsum_hi, bias_hi),
                15);

        vout_lo =
            vmaxq_s32(
                clamp_min,
                vminq_s32(vout_lo, clamp_max));

        vout_hi =
            vmaxq_s32(
                clamp_min,
                vminq_s32(vout_hi, clamp_max));

        const int16x8_t vout =
            vcombine_s16(
                vmovn_s32(vout_lo),
                vmovn_s32(vout_hi));

        vst1q_s16(
            output + i,
            vout);
    }

    for (; i < numel; ++i)
    {
        int32_t buffer =
            branch[i] +
            static_cast<int32_t>(residual[i]) *
            multiplier_s32;

        buffer =
            fixedpoint::rounding_rshift(
                buffer,
                15);

        buffer =
            std::clamp(
                buffer,
                kInt15Min,
                kInt15Max);

        output[i] =
            static_cast<int16_t>(buffer);
    }
}

} // namespace iwhisper
