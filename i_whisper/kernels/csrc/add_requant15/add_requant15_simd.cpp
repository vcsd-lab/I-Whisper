#include "add_requant15.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

#include <algorithm>
#include <cstdint>

namespace iwhisper
{
namespace
{

static inline int32x4_t rounding_rshift_variable_s32x4(
    int32x4_t x,
    int32x4_t shift)
{
    const int32x4_t zero = vdupq_n_s32(0);
    const int32x4_t max_shift = vdupq_n_s32(32);

    shift = vmaxq_s32(shift, zero);
    shift = vminq_s32(shift, max_shift);

    const uint32x4_t zero_shift_mask = vceqq_s32(shift, zero);
    const uint32x4_t large_shift_mask = vcgeq_s32(shift, max_shift);

    int32x4_t rounding_shift = vsubq_s32(shift, vdupq_n_s32(1));
    rounding_shift = vmaxq_s32(rounding_shift, zero);

    const uint32x4_t rounding_base =
        vshlq_u32(vdupq_n_u32(1), rounding_shift);

    const uint32x4_t ux = vreinterpretq_u32_s32(x);
    const uint32x4_t sign = vshrq_n_u32(ux, 31);
    const uint32x4_t rounding = vsubq_u32(rounding_base, sign);
    const uint32x4_t rounded = vaddq_u32(ux, rounding);

    const int32x4_t negative_shift = vnegq_s32(shift);
    const int32x4_t shifted =
        vshlq_s32(vreinterpretq_s32_u32(rounded), negative_shift);

    int32x4_t result = vbslq_s32(zero_shift_mask, x, shifted);
    result = vbslq_s32(large_shift_mask, zero, result);
    return result;
}

static inline void load_multiplier_u16x8_as_s32(
    const uint16_t *multiplier,
    int64_t multiplier_count,
    int64_t channel,
    int32x4_t &mul_lo,
    int32x4_t &mul_hi)
{
    if (multiplier_count == 1)
    {
        const int32_t value = static_cast<int32_t>(multiplier[0]);
        mul_lo = vdupq_n_s32(value);
        mul_hi = mul_lo;
    }
    else
    {
        const uint16x8_t m16 = vld1q_u16(multiplier + channel);
        const uint32x4_t m_lo_u32 = vmovl_u16(vget_low_u16(m16));
        const uint32x4_t m_hi_u32 = vmovl_u16(vget_high_u16(m16));

        mul_lo = vreinterpretq_s32_u32(m_lo_u32);
        mul_hi = vreinterpretq_s32_u32(m_hi_u32);
    }
}

static inline void load_shift_i8x8_as_s32(
    const int8_t *shift,
    int64_t shift_count,
    int64_t channel,
    int32x4_t &shift_lo,
    int32x4_t &shift_hi)
{
    if (shift_count == 1)
    {
        const int32_t value = static_cast<int32_t>(shift[0]);
        shift_lo = vdupq_n_s32(value);
        shift_hi = shift_lo;
    }
    else
    {
        const int8x8_t s8 = vld1_s8(shift + channel);
        const int16x8_t s16 = vmovl_s8(s8);

        shift_lo = vmovl_s16(vget_low_s16(s16));
        shift_hi = vmovl_s16(vget_high_s16(s16));
    }
}

static inline int16_t scalar_one(
    int8_t input,
    int8_t addend,
    uint16_t multiplier,
    int8_t shift)
{
    int32_t value =
        static_cast<int32_t>(input) +
        static_cast<int32_t>(addend);

    value *= static_cast<int32_t>(multiplier);

    const uint32_t requant_shift =
        static_cast<uint32_t>(
            std::max<int32_t>(static_cast<int32_t>(shift), 0));

    value = iwhisper::fixedpoint::rounding_rshift(
        value,
        requant_shift);

    value = std::clamp(value, int32_t{-16383}, int32_t{16383});
    return static_cast<int16_t>(value);
}

} // namespace

void add_requant15_simd(
    const int8_t *input,
    const int8_t *addend,
    const uint16_t *multiplier,
    const int8_t *shift,
    int16_t *output,
    int64_t batch,
    int64_t positions,
    int64_t channels,
    int64_t multiplier_count,
    int64_t shift_count)
{
    const int32x4_t clamp_min = vdupq_n_s32(-16383);
    const int32x4_t clamp_max = vdupq_n_s32(16383);

    for (int64_t b = 0; b < batch; ++b)
    {
        for (int64_t t = 0; t < positions; ++t)
        {
            const int64_t input_row_base = (b * positions + t) * channels;
            const int64_t addend_row_base = t * channels;

            int64_t c = 0;

            for (; c + 8 <= channels; c += 8)
            {
                const int8x8_t x8 = vld1_s8(input + input_row_base + c);
                const int8x8_t a8 = vld1_s8(addend + addend_row_base + c);

                const int16x8_t x16 = vmovl_s8(x8);
                const int16x8_t a16 = vmovl_s8(a8);
                const int16x8_t sum16 = vaddq_s16(x16, a16);

                int32x4_t value_lo = vmovl_s16(vget_low_s16(sum16));
                int32x4_t value_hi = vmovl_s16(vget_high_s16(sum16));

                int32x4_t mul_lo;
                int32x4_t mul_hi;
                load_multiplier_u16x8_as_s32(
                    multiplier,
                    multiplier_count,
                    c,
                    mul_lo,
                    mul_hi);

                value_lo = vmulq_s32(value_lo, mul_lo);
                value_hi = vmulq_s32(value_hi, mul_hi);

                int32x4_t shift_lo;
                int32x4_t shift_hi;
                load_shift_i8x8_as_s32(
                    shift,
                    shift_count,
                    c,
                    shift_lo,
                    shift_hi);

                value_lo = rounding_rshift_variable_s32x4(value_lo, shift_lo);
                value_hi = rounding_rshift_variable_s32x4(value_hi, shift_hi);

                value_lo = vminq_s32(vmaxq_s32(value_lo, clamp_min), clamp_max);
                value_hi = vminq_s32(vmaxq_s32(value_hi, clamp_min), clamp_max);

                const int16x8_t out16 = vcombine_s16(
                    vmovn_s32(value_lo),
                    vmovn_s32(value_hi));

                vst1q_s16(output + input_row_base + c, out16);
            }

            for (; c < channels; ++c)
            {
                const int64_t mi = (multiplier_count == 1) ? 0 : c;
                const int64_t si = (shift_count == 1) ? 0 : c;

                output[input_row_base + c] = scalar_one(
                    input[input_row_base + c],
                    addend[addend_row_base + c],
                    multiplier[mi],
                    shift[si]);
            }
        }
    }
}

} // namespace iwhisper
