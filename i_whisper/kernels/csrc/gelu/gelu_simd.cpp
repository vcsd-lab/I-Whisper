#include "gelu.h"
#include "../common/fixedpoint.h"

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>


namespace iwhisper
{

namespace
{

constexpr int32_t kLutZero = 8;
constexpr int32_t kLutUpper = 15;

constexpr int32_t kActMin = -127;
constexpr int32_t kActMax = 127;


// ============================================================
// Signed int32 x unsigned uint32
// -> high signed 32 bits
//
// Per-lane multiplier version.
//
// Equivalent to:
//
//   int32_t(
//       (int64_t(x) * int64_t(uint32_multiplier))
//       >> 32
//   )
//
// Unlike LayerNorm, multiplier is not necessarily shared
// across lanes.
// ============================================================

static inline int32x4_t mulhi_s32_u32x4(
    int32x4_t x,
    uint32x4_t multiplier)
{
    const uint32x4_t ux =
        vreinterpretq_u32_s32(x);

    const uint64x2_t product_lo =
        vmull_u32(
            vget_low_u32(ux),
            vget_low_u32(multiplier));

    const uint64x2_t product_hi =
        vmull_u32(
            vget_high_u32(ux),
            vget_high_u32(multiplier));

    uint32x4_t high =
        vcombine_u32(
            vshrn_n_u64(
                product_lo,
                32),
            vshrn_n_u64(
                product_hi,
                32));

    // ux = x + 2^32 for negative signed x.
    //
    // Therefore unsigned high contains one extra
    // multiplier for negative lanes.
    const uint32x4_t sign =
        vshrq_n_u32(
            ux,
            31);

    const uint32x4_t correction =
        vmulq_u32(
            sign,
            multiplier);

    high =
        vsubq_u32(
            high,
            correction);

    return
        vreinterpretq_s32_u32(high);
}


// ============================================================
// Variable per-lane rounded right shift
//
// Frozen Rshift semantics:
//
// shift == 0:
//     y = x
//
// shift > 0:
//     bias = 2^(shift - 1) - signbit(x)
//     y = (x + bias) >> shift
//
// shift is clamped into [0, 32].
//
// Addition is explicit 32-bit wrap, matching torch.int32.
// ============================================================

static inline int32x4_t rounding_rshift_variable_s32x4(
    int32x4_t x,
    int32x4_t shift)
{
    const int32x4_t zero =
        vdupq_n_s32(0);

    const int32x4_t max_shift =
        vdupq_n_s32(32);

    // Clamp negative shifts to 0.
    shift =
        vmaxq_s32(
            shift,
            zero);

    // Record >= 32 semantics before/after clamp.
    // Since values >32 become 32 below, ==32 is sufficient
    // after clamping.
    shift =
        vminq_s32(
            shift,
            max_shift);

    const uint32x4_t zero_shift_mask =
        vceqq_s32(
            shift,
            zero);

    const uint32x4_t large_shift_mask =
        vcgeq_s32(
            shift,
            max_shift);


    // max(shift - 1, 0)
    int32x4_t rounding_shift =
        vsubq_s32(
            shift,
            vdupq_n_s32(1));

    rounding_shift =
        vmaxq_s32(
            rounding_shift,
            zero);


    // 1 << rounding_shift
    const uint32x4_t rounding_base =
        vshlq_u32(
            vdupq_n_u32(1),
            rounding_shift);


    const uint32x4_t ux =
        vreinterpretq_u32_s32(x);

    const uint32x4_t sign =
        vshrq_n_u32(
            ux,
            31);


    const uint32x4_t rounding =
        vsubq_u32(
            rounding_base,
            sign);


    // Explicit INT32 wrap semantics.
    const uint32x4_t rounded =
        vaddq_u32(
            ux,
            rounding);


    const int32x4_t negative_shift =
        vnegq_s32(
            shift);


    const int32x4_t shifted =
        vshlq_s32(
            vreinterpretq_s32_u32(
                rounded),
            negative_shift);


    // shift == 0 -> identity
    int32x4_t result =
        vbslq_s32(
            zero_shift_mask,
            x,
            shifted);


    // shift >= 32 -> 0
    result =
        vbslq_s32(
            large_shift_mask,
            zero,
            result);


    return result;
}


// ============================================================
// int8x16 -> four int32x4 vectors
// ============================================================

static inline void widen_s8_to_s32x4(
    int8x16_t x,
    int32x4_t &x0,
    int32x4_t &x1,
    int32x4_t &x2,
    int32x4_t &x3)
{
    const int16x8_t lo16 =
        vmovl_s8(
            vget_low_s8(x));

    const int16x8_t hi16 =
        vmovl_s8(
            vget_high_s8(x));

    x0 =
        vmovl_s16(
            vget_low_s16(lo16));

    x1 =
        vmovl_s16(
            vget_high_s16(lo16));

    x2 =
        vmovl_s16(
            vget_low_s16(hi16));

    x3 =
        vmovl_s16(
            vget_high_s16(hi16));
}


// ============================================================
// Pack four int32x4 vectors into int8x16.
//
// Caller guarantees values fit int8.
// ============================================================

static inline int8x16_t pack_s32x4_to_s8(
    int32x4_t x0,
    int32x4_t x1,
    int32x4_t x2,
    int32x4_t x3)
{
    const int16x8_t lo16 =
        vcombine_s16(
            vmovn_s32(x0),
            vmovn_s32(x1));

    const int16x8_t hi16 =
        vcombine_s16(
            vmovn_s32(x2),
            vmovn_s32(x3));

    return
        vcombine_s8(
            vmovn_s16(lo16),
            vmovn_s16(hi16));
}


// ============================================================
// Scalar tail
// ============================================================

static inline int8_t gelu_one_scalar(
    int8_t input,
    const uint8_t *lut,
    uint32_t luti_multiplier,
    int8_t luti_shift,
    uint8_t luto_multiplier,
    int8_t luto_shift)
{

    int32_t index =
        iwhisper::fixedpoint::mulh(
            static_cast<int32_t>(input),
            luti_multiplier);

    index =
        iwhisper::fixedpoint::rounding_rshift(
            index,
            static_cast<uint32_t>(
                std::clamp<int32_t>(
                    static_cast<int32_t>(
                        luti_shift),
                    0,
                    32)));

    index += kLutZero;

    index =
        std::clamp(
            index,
            int32_t{0},
            kLutUpper);

    const int32_t gate =
        static_cast<int32_t>(
            lut[index]);

    int32_t value =
        static_cast<int32_t>(input)
        * gate;

    value *=
        static_cast<int32_t>(
            luto_multiplier);

    value =
        iwhisper::fixedpoint::rounding_rshift(
            value,
            static_cast<uint32_t>(
                std::clamp<int32_t>(
                    static_cast<int32_t>(
                        luto_shift),
                    0,
                    32)));

    value =
        std::clamp(
            value,
            kActMin,
            kActMax);

    return
        static_cast<int8_t>(
            value);
}

} // namespace


// ============================================================
// Complete SIMD GeLU
// ============================================================

void gelu_simd(
    const int8_t *input,
    const uint8_t *lut,
    const uint32_t *luti_multiplier,
    const int8_t *luti_shift,
    const uint8_t *luto_multiplier,
    const int8_t *luto_shift,
    int8_t *output,
    int64_t numel,
    int64_t channels,
    int64_t luti_param_count,
    int64_t luto_param_count)
{
    const uint8x16_t lut_vec =
        vld1q_u8(lut);

    const int32x4_t lut_zero_vec =
        vdupq_n_s32(
            kLutZero);

    const int32x4_t lut_min_vec =
        vdupq_n_s32(0);

    const int32x4_t lut_max_vec =
        vdupq_n_s32(
            kLutUpper);

    const int32x4_t act_min_vec =
        vdupq_n_s32(
            kActMin);

    const int32x4_t act_max_vec =
        vdupq_n_s32(
            kActMax);


    const int64_t rows =
        numel / channels;


    for (int64_t row = 0;
         row < rows;
         ++row)
    {
        const int8_t *row_in =
            input + row * channels;

        int8_t *row_out =
            output + row * channels;


        int64_t col = 0;


        // ====================================================
        // SIMD body: 16 elements
        // ====================================================

        for (;
             col + 16 <= channels;
             col += 16)
        {
            // ------------------------------------------------
            // Load input
            // ------------------------------------------------

            const int8x16_t x8 =
                vld1q_s8(
                    row_in + col);


            int32x4_t x0;
            int32x4_t x1;
            int32x4_t x2;
            int32x4_t x3;

            widen_s8_to_s32x4(
                x8,
                x0,
                x1,
                x2,
                x3);


            // ------------------------------------------------
            // Load LUT-input multipliers
            // ------------------------------------------------

            uint32x4_t luti_m0;
            uint32x4_t luti_m1;
            uint32x4_t luti_m2;
            uint32x4_t luti_m3;


            if (luti_param_count == 1)
            {
                const uint32x4_t m =
                    vdupq_n_u32(
                        luti_multiplier[0]);

                luti_m0 = m;
                luti_m1 = m;
                luti_m2 = m;
                luti_m3 = m;
            }
            else
            {
                luti_m0 =
                    vld1q_u32(
                        luti_multiplier
                        + col);

                luti_m1 =
                    vld1q_u32(
                        luti_multiplier
                        + col + 4);

                luti_m2 =
                    vld1q_u32(
                        luti_multiplier
                        + col + 8);

                luti_m3 =
                    vld1q_u32(
                        luti_multiplier
                        + col + 12);
            }


            // ------------------------------------------------
            // MULH for LUT input
            // ------------------------------------------------

            int32x4_t idx0 =
                mulhi_s32_u32x4(
                    x0,
                    luti_m0);

            int32x4_t idx1 =
                mulhi_s32_u32x4(
                    x1,
                    luti_m1);

            int32x4_t idx2 =
                mulhi_s32_u32x4(
                    x2,
                    luti_m2);

            int32x4_t idx3 =
                mulhi_s32_u32x4(
                    x3,
                    luti_m3);


            // ------------------------------------------------
            // Load LUT-input shifts
            // ------------------------------------------------

            int32x4_t luti_s0;
            int32x4_t luti_s1;
            int32x4_t luti_s2;
            int32x4_t luti_s3;


            if (luti_param_count == 1)
            {
                const int32x4_t s =
                    vdupq_n_s32(
                        static_cast<int32_t>(
                            luti_shift[0]));

                luti_s0 = s;
                luti_s1 = s;
                luti_s2 = s;
                luti_s3 = s;
            }
            else
            {
                const int8x16_t shifts8 =
                    vld1q_s8(
                        luti_shift
                        + col);

                widen_s8_to_s32x4(
                    shifts8,
                    luti_s0,
                    luti_s1,
                    luti_s2,
                    luti_s3);
            }


            // ------------------------------------------------
            // Rounded Rshift
            // ------------------------------------------------

            idx0 =
                rounding_rshift_variable_s32x4(
                    idx0,
                    luti_s0);

            idx1 =
                rounding_rshift_variable_s32x4(
                    idx1,
                    luti_s1);

            idx2 =
                rounding_rshift_variable_s32x4(
                    idx2,
                    luti_s2);

            idx3 =
                rounding_rshift_variable_s32x4(
                    idx3,
                    luti_s3);


            // ------------------------------------------------
            // + LUT zero
            // ------------------------------------------------

            idx0 =
                vaddq_s32(
                    idx0,
                    lut_zero_vec);

            idx1 =
                vaddq_s32(
                    idx1,
                    lut_zero_vec);

            idx2 =
                vaddq_s32(
                    idx2,
                    lut_zero_vec);

            idx3 =
                vaddq_s32(
                    idx3,
                    lut_zero_vec);


            // ------------------------------------------------
            // Clamp LUT index [0, 15]
            // ------------------------------------------------

            idx0 =
                vminq_s32(
                    vmaxq_s32(
                        idx0,
                        lut_min_vec),
                    lut_max_vec);

            idx1 =
                vminq_s32(
                    vmaxq_s32(
                        idx1,
                        lut_min_vec),
                    lut_max_vec);

            idx2 =
                vminq_s32(
                    vmaxq_s32(
                        idx2,
                        lut_min_vec),
                    lut_max_vec);

            idx3 =
                vminq_s32(
                    vmaxq_s32(
                        idx3,
                        lut_min_vec),
                    lut_max_vec);


            // ------------------------------------------------
            // int32 indices -> uint8x16
            // ------------------------------------------------

            const int8x16_t index_s8 =
                pack_s32x4_to_s8(
                    idx0,
                    idx1,
                    idx2,
                    idx3);

            const uint8x16_t index_u8 =
                vreinterpretq_u8_s8(
                    index_s8);


            // ------------------------------------------------
            // 16-entry uint8 LUT lookup
            //
            // vqtbl1q_u8:
            //
            // table = 16 LUT values
            // index = 16 independent indices
            // ------------------------------------------------

            const uint8x16_t gate8 =
                vqtbl1q_u8(
                    lut_vec,
                    index_u8);


            // ------------------------------------------------
            // x * LUT
            //
            // x      : signed int8
            // LUT    : unsigned uint8 [0,255]
            //
            // Product bound:
            //
            // 127 * 255 = 32385
            //
            // fits signed INT16.
            // ------------------------------------------------

            const int16x8_t x_lo16 =
                vmovl_s8(
                    vget_low_s8(x8));

            const int16x8_t x_hi16 =
                vmovl_s8(
                    vget_high_s8(x8));


            const int16x8_t gate_lo16 =
                vreinterpretq_s16_u16(
                    vmovl_u8(
                        vget_low_u8(
                            gate8)));

            const int16x8_t gate_hi16 =
                vreinterpretq_s16_u16(
                    vmovl_u8(
                        vget_high_u8(
                            gate8)));


            const int16x8_t gated_lo16 =
                vmulq_s16(
                    x_lo16,
                    gate_lo16);

            const int16x8_t gated_hi16 =
                vmulq_s16(
                    x_hi16,
                    gate_hi16);


            // ------------------------------------------------
            // Load LUT-output multiplier
            // ------------------------------------------------

            uint8x16_t luto_m8;

            if (luto_param_count == 1)
            {
                luto_m8 =
                    vdupq_n_u8(
                        luto_multiplier[0]);
            }
            else
            {
                luto_m8 =
                    vld1q_u8(
                        luto_multiplier
                        + col);
            }


            const int16x8_t luto_m_lo16 =
                vreinterpretq_s16_u16(
                    vmovl_u8(
                        vget_low_u8(
                            luto_m8)));

            const int16x8_t luto_m_hi16 =
                vreinterpretq_s16_u16(
                    vmovl_u8(
                        vget_high_u8(
                            luto_m8)));


            // ------------------------------------------------
            // Normal INT32 multiplication.
            //
            // IMPORTANT:
            // luto path is NOT MULH.
            // ------------------------------------------------

            int32x4_t out0 =
                vmull_s16(
                    vget_low_s16(
                        gated_lo16),
                    vget_low_s16(
                        luto_m_lo16));

            int32x4_t out1 =
                vmull_s16(
                    vget_high_s16(
                        gated_lo16),
                    vget_high_s16(
                        luto_m_lo16));

            int32x4_t out2 =
                vmull_s16(
                    vget_low_s16(
                        gated_hi16),
                    vget_low_s16(
                        luto_m_hi16));

            int32x4_t out3 =
                vmull_s16(
                    vget_high_s16(
                        gated_hi16),
                    vget_high_s16(
                        luto_m_hi16));


            // ------------------------------------------------
            // LUT-output shifts
            // ------------------------------------------------

            int32x4_t luto_s0;
            int32x4_t luto_s1;
            int32x4_t luto_s2;
            int32x4_t luto_s3;


            if (luto_param_count == 1)
            {
                const int32x4_t s =
                    vdupq_n_s32(
                        static_cast<int32_t>(
                            luto_shift[0]));

                luto_s0 = s;
                luto_s1 = s;
                luto_s2 = s;
                luto_s3 = s;
            }
            else
            {
                const int8x16_t shifts8 =
                    vld1q_s8(
                        luto_shift
                        + col);

                widen_s8_to_s32x4(
                    shifts8,
                    luto_s0,
                    luto_s1,
                    luto_s2,
                    luto_s3);
            }


            out0 =
                rounding_rshift_variable_s32x4(
                    out0,
                    luto_s0);

            out1 =
                rounding_rshift_variable_s32x4(
                    out1,
                    luto_s1);

            out2 =
                rounding_rshift_variable_s32x4(
                    out2,
                    luto_s2);

            out3 =
                rounding_rshift_variable_s32x4(
                    out3,
                    luto_s3);


            // ------------------------------------------------
            // INT8 symmetric clamp
            // ------------------------------------------------

            out0 =
                vminq_s32(
                    vmaxq_s32(
                        out0,
                        act_min_vec),
                    act_max_vec);

            out1 =
                vminq_s32(
                    vmaxq_s32(
                        out1,
                        act_min_vec),
                    act_max_vec);

            out2 =
                vminq_s32(
                    vmaxq_s32(
                        out2,
                        act_min_vec),
                    act_max_vec);

            out3 =
                vminq_s32(
                    vmaxq_s32(
                        out3,
                        act_min_vec),
                    act_max_vec);


            // ------------------------------------------------
            // Pack + store
            // ------------------------------------------------

            const int8x16_t y8 =
                pack_s32x4_to_s8(
                    out0,
                    out1,
                    out2,
                    out3);

            vst1q_s8(
                row_out + col,
                y8);
        }


        // ====================================================
        // Scalar tail
        //
        // Important for encoder gelu1 because last dimension
        // can be 750, which is not divisible by 16.
        // ====================================================

        for (;
             col < channels;
             ++col)
        {
            const int64_t luti_idx =
                (luti_param_count == 1)
                    ? 0
                    : col;

            const int64_t luto_idx =
                (luto_param_count == 1)
                    ? 0
                    : col;

            row_out[col] =
                gelu_one_scalar(
                    row_in[col],
                    lut,
                    luti_multiplier[
                        luti_idx],
                    luti_shift[
                        luti_idx],
                    luto_multiplier[
                        luto_idx],
                    luto_shift[
                        luto_idx]);
        }
    }
}

} // namespace iwhisper