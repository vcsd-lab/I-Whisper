#include "layernorm.h"
#include "../common/fixedpoint.h"

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace iwhisper
{
    namespace
    {

        constexpr int32_t kNormMin = -16383;
        constexpr int32_t kNormMax = 16383;

        constexpr int32_t kOutMin = -127;
        constexpr int32_t kOutMax = 127;

        // ============================================================
        // LayerNorm reduction approximation
        // ============================================================

        static inline int32_t apply_norm_reduce(
            int32_t x,
            int64_t hidden_size)
        {
            switch (hidden_size)
            {
            case 384:
                return (
                           x + (x >> 2) + (x >> 3)) >>
                       9;

            case 512:
                return x >> 9;

            case 768:
                return (
                           x + (x >> 2) + (x >> 3)) >>
                       10;

            case 1024:
                return x >> 10;

            case 1280:
                return (
                           x + (x >> 1)) >>
                       11;

            default:
                throw std::invalid_argument(
                    "unsupported LayerNorm hidden size: " + std::to_string(hidden_size));
            }
        }

        static inline int32_t bit_length_nonnegative(
            int32_t x)
        {
            if (x == 0)
            {
                return 0;
            }

            return 32 - __builtin_clz(
                            static_cast<uint32_t>(x));
        }

        // ============================================================
        // Explicit wrap helpers
        // ============================================================

        static inline int32_t wrap_mul_s32(
            int32_t a,
            int32_t b)
        {
            return static_cast<int32_t>(
                static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
        }

        static inline int32_t wrap_add_s32(
            int32_t a,
            int32_t b)
        {
            return static_cast<int32_t>(
                static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
        }

        // ============================================================
        // Bit-exact SIMD Rshift
        //
        // Match:
        //
        // bias = 2^(shift-1) - signbit(x)
        // y    = (x + bias) >> shift
        //
        // Addition is explicitly 32-bit wrapping.
        // shift == 0 -> identity.
        // ============================================================

        static inline int32x4_t rounding_rshift_s32x4(
            int32x4_t x,
            uint32_t shift)
        {
            if (shift == 0)
            {
                return x;
            }

            const uint32x4_t ux =
                vreinterpretq_u32_s32(x);

            // 0 for positive, 1 for negative
            const uint32x4_t sign =
                vshrq_n_u32(
                    ux,
                    31);

            const uint32x4_t rounding =
                vsubq_u32(
                    vdupq_n_u32(
                        uint32_t{1} << (shift - 1)),
                    sign);

            // uint32 addition gives explicit low-32-bit wrap
            const uint32x4_t rounded =
                vaddq_u32(
                    ux,
                    rounding);

            // negative shift count => arithmetic right shift
            const int32x4_t shift_vec =
                vdupq_n_s32(
                    -static_cast<int32_t>(shift));

            return vshlq_s32(
                vreinterpretq_s32_u32(rounded),
                shift_vec);
        }

        // ============================================================
        // Signed int32 x unsigned Q32 multiplier -> signed high 32 bits
        //
        // We need:
        //
        //   high32(int64(x) * uint32(multiplier))
        //
        // NEON has signed*signed and unsigned*unsigned widening multiply,
        // but not signed*unsigned directly.
        //
        // For negative x:
        //
        //   uint32(x) = x + 2^32
        //
        // therefore:
        //
        //   high(uint32(x) * m) - m
        //
        // reconstructs high(signed(x) * unsigned(m)).
        // ============================================================

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
                    vshrn_n_u64(
                        product_lo,
                        32),
                    vshrn_n_u64(
                        product_hi,
                        32));

            const uint32x4_t sign =
                vshrq_n_u32(
                    ux,
                    31);

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

        static inline int32_t mulhi_s32_u32_scalar(
            int32_t x,
            uint32_t multiplier)
        {
            const int64_t product =
                static_cast<int64_t>(x) * static_cast<int64_t>(multiplier);

            return static_cast<int32_t>(
                product >> 32);
        }

        // ============================================================
        // Row stats
        // ============================================================

        struct LayerNormRowStats
        {
            int32_t mean;
            int32_t variance;
            int32_t renorm_shift;
        };

        static inline LayerNormRowStats
        compute_row_stats_simd(
            const int16_t *row_ptr,
            int64_t hidden_size)
        {
            // --------------------------------------------------------
            // Pass 1:
            // mean
            // --------------------------------------------------------

            int32x4_t sum_lo =
                vdupq_n_s32(0);

            int32x4_t sum_hi =
                vdupq_n_s32(0);

            int64_t col = 0;

            for (;
                 col + 8 <= hidden_size;
                 col += 8)
            {
                const int16x8_t x16 =
                    vld1q_s16(
                        row_ptr + col);

                const int32x4_t x_lo =
                    vmovl_s16(
                        vget_low_s16(x16));

                const int32x4_t x_hi =
                    vmovl_s16(
                        vget_high_s16(x16));

                sum_lo =
                    vaddq_s32(
                        sum_lo,
                        x_lo);

                sum_hi =
                    vaddq_s32(
                        sum_hi,
                        x_hi);
            }

            int32_t row_sum =
                vaddvq_s32(sum_lo) + vaddvq_s32(sum_hi);

            for (;
                 col < hidden_size;
                 ++col)
            {
                row_sum +=
                    static_cast<int32_t>(
                        row_ptr[col]);
            }

            const int32_t row_mean =
                apply_norm_reduce(
                    row_sum,
                    hidden_size);

            // --------------------------------------------------------
            // Pass 2:
            // centered square maximum
            // --------------------------------------------------------

            const int32x4_t mean_vec =
                vdupq_n_s32(
                    row_mean);

            int32x4_t max_lo =
                vdupq_n_s32(0);

            int32x4_t max_hi =
                vdupq_n_s32(0);

            col = 0;

            for (;
                 col + 8 <= hidden_size;
                 col += 8)
            {
                const int16x8_t x16 =
                    vld1q_s16(
                        row_ptr + col);

                int32x4_t x_lo =
                    vmovl_s16(
                        vget_low_s16(x16));

                int32x4_t x_hi =
                    vmovl_s16(
                        vget_high_s16(x16));

                x_lo =
                    vsubq_s32(
                        x_lo,
                        mean_vec);

                x_hi =
                    vsubq_s32(
                        x_hi,
                        mean_vec);

                const int32x4_t square_lo =
                    vmulq_s32(
                        x_lo,
                        x_lo);

                const int32x4_t square_hi =
                    vmulq_s32(
                        x_hi,
                        x_hi);

                max_lo =
                    vmaxq_s32(
                        max_lo,
                        square_lo);

                max_hi =
                    vmaxq_s32(
                        max_hi,
                        square_hi);
            }

            int32_t max_square =
                std::max(
                    vmaxvq_s32(max_lo),
                    vmaxvq_s32(max_hi));

            for (;
                 col < hidden_size;
                 ++col)
            {
                const int32_t centered =
                    static_cast<int32_t>(
                        row_ptr[col]) -
                    row_mean;

                const int32_t square =
                    centered * centered;

                max_square =
                    std::max(
                        max_square,
                        square);
            }

            const int32_t max_bit_length =
                bit_length_nonnegative(
                    max_square);

            int32_t renorm_shift =
                std::max(
                    max_bit_length - 16,
                    int32_t{0});

            renorm_shift =
                (renorm_shift + 1) & ~int32_t{1};

            // --------------------------------------------------------
            // Pass 3:
            // rounded shifted-square accumulation
            // --------------------------------------------------------

            int32x4_t var_lo =
                vdupq_n_s32(0);

            int32x4_t var_hi =
                vdupq_n_s32(0);

            col = 0;

            for (;
                 col + 8 <= hidden_size;
                 col += 8)
            {
                const int16x8_t x16 =
                    vld1q_s16(
                        row_ptr + col);

                int32x4_t x_lo =
                    vmovl_s16(
                        vget_low_s16(x16));

                int32x4_t x_hi =
                    vmovl_s16(
                        vget_high_s16(x16));

                x_lo =
                    vsubq_s32(
                        x_lo,
                        mean_vec);

                x_hi =
                    vsubq_s32(
                        x_hi,
                        mean_vec);

                int32x4_t square_lo =
                    vmulq_s32(
                        x_lo,
                        x_lo);

                int32x4_t square_hi =
                    vmulq_s32(
                        x_hi,
                        x_hi);

                square_lo =
                    rounding_rshift_s32x4(
                        square_lo,
                        static_cast<uint32_t>(
                            renorm_shift));

                square_hi =
                    rounding_rshift_s32x4(
                        square_hi,
                        static_cast<uint32_t>(
                            renorm_shift));

                var_lo =
                    vaddq_s32(
                        var_lo,
                        square_lo);

                var_hi =
                    vaddq_s32(
                        var_hi,
                        square_hi);
            }

            int32_t var_sum =
                vaddvq_s32(var_lo) + vaddvq_s32(var_hi);

            for (;
                 col < hidden_size;
                 ++col)
            {
                const int32_t centered =
                    static_cast<int32_t>(
                        row_ptr[col]) -
                    row_mean;

                const int32_t square =
                    centered * centered;

                var_sum +=
                    iwhisper::fixedpoint::rounding_rshift(
                        square,
                        static_cast<uint32_t>(
                            renorm_shift));
            }

            const int32_t variance =
                apply_norm_reduce(
                    var_sum,
                    hidden_size);

            return {
                row_mean,
                variance,
                renorm_shift};
        }

    } // namespace

    // ============================================================
    // Stats SIMD
    // ============================================================

    void layernorm32_stats_simd(
        const int16_t *input,
        int32_t *mean,
        int32_t *variance,
        int32_t *renorm_shift,
        int64_t rows,
        int64_t hidden_size)
    {
        for (int64_t row = 0;
             row < rows;
             ++row)
        {
            const int16_t *row_ptr =
                input + row * hidden_size;

            const LayerNormRowStats stats =
                compute_row_stats_simd(
                    row_ptr,
                    hidden_size);

            mean[row] =
                stats.mean;

            variance[row] =
                stats.variance;

            renorm_shift[row] =
                stats.renorm_shift;
        }
    }

    // ============================================================
    // Full LayerNorm SIMD
    // ============================================================

    void layernorm32_simd(
        const int16_t *input,
        const int8_t *weight,
        const int32_t *bias,
        uint8_t ln_multiplier,
        int8_t ln_shift,
        uint32_t out_multiplier,
        int8_t out_shift,
        uint32_t rsqrt_offset,
        int8_t *output,
        int64_t rows,
        int64_t hidden_size,
        bool has_bias)

    {
        // For base-fixln.
        //
        // Keep this in the caller rather than hard-coding it
        // inside FISR itself.

        const int32x4_t norm_min =
            vdupq_n_s32(
                kNormMin);

        const int32x4_t norm_max =
            vdupq_n_s32(
                kNormMax);

        const int32x4_t out_min =
            vdupq_n_s32(
                kOutMin);

        const int32x4_t out_max =
            vdupq_n_s32(
                kOutMax);

        for (int64_t row = 0;
             row < rows;
             ++row)
        {
            const int16_t *row_ptr =
                input + row * hidden_size;

            int8_t *out_ptr =
                output + row * hidden_size;

            // ----------------------------------------------------
            // Stats
            // ----------------------------------------------------

            const LayerNormRowStats stats =
                compute_row_stats_simd(
                    row_ptr,
                    hidden_size);

            // ----------------------------------------------------
            // FISR
            //
            // Keep the existing bit-exact scalar implementation.
            // One invocation per row is not worth vectorizing yet.
            // ----------------------------------------------------

            int32_t invstd = 0;

            layernorm32_rsqrt_scalar(
                &stats.variance,
                &invstd,
                1,
                rsqrt_offset);

            // ----------------------------------------------------
            // Final normalization shift
            // ----------------------------------------------------

            const int32_t norm_shift_signed =
                static_cast<int32_t>(
                    ln_shift) +
                (stats.renorm_shift >> 1);

            const uint32_t norm_shift =
                static_cast<uint32_t>(
                    norm_shift_signed);

            const uint32_t final_shift =
                static_cast<uint32_t>(
                    static_cast<int32_t>(
                        out_shift));

            const int32x4_t mean_vec =
                vdupq_n_s32(
                    stats.mean);

            const int32x4_t invstd_vec =
                vdupq_n_s32(
                    invstd);

            const int32x4_t ln_mul_vec =
                vdupq_n_s32(
                    static_cast<int32_t>(
                        ln_multiplier));

            // ----------------------------------------------------
            // Normalize + affine + output requant
            // ----------------------------------------------------

            int64_t col = 0;

            for (;
                 col + 8 <= hidden_size;
                 col += 8)
            {
                const int16x8_t x16 =
                    vld1q_s16(
                        row_ptr + col);

                int32x4_t x_lo =
                    vmovl_s16(
                        vget_low_s16(x16));

                int32x4_t x_hi =
                    vmovl_s16(
                        vget_high_s16(x16));

                // centered
                x_lo =
                    vsubq_s32(
                        x_lo,
                        mean_vec);

                x_hi =
                    vsubq_s32(
                        x_hi,
                        mean_vec);

                // centered * invstd
                //
                // vmulq_s32 keeps the low 32 bits,
                // matching Torch int32 wrap semantics.
                x_lo =
                    vmulq_s32(
                        x_lo,
                        invstd_vec);

                x_hi =
                    vmulq_s32(
                        x_hi,
                        invstd_vec);

                // * ln_multiplier
                x_lo =
                    vmulq_s32(
                        x_lo,
                        ln_mul_vec);

                x_hi =
                    vmulq_s32(
                        x_hi,
                        ln_mul_vec);

                // rounded LN requant
                x_lo =
                    rounding_rshift_s32x4(
                        x_lo,
                        norm_shift);

                x_hi =
                    rounding_rshift_s32x4(
                        x_hi,
                        norm_shift);

                // INT15 clamp
                x_lo =
                    vminq_s32(
                        vmaxq_s32(
                            x_lo,
                            norm_min),
                        norm_max);

                x_hi =
                    vminq_s32(
                        vmaxq_s32(
                            x_hi,
                            norm_min),
                        norm_max);

                // ------------------------------------------------
                // Load INT8 affine weight -> INT32
                // ------------------------------------------------

                const int8x8_t w8 =
                    vld1_s8(
                        weight + col);

                const int16x8_t w16 =
                    vmovl_s8(w8);

                const int32x4_t w_lo =
                    vmovl_s16(
                        vget_low_s16(w16));

                const int32x4_t w_hi =
                    vmovl_s16(
                        vget_high_s16(w16));

                // weight multiplication
                x_lo =
                    vmulq_s32(
                        x_lo,
                        w_lo);

                x_hi =
                    vmulq_s32(
                        x_hi,
                        w_hi);

                // bias
                if (has_bias)
                {
                    x_lo =
                        vaddq_s32(
                            x_lo,
                            vld1q_s32(
                                bias + col));

                    x_hi =
                        vaddq_s32(
                            x_hi,
                            vld1q_s32(
                                bias + col + 4));
                }

                // ------------------------------------------------
                // Truncating MULH
                // ------------------------------------------------

                x_lo =
                    mulhi_s32_u32x4(
                        x_lo,
                        out_multiplier);

                x_hi =
                    mulhi_s32_u32x4(
                        x_hi,
                        out_multiplier);

                // ------------------------------------------------
                // Final rounded right shift
                // ------------------------------------------------

                x_lo =
                    rounding_rshift_s32x4(
                        x_lo,
                        final_shift);

                x_hi =
                    rounding_rshift_s32x4(
                        x_hi,
                        final_shift);

                // INT8 symmetric clamp [-127, 127]
                x_lo =
                    vminq_s32(
                        vmaxq_s32(
                            x_lo,
                            out_min),
                        out_max);

                x_hi =
                    vminq_s32(
                        vmaxq_s32(
                            x_hi,
                            out_min),
                        out_max);

                // ------------------------------------------------
                // Narrow:
                // int32 x8 -> int16 x8 -> int8 x8
                //
                // Values are already clamped, so plain narrowing
                // is exact.
                // ------------------------------------------------

                const int16x8_t y16 =
                    vcombine_s16(
                        vmovn_s32(x_lo),
                        vmovn_s32(x_hi));

                const int8x8_t y8 =
                    vmovn_s16(y16);

                vst1_s8(
                    out_ptr + col,
                    y8);
            }

            // ----------------------------------------------------
            // Scalar tail.
            //
            // Current supported hidden sizes are all multiples of 8,
            // so this should normally never execute.
            // ----------------------------------------------------

            for (;
                 col < hidden_size;
                 ++col)
            {
                int32_t x =
                    static_cast<int32_t>(
                        row_ptr[col]) -
                    stats.mean;

                x =
                    wrap_mul_s32(
                        x,
                        invstd);

                x =
                    wrap_mul_s32(
                        x,
                        static_cast<int32_t>(
                            ln_multiplier));

                x =
                    iwhisper::fixedpoint::rounding_rshift(
                        x,
                        norm_shift);

                x =
                    std::clamp(
                        x,
                        kNormMin,
                        kNormMax);

                x =
                    wrap_mul_s32(
                        x,
                        static_cast<int32_t>(
                            weight[col]));

                if (has_bias)
                {
                    x =
                        wrap_add_s32(
                            x,
                            bias[col]);
                }

                x =
                    mulhi_s32_u32_scalar(
                        x,
                        out_multiplier);

                x =
                    iwhisper::fixedpoint::rounding_rshift(
                        x,
                        final_shift);

                x =
                    std::clamp(
                        x,
                        kOutMin,
                        kOutMax);

                out_ptr[col] =
                    static_cast<int8_t>(x);
            }
        }
    }

} // namespace iwhisper