#include "layernorm.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <limits>

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

        static inline int64_t apply_norm_reduce64(
            int64_t x,
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
                    "unsupported LayerNorm hidden size: " +
                    std::to_string(hidden_size));
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

        template <int Shift>
        static inline int32_t
        accumulate_variance_shifted_simd(
            const int16_t *row_ptr,
            int64_t hidden_size,
            int32_t row_mean)
        {
            static_assert(
                Shift >= 0 &&
                Shift <= 16 &&
                (Shift % 2) == 0);

            const int32x4_t mean_vec =
                vdupq_n_s32(
                    row_mean);

            int32x4_t var0 =
                vdupq_n_s32(0);

            int32x4_t var1 =
                vdupq_n_s32(0);

            int32x4_t var2 =
                vdupq_n_s32(0);

            int32x4_t var3 =
                vdupq_n_s32(0);

            int64_t col = 0;

            for (;
                 col + 16 <= hidden_size;
                 col += 16)
            {
                const int16x8_t x16_0 =
                    vld1q_s16(
                        row_ptr + col);

                const int16x8_t x16_1 =
                    vld1q_s16(
                        row_ptr + col + 8);

                int32x4_t x0_lo =
                    vmovl_s16(
                        vget_low_s16(x16_0));

                int32x4_t x0_hi =
                    vmovl_s16(
                        vget_high_s16(x16_0));

                int32x4_t x1_lo =
                    vmovl_s16(
                        vget_low_s16(x16_1));

                int32x4_t x1_hi =
                    vmovl_s16(
                        vget_high_s16(x16_1));

                x0_lo =
                    vsubq_s32(
                        x0_lo,
                        mean_vec);

                x0_hi =
                    vsubq_s32(
                        x0_hi,
                        mean_vec);

                x1_lo =
                    vsubq_s32(
                        x1_lo,
                        mean_vec);

                x1_hi =
                    vsubq_s32(
                        x1_hi,
                        mean_vec);

                const int32x4_t square0_lo =
                    vmulq_s32(
                        x0_lo,
                        x0_lo);

                const int32x4_t square0_hi =
                    vmulq_s32(
                        x0_hi,
                        x0_hi);

                const int32x4_t square1_lo =
                    vmulq_s32(
                        x1_lo,
                        x1_lo);

                const int32x4_t square1_hi =
                    vmulq_s32(
                        x1_hi,
                        x1_hi);

                if constexpr (Shift == 0)
                {
                    // No shift required.
                    var0 =
                        vaddq_s32(
                            var0,
                            square0_lo);

                    var1 =
                        vaddq_s32(
                            var1,
                            square0_hi);

                    var2 =
                        vaddq_s32(
                            var2,
                            square1_lo);

                    var3 =
                        vaddq_s32(
                            var3,
                            square1_hi);
                }
                else
                {
                    // acc += rounded_right_shift(square, Shift)
                    //
                    // Compile-time Shift allows AArch64 to emit:
                    //
                    // SRSRA Vd.4S, Vn.4S, #Shift

                    var0 =
                        vrsraq_n_s32(
                            var0,
                            square0_lo,
                            Shift);

                    var1 =
                        vrsraq_n_s32(
                            var1,
                            square0_hi,
                            Shift);

                    var2 =
                        vrsraq_n_s32(
                            var2,
                            square1_lo,
                            Shift);

                    var3 =
                        vrsraq_n_s32(
                            var3,
                            square1_hi,
                            Shift);
                }
            }

            const int32x4_t var01 =
                vaddq_s32(
                    var0,
                    var1);

            const int32x4_t var23 =
                vaddq_s32(
                    var2,
                    var3);

            int32_t var_sum =
                vaddvq_s32(
                    vaddq_s32(
                        var01,
                        var23));

            // Generic tail.
            //
            // All currently supported hidden sizes are divisible by 16,
            // but retain this path for completeness.
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

                if constexpr (Shift == 0)
                {
                    var_sum += square;
                }
                else
                {
                    var_sum +=
                        iwhisper::fixedpoint::rounding_rshift(
                            square,
                            Shift);
                }
            }

            return var_sum;
        }

        static inline int32_t
        accumulate_variance_shifted_dispatch(
            const int16_t *row_ptr,
            int64_t hidden_size,
            int32_t row_mean,
            int32_t renorm_shift)
        {
            switch (renorm_shift)
            {
            case 0:
                return accumulate_variance_shifted_simd<0>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            case 2:
                return accumulate_variance_shifted_simd<2>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            case 4:
                return accumulate_variance_shifted_simd<4>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            case 6:
                return accumulate_variance_shifted_simd<6>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            case 8:
                return accumulate_variance_shifted_simd<8>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            case 10:
                return accumulate_variance_shifted_simd<10>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            case 12:
                return accumulate_variance_shifted_simd<12>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            case 14:
                return accumulate_variance_shifted_simd<14>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            case 16:
                return accumulate_variance_shifted_simd<16>(
                    row_ptr,
                    hidden_size,
                    row_mean);

            default:
                throw std::runtime_error(
                    "unexpected LayerNorm32 renorm shift: " + std::to_string(
                                                                  renorm_shift));
            }
        }

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

            // Track input extrema while computing the row sum.
            // These extrema are sufficient to recover the maximum
            // absolute centered value after the mean is known.
            int16x8_t min_vec =
                vdupq_n_s16(
                    std::numeric_limits<int16_t>::max());

            int16x8_t max_vec =
                vdupq_n_s16(
                    std::numeric_limits<int16_t>::min());

            int64_t col = 0;

            for (;
                 col + 8 <= hidden_size;
                 col += 8)
            {
                const int16x8_t x16 =
                    vld1q_s16(
                        row_ptr + col);

                // Sum
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

                // Min / max
                min_vec =
                    vminq_s16(
                        min_vec,
                        x16);

                max_vec =
                    vmaxq_s16(
                        max_vec,
                        x16);
            }

            int32_t row_sum =
                vaddvq_s32(sum_lo) + vaddvq_s32(sum_hi);

            int32_t row_min =
                static_cast<int32_t>(
                    vminvq_s16(min_vec));

            int32_t row_max =
                static_cast<int32_t>(
                    vmaxvq_s16(max_vec));

            for (;
                 col < hidden_size;
                 ++col)
            {
                const int32_t value =
                    static_cast<int32_t>(
                        row_ptr[col]);

                row_sum += value;

                row_min =
                    std::min(
                        row_min,
                        value);

                row_max =
                    std::max(
                        row_max,
                        value);
            }

            const int32_t row_mean =
                apply_norm_reduce(
                    row_sum,
                    hidden_size);

            // --------------------------------------------------------
            // Determine the maximum centered magnitude without an
            // additional row scan.
            //
            // For any fixed mean:
            //
            //   max_i |x_i - mean|
            //     = max(
            //         |min(x) - mean|,
            //         |max(x) - mean|)
            //
            // Therefore the maximum centered square can be obtained
            // exactly from the row extrema collected during Pass 1.
            // --------------------------------------------------------

            const int64_t dist_min =
                std::max(
                    static_cast<int64_t>(row_min),
                    static_cast<int64_t>(row_mean)) -
                std::min(
                    static_cast<int64_t>(row_min),
                    static_cast<int64_t>(row_mean));

            const int64_t dist_max =
                std::max(
                    static_cast<int64_t>(row_max),
                    static_cast<int64_t>(row_mean)) -
                std::min(
                    static_cast<int64_t>(row_max),
                    static_cast<int64_t>(row_mean));

            const int64_t max_abs =
                std::max(
                    dist_min,
                    dist_max);

            const int64_t max_square64 =
                max_abs * max_abs;

            if (
                max_square64 >
                std::numeric_limits<int32_t>::max())
            {
                throw std::overflow_error(
                    "LayerNorm32 centered square exceeds INT32");
            }

            const int32_t max_square =
                static_cast<int32_t>(
                    max_square64);

            const int32_t max_bit_length =
                bit_length_nonnegative(
                    max_square);

            int32_t renorm_shift =
                std::max(
                    max_bit_length - 16,
                    int32_t{0});

            renorm_shift =
                (renorm_shift + 1) & ~int32_t{1};

            const int32_t var_sum =
                accumulate_variance_shifted_dispatch(
                    row_ptr,
                    hidden_size,
                    row_mean,
                    renorm_shift);

            const int32_t variance =
                apply_norm_reduce(
                    var_sum,
                    hidden_size);

            return {
                row_mean,
                variance,
                renorm_shift};
        }

        // ============================================================
        // Row stats for 64-bit variance
        // ============================================================
        struct LayerNorm64RowStats
        {
            int32_t mean;
            int64_t variance;
        };

        static inline LayerNorm64RowStats
        compute_row_stats64_simd(
            const int16_t *row_ptr,
            int64_t hidden_size)
        {
            // --------------------------------------------------------
            // Pass 1:
            // Mean.
            //
            // Intentionally identical to LayerNorm32 SIMD.
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
            // Full INT64 centered-square accumulation.
            //
            // 8 INT16 values
            //   -> 2 x INT32x4 centered vectors
            //   -> 4 x INT64x2 accumulators
            //
            // No max-square pass.
            // No pre-shift.
            // --------------------------------------------------------

            const int32x4_t mean_vec =
                vdupq_n_s32(
                    row_mean);

            int64x2_t var0 =
                vdupq_n_s64(0);

            int64x2_t var1 =
                vdupq_n_s64(0);

            int64x2_t var2 =
                vdupq_n_s64(0);

            int64x2_t var3 =
                vdupq_n_s64(0);

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

                // Each vmull/vmlal performs:
                //
                // INT32 x INT32 -> INT64
                //
                // so the square is never first truncated to INT32.

                var0 =
                    vmlal_s32(
                        var0,
                        vget_low_s32(x_lo),
                        vget_low_s32(x_lo));

                var1 =
                    vmlal_s32(
                        var1,
                        vget_high_s32(x_lo),
                        vget_high_s32(x_lo));

                var2 =
                    vmlal_s32(
                        var2,
                        vget_low_s32(x_hi),
                        vget_low_s32(x_hi));

                var3 =
                    vmlal_s32(
                        var3,
                        vget_high_s32(x_hi),
                        vget_high_s32(x_hi));
            }

            int64_t var_sum =
                vaddvq_s64(var0) + vaddvq_s64(var1) + vaddvq_s64(var2) + vaddvq_s64(var3);

            // Current supported hidden sizes are multiples of 8,
            // but retain a scalar tail for completeness.
            for (;
                 col < hidden_size;
                 ++col)
            {
                const int64_t centered =
                    static_cast<int64_t>(
                        row_ptr[col]) -
                    static_cast<int64_t>(
                        row_mean);

                var_sum +=
                    centered * centered;
            }

            const int64_t variance =
                apply_norm_reduce64(
                    var_sum,
                    hidden_size);

            return {
                row_mean,
                variance};
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

    void layernorm64_stats_simd(
        const int16_t *input,
        int32_t *mean,
        int64_t *variance,
        int64_t rows,
        int64_t hidden_size)
    {
        for (int64_t row = 0;
             row < rows;
             ++row)
        {
            const int16_t *row_ptr =
                input + row * hidden_size;

            const LayerNorm64RowStats stats =
                compute_row_stats64_simd(
                    row_ptr,
                    hidden_size);

            mean[row] =
                stats.mean;

            variance[row] =
                stats.variance;
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

    void layernorm64_simd(
        const int16_t *input,
        const int8_t *weight,
        const int32_t *bias,
        uint8_t ln_multiplier,
        int8_t ln_shift,
        uint32_t out_multiplier,
        int8_t out_shift,
        uint32_t rsqrt_offset,
        int32_t additional_offset,
        int8_t *output,
        int64_t rows,
        int64_t hidden_size,
        bool has_bias)
    {
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
            // INT64 stats
            // ----------------------------------------------------

            const LayerNorm64RowStats stats =
                compute_row_stats64_simd(
                    row_ptr,
                    hidden_size);

            if (stats.variance < 0 ||
                stats.variance >
                    static_cast<int64_t>(
                        std::numeric_limits<int32_t>::max()))
            {
                throw std::overflow_error(
                    "LayerNorm64 variance does not fit INT32");
            }

            const int32_t variance =
                static_cast<int32_t>(
                    stats.variance);

            // ----------------------------------------------------
            // Reuse existing bit-exact integer FISR.
            //
            // LayerNorm64:
            //   rsqrt_offset = 15
            // ----------------------------------------------------

            int32_t invstd = 0;

            layernorm32_rsqrt_scalar(
                &variance,
                &invstd,
                1,
                rsqrt_offset);

            // ----------------------------------------------------
            // FISR scale compensation.
            //
            // offset 15 vs original offset 13 => +2 bits.
            // ----------------------------------------------------

            const int32_t norm_shift_signed =
                static_cast<int32_t>(
                    ln_shift) +
                additional_offset;

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
            // Same normalization / affine / requant path
            // as LayerNorm32 SIMD.
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
                // Keep low-32-bit semantics identical to
                // the existing LayerNorm32 path.
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

                // FISR scale compensation + LN requant
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
                // INT8 weight -> INT32
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

                x_lo =
                    vmulq_s32(
                        x_lo,
                        w_lo);

                x_hi =
                    vmulq_s32(
                        x_hi,
                        w_hi);

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

                // signed INT32 x UINT32 -> high 32 bits
                x_lo =
                    mulhi_s32_u32x4(
                        x_lo,
                        out_multiplier);

                x_hi =
                    mulhi_s32_u32x4(
                        x_hi,
                        out_multiplier);

                x_lo =
                    rounding_rshift_s32x4(
                        x_lo,
                        final_shift);

                x_hi =
                    rounding_rshift_s32x4(
                        x_hi,
                        final_shift);

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