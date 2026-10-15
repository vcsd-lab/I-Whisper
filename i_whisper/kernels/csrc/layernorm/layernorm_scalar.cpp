#include "layernorm.h"
#include "../common/fixedpoint.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace iwhisper
{
    namespace
    {

        static inline int32_t apply_384_norm_reduce(int32_t x)
        {
            return (x + (x >> 2) + (x >> 3)) >> 9;
        }
        static inline int32_t apply_512_norm_reduce(int32_t x)
        {
            return (x) >> 9;
        }
        static inline int32_t apply_768_norm_reduce(int32_t x)
        {
            return (x + (x >> 2) + (x >> 3)) >> 10;
        }
        static inline int32_t apply_1024_norm_reduce(int32_t x)
        {
            return (x) >> 10;
        }
        static inline int32_t apply_1280_norm_reduce(int32_t x)
        {
            return (x + (x >> 1)) >> 11;
        }

        static inline int32_t apply_norm_reduce(
            int32_t x,
            int64_t hidden_size)
        {
            switch (hidden_size)
            {
            case 1280:
                return apply_1280_norm_reduce(x);
            case 1024:
                return apply_1024_norm_reduce(x);
            case 768:
                return apply_768_norm_reduce(x);
            case 512:
                return apply_512_norm_reduce(x);
            case 384:
                return apply_384_norm_reduce(x);
            default:
                throw std::invalid_argument(
                    "unsupported LayerNorm hidden size: " +
                    std::to_string(hidden_size));
            }
        }

        static inline int64_t apply_norm_reduce64(
            int64_t x,
            int64_t hidden_size)
        {
            switch (hidden_size)
            {
            case 1280:
                return (
                           x + (x >> 1)) >>
                       11;

            case 1024:
                return x >> 10;

            case 768:
                return (
                           x + (x >> 2) + (x >> 3)) >>
                       10;

            case 512:
                return x >> 9;

            case 384:
                return (
                           x + (x >> 2) + (x >> 3)) >>
                       9;

            default:
                throw std::invalid_argument(
                    "unsupported LayerNorm hidden size: " +
                    std::to_string(hidden_size));
            }
        }

        inline int32_t bit_length_nonnegative(int32_t x)
        {
            if (x == 0)
                return 0;

            return 32 - __builtin_clz(
                            static_cast<uint32_t>(x));
        }

        // Match the current frozen Torch Rshift() semantics for non-negative values.
        // For shift > 0: (x + 2^(shift-1)) >> shift
        // For shift == 0, the frozen Torch helper currently returns x + 1.
        inline int32_t rounding_rshift_nonnegative_golden(
            int32_t x,
            int32_t shift)
        {
            if (shift <= 0)
                return x + 1;

            if (shift > 32)
                shift = 32;

            const int64_t rounding_bias =
                int64_t{1} << (shift - 1);

            return static_cast<int32_t>(
                (static_cast<int64_t>(x) + rounding_bias) >> shift);
        }

        inline int32_t centered_square(
            int16_t x,
            int32_t mean)
        {
            const int32_t centered =
                static_cast<int32_t>(x) - mean;

            return centered * centered;
        }

    } // namespace

    void layernorm32_stats_scalar(
        const int16_t *input,
        int32_t *mean,
        int32_t *variance,
        int32_t *renorm_shift,
        int64_t rows,
        int64_t hidden_size)
    {

        for (int64_t row = 0; row < rows; ++row)
        {
            const int16_t *row_ptr =
                input + row * hidden_size;

            // Pass 1: mean
            int32_t row_sum = 0;

            for (int64_t col = 0; col < hidden_size; ++col)
            {
                row_sum +=
                    static_cast<int32_t>(row_ptr[col]);
            }

            const int32_t row_mean =
                apply_norm_reduce(
                    row_sum,
                    hidden_size);

            mean[row] = row_mean;

            // Pass 2: max centered square -> renorm shift
            int32_t max_square = 0;

            for (int64_t col = 0; col < hidden_size; ++col)
            {
                const int32_t square =
                    centered_square(
                        row_ptr[col],
                        row_mean);

                max_square =
                    std::max(
                        max_square,
                        square);
            }

            const int32_t max_bit_length =
                bit_length_nonnegative(max_square);

            int32_t row_renorm_shift =
                std::max(
                    max_bit_length - 16,
                    int32_t{0});

            row_renorm_shift =
                (row_renorm_shift + 1) & ~int32_t{1};

            renorm_shift[row] =
                row_renorm_shift;

            // Pass 3: shifted-square reduction -> variance
            int32_t var_sum = 0;

            for (int64_t col = 0; col < hidden_size; ++col)
            {
                const int32_t square =
                    centered_square(
                        row_ptr[col],
                        row_mean);

                var_sum +=
                    iwhisper::fixedpoint::rounding_rshift(
                        square,
                        row_renorm_shift);
            }

            variance[row] =
                apply_norm_reduce(
                    var_sum,
                    hidden_size);
        }
    }

    static inline int32_t wrap_mul_s32(
        int32_t a,
        int32_t b)
    {
        const uint32_t ua =
            static_cast<uint32_t>(a);

        const uint32_t ub =
            static_cast<uint32_t>(b);

        return static_cast<int32_t>(
            ua * ub);
    }

    void layernorm64_stats_scalar(
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

            // ----------------------------------------------------
            // Pass 1:
            // Keep mean computation identical to LayerNorm32.
            // ----------------------------------------------------

            int32_t row_sum = 0;

            for (int64_t col = 0;
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

            mean[row] =
                row_mean;

            // ----------------------------------------------------
            // Pass 2:
            // Full-precision centered-square accumulation.
            //
            // No row-wise pre-shift is applied here.
            // ----------------------------------------------------

            int64_t var_sum = 0;

            for (int64_t col = 0;
                 col < hidden_size;
                 ++col)
            {
                const int64_t centered =
                    static_cast<int64_t>(
                        row_ptr[col]) -
                    static_cast<int64_t>(
                        row_mean);

                const int64_t square =
                    centered * centered;

                var_sum += square;
            }

            variance[row] =
                apply_norm_reduce64(
                    var_sum,
                    hidden_size);
        }
    }


    static inline int32_t wrap_sub_s32(
        int32_t a,
        int32_t b)
    {
        return static_cast<int32_t>(
            static_cast<uint32_t>(a) - static_cast<uint32_t>(b));
    }

    static inline int32_t int32_to_fp32_bits_trunc(
        int32_t x)
    {
        if (x == 0)
            return 0;

        const int32_t e =
            bit_length_nonnegative(x) - 1;

        uint64_t significand;

        if (e <= 23)
        {
            significand =
                static_cast<uint64_t>(
                    static_cast<uint32_t>(x))
                << (23 - e);
        }
        else
        {
            significand =
                static_cast<uint64_t>(
                    static_cast<uint32_t>(x)) >>
                (e - 23);
        }

        const uint32_t exponent =
            static_cast<uint32_t>(e + 127);

        const uint32_t mantissa =
            static_cast<uint32_t>(significand) & 0x007FFFFFu;

        return static_cast<int32_t>(
            (exponent << 23) | mantissa);
    }

    static inline int32_t fp32_bits_to_int32_trunc(
        int32_t bits)
    {
        const uint32_t raw =
            static_cast<uint32_t>(bits);

        const int32_t exponent_raw =
            static_cast<int32_t>(
                (raw >> 23) & 0xFFu);

        const int32_t e =
            exponent_raw - 127;

        const uint64_t significand =
            static_cast<uint64_t>(
                (1u << 23) | (raw & 0x007FFFFFu));

        if (e < 0)
            return 0;

        uint64_t value;

        if (e >= 23)
        {
            value =
                significand
                << (e - 23);
        }
        else
        {
            value =
                significand >> (23 - e);
        }

        return static_cast<int32_t>(
            static_cast<uint32_t>(value));
    }

    static inline int32_t layernorm32_fisr_one(
        int32_t x,
        uint32_t kOffset)
    {
        constexpr int32_t kNewtonIter = 2;
        constexpr int32_t kMagic =
            static_cast<int32_t>(0x5f3759dfu);

        int32_t i =
            int32_to_fp32_bits_trunc(x);

        i =
            kMagic - (i >> 1);

        int32_t y =
            fp32_bits_to_int32_trunc(
                i + (kOffset << 23));

        const int32_t three =
            3 << kOffset;

        for (int32_t iter = 0;
             iter < kNewtonIter;
             ++iter)
        {
            const int32_t xy =
                wrap_mul_s32(x, y);

            const int32_t xy_scaled =
                iwhisper::fixedpoint::rounding_rshift(
                    xy,
                    kOffset);

            const int32_t t =
                wrap_mul_s32(
                    xy_scaled,
                    y);

            const int32_t correction =
                wrap_sub_s32(
                    three,
                    t);

            const int32_t product =
                wrap_mul_s32(
                    y,
                    correction);

            y =
                iwhisper::fixedpoint::rounding_rshift(
                    product,
                    kOffset + 1);
        }

        return y;
    }

    void layernorm32_rsqrt_scalar(
        const int32_t *variance,
        int32_t *invstd,
        int64_t numel,
        uint32_t offset)
    {
        for (int64_t i = 0;
             i < numel;
             ++i)
        {
            invstd[i] =
                layernorm32_fisr_one(
                    variance[i], offset);
        }
    }

    static inline int32_t wrap_add_s32(
        int32_t a,
        int32_t b)
    {
        return static_cast<int32_t>(
            static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
    }

    void layernorm32_scalar(
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
        constexpr int32_t kInt15Min = -16383;
        constexpr int32_t kInt15Max = 16383;
        constexpr int32_t kInt8Min = -127;
        constexpr int32_t kInt8Max = 127;

        for (int64_t row = 0; row < rows; ++row)
        {
            const int16_t *row_in =
                input + row * hidden_size;

            int8_t *row_out =
                output + row * hidden_size;

            int32_t row_sum = 0;

            for (int64_t col = 0; col < hidden_size; ++col)
            {
                row_sum +=
                    static_cast<int32_t>(
                        row_in[col]);
            }

            const int32_t row_mean =
                apply_norm_reduce(
                    row_sum,
                    hidden_size);

            int32_t max_square = 0;

            for (int64_t col = 0; col < hidden_size; ++col)
            {
                const int32_t square =
                    centered_square(
                        row_in[col],
                        row_mean);

                max_square =
                    std::max(
                        max_square,
                        square);
            }

            const int32_t max_bit_length =
                bit_length_nonnegative(
                    max_square);

            int32_t row_renorm_shift =
                std::max(
                    max_bit_length - 16,
                    int32_t{0});

            row_renorm_shift =
                (row_renorm_shift + 1) & ~int32_t{1};

            int32_t var_sum = 0;

            for (int64_t col = 0; col < hidden_size; ++col)
            {
                const int32_t square =
                    centered_square(
                        row_in[col],
                        row_mean);

                var_sum +=
                    iwhisper::fixedpoint::rounding_rshift(
                        square,
                        row_renorm_shift);
            }

            const int32_t variance =
                apply_norm_reduce(
                    var_sum,
                    hidden_size);

            const int32_t invstd =
                layernorm32_fisr_one(
                    variance, rsqrt_offset);

            const int32_t normalize_shift =
                static_cast<int32_t>(ln_shift) + (row_renorm_shift >> 1);

            for (int64_t col = 0; col < hidden_size; ++col)
            {
                const int32_t centered =
                    static_cast<int32_t>(
                        row_in[col]) -
                    row_mean;

                int32_t value =
                    wrap_mul_s32(
                        centered,
                        invstd);

                value =
                    wrap_mul_s32(
                        value,
                        static_cast<int32_t>(
                            ln_multiplier));

                value =
                    iwhisper::fixedpoint::rounding_rshift(
                        value,
                        normalize_shift);

                value =
                    std::clamp(
                        value,
                        kInt15Min,
                        kInt15Max);

                value =
                    wrap_mul_s32(
                        value,
                        static_cast<int32_t>(
                            weight[col]));

                if (has_bias)
                {
                    value =
                        wrap_add_s32(
                            value,
                            bias[col]);
                }

                value =
                    iwhisper::fixedpoint::mulh(
                        value,
                        out_multiplier);

                value =
                    iwhisper::fixedpoint::rounding_rshift(
                        value,
                        static_cast<int32_t>(
                            out_shift));

                value =
                    std::clamp(
                        value,
                        kInt8Min,
                        kInt8Max);

                row_out[col] =
                    static_cast<int8_t>(
                        value);
            }
        }
    }

    void layernorm64_scalar(
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
        constexpr int32_t kInt15Min = -16383;
        constexpr int32_t kInt15Max = 16383;
        constexpr int32_t kInt8Min = -127;
        constexpr int32_t kInt8Max = 127;

        for (int64_t row = 0;
             row < rows;
             ++row)
        {
            const int16_t *row_in =
                input + row * hidden_size;

            int8_t *row_out =
                output + row * hidden_size;

            // ----------------------------------------------------
            // Pass 1:
            // Mean computation is intentionally identical to
            // LayerNorm32.
            // ----------------------------------------------------

            int32_t row_sum = 0;

            for (int64_t col = 0;
                 col < hidden_size;
                 ++col)
            {
                row_sum +=
                    static_cast<int32_t>(
                        row_in[col]);
            }

            const int32_t row_mean =
                apply_norm_reduce(
                    row_sum,
                    hidden_size);

            // ----------------------------------------------------
            // Pass 2:
            // Full INT64 centered-square accumulation.
            //
            // Unlike LayerNorm32, no row-wise pre-shift is
            // applied before accumulation.
            // ----------------------------------------------------

            int64_t var_sum = 0;

            for (int64_t col = 0;
                 col < hidden_size;
                 ++col)
            {
                const int64_t centered =
                    static_cast<int64_t>(
                        row_in[col]) -
                    static_cast<int64_t>(
                        row_mean);

                const int64_t square =
                    centered * centered;

                var_sum += square;
            }

            const int64_t variance64 =
                apply_norm_reduce64(
                    var_sum,
                    hidden_size);

            // Full variance must fit the existing INT32 FISR
            // interface. Keep this explicit rather than silently
            // narrowing.
            if (variance64 < 0 ||
                variance64 >
                    static_cast<int64_t>(
                        std::numeric_limits<int32_t>::max()))
            {
                throw std::overflow_error(
                    "LayerNorm64 variance does not fit INT32");
            }

            const int32_t variance =
                static_cast<int32_t>(
                    variance64);

            // ----------------------------------------------------
            // Existing integer FISR.
            //
            // LayerNorm64 uses a larger rsqrt offset because the
            // unshifted variance has a larger dynamic range.
            // ----------------------------------------------------

            const int32_t invstd =
                layernorm32_fisr_one(
                    variance,
                    rsqrt_offset);

            // ----------------------------------------------------
            // Compensate for the increased FISR fixed-point scale.
            //
            // For rsqrt_offset = 15 versus the original 13:
            //
            // additional_offset = 2
            // ----------------------------------------------------

            const int32_t normalize_shift =
                static_cast<int32_t>(
                    ln_shift) +
                additional_offset;

            // ----------------------------------------------------
            // Normalize + affine + output requant.
            //
            // Keep this path identical to LayerNorm32.
            // ----------------------------------------------------

            for (int64_t col = 0;
                 col < hidden_size;
                 ++col)
            {
                const int32_t centered =
                    static_cast<int32_t>(
                        row_in[col]) -
                    row_mean;

                int32_t value =
                    wrap_mul_s32(
                        centered,
                        invstd);

                value =
                    wrap_mul_s32(
                        value,
                        static_cast<int32_t>(
                            ln_multiplier));

                value =
                    iwhisper::fixedpoint::rounding_rshift(
                        value,
                        static_cast<uint32_t>(
                            normalize_shift));

                value =
                    std::clamp(
                        value,
                        kInt15Min,
                        kInt15Max);

                value =
                    wrap_mul_s32(
                        value,
                        static_cast<int32_t>(
                            weight[col]));

                if (has_bias)
                {
                    value =
                        wrap_add_s32(
                            value,
                            bias[col]);
                }

                value =
                    iwhisper::fixedpoint::mulh(
                        value,
                        out_multiplier);

                value =
                    iwhisper::fixedpoint::rounding_rshift(
                        value,
                        static_cast<uint32_t>(
                            static_cast<int32_t>(
                                out_shift)));

                value =
                    std::clamp(
                        value,
                        kInt8Min,
                        kInt8Max);

                row_out[col] =
                    static_cast<int8_t>(
                        value);
            }
        }
    }

} // namespace iwhisper
