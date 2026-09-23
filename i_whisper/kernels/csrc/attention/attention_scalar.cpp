#include "attention.h"

#include <cstdint>

namespace iwhisper
{

    namespace
    {

        // ============================================================
        // Explicit INT16 wrap helpers.
        //
        // These are kept only for operations where the Torch INT16
        // datapath can actually cross the signed 16-bit boundary.
        // ============================================================

        static inline int16_t s16_from_bits(
            uint16_t bits)
        {
            const int32_t value =
                (bits & uint16_t{0x8000})
                    ? static_cast<int32_t>(bits) - 65536
                    : static_cast<int32_t>(bits);

            return static_cast<int16_t>(value);
        }

        static inline int16_t wrap_add_s16(
            int16_t a,
            int16_t b)
        {
            const uint16_t ua =
                static_cast<uint16_t>(a);

            const uint16_t ub =
                static_cast<uint16_t>(b);

            const uint16_t result =
                static_cast<uint16_t>(
                    ua + ub);

            return s16_from_bits(result);
        }

        static inline int16_t wrap_sub_s16(
            int16_t a,
            int16_t b)
        {
            const uint16_t ua =
                static_cast<uint16_t>(a);

            const uint16_t ub =
                static_cast<uint16_t>(b);

            const uint16_t result =
                static_cast<uint16_t>(
                    ua - ub);

            return s16_from_bits(result);
        }

    } // namespace

    void qk_scalar(
        const int8_t *q,
        const int8_t *k,
        int32_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim)
    {
        for (int64_t b = 0; b < batch; ++b)
        {
            for (int64_t h = 0; h < heads; ++h)
            {
                const int64_t q_head_base =
                    ((b * heads + h) * q_length * head_dim);

                const int64_t k_head_base =
                    ((b * heads + h) * k_length * head_dim);

                const int64_t out_head_base =
                    ((b * heads + h) * q_length * k_length);

                for (int64_t qi = 0;
                     qi < q_length;
                     ++qi)
                {
                    const int8_t *q_ptr =
                        q + q_head_base + qi * head_dim;

                    int32_t *out_ptr =
                        output + out_head_base + qi * k_length;

                    for (int64_t ki = 0;
                         ki < k_length;
                         ++ki)
                    {
                        const int8_t *k_ptr =
                            k + k_head_base + ki * head_dim;

                        int32_t acc = 0;

                        for (int64_t d = 0;
                             d < head_dim;
                             ++d)
                        {
                            acc +=
                                static_cast<int32_t>(
                                    q_ptr[d]) *
                                static_cast<int32_t>(
                                    k_ptr[d]);
                        }

                        out_ptr[ki] = acc;
                    }
                }
            }
        }
    }

    // ============================================================
    // Integer shift-exp
    //
    // Torch golden:
    //
    //     x[mask] = INT16_MIN
    //
    //     x_max = x.max(dim=-1, keepdim=True).values + 1
    //     x_sub = x - x_max
    //
    //     x_mul_fix =
    //         x_sub
    //         + (x_sub >> 1)
    //         - (x_sub >> 4)
    //
    //     x_mul_int = x_mul_fix >> frac_bits
    //
    //     x_mul_frac =
    //         x_mul_fix
    //         - (x_mul_int << frac_bits)
    //
    //     x_exp =
    //         ((1 << frac_bits)
    //          + (x_mul_frac >> 1))
    //         >> (-x_mul_int)
    //
    //     x_exp[mask] = 0
    //     x_exp[x_sub > 0] = 0
    //
    //     if frac_bits > 7:
    //         x_exp =
    //             x_exp.int()
    //             >> (frac_bits - 7)
    //
    //     return x_exp.uint8()
    //
    // Input:
    //     [B, H, Tq, Tk] INT16
    //
    // Mask:
    //     [Tq, Tk] BOOL
    //     shared across B/H
    //
    // Output:
    //     [B, H, Tq, Tk] UINT8
    //
    // Important:
    //     input is NOT modified.
    // ============================================================

    void shift_exp_scalar(
        const int16_t *input,
        const bool *mask,
        uint8_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int32_t frac_bits,
        bool has_mask)
    {
        constexpr int16_t kInt16Min =
            static_cast<int16_t>(-32768);

        const int32_t frac_scale =
            int32_t{1} << frac_bits;

        for (int64_t b = 0;
             b < batch;
             ++b)
        {
            for (int64_t h = 0;
                 h < heads;
                 ++h)
            {
                const int64_t head_base =
                    (b * heads + h) * q_length * k_length;

                for (int64_t qi = 0;
                     qi < q_length;
                     ++qi)
                {
                    const int64_t row_base =
                        head_base + qi * k_length;

                    // ============================================
                    // Pass 1:
                    //     masked row maximum
                    //
                    // Torch first replaces masked elements with
                    // INT16_MIN and then performs max().
                    // ============================================

                    int16_t row_max =
                        kInt16Min;

                    for (int64_t ki = 0;
                         ki < k_length;
                         ++ki)
                    {
                        const bool masked =
                            has_mask &&
                            mask[qi * k_length + ki];

                        const int16_t value =
                            masked
                                ? kInt16Min
                                : input[row_base + ki];

                        if (value > row_max)
                        {
                            row_max = value;
                        }
                    }

                    // Torch INT16:
                    //
                    //     x_max = row_max + 1
                    //
                    // Preserve INT16 wrap semantics.
                    const int16_t x_max =
                        wrap_add_s16(
                            row_max,
                            int16_t{1});

                    // ============================================
                    // Pass 2:
                    //     subtract max
                    //     approximate exp
                    // ============================================

                    for (int64_t ki = 0;
                         ki < k_length;
                         ++ki)
                    {
                        const int64_t idx =
                            row_base + ki;

                        const bool masked =
                            has_mask &&
                            mask[qi * k_length + ki];

                        // Masked output is unconditionally zero.
                        //
                        // Torch computes intermediate values first
                        // and overwrites them afterwards, but the
                        // final result is identical.
                        if (masked)
                        {
                            output[idx] = 0;
                            continue;
                        }

                        const int16_t x =
                            input[idx];

                        // ----------------------------------------
                        // x_sub = x - x_max
                        //
                        // Normally <= -1 because:
                        //
                        //     x_max = row_max + 1
                        //
                        // but INT16 wrap can make pathological
                        // boundary cases positive.
                        // ----------------------------------------

                        const int16_t x_sub =
                            wrap_sub_s16(
                                x,
                                x_max);

                        // Torch eventually forces these lanes to 0:
                        //
                        //     x_exp[x_sub > 0] = 0
                        //
                        // so there is no need to evaluate their
                        // exponential approximation.
                        if (x_sub > 0)
                        {
                            output[idx] = 0;
                            continue;
                        }

                        // ----------------------------------------
                        // x_mul_fix
                        //
                        // log2(e) approximation:
                        //
                        //     1 + 1/2 - 1/16
                        //     = 1.4375
                        //
                        // Keep each add/sub in INT16 because Torch
                        // performs this expression in INT16.
                        //
                        // Signed >> is arithmetic on our AArch64
                        // deployment target.
                        // ----------------------------------------

                        const int16_t half =
                            static_cast<int16_t>(
                                x_sub >> 1);

                        const int16_t sixteenth =
                            static_cast<int16_t>(
                                x_sub >> 4);

                        const int16_t add_half =
                            wrap_add_s16(
                                x_sub,
                                half);

                        const int16_t x_mul_fix =
                            wrap_sub_s16(
                                add_half,
                                sixteenth);

                        // ----------------------------------------
                        // Split:
                        //
                        //     x_mul_fix
                        //       = integer * 2^frac_bits
                        //         + fraction
                        // ----------------------------------------

                        const int16_t x_mul_int =
                            static_cast<int16_t>(
                                x_mul_fix >> frac_bits);

                        // x_mul_int is derived from INT16 and
                        // frac_bits, so this multiplication is
                        // representable for the supported range.
                        //
                        // Use multiplication instead of signed
                        // left shift to avoid relying on negative
                        // signed-left-shift behavior in C++.
                        const int16_t integer_part =
                            static_cast<int16_t>(
                                static_cast<int32_t>(
                                    x_mul_int) *
                                frac_scale);

                        const int16_t x_mul_frac =
                            static_cast<int16_t>(
                                x_mul_fix - integer_part);

                        // For normal fixed-point decomposition:
                        //
                        //     0 <= x_mul_frac < 2^frac_bits
                        //
                        // therefore numerator is positive.
                        const int16_t exp_numerator =
                            static_cast<int16_t>(
                                frac_scale +
                                static_cast<int32_t>(
                                    x_mul_frac >> 1));

                        // ----------------------------------------
                        // Divide by 2^(-x_mul_int)
                        //
                        // Torch:
                        //
                        //     numerator >> (-x_mul_int)
                        //
                        // In ordinary operation x_mul_int <= 0.
                        //
                        // A severe INT16 wrap in x_mul_fix can,
                        // however, make x_mul_int > 0. PyTorch's
                        // negative right-shift count produces 0
                        // here, so preserve that final behavior.
                        // ----------------------------------------

                        int16_t x_exp;

                        if (x_mul_int > 0)
                        {
                            x_exp = 0;
                        }
                        else
                        {
                            const int32_t exp_shift =
                                -static_cast<int32_t>(
                                    x_mul_int);

                            // exp_numerator is positive.
                            //
                            // Any right shift >= 16 therefore
                            // produces zero for INT16.
                            if (exp_shift >= 16)
                            {
                                x_exp = 0;
                            }
                            else
                            {
                                x_exp =
                                    static_cast<int16_t>(
                                        exp_numerator >> exp_shift);
                            }
                        }

                        // ----------------------------------------
                        // Torch converts higher-Q representations
                        // back to Q1.7.
                        //
                        // Production I-Whisper currently uses:
                        //
                        //     frac_bits = 7
                        // ----------------------------------------

                        if (frac_bits > 7)
                        {
                            const int32_t output_value =
                                static_cast<int32_t>(
                                    x_exp) >>
                                (frac_bits - 7);

                            output[idx] =
                                static_cast<uint8_t>(
                                    output_value);
                        }
                        else
                        {
                            output[idx] =
                                static_cast<uint8_t>(
                                    x_exp);
                        }
                    }
                }
            }
        }
    }

} // namespace iwhisper