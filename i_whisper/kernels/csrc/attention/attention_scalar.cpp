#include "attention.h"
#include "../layernorm/layernorm.h"

#include <cstdint>
#include <algorithm>
#include <stdexcept>

#include "../common/fixedpoint.h"

namespace iwhisper
{

    namespace
    {
        static inline int32_t wrap_mul_s32_attn(
            int32_t a,
            int32_t b)
        {
            return static_cast<int32_t>(
                static_cast<uint32_t>(a)
                * static_cast<uint32_t>(b));
        }




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

        // ============================================================
        // Explicit INT32 wrap helper for Torch int32 multiplication.
        //
        // Torch computes:
        //
        //     qk.int32 * multiplier.int32
        //
        // in INT32.  Production Whisper values are comfortably inside
        // the range, but keeping modulo-2^32 behavior here makes the
        // scalar reference deterministic even for boundary tests.
        // ============================================================

        static inline int32_t s32_from_bits(
            uint32_t bits)
        {
            const int64_t value =
                (bits & uint32_t{0x80000000u})
                    ? static_cast<int64_t>(bits) -
                          (int64_t{1} << 32)
                    : static_cast<int64_t>(bits);

            return static_cast<int32_t>(value);
        }

        static inline int32_t wrap_mul_s32_u8(
            int32_t a,
            uint8_t b)
        {
            const uint64_t product =
                static_cast<uint64_t>(
                    static_cast<uint32_t>(a)) *
                static_cast<uint64_t>(b);

            return s32_from_bits(
                static_cast<uint32_t>(product));
        }

        static inline int16_t clamp_s16(
            int32_t value)
        {
            if (value > 32767)
            {
                return static_cast<int16_t>(32767);
            }

            if (value < -32768)
            {
                return static_cast<int16_t>(-32768);
            }

            return static_cast<int16_t>(value);
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
    // Fused QK^T + requantization + masked row maximum.
    //
    // Torch golden:
    //
    //     qk = Q @ K^T                              # INT32
    //     qk = Rshift(qk * multiplier, shift)       # INT32
    //     qk = clamp(qk, INT16_MIN, INT16_MAX)      # INT16
    //     qk[mask] = INT16_MIN
    //     row_max_plus1 = qk.max(-1, keepdim=True) + 1
    //
    // Input:
    //     q:          [B,H,Tq,D]  INT8
    //     k:          [B,H,Tk,D]  INT8
    //     mask:       [Tq,Tk]     BOOL, shared across B/H
    //     multiplier: scalar      UINT8
    //     shift:      scalar      INT8 semantics; negative -> 0
    //
    // Output:
    //     qk_output:      [B,H,Tq,Tk] INT16
    //     row_max_output: [B,H,Tq,1]  INT16, stores max + 1
    //
    // Masked lanes do not need their dot product evaluated because the
    // Torch golden overwrites them with INT16_MIN before the reduction.
    // ============================================================

    void qk_requant_rowmax_scalar(
        const int8_t *q,
        const int8_t *k,
        const bool *mask,
        int16_t *qk_output,
        int16_t *row_max_output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim,
        uint8_t multiplier,
        int32_t shift,
        bool has_mask)
    {
        constexpr int16_t kInt16Min =
            static_cast<int16_t>(-32768);

        const uint32_t requant_shift =
            shift > 0
                ? static_cast<uint32_t>(shift)
                : uint32_t{0};

        for (int64_t b = 0; b < batch; ++b)
        {
            for (int64_t h = 0; h < heads; ++h)
            {
                const int64_t q_head_base =
                    (b * heads + h) * q_length * head_dim;

                const int64_t k_head_base =
                    (b * heads + h) * k_length * head_dim;

                const int64_t qk_head_base =
                    (b * heads + h) * q_length * k_length;

                const int64_t max_head_base =
                    (b * heads + h) * q_length;

                for (int64_t qi = 0;
                     qi < q_length;
                     ++qi)
                {
                    const int8_t *q_ptr =
                        q + q_head_base + qi * head_dim;

                    int16_t *qk_ptr =
                        qk_output + qk_head_base + qi * k_length;

                    int16_t row_max =
                        kInt16Min;

                    for (int64_t ki = 0;
                         ki < k_length;
                         ++ki)
                    {
                        const bool masked =
                            has_mask &&
                            mask[qi * k_length + ki];

                        if (masked)
                        {
                            qk_ptr[ki] = kInt16Min;
                            continue;
                        }

                        const int8_t *k_ptr =
                            k + k_head_base + ki * head_dim;

                        int32_t acc = 0;

                        for (int64_t d = 0;
                             d < head_dim;
                             ++d)
                        {
                            acc +=
                                static_cast<int32_t>(q_ptr[d]) *
                                static_cast<int32_t>(k_ptr[d]);
                        }

                        // Torch:
                        //     qk.mul(multiplier.int())
                        // Both operands are INT32. Preserve INT32 wrapping
                        // explicitly for deterministic boundary behavior.
                        const int32_t scaled =
                            wrap_mul_s32_u8(
                                acc,
                                multiplier);

                        // Torch Rshift first clamps negative shift counts to
                        // zero and returns zero for shifts >= 32.  The shared
                        // helper implements the latter two cases directly.
                        const int32_t requantized =
                            iwhisper::fixedpoint::rounding_rshift(
                                scaled,
                                requant_shift);

                        const int16_t qk_value =
                            clamp_s16(requantized);

                        qk_ptr[ki] = qk_value;

                        if (qk_value > row_max)
                        {
                            row_max = qk_value;
                        }
                    }

                    // Match Torch INT16:
                    //     row_max_plus1 = row_max + 1
                    // including 32767 -> -32768 wrapping.
                    row_max_output[max_head_base + qi] =
                        wrap_add_s16(
                            row_max,
                            int16_t{1});
                }
            }
        }
    }

    // ============================================================
    // Integer shift-exp with externally supplied row maximum.
    //
    // The preceding fused QK kernel already computes:
    //
    //     row_max_plus1 =
    //         masked_qk.max(dim=-1, keepdim=True) + 1
    //
    // Torch golden for this stage:
    //
    //     x_sub = x - row_max_plus1
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
    //         x_exp = x_exp >> (frac_bits - 7)
    //
    //     return x_exp.uint8()
    //
    // Input:
    //     input:         [B,H,Tq,Tk] INT16
    //     row_max_plus1: [B,H,Tq,1]  INT16
    //
    // Mask:
    //     [Tq,Tk] BOOL, shared across B/H
    //
    // Output:
    //     [B, H, Tq, Tk] UINT8
    //
    // Important:
    //     [B,H,Tq,Tk] UINT8
    //
    // This is now a single O(Tk) pass per row.  Row-max reduction is
    // intentionally not performed here.
    // ============================================================

    void shift_exp_scalar(
        const int16_t *input,
        const int16_t *row_max_plus1,
        const bool *mask,
        uint8_t *output,
        int32_t *row_sum_output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int32_t frac_bits,
        bool has_mask)
    {
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

                const int64_t max_head_base =
                    (b * heads + h) * q_length;

                for (int64_t qi = 0;
                     qi < q_length;
                     ++qi)
                {
                    const int64_t row_base =
                        head_base + qi * k_length;

                    const int16_t x_max = row_max_plus1[max_head_base + qi];

                    int32_t row_sum = 0;

                    for (int64_t ki = 0;
                         ki < k_length;
                         ++ki)
                    {
                        const int64_t idx =
                            row_base + ki;

                        const bool masked =
                            has_mask &&
                            mask[qi * k_length + ki];

                        // Masked attention probability is always zero.
                        // No remaining arithmetic is observable for this lane.
                        if (masked)
                        {
                            output[idx] = 0;
                            continue;
                        }

                        // Torch INT16 subtraction wraps modulo 2^16.
                        const int16_t x_sub =
                            wrap_sub_s16(
                                input[idx],
                                x_max);

                        // Normally x_sub <= -1 because x_max stores
                        // row_max + 1.  INT16 wrap at pathological
                        // boundaries can make the result positive.
                        // Torch forces those lanes to zero.
                        if (x_sub > 0)
                        {
                            output[idx] = 0;
                            continue;
                        }

                        // ----------------------------------------
                        // x_mul_fix
                        //
                        // log2(e) ~= 1 + 1/2 - 1/16 = 1.4375
                        //
                        // Keep every add/sub in INT16 to reproduce
                        // Torch's INT16 wrap semantics.
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

                        const int16_t x_mul_int =
                            static_cast<int16_t>(
                                x_mul_fix >> frac_bits);

                        // Avoid negative signed left-shift in C++.
                        const int16_t integer_part =
                            static_cast<int16_t>(
                                static_cast<int32_t>(
                                    x_mul_int) *
                                frac_scale);

                        const int16_t x_mul_frac =
                            static_cast<int16_t>(
                                x_mul_fix - integer_part);

                        const int16_t exp_numerator =
                            static_cast<int16_t>(
                                frac_scale +
                                static_cast<int32_t>(
                                    x_mul_frac >> 1));

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
                                // Equivalent to the Torch negative-shift-count
                                // boundary behavior observed by the old scalar
                                // differential test.
                                x_exp = 0;
                            }
                            else
                            {
                                x_exp =
                                    static_cast<int16_t>(
                                        exp_numerator >> exp_shift);
                            }
                        }

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

                        row_sum += static_cast<int32_t>(output[idx]);
                    }

                    // row_sum store back
                    row_sum_output[max_head_base + qi] = row_sum;
                }
            }
        }
    }

    // ============================================================
    // P @ V scalar accumulation
    //
    // p:
    //     [B,H,Tq,Tk] UINT8
    //
    // v:
    //     [B,H,D,Tk] INT8
    //
    // output:
    //     [B,H,Tq,D] INT32
    //
    // Torch golden:
    //
    //     p.to(torch.int32)
    //       @ v.to(torch.int32).transpose(-1, -2)
    //
    // Keep the accumulator as uint32 bits so addition wraps
    // modulo 2^32 without invoking C++ signed-overflow UB.
    // ============================================================
    void pv_scalar(
        const uint8_t *p,
        const int8_t *v,
        int32_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim)
    {
        for (int64_t b = 0;
             b < batch;
             ++b)
        {
            for (int64_t h = 0;
                 h < heads;
                 ++h)
            {
                const int64_t p_head_base =
                    (b * heads + h) * q_length * k_length;

                const int64_t v_head_base =
                    (b * heads + h) * head_dim * k_length;

                const int64_t out_head_base =
                    (b * heads + h) * q_length * head_dim;

                for (int64_t qi = 0;
                     qi < q_length;
                     ++qi)
                {
                    const uint8_t *p_row =
                        p + p_head_base + qi * k_length;

                    int32_t *out_row =
                        output + out_head_base + qi * head_dim;

                    for (int64_t d = 0;
                         d < head_dim;
                         ++d)
                    {
                        const int8_t *v_row =
                            v + v_head_base + d * k_length;

                        uint32_t acc_bits = 0;

                        for (int64_t ki = 0;
                             ki < k_length;
                             ++ki)
                        {
                            const int32_t product =
                                static_cast<int32_t>(
                                    p_row[ki]) *
                                static_cast<int32_t>(
                                    v_row[ki]);

                            acc_bits +=
                                static_cast<uint32_t>(
                                    product);
                        }

                        // Same two's-complement bit pattern as Torch INT32.
                        out_row[d] =
                            static_cast<int32_t>(
                                acc_bits);
                    }
                }
            }
        }
    }





    // ============================================================
    // Fused P @ V + normalize + AV requant + final layout
    //
    // Torch golden:
    //
    // out = P.int32 @ V.int32.transpose(-1, -2)
    // inv = Fast_Inverse_Square_Root(row_sum, rsqrt_offset, 2)
    // out = Rshift(out * inv, rsqrt_offset)
    // out = Rshift(out * inv, rsqrt_offset)
    // out = Rshift(out * av_multiplier, av_shift)
    // out = clamp(out, -127, 127).int8()
    // out = out.permute(0, 2, 1, 3).flatten(2)
    //
    // The INT32 P@V tensor is never materialized.
    // ============================================================
    void pv_norm_requant_scalar(
        const uint8_t *p,
        const int8_t *v,
        const int32_t *row_sum,
        uint8_t av_multiplier,
        int8_t av_shift,
        uint32_t rsqrt_offset,
        int8_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim)
    {
        constexpr int32_t kActMin = -127;
        constexpr int32_t kActMax = 127;

        const uint32_t requant_shift =
            static_cast<uint32_t>(
                std::clamp<int32_t>(
                    static_cast<int32_t>(av_shift),
                    0,
                    32));

        for (int64_t b = 0;
             b < batch;
             ++b)
        {
            for (int64_t h = 0;
                 h < heads;
                 ++h)
            {
                const int64_t p_head_base =
                    (b * heads + h)
                    * q_length
                    * k_length;

                const int64_t v_head_base =
                    (b * heads + h)
                    * head_dim
                    * k_length;

                const int64_t sum_head_base =
                    (b * heads + h)
                    * q_length;

                for (int64_t qi = 0;
                     qi < q_length;
                     ++qi)
                {
                    const int32_t sum =
                        row_sum[
                            sum_head_base
                            + qi];

                    if (sum < 0)
                    {
                        throw std::invalid_argument(
                            "row_sum must be non-negative");
                    }

                    // One inverse denominator per attention row.
                    int32_t inv = 0;

                    iwhisper::layernorm32_rsqrt_scalar(
                        &sum,
                        &inv,
                        1,
                        rsqrt_offset);

                    const uint8_t *p_row =
                        p
                        + p_head_base
                        + qi * k_length;

                    for (int64_t d = 0;
                         d < head_dim;
                         ++d)
                    {
                        const int8_t *v_row =
                            v
                            + v_head_base
                            + d * k_length;

                        // Explicit modulo-2^32 accumulation.
                        uint32_t acc_bits = 0;

                        for (int64_t ki = 0;
                             ki < k_length;
                             ++ki)
                        {
                            const int32_t product =
                                static_cast<int32_t>(
                                    p_row[ki])
                                *
                                static_cast<int32_t>(
                                    v_row[ki]);

                            acc_bits +=
                                static_cast<uint32_t>(
                                    product);
                        }

                        int32_t value =
                            static_cast<int32_t>(
                                acc_bits);

                        // Divide by sum(E) by multiplying twice by
                        // the Q(rsqrt_offset) inverse square root.
                        value =
                            iwhisper::fixedpoint::rounding_rshift(
                                wrap_mul_s32_attn(
                                    value,
                                    inv),
                                rsqrt_offset);

                        value =
                            iwhisper::fixedpoint::rounding_rshift(
                                wrap_mul_s32_attn(
                                    value,
                                    inv),
                                rsqrt_offset);

                        // Final AV requantization.
                        value =
                            iwhisper::fixedpoint::rounding_rshift(
                                wrap_mul_s32_attn(
                                    value,
                                    static_cast<int32_t>(
                                        av_multiplier)),
                                requant_shift);

                        value =
                            std::clamp(
                                value,
                                kActMin,
                                kActMax);

                        // Directly emit final Whisper attention layout:
                        // [B,Tq,H*D].
                        const int64_t out_index =
                            (b * q_length + qi)
                                * (heads * head_dim)
                            + h * head_dim
                            + d;

                        output[out_index] =
                            static_cast<int8_t>(
                                value);
                    }
                }
            }
        }
    }













} // namespace iwhisper