#include "attention.h"
#include "../common/simd.h"
#include "../common/fixedpoint.h"
#include "../layernorm/layernorm.h"

#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace iwhisper
{
    namespace
    {

        // ============================================================
        // INT8 x INT8 dot product.
        //
        // 16 elements / iteration:
        //     s8 -> s16 products
        //     pairwise widen s16 -> s32
        //
        // Uses baseline NEON rather than the optional DOTPROD extension,
        // so the same source remains usable through SIMDe on x86.
        // ============================================================
        static inline int32_t dot_s8_neon(
            const int8_t *a,
            const int8_t *b,
            int64_t length)
        {
            int32x4_t acc = vdupq_n_s32(0);

            int64_t d = 0;

            for (; d + 16 <= length; d += 16)
            {
                const int8x16_t va =
                    vld1q_s8(a + d);

                const int8x16_t vb =
                    vld1q_s8(b + d);

#ifdef IWHISPER_USE_DOTPROD
                acc = vdotq_s32(
                    acc,
                    va,
                    vb);
#else
                const int16x8_t prod_lo =
                    vmull_s8(
                        vget_low_s8(va),
                        vget_low_s8(vb));

                const int16x8_t prod_hi =
                    vmull_s8(
                        vget_high_s8(va),
                        vget_high_s8(vb));

                acc =
                    vaddq_s32(
                        acc,
                        vpaddlq_s16(prod_lo));

                acc =
                    vaddq_s32(
                        acc,
                        vpaddlq_s16(prod_hi));
#endif
            }

            int32_t sum =
                vaddvq_s32(acc);

            for (; d < length; ++d)
            {
                sum +=
                    static_cast<int32_t>(a[d]) *
                    static_cast<int32_t>(b[d]);
            }

            return sum;
        }

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
            return s16_from_bits(
                static_cast<uint16_t>(
                    static_cast<uint16_t>(a) + static_cast<uint16_t>(b)));
        }

        static inline int16_t wrap_sub_s16(
            int16_t a,
            int16_t b)
        {
            return s16_from_bits(
                static_cast<uint16_t>(
                    static_cast<uint16_t>(a) - static_cast<uint16_t>(b)));
        }

        static inline int32_t s32_from_bits(
            uint32_t bits)
        {
            const int64_t value =
                (bits & uint32_t{0x80000000u})
                    ? static_cast<int64_t>(bits) - (int64_t{1} << 32)
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
            return static_cast<int16_t>(
                std::clamp(
                    value,
                    int32_t{-32768},
                    int32_t{32767}));
        }

        static inline uint8_t shift_exp_one_scalar(
            int16_t input,
            int16_t x_max,
            int32_t frac_bits)
        {
            const int32_t frac_scale =
                int32_t{1} << frac_bits;

            const int16_t x_sub =
                wrap_sub_s16(input, x_max);

            if (x_sub > 0)
                return uint8_t{0};

            const int16_t half =
                static_cast<int16_t>(x_sub >> 1);
            const int16_t sixteenth =
                static_cast<int16_t>(x_sub >> 4);

            const int16_t x_mul_fix =
                wrap_sub_s16(
                    wrap_add_s16(x_sub, half),
                    sixteenth);

            const int16_t x_mul_int =
                static_cast<int16_t>(
                    x_mul_fix >> frac_bits);

            const int16_t integer_part =
                static_cast<int16_t>(
                    static_cast<int32_t>(x_mul_int) * frac_scale);

            const int16_t x_mul_frac =
                static_cast<int16_t>(
                    x_mul_fix - integer_part);

            const int16_t exp_numerator =
                static_cast<int16_t>(
                    frac_scale + static_cast<int32_t>(
                                     x_mul_frac >> 1));

            int16_t x_exp = 0;

            if (x_mul_int <= 0)
            {
                const int32_t exp_shift =
                    -static_cast<int32_t>(x_mul_int);

                if (exp_shift < 16)
                {
                    x_exp =
                        static_cast<int16_t>(
                            exp_numerator >> exp_shift);
                }
            }

            if (frac_bits > 7)
            {
                x_exp =
                    static_cast<int16_t>(
                        static_cast<int32_t>(x_exp) >> (frac_bits - 7));
            }

            return static_cast<uint8_t>(x_exp);
        }

        // ============================================================
        // Bit-exact vector Rshift.
        //
        // Matches fixedpoint::rounding_rshift():
        //   shift == 0  -> identity
        //   1..31       -> rounded arithmetic right shift
        //   shift >= 32 -> zero
        // ============================================================
        static inline int32x4_t rounding_rshift_s32x4_attn(
            int32x4_t x,
            uint32_t shift)
        {
            if (shift == 0)
            {
                return x;
            }

            if (shift >= 32)
            {
                return vdupq_n_s32(0);
            }

            const uint32x4_t ux =
                vreinterpretq_u32_s32(x);

            const uint32x4_t sign =
                vshrq_n_u32(
                    ux,
                    31);

            const uint32x4_t rounding =
                vsubq_u32(
                    vdupq_n_u32(
                        uint32_t{1}
                        << (shift - 1)),
                    sign);

            const uint32x4_t rounded =
                vaddq_u32(
                    ux,
                    rounding);

            const int32x4_t shift_vec =
                vdupq_n_s32(
                    -static_cast<int32_t>(
                        shift));

            return vshlq_s32(
                vreinterpretq_s32_u32(
                    rounded),
                shift_vec);
        }

        // ============================================================
        // Four-output UINT8 x INT8 dot microkernel.
        //
        // Computes:
        //
        //   out_j = sum_k p[k] * v_j[k], j=0..3
        //
        // with explicit modulo-2^32 accumulation.
        //
        // DOTPROD path:
        //
        //   p_u8 = p_s8 + 128
        //
        // therefore
        //
        //   sum(p_u8 * v)
        // = sum(p_s8 * v) + 128 * sum(v)
        //
        // where p_s8 is obtained exactly with XOR 0x80.
        //
        // Baseline path widens UINT8 to signed INT16 (0..255 fits)
        // and multiplies by widened signed INT8 directly.
        // ============================================================
        static inline void dot_u8_s8_4_neon(
            const uint8_t *p,
            const int8_t *v0,
            const int8_t *v1,
            const int8_t *v2,
            const int8_t *v3,
            int64_t length,
            int32_t &out0,
            int32_t &out1,
            int32_t &out2,
            int32_t &out3)
        {
            int64_t k = 0;

#ifdef IWHISPER_USE_DOTPROD
            int32x4_t acc0 =
                vdupq_n_s32(0);
            int32x4_t acc1 =
                vdupq_n_s32(0);
            int32x4_t acc2 =
                vdupq_n_s32(0);
            int32x4_t acc3 =
                vdupq_n_s32(0);

            int32_t sum_v0 = 0;
            int32_t sum_v1 = 0;
            int32_t sum_v2 = 0;
            int32_t sum_v3 = 0;

            const uint8x16_t sign_flip =
                vdupq_n_u8(
                    uint8_t{0x80});

            for (;
                 k + 16 <= length;
                 k += 16)
            {
                const uint8x16_t pu =
                    vld1q_u8(
                        p + k);

                // XOR 0x80 maps:
                //   [0,255] -> [-128,127]
                // exactly as p_signed = p_unsigned - 128.
                const int8x16_t ps =
                    vreinterpretq_s8_u8(
                        veorq_u8(
                            pu,
                            sign_flip));

                const int8x16_t x0 =
                    vld1q_s8(v0 + k);
                const int8x16_t x1 =
                    vld1q_s8(v1 + k);
                const int8x16_t x2 =
                    vld1q_s8(v2 + k);
                const int8x16_t x3 =
                    vld1q_s8(v3 + k);

                acc0 =
                    vdotq_s32(
                        acc0,
                        ps,
                        x0);
                acc1 =
                    vdotq_s32(
                        acc1,
                        ps,
                        x1);
                acc2 =
                    vdotq_s32(
                        acc2,
                        ps,
                        x2);
                acc3 =
                    vdotq_s32(
                        acc3,
                        ps,
                        x3);

                // Widening horizontal sums are exact for 16 INT8 lanes.
                sum_v0 +=
                    static_cast<int32_t>(
                        vaddlvq_s8(x0));
                sum_v1 +=
                    static_cast<int32_t>(
                        vaddlvq_s8(x1));
                sum_v2 +=
                    static_cast<int32_t>(
                        vaddlvq_s8(x2));
                sum_v3 +=
                    static_cast<int32_t>(
                        vaddlvq_s8(x3));
            }

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

            bits0 +=
                static_cast<uint32_t>(
                    sum_v0) *
                uint32_t{128};
            bits1 +=
                static_cast<uint32_t>(
                    sum_v1) *
                uint32_t{128};
            bits2 +=
                static_cast<uint32_t>(
                    sum_v2) *
                uint32_t{128};
            bits3 +=
                static_cast<uint32_t>(
                    sum_v3) *
                uint32_t{128};

#else
            int32x4_t acc0 =
                vdupq_n_s32(0);
            int32x4_t acc1 =
                vdupq_n_s32(0);
            int32x4_t acc2 =
                vdupq_n_s32(0);
            int32x4_t acc3 =
                vdupq_n_s32(0);

            for (;
                 k + 8 <= length;
                 k += 8)
            {
                const uint8x8_t p8 =
                    vld1_u8(
                        p + k);

                const int16x8_t p16 =
                    vreinterpretq_s16_u16(
                        vmovl_u8(
                            p8));

                const int16x8_t x0 =
                    vmovl_s8(
                        vld1_s8(v0 + k));
                const int16x8_t x1 =
                    vmovl_s8(
                        vld1_s8(v1 + k));
                const int16x8_t x2 =
                    vmovl_s8(
                        vld1_s8(v2 + k));
                const int16x8_t x3 =
                    vmovl_s8(
                        vld1_s8(v3 + k));

                acc0 =
                    vaddq_s32(
                        acc0,
                        vpaddlq_s16(
                            vmulq_s16(
                                p16,
                                x0)));
                acc1 =
                    vaddq_s32(
                        acc1,
                        vpaddlq_s16(
                            vmulq_s16(
                                p16,
                                x1)));
                acc2 =
                    vaddq_s32(
                        acc2,
                        vpaddlq_s16(
                            vmulq_s16(
                                p16,
                                x2)));
                acc3 =
                    vaddq_s32(
                        acc3,
                        vpaddlq_s16(
                            vmulq_s16(
                                p16,
                                x3)));
            }

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
#endif

            // Scalar tail uses the original UINT8 x INT8 product,
            // so no zero-point correction is required here.
            for (;
                 k < length;
                 ++k)
            {
                const int32_t pv =
                    static_cast<int32_t>(
                        p[k]);

                bits0 +=
                    static_cast<uint32_t>(
                        pv * static_cast<int32_t>(
                                 v0[k]));
                bits1 +=
                    static_cast<uint32_t>(
                        pv * static_cast<int32_t>(
                                 v1[k]));
                bits2 +=
                    static_cast<uint32_t>(
                        pv * static_cast<int32_t>(
                                 v2[k]));
                bits3 +=
                    static_cast<uint32_t>(
                        pv * static_cast<int32_t>(
                                 v3[k]));
            }

            out0 =
                s32_from_bits(bits0);
            out1 =
                s32_from_bits(bits1);
            out2 =
                s32_from_bits(bits2);
            out3 =
                s32_from_bits(bits3);
        }

        // Scalar-D tail helper.  Production Whisper D=64, so the
        // four-output path handles the full head in normal inference.
        static inline int32_t dot_u8_s8_tail_scalar(
            const uint8_t *p,
            const int8_t *v,
            int64_t length)
        {
            uint32_t bits = 0;

            for (int64_t k = 0;
                 k < length;
                 ++k)
            {
                bits +=
                    static_cast<uint32_t>(
                        static_cast<int32_t>(
                            p[k]) *
                        static_cast<int32_t>(
                            v[k]));
            }

            return s32_from_bits(bits);
        }

        static inline int32x4_t
        postprocess_pv_s32x4(
            int32x4_t value,
            int32_t inv,
            uint32_t rsqrt_offset,
            uint8_t av_multiplier,
            uint32_t requant_shift)
        {
            const int32x4_t inv_vec =
                vdupq_n_s32(
                    inv);

            value =
                vmulq_s32(
                    value,
                    inv_vec);

            value =
                rounding_rshift_s32x4_attn(
                    value,
                    rsqrt_offset);

            value =
                vmulq_s32(
                    value,
                    inv_vec);

            value =
                rounding_rshift_s32x4_attn(
                    value,
                    rsqrt_offset);

            value =
                vmulq_n_s32(
                    value,
                    static_cast<int32_t>(
                        av_multiplier));

            value =
                rounding_rshift_s32x4_attn(
                    value,
                    requant_shift);

            const int32x4_t min_vec =
                vdupq_n_s32(-127);

            const int32x4_t max_vec =
                vdupq_n_s32(127);

            return vminq_s32(
                vmaxq_s32(
                    value,
                    min_vec),
                max_vec);
        }

    } // namespace

    void qk_simd(
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
                    (b * heads + h) * q_length * head_dim;

                const int64_t k_head_base =
                    (b * heads + h) * k_length * head_dim;

                const int64_t out_head_base =
                    (b * heads + h) * q_length * k_length;

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

                        out_ptr[ki] =
                            dot_s8_neon(
                                q_ptr,
                                k_ptr,
                                head_dim);
                    }
                }
            }
        }
    }

    // ============================================================
    // Fused SIMD QK^T + requantization + masked row maximum.
    //
    // Same contract / arithmetic semantics as
    // qk_requant_rowmax_scalar().
    //
    // Only the D-dimensional dot product is vectorized in this
    // first production SIMD version.  Requantization, masking and
    // row-max reduction remain scalar so the numerical boundary is
    // identical to the verified scalar kernel.
    // ============================================================
    void qk_requant_rowmax_simd(
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

        for (int64_t b = 0;
             b < batch;
             ++b)
        {
            for (int64_t h = 0;
                 h < heads;
                 ++h)
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
                            has_mask && mask[qi * k_length + ki];

                        if (masked)
                        {
                            qk_ptr[ki] =
                                kInt16Min;
                            continue;
                        }

                        const int8_t *k_ptr =
                            k + k_head_base + ki * head_dim;

                        const int32_t acc =
                            dot_s8_neon(
                                q_ptr,
                                k_ptr,
                                head_dim);

                        const int32_t scaled =
                            wrap_mul_s32_u8(
                                acc,
                                multiplier);

                        const int32_t requantized =
                            iwhisper::fixedpoint::rounding_rshift(
                                scaled,
                                requant_shift);

                        const int16_t qk_value =
                            clamp_s16(
                                requantized);

                        qk_ptr[ki] =
                            qk_value;

                        if (qk_value > row_max)
                        {
                            row_max =
                                qk_value;
                        }
                    }

                    row_max_output[max_head_base + qi] =
                        wrap_add_s16(
                            row_max,
                            int16_t{1});
                }
            }
        }
    }

    void shift_exp_simd(
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
        const int16_t frac_scale =
            static_cast<int16_t>(
                int32_t{1} << frac_bits);

        const int16x8_t zero_s16 =
            vdupq_n_s16(0);
        const int16x8_t minus_sixteen =
            vdupq_n_s16(-16);
        const int16x8_t minus_fifteen =
            vdupq_n_s16(-15);
        const int16x8_t frac_scale_vec =
            vdupq_n_s16(frac_scale);
        const int16x8_t frac_shift_vec =
            vdupq_n_s16(
                static_cast<int16_t>(-frac_bits));
        const int16x8_t out_shift_vec =
            vdupq_n_s16(
                static_cast<int16_t>(
                    -std::max(frac_bits - 7, 0)));

        for (int64_t b = 0; b < batch; ++b)
        {
            for (int64_t h = 0; h < heads; ++h)
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
                    const int16_t x_max =
                        row_max_plus1[max_head_base + qi];
                    const int16x8_t x_max_vec =
                        vdupq_n_s16(x_max);

                    uint32x4_t sum_vec =
                        vdupq_n_u32(0);
                    int64_t ki = 0;

                    for (;
                         ki + 8 <= k_length;
                         ki += 8)
                    {
                        const int64_t idx =
                            row_base + ki;
                        const int16x8_t x =
                            vld1q_s16(input + idx);

                        const int16x8_t x_sub =
                            vsubq_s16(x, x_max_vec);

                        uint16x8_t active =
                            vcleq_s16(
                                x_sub,
                                zero_s16);

                        if (has_mask)
                        {
                            const uint8x8_t mask8 =
                                vld1_u8(
                                    reinterpret_cast<
                                        const uint8_t *>(
                                        mask + qi * k_length + ki));

                            const uint16x8_t mask16 =
                                vmovl_u8(mask8);
                            const uint16x8_t unmasked =
                                vceqq_u16(
                                    mask16,
                                    vdupq_n_u16(0));
                            active =
                                vandq_u16(
                                    active,
                                    unmasked);
                        }

                        const int16x8_t half =
                            vshrq_n_s16(x_sub, 1);
                        const int16x8_t sixteenth =
                            vshrq_n_s16(x_sub, 4);
                        const int16x8_t add_half =
                            vaddq_s16(x_sub, half);
                        const int16x8_t x_mul_fix =
                            vsubq_s16(
                                add_half,
                                sixteenth);

                        const int16x8_t x_mul_int =
                            vshlq_s16(
                                x_mul_fix,
                                frac_shift_vec);
                        const int16x8_t integer_part =
                            vmulq_s16(
                                x_mul_int,
                                frac_scale_vec);
                        const int16x8_t x_mul_frac =
                            vsubq_s16(
                                x_mul_fix,
                                integer_part);
                        const int16x8_t exp_numerator =
                            vaddq_s16(
                                frac_scale_vec,
                                vshrq_n_s16(
                                    x_mul_frac,
                                    1));

                        const uint16x8_t exp_range_valid =
                            vandq_u16(
                                vcleq_s16(
                                    x_mul_int,
                                    zero_s16),
                                vcgtq_s16(
                                    x_mul_int,
                                    minus_sixteen));

                        active =
                            vandq_u16(
                                active,
                                exp_range_valid);

                        int16x8_t safe_exp_shift =
                            vmaxq_s16(
                                x_mul_int,
                                minus_fifteen);
                        safe_exp_shift =
                            vminq_s16(
                                safe_exp_shift,
                                zero_s16);

                        int16x8_t x_exp =
                            vshlq_s16(
                                exp_numerator,
                                safe_exp_shift);

                        if (frac_bits > 7)
                        {
                            x_exp =
                                vshlq_s16(
                                    x_exp,
                                    out_shift_vec);
                        }

                        x_exp =
                            vandq_s16(
                                x_exp,
                                vreinterpretq_s16_u16(
                                    active));

                        const uint8x8_t out8 =
                            vqmovun_s16(x_exp);

                        vst1_u8(
                            output + idx,
                            out8);

                        const uint16x8_t out16 =
                            vmovl_u8(out8);

                        sum_vec =
                            vaddq_u32(
                                sum_vec,
                                vmovl_u16(
                                    vget_low_u16(
                                        out16)));
                        sum_vec =
                            vaddq_u32(
                                sum_vec,
                                vmovl_u16(
                                    vget_high_u16(
                                        out16)));
                    }

                    uint32_t row_sum_bits =
                        vaddvq_u32(sum_vec);

                    for (;
                         ki < k_length;
                         ++ki)
                    {
                        const int64_t idx =
                            row_base + ki;
                        const bool masked =
                            has_mask && mask[qi * k_length + ki];

                        uint8_t value = 0;
                        if (!masked)
                        {
                            value =
                                shift_exp_one_scalar(
                                    input[idx],
                                    x_max,
                                    frac_bits);
                        }

                        output[idx] = value;
                        row_sum_bits +=
                            static_cast<uint32_t>(
                                value);
                    }

                    row_sum_output[max_head_base + qi] =
                        static_cast<int32_t>(
                            row_sum_bits);
                }
            }
        }
    }

    // ============================================================
    // Raw P @ V SIMD debug/reference kernel.
    //
    // p:      [B,H,Tq,Tk] UINT8
    // v:      [B,H,D,Tk]  INT8
    // output: [B,H,Tq,D]  INT32
    // ============================================================
    void pv_simd(
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

                    int64_t d = 0;

                    for (;
                         d + 4 <= head_dim;
                         d += 4)
                    {
                        const int8_t *v0 =
                            v + v_head_base + (d + 0) * k_length;
                        const int8_t *v1 =
                            v + v_head_base + (d + 1) * k_length;
                        const int8_t *v2 =
                            v + v_head_base + (d + 2) * k_length;
                        const int8_t *v3 =
                            v + v_head_base + (d + 3) * k_length;

                        dot_u8_s8_4_neon(
                            p_row,
                            v0,
                            v1,
                            v2,
                            v3,
                            k_length,
                            out_row[d + 0],
                            out_row[d + 1],
                            out_row[d + 2],
                            out_row[d + 3]);
                    }

                    for (;
                         d < head_dim;
                         ++d)
                    {
                        const int8_t *v_row =
                            v + v_head_base + d * k_length;

                        out_row[d] =
                            dot_u8_s8_tail_scalar(
                                p_row,
                                v_row,
                                k_length);
                    }
                }
            }
        }
    }

    // ============================================================
    // Fused SIMD P @ V + normalize + AV requant + final layout.
    //
    // The INT32 PV tensor is never materialized.
    // FISR remains scalar: one invocation per (B,H,Tq) row.
    // Four D outputs are accumulated and postprocessed together.
    // ============================================================
    void pv_norm_requant_simd(
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
        const uint32_t requant_shift =
            static_cast<uint32_t>(
                std::clamp<int32_t>(
                    static_cast<int32_t>(
                        av_shift),
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
                    (b * heads + h) * q_length * k_length;

                const int64_t v_head_base =
                    (b * heads + h) * head_dim * k_length;

                const int64_t sum_head_base =
                    (b * heads + h) * q_length;

                for (int64_t qi = 0;
                     qi < q_length;
                     ++qi)
                {
                    const int32_t sum =
                        row_sum[sum_head_base + qi];

                    if (sum < 0)
                    {
                        throw std::invalid_argument(
                            "row_sum must be non-negative");
                    }

                    int32_t inv = 0;

                    iwhisper::layernorm32_rsqrt_scalar(
                        &sum,
                        &inv,
                        1,
                        rsqrt_offset);

                    const uint8_t *p_row =
                        p + p_head_base + qi * k_length;

                    int8_t *out_row =
                        output + (b * q_length + qi) * (heads * head_dim) + h * head_dim;

                    int64_t d = 0;

                    for (;
                         d + 4 <= head_dim;
                         d += 4)
                    {
                        const int8_t *v0 =
                            v + v_head_base + (d + 0) * k_length;
                        const int8_t *v1 =
                            v + v_head_base + (d + 1) * k_length;
                        const int8_t *v2 =
                            v + v_head_base + (d + 2) * k_length;
                        const int8_t *v3 =
                            v + v_head_base + (d + 3) * k_length;

                        alignas(16) int32_t dots[4];

                        dot_u8_s8_4_neon(
                            p_row,
                            v0,
                            v1,
                            v2,
                            v3,
                            k_length,
                            dots[0],
                            dots[1],
                            dots[2],
                            dots[3]);

                        int32x4_t values =
                            vld1q_s32(
                                dots);

                        values =
                            postprocess_pv_s32x4(
                                values,
                                inv,
                                rsqrt_offset,
                                av_multiplier,
                                requant_shift);

                        // Values are already clamped to INT8 range.
                        out_row[d + 0] =
                            static_cast<int8_t>(
                                vgetq_lane_s32(
                                    values,
                                    0));
                        out_row[d + 1] =
                            static_cast<int8_t>(
                                vgetq_lane_s32(
                                    values,
                                    1));
                        out_row[d + 2] =
                            static_cast<int8_t>(
                                vgetq_lane_s32(
                                    values,
                                    2));
                        out_row[d + 3] =
                            static_cast<int8_t>(
                                vgetq_lane_s32(
                                    values,
                                    3));
                    }

                    // D tail.  Production Whisper D=64, so this path
                    // normally does not execute.
                    for (;
                         d < head_dim;
                         ++d)
                    {
                        const int8_t *v_row =
                            v + v_head_base + d * k_length;

                        int32_t value =
                            dot_u8_s8_tail_scalar(
                                p_row,
                                v_row,
                                k_length);

                        value =
                            iwhisper::fixedpoint::rounding_rshift(
                                s32_from_bits(
                                    static_cast<uint32_t>(
                                        value) *
                                    static_cast<uint32_t>(
                                        inv)),
                                rsqrt_offset);

                        value =
                            iwhisper::fixedpoint::rounding_rshift(
                                s32_from_bits(
                                    static_cast<uint32_t>(
                                        value) *
                                    static_cast<uint32_t>(
                                        inv)),
                                rsqrt_offset);

                        value =
                            iwhisper::fixedpoint::rounding_rshift(
                                s32_from_bits(
                                    static_cast<uint32_t>(
                                        value) *
                                    static_cast<uint32_t>(
                                        av_multiplier)),
                                requant_shift);

                        value =
                            std::clamp(
                                value,
                                int32_t{-127},
                                int32_t{127});

                        out_row[d] =
                            static_cast<int8_t>(
                                value);
                    }
                }
            }
        }
    }

} // namespace iwhisper