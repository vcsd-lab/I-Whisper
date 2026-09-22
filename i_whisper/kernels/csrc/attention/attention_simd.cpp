#include "attention.h"
#include <arm_neon.h>
#include <cstdint>

namespace iwhisper
{

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
} // namespace iwhisper