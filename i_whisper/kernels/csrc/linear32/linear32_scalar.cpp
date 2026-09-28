#include "linear32.h"
#include "../common/fixedpoint.h"

#include <cstdint>

namespace iwhisper
{

    void linear32_accum_scalar(
        const int8_t *x,
        const int8_t *weight,
        const int32_t *bias,
        const int8_t *weight_zero,
        int32_t *output,
        int64_t M,
        int64_t K,
        int64_t N,
        bool has_bias,
        bool asymmetric)
    {
        for (int64_t m = 0; m < M; ++m)
        {
            const int8_t *x_row = x + m * K;

            int32_t row_sum = 0;

            if (asymmetric)
            {
                for (int64_t k = 0; k < K; ++k)
                {
                    row_sum += static_cast<int32_t>(x_row[k]);
                }
            }

            for (int64_t n = 0; n < N; ++n)
            {
                const int8_t *w_row = weight + n * K;

                int32_t acc = 0;

                for (int64_t k = 0; k < K; ++k)
                {
                    acc +=
                        static_cast<int32_t>(x_row[k]) *
                        static_cast<int32_t>(w_row[k]);
                }

                if (has_bias)
                {
                    acc += bias[n];
                }

                if (asymmetric)
                {
                    acc -=
                        static_cast<int32_t>(weight_zero[n]) *
                        row_sum;
                }

                output[m * N + n] = acc;
            }
        }
    }

    void linear32_requant_scalar(
        const int32_t *input,
        const uint32_t *multiplier,
        const int8_t *shift,
        int32_t *output,
        int64_t M,
        int64_t N)
    {
        for (int64_t m = 0; m < M; ++m)
        {
            for (int64_t n = 0; n < N; ++n)
            {
                const int64_t idx = m * N + n;

                const int32_t scaled =
                    fixedpoint::mulh(
                        input[idx],
                        multiplier[n]);

                const int64_t widened =
                    static_cast<int64_t>(scaled);

                const int64_t shifted =
                    widened * (int64_t{1} << shift[n]);

                output[idx] =
                    static_cast<int32_t>(shifted);
            }
        }
    }

    void linear32_scalar(
        const int8_t *x,
        const int8_t *weight,
        const int32_t *bias,
        const int8_t *weight_zero,
        const uint32_t *multiplier,
        const int8_t *shift,
        int32_t *output,
        int64_t M,
        int64_t K,
        int64_t N,
        bool has_bias,
        bool asymmetric)
    {
        linear32_accum_scalar(
            x,
            weight,
            bias,
            weight_zero,
            output,
            M,
            K,
            N,
            has_bias,
            asymmetric);

        linear32_requant_scalar(
            output, // input
            multiplier,
            shift,
            output, // output
            M,
            N);
    }

} // namespace iwhisper
