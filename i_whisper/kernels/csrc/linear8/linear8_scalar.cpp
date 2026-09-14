#include "linear8.h"
#include "../common/fixedpoint.h"

#include <algorithm>
#include <cstdint>

namespace iwhisper
{

    /**
     * small helper function to accumulate one row of x and one row of weight
     */
    static inline int32_t accumulate_one_row(
        const int8_t *x,
        const int8_t *weight,
        int64_t K)
    {
        int32_t acc = 0;

        for (int64_t k = 0; k < K; ++k)
        {
            acc +=
                static_cast<int32_t>(x[k]) *
                static_cast<int32_t>(weight[k]);
        }

        return acc;
    }

    void linear8_scalar(
        const int8_t *x,
        const int8_t *weight,
        const int32_t *bias,
        const int8_t *weight_zero,
        const uint32_t *multiplier,
        const int8_t *shift,

        int8_t *output,

        int64_t M,
        int64_t K,
        int64_t N,

        bool has_bias,
        bool asymmetric)
    {
        for (int64_t m = 0; m < M; ++m)
        {
            for (int64_t n = 0; n < N; ++n)
            {

                int32_t acc = accumulate_one_row(
                    &x[m * K],
                    &weight[n * K],
                    K);

                if (asymmetric)
                {
                    int32_t x_sum_acc = 0;
                    for (int64_t k = 0; k < K; ++k)
                    {
                        x_sum_acc += static_cast<int32_t>(x[m * K + k]);
                    }
                    acc -= static_cast<int32_t>(weight_zero[n]) * x_sum_acc;
                }
                if (has_bias)
                {
                    acc += bias[n]; // Assuming bias is a single value for the row
                }

                acc = fixedpoint::requant(
                    acc,
                    multiplier[n],
                    static_cast<uint32_t>(shift[n]));

                // Clamp (Saturate) the result to int8_t range
                acc = std::clamp(
                        acc,
                        int32_t{-127},
                        int32_t{127});

                output[m * N + n] = static_cast<int8_t>(acc);
            }
        }
    }

    void linear8_accum_scalar(
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
            for (int64_t n = 0; n < N; ++n)
            {

                int32_t acc = accumulate_one_row(
                    &x[m * K],
                    &weight[n * K],
                    K);

                if (asymmetric)
                {
                    int32_t x_sum_acc = 0;
                    for (int64_t k = 0; k < K; ++k)
                    {
                        x_sum_acc += static_cast<int32_t>(x[m * K + k]);
                    }
                    acc -= static_cast<int32_t>(weight_zero[n]) * x_sum_acc;
                }
                if (has_bias)
                {
                    acc += bias[n]; // Assuming bias is a single value for the row
                }

                output[m * N + n] = acc;
            }
        }
    }

    void linear8_requant_scalar(
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
                const int64_t index = m * N + n;

                output[index] = fixedpoint::requant(
                    input[index],
                    multiplier[n],
                    static_cast<uint32_t>(shift[n]));
            }
        }
    }

} // namespace iwhisper