#include "classifier.h"
#include "../common/fixedpoint.h"

#include <algorithm>
#include <cstdint>

namespace iwhisper
{
namespace
{

inline int32_t s32_from_bits(
    uint32_t bits)
{
    const int64_t value =
        (bits & uint32_t{0x80000000u})
            ? static_cast<int64_t>(bits)
                - (int64_t{1} << 32)
            : static_cast<int64_t>(bits);

    return static_cast<int32_t>(
        value);
}


inline int32_t dot_s8_s8_wrap(
    const int8_t *x,
    const int8_t *w,
    int64_t K)
{
    uint32_t acc_bits = 0;

    for (int64_t k = 0;
         k < K;
         ++k)
    {
        const int32_t product =
            static_cast<int32_t>(
                x[k])
            *
            static_cast<int32_t>(
                w[k]);

        acc_bits +=
            static_cast<uint32_t>(
                product);
    }

    return s32_from_bits(
        acc_bits);
}

} // namespace


void classifier_accum_scalar(
    const int8_t *input,
    const int8_t *weight,
    int32_t *output,
    int64_t M,
    int64_t K,
    int64_t N)
{
    for (int64_t m = 0;
         m < M;
         ++m)
    {
        const int8_t *x_row =
            input + m * K;

        int32_t *out_row =
            output + m * N;

        for (int64_t n = 0;
             n < N;
             ++n)
        {
            const int8_t *w_row =
                weight + n * K;

            out_row[n] =
                dot_s8_s8_wrap(
                    x_row,
                    w_row,
                    K);
        }
    }
}


void classifier_scalar(
    const int8_t *input,
    const int8_t *weight,
    uint32_t multiplier,
    int8_t shift,
    int8_t *output,
    int64_t M,
    int64_t K,
    int64_t N)
{
    const uint32_t requant_shift =
        static_cast<uint32_t>(
            std::max<int32_t>(
                static_cast<int32_t>(
                    shift),
                0));

    for (int64_t m = 0;
         m < M;
         ++m)
    {
        const int8_t *x_row =
            input + m * K;

        int8_t *out_row =
            output + m * N;

        for (int64_t n = 0;
             n < N;
             ++n)
        {
            const int8_t *w_row =
                weight + n * K;

            int32_t value =
                dot_s8_s8_wrap(
                    x_row,
                    w_row,
                    K);

            value =
                iwhisper::fixedpoint::mulh(
                    value,
                    multiplier);

            value =
                iwhisper::fixedpoint::rounding_rshift(
                    value,
                    requant_shift);

            value =
                std::clamp(
                    value,
                    int32_t{-127},
                    int32_t{127});

            out_row[n] =
                static_cast<int8_t>(
                    value);
        }
    }
}

} // namespace iwhisper
