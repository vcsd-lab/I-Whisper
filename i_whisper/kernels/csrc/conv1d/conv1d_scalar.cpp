#include "conv1d.h"
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


inline int64_t conv1d_output_length(
    int64_t input_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    // dilation = 1
    return
        (input_length
         + 2 * padding
         - kernel_size)
        / stride
        + 1;
}


inline int32_t conv1d_one_accum(
    const int8_t *input_batch,
    const int8_t *weight_row,
    int32_t bias,
    int64_t in_channels,
    int64_t input_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_position)
{
    // Keep the accumulation as modulo-2^32 bits.
    // This avoids signed-overflow UB and matches Torch INT32
    // accumulation / bias-add semantics.
    uint32_t acc_bits =
        static_cast<uint32_t>(
            bias);

    const int64_t input_origin =
        output_position * stride
        - padding;

    for (int64_t ic = 0;
         ic < in_channels;
         ++ic)
    {
        const int8_t *input_channel =
            input_batch
            + ic * input_length;

        // Conv1d_int stores weight flattened from
        // [Cout, Cin, K] -> [Cout, Cin*K].
        // Therefore K is the fastest-changing dimension.
        const int8_t *weight_channel =
            weight_row
            + ic * kernel_size;

        for (int64_t k = 0;
             k < kernel_size;
             ++k)
        {
            const int64_t input_position =
                input_origin + k;

            // Zero padding.
            if (input_position < 0
                || input_position >= input_length)
            {
                continue;
            }

            const int32_t product =
                static_cast<int32_t>(
                    input_channel[
                        input_position])
                *
                static_cast<int32_t>(
                    weight_channel[k]);

            acc_bits +=
                static_cast<uint32_t>(
                    product);
        }
    }

    return s32_from_bits(
        acc_bits);
}

} // namespace


void conv1d_accum_scalar(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    int32_t *output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_length,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool has_bias)
{
    const int64_t output_length =
        conv1d_output_length(
            input_length,
            kernel_size,
            stride,
            padding);

    for (int64_t b = 0;
         b < batch;
         ++b)
    {
        const int8_t *input_batch =
            input
            + b
                * in_channels
                * input_length;

        for (int64_t oc = 0;
             oc < out_channels;
             ++oc)
        {
            const int8_t *weight_row =
                weight
                + oc
                    * in_channels
                    * kernel_size;

            const int32_t bias_value =
                has_bias
                    ? bias[oc]
                    : int32_t{0};

            int32_t *output_row =
                output
                + (b * out_channels + oc)
                    * output_length;

            for (int64_t t = 0;
                 t < output_length;
                 ++t)
            {
                output_row[t] =
                    conv1d_one_accum(
                        input_batch,
                        weight_row,
                        bias_value,
                        in_channels,
                        input_length,
                        kernel_size,
                        stride,
                        padding,
                        t);
            }
        }
    }
}


void conv1d_scalar(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    uint32_t multiplier,
    int8_t shift,
    int8_t *output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_length,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool has_bias)
{
    const int64_t output_length =
        conv1d_output_length(
            input_length,
            kernel_size,
            stride,
            padding);

    const uint32_t requant_shift =
        static_cast<uint32_t>(
            std::max<int32_t>(
                static_cast<int32_t>(
                    shift),
                0));

    for (int64_t b = 0;
         b < batch;
         ++b)
    {
        const int8_t *input_batch =
            input
            + b
                * in_channels
                * input_length;

        for (int64_t oc = 0;
             oc < out_channels;
             ++oc)
        {
            const int8_t *weight_row =
                weight
                + oc
                    * in_channels
                    * kernel_size;

            const int32_t bias_value =
                has_bias
                    ? bias[oc]
                    : int32_t{0};

            int8_t *output_row =
                output
                + (b * out_channels + oc)
                    * output_length;

            for (int64_t t = 0;
                 t < output_length;
                 ++t)
            {
                int32_t value =
                    conv1d_one_accum(
                        input_batch,
                        weight_row,
                        bias_value,
                        in_channels,
                        input_length,
                        kernel_size,
                        stride,
                        padding,
                        t);

                // Exact current model_int.py requant:
                //
                //   (value.long() * multiplier.long()) >> 32
                //   then Rshift(...)
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

                output_row[t] =
                    static_cast<int8_t>(
                        value);
            }
        }
    }
}

} // namespace iwhisper
