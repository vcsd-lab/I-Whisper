#include "conv1d.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

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

    return static_cast<int32_t>(value);
}


inline int64_t conv1d_output_length_simd(
    int64_t input_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    return
        (input_length
         + 2 * padding
         - kernel_size)
        / stride
        + 1;
}


inline int32_t conv1d_one_accum_simd_tail(
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

        const int8_t *weight_channel =
            weight_row
            + ic * kernel_size;

        for (int64_t k = 0;
             k < kernel_size;
             ++k)
        {
            const int64_t input_position =
                input_origin + k;

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


// Whether eight consecutive output positions can be read safely
// without padding checks.
//
// stride=1:
//   vld1_s8 reads exactly the eight required samples.
//
// stride=2:
//   vld2_s8 reads 16 bytes and returns the even-indexed eight
//   samples in val[0].  Therefore one additional byte beyond the
//   last mathematically-used sample must also be in-bounds.
inline bool can_vectorize_8(
    int64_t output_position,
    int64_t input_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    if (stride != 1
        && stride != 2)
    {
        return false;
    }

    const int64_t first_input =
        output_position * stride
        - padding;

    if (first_input < 0)
    {
        return false;
    }

    const int64_t last_output =
        output_position + 7;

    int64_t last_read =
        last_output * stride
        - padding
        + (kernel_size - 1);

    if (stride == 2)
    {
        // vld2_s8 loads the interleaved odd byte as well.
        ++last_read;
    }

    return last_read
        < input_length;
}


// Direct Conv1d microkernel over 8 output-time positions.
//
// Vectorization axis is time, not Cin*K.
// This is important for the native [B,C,T] input layout:
// neighboring time samples are contiguous in memory.
//
// Each scalar weight is reused across 8 output positions.
inline void conv1d_accum_time8(
    const int8_t *input_batch,
    const int8_t *weight_row,
    int32_t bias,
    int64_t in_channels,
    int64_t input_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_position,
    int32_t *out8)
{
    int32x4_t acc_lo =
        vdupq_n_s32(
            bias);

    int32x4_t acc_hi =
        vdupq_n_s32(
            bias);

    for (int64_t ic = 0;
         ic < in_channels;
         ++ic)
    {
        const int8_t *input_channel =
            input_batch
            + ic * input_length;

        const int8_t *weight_channel =
            weight_row
            + ic * kernel_size;

        for (int64_t k = 0;
             k < kernel_size;
             ++k)
        {
            const int64_t base =
                output_position * stride
                - padding
                + k;

            int8x8_t x8;

            if (stride == 1)
            {
                x8 =
                    vld1_s8(
                        input_channel
                        + base);
            }
            else
            {
                // stride == 2
                //
                // memory:
                //   x0, gap, x1, gap, ... x7, gap
                //
                // val[0] = x0..x7
                const int8x8x2_t deinterleaved =
                    vld2_s8(
                        input_channel
                        + base);

                x8 =
                    deinterleaved.val[0];
            }

            const int16x8_t x16 =
                vmovl_s8(
                    x8);

            const int16_t w =
                static_cast<int16_t>(
                    weight_channel[k]);

            // INT8 * INT8 fits exactly in INT16:
            // max magnitude = 16384.
            const int16x8_t product =
                vmulq_n_s16(
                    x16,
                    w);

            // INT32 lane additions wrap modulo 2^32,
            // matching the scalar/Torch accumulation semantics.
            acc_lo =
                vaddw_s16(
                    acc_lo,
                    vget_low_s16(
                        product));

            acc_hi =
                vaddw_s16(
                    acc_hi,
                    vget_high_s16(
                        product));
        }
    }

    vst1q_s32(
        out8,
        acc_lo);

    vst1q_s32(
        out8 + 4,
        acc_hi);
}


inline int8_t conv1d_requant_one(
    int32_t value,
    uint32_t multiplier,
    uint32_t shift)
{
    value =
        iwhisper::fixedpoint::requant(
            value,
            multiplier,
            shift);

    value =
        std::clamp(
            value,
            int32_t{-127},
            int32_t{127});

    return static_cast<int8_t>(
        value);
}

} // namespace


void conv1d_accum_simd(
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
        conv1d_output_length_simd(
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

            int64_t t = 0;

            while (t < output_length)
            {
                if (t + 8 <= output_length
                    && can_vectorize_8(
                        t,
                        input_length,
                        kernel_size,
                        stride,
                        padding))
                {
                    conv1d_accum_time8(
                        input_batch,
                        weight_row,
                        bias_value,
                        in_channels,
                        input_length,
                        kernel_size,
                        stride,
                        padding,
                        t,
                        output_row + t);

                    t += 8;
                }
                else
                {
                    output_row[t] =
                        conv1d_one_accum_simd_tail(
                            input_batch,
                            weight_row,
                            bias_value,
                            in_channels,
                            input_length,
                            kernel_size,
                            stride,
                            padding,
                            t);

                    ++t;
                }
            }
        }
    }
}


void conv1d_simd(
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
        conv1d_output_length_simd(
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

            int64_t t = 0;

            while (t < output_length)
            {
                if (t + 8 <= output_length
                    && can_vectorize_8(
                        t,
                        input_length,
                        kernel_size,
                        stride,
                        padding))
                {
                    alignas(16) int32_t accum[8];

                    conv1d_accum_time8(
                        input_batch,
                        weight_row,
                        bias_value,
                        in_channels,
                        input_length,
                        kernel_size,
                        stride,
                        padding,
                        t,
                        accum);

                    // Convolution dominates this kernel.
                    // Keep requant scalar in v1 so MULH/Rshift semantics
                    // reuse the already-sealed fixedpoint primitive.
                    for (int lane = 0;
                         lane < 8;
                         ++lane)
                    {
                        output_row[t + lane] =
                            conv1d_requant_one(
                                accum[lane],
                                multiplier,
                                requant_shift);
                    }

                    t += 8;
                }
                else
                {
                    const int32_t accum =
                        conv1d_one_accum_simd_tail(
                            input_batch,
                            weight_row,
                            bias_value,
                            in_channels,
                            input_length,
                            kernel_size,
                            stride,
                            padding,
                            t);

                    output_row[t] =
                        conv1d_requant_one(
                            accum,
                            multiplier,
                            requant_shift);

                    ++t;
                }
            }
        }
    }
}

} // namespace iwhisper
