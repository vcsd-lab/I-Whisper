#include "conv1d.h"
#include "../common/fixedpoint.h"
#include "../common/simd.h"

#include <algorithm>
#include <cstdint>
#include <ATen/Parallel.h>

namespace iwhisper
{
    namespace
    {

        inline int32_t s32_from_bits(
            uint32_t bits)
        {
            const int64_t value =
                (bits & uint32_t{0x80000000u})
                    ? static_cast<int64_t>(bits) - (int64_t{1} << 32)
                    : static_cast<int64_t>(bits);

            return static_cast<int32_t>(value);
        }

        inline int64_t conv1d_output_length_simd(
            int64_t input_length,
            int64_t kernel_size,
            int64_t stride,
            int64_t padding)
        {
            return (input_length + 2 * padding - kernel_size) / stride + 1;
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
                output_position * stride - padding;

            for (int64_t ic = 0;
                 ic < in_channels;
                 ++ic)
            {
                const int8_t *input_channel =
                    input_batch + ic * input_length;

                const int8_t *weight_channel =
                    weight_row + ic * kernel_size;

                for (int64_t k = 0;
                     k < kernel_size;
                     ++k)
                {
                    const int64_t input_position =
                        input_origin + k;

                    if (input_position < 0 || input_position >= input_length)
                    {
                        continue;
                    }

                    const int32_t product =
                        static_cast<int32_t>(
                            input_channel[input_position]) *
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
            if (stride != 1 && stride != 2)
            {
                return false;
            }

            const int64_t first_input =
                output_position * stride - padding;

            if (first_input < 0)
            {
                return false;
            }

            const int64_t last_output =
                output_position + 7;

            int64_t last_read =
                last_output * stride - padding + (kernel_size - 1);

            if (stride == 2)
            {
                // vld2_s8 loads the interleaved odd byte as well.
                ++last_read;
            }

            return last_read < input_length;
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
                    input_batch + ic * input_length;

                const int8_t *weight_channel =
                    weight_row + ic * kernel_size;

                for (int64_t k = 0;
                     k < kernel_size;
                     ++k)
                {
                    const int64_t base =
                        output_position * stride - padding + k;

                    int8x8_t x8;

                    if (stride == 1)
                    {
                        x8 =
                            vld1_s8(
                                input_channel + base);
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
                                input_channel + base);

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

        inline void conv1d_accum_time8_oc4(
            const int8_t *input_batch,
            const int8_t *weight_group,
            const int32_t *bias,
            bool has_bias,
            int64_t in_channels,
            int64_t input_length,
            int64_t kernel_size,
            int64_t stride,
            int64_t padding,
            int64_t output_position,
            int32_t *out0,
            int32_t *out1,
            int32_t *out2,
            int32_t *out3)
        {
            const int64_t weight_row_size =
                in_channels * kernel_size;

            const int8_t *weight_row0 =
                weight_group;

            const int8_t *weight_row1 =
                weight_group + weight_row_size;

            const int8_t *weight_row2 =
                weight_group + 2 * weight_row_size;

            const int8_t *weight_row3 =
                weight_group + 3 * weight_row_size;

            const int32_t bias0 =
                has_bias ? bias[0] : int32_t{0};

            const int32_t bias1 =
                has_bias ? bias[1] : int32_t{0};

            const int32_t bias2 =
                has_bias ? bias[2] : int32_t{0};

            const int32_t bias3 =
                has_bias ? bias[3] : int32_t{0};

            int32x4_t acc0_lo = vdupq_n_s32(bias0);
            int32x4_t acc0_hi = vdupq_n_s32(bias0);

            int32x4_t acc1_lo = vdupq_n_s32(bias1);
            int32x4_t acc1_hi = vdupq_n_s32(bias1);

            int32x4_t acc2_lo = vdupq_n_s32(bias2);
            int32x4_t acc2_hi = vdupq_n_s32(bias2);

            int32x4_t acc3_lo = vdupq_n_s32(bias3);
            int32x4_t acc3_hi = vdupq_n_s32(bias3);

            for (int64_t ic = 0;
                 ic < in_channels;
                 ++ic)
            {
                const int8_t *input_channel =
                    input_batch + ic * input_length;

                const int8_t *weight_channel0 =
                    weight_row0 + ic * kernel_size;

                const int8_t *weight_channel1 =
                    weight_row1 + ic * kernel_size;

                const int8_t *weight_channel2 =
                    weight_row2 + ic * kernel_size;

                const int8_t *weight_channel3 =
                    weight_row3 + ic * kernel_size;

                for (int64_t k = 0;
                     k < kernel_size;
                     ++k)
                {
                    const int64_t base =
                        output_position * stride - padding + k;

                    int8x8_t x8;

                    if (stride == 1)
                    {
                        x8 =
                            vld1_s8(
                                input_channel + base);
                    }
                    else
                    {
                        const int8x8x2_t deinterleaved =
                            vld2_s8(
                                input_channel + base);

                        x8 =
                            deinterleaved.val[0];
                    }

                    const int16x8_t x16 =
                        vmovl_s8(x8);

                    const int16x4_t x_lo =
                        vget_low_s16(x16);

                    const int16x4_t x_hi =
                        vget_high_s16(x16);

                    const int16_t w0 =
                        static_cast<int16_t>(
                            weight_channel0[k]);

                    const int16_t w1 =
                        static_cast<int16_t>(
                            weight_channel1[k]);

                    const int16_t w2 =
                        static_cast<int16_t>(
                            weight_channel2[k]);

                    const int16_t w3 =
                        static_cast<int16_t>(
                            weight_channel3[k]);

                    acc0_lo =
                        vmlal_n_s16(
                            acc0_lo,
                            x_lo,
                            w0);

                    acc0_hi =
                        vmlal_n_s16(
                            acc0_hi,
                            x_hi,
                            w0);

                    acc1_lo =
                        vmlal_n_s16(
                            acc1_lo,
                            x_lo,
                            w1);

                    acc1_hi =
                        vmlal_n_s16(
                            acc1_hi,
                            x_hi,
                            w1);

                    acc2_lo =
                        vmlal_n_s16(
                            acc2_lo,
                            x_lo,
                            w2);

                    acc2_hi =
                        vmlal_n_s16(
                            acc2_hi,
                            x_hi,
                            w2);

                    acc3_lo =
                        vmlal_n_s16(
                            acc3_lo,
                            x_lo,
                            w3);

                    acc3_hi =
                        vmlal_n_s16(
                            acc3_hi,
                            x_hi,
                            w3);
                }
            }

            vst1q_s32(out0, acc0_lo);
            vst1q_s32(out0 + 4, acc0_hi);

            vst1q_s32(out1, acc1_lo);
            vst1q_s32(out1 + 4, acc1_hi);

            vst1q_s32(out2, acc2_lo);
            vst1q_s32(out2 + 4, acc2_hi);

            vst1q_s32(out3, acc3_lo);
            vst1q_s32(out3 + 4, acc3_hi);
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
                input + b * in_channels * input_length;

            for (int64_t oc = 0;
                 oc < out_channels;
                 ++oc)
            {
                const int8_t *weight_row =
                    weight + oc * in_channels * kernel_size;

                const int32_t bias_value =
                    has_bias
                        ? bias[oc]
                        : int32_t{0};

                int32_t *output_row =
                    output + (b * out_channels + oc) * output_length;

                int64_t t = 0;

                while (t < output_length)
                {
                    if (t + 8 <= output_length && can_vectorize_8(
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

    // void conv1d_simd(
    //     const int8_t *input,
    //     const int8_t *weight,
    //     const int32_t *bias,
    //     uint32_t multiplier,
    //     int8_t shift,
    //     int8_t *output,
    //     int64_t batch,
    //     int64_t in_channels,
    //     int64_t input_length,
    //     int64_t out_channels,
    //     int64_t kernel_size,
    //     int64_t stride,
    //     int64_t padding,
    //     bool has_bias)
    // {
    //     const int64_t output_length =
    //         conv1d_output_length_simd(
    //             input_length,
    //             kernel_size,
    //             stride,
    //             padding);

    //     const uint32_t requant_shift =
    //         static_cast<uint32_t>(
    //             std::max<int32_t>(
    //                 static_cast<int32_t>(
    //                     shift),
    //                 0));

    //     const int64_t work_items =
    //         batch * out_channels;

    //     at::parallel_for(
    //         0,
    //         work_items,
    //         1,
    //         [&](int64_t begin, int64_t end)
    //         {
    //             for (int64_t work = begin;
    //                  work < end;
    //                  ++work)
    //             {
    //                 const int64_t oc =
    //                     work % out_channels;

    //                 const int64_t b =
    //                     work / out_channels;

    //                 const int8_t *input_batch =
    //                     input + b * in_channels * input_length;

    //                 const int8_t *weight_row =
    //                     weight + oc * in_channels * kernel_size;

    //                 const int32_t bias_value =
    //                     has_bias
    //                         ? bias[oc]
    //                         : int32_t{0};

    //                 int8_t *output_row =
    //                     output + (b * out_channels + oc) * output_length;

    //                 int64_t t = 0;

    //                 while (t < output_length)
    //                 {
    //                     if (t + 8 <= output_length && can_vectorize_8(
    //                                                       t,
    //                                                       input_length,
    //                                                       kernel_size,
    //                                                       stride,
    //                                                       padding))
    //                     {
    //                         alignas(16) int32_t accum[8];

    //                         conv1d_accum_time8(
    //                             input_batch,
    //                             weight_row,
    //                             bias_value,
    //                             in_channels,
    //                             input_length,
    //                             kernel_size,
    //                             stride,
    //                             padding,
    //                             t,
    //                             accum);

    //                         for (int lane = 0;
    //                              lane < 8;
    //                              ++lane)
    //                         {
    //                             output_row[t + lane] =
    //                                 conv1d_requant_one(
    //                                     accum[lane],
    //                                     multiplier,
    //                                     requant_shift);
    //                         }

    //                         t += 8;
    //                     }
    //                     else
    //                     {
    //                         const int32_t accum =
    //                             conv1d_one_accum_simd_tail(
    //                                 input_batch,
    //                                 weight_row,
    //                                 bias_value,
    //                                 in_channels,
    //                                 input_length,
    //                                 kernel_size,
    //                                 stride,
    //                                 padding,
    //                                 t);

    //                         output_row[t] =
    //                             conv1d_requant_one(
    //                                 accum,
    //                                 multiplier,
    //                                 requant_shift);

    //                         ++t;
    //                     }
    //                 }
    //             }
    //         });
    // }

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
                    static_cast<int32_t>(shift),
                    0));

        /*
         * Process output channels in groups of 4.
         *
         * Example:
         *   Cout = 512
         *
         *   group 0 -> oc 0,1,2,3
         *   group 1 -> oc 4,5,6,7
         *   ...
         *
         * If Cout is not divisible by 4, the final group
         * falls back to the original single-OC path.
         */
        const int64_t oc_groups =
            (out_channels + 3) / 4;

        const int64_t work_items =
            batch * oc_groups;

        at::parallel_for(
            0,
            work_items,
            1,
            [&](int64_t begin, int64_t end)
            {
                for (int64_t work = begin;
                     work < end;
                     ++work)
                {
                    /*
                     * Map one parallel work item to:
                     *
                     *   one batch
                     *   ×
                     *   one group of four output channels
                     */
                    const int64_t group =
                        work % oc_groups;

                    const int64_t b =
                        work / oc_groups;

                    const int64_t oc =
                        group * 4;

                    const int8_t *input_batch =
                        input +
                        b * in_channels * input_length;

                    /*
                     * ==========================================================
                     * Fast path:
                     *   four complete output channels are available.
                     * ==========================================================
                     */
                    if (oc + 3 < out_channels)
                    {
                        const int64_t weight_row_size =
                            in_channels * kernel_size;

                        /*
                         * weight_group points to weight[oc].
                         *
                         * conv1d_accum_time8_oc4() derives
                         * weight[oc+1], weight[oc+2], weight[oc+3]
                         * internally from this pointer.
                         */
                        const int8_t *weight_group =
                            weight +
                            oc * weight_row_size;

                        /*
                         * Do NOT write:
                         *
                         *     bias + oc
                         *
                         * unconditionally, because bias may be nullptr.
                         */
                        const int32_t *bias_group =
                            has_bias
                                ? bias + oc
                                : nullptr;

                        int8_t *output_row0 =
                            output +
                            (b * out_channels + oc + 0) * output_length;

                        int8_t *output_row1 =
                            output +
                            (b * out_channels + oc + 1) * output_length;

                        int8_t *output_row2 =
                            output +
                            (b * out_channels + oc + 2) * output_length;

                        int8_t *output_row3 =
                            output +
                            (b * out_channels + oc + 3) * output_length;

                        int64_t t = 0;

                        while (t < output_length)
                        {
                            /*
                             * --------------------------------------------------
                             * 4 OC × 8 time fast path
                             * --------------------------------------------------
                             *
                             * Same x8 input vector is reused by:
                             *
                             *   oc + 0
                             *   oc + 1
                             *   oc + 2
                             *   oc + 3
                             */
                            if (t + 8 <= output_length &&
                                can_vectorize_8(
                                    t,
                                    input_length,
                                    kernel_size,
                                    stride,
                                    padding))
                            {
                                alignas(16) int32_t accum0[8];
                                alignas(16) int32_t accum1[8];
                                alignas(16) int32_t accum2[8];
                                alignas(16) int32_t accum3[8];

                                conv1d_accum_time8_oc4(
                                    input_batch,
                                    weight_group,
                                    bias_group,
                                    has_bias,
                                    in_channels,
                                    input_length,
                                    kernel_size,
                                    stride,
                                    padding,
                                    t,
                                    accum0,
                                    accum1,
                                    accum2,
                                    accum3);

                                /*
                                 * Requantization stays operator-specific
                                 * and unchanged.
                                 */
                                for (int lane = 0;
                                     lane < 8;
                                     ++lane)
                                {
                                    output_row0[t + lane] =
                                        conv1d_requant_one(
                                            accum0[lane],
                                            multiplier,
                                            requant_shift);

                                    output_row1[t + lane] =
                                        conv1d_requant_one(
                                            accum1[lane],
                                            multiplier,
                                            requant_shift);

                                    output_row2[t + lane] =
                                        conv1d_requant_one(
                                            accum2[lane],
                                            multiplier,
                                            requant_shift);

                                    output_row3[t + lane] =
                                        conv1d_requant_one(
                                            accum3[lane],
                                            multiplier,
                                            requant_shift);
                                }

                                t += 8;
                            }

                            /*
                             * --------------------------------------------------
                             * Time / padding tail
                             * --------------------------------------------------
                             *
                             * We intentionally keep the original scalar
                             * accumulation semantics here.
                             *
                             * This handles:
                             *   - left padding
                             *   - right padding
                             *   - final <8 output positions
                             */
                            else
                            {
                                const int32_t accum0 =
                                    conv1d_one_accum_simd_tail(
                                        input_batch,
                                        weight_group +
                                            0 * weight_row_size,
                                        has_bias
                                            ? bias[oc + 0]
                                            : int32_t{0},
                                        in_channels,
                                        input_length,
                                        kernel_size,
                                        stride,
                                        padding,
                                        t);

                                const int32_t accum1 =
                                    conv1d_one_accum_simd_tail(
                                        input_batch,
                                        weight_group +
                                            1 * weight_row_size,
                                        has_bias
                                            ? bias[oc + 1]
                                            : int32_t{0},
                                        in_channels,
                                        input_length,
                                        kernel_size,
                                        stride,
                                        padding,
                                        t);

                                const int32_t accum2 =
                                    conv1d_one_accum_simd_tail(
                                        input_batch,
                                        weight_group +
                                            2 * weight_row_size,
                                        has_bias
                                            ? bias[oc + 2]
                                            : int32_t{0},
                                        in_channels,
                                        input_length,
                                        kernel_size,
                                        stride,
                                        padding,
                                        t);

                                const int32_t accum3 =
                                    conv1d_one_accum_simd_tail(
                                        input_batch,
                                        weight_group +
                                            3 * weight_row_size,
                                        has_bias
                                            ? bias[oc + 3]
                                            : int32_t{0},
                                        in_channels,
                                        input_length,
                                        kernel_size,
                                        stride,
                                        padding,
                                        t);

                                output_row0[t] =
                                    conv1d_requant_one(
                                        accum0,
                                        multiplier,
                                        requant_shift);

                                output_row1[t] =
                                    conv1d_requant_one(
                                        accum1,
                                        multiplier,
                                        requant_shift);

                                output_row2[t] =
                                    conv1d_requant_one(
                                        accum2,
                                        multiplier,
                                        requant_shift);

                                output_row3[t] =
                                    conv1d_requant_one(
                                        accum3,
                                        multiplier,
                                        requant_shift);

                                ++t;
                            }
                        }
                    }

                    /*
                     * ==========================================================
                     * Output-channel tail
                     * ==========================================================
                     *
                     * Example:
                     *
                     *   Cout = 7
                     *
                     *   group 0 -> oc 0,1,2,3    fast path
                     *   group 1 -> oc 4,5,6      this path
                     *
                     * Keep original single-OC implementation.
                     */
                    else
                    {
                        const int64_t oc_end =
                            std::min<int64_t>(
                                oc + 4,
                                out_channels);

                        for (int64_t tail_oc = oc;
                             tail_oc < oc_end;
                             ++tail_oc)
                        {
                            const int8_t *weight_row =
                                weight +
                                tail_oc *
                                    in_channels *
                                    kernel_size;

                            const int32_t bias_value =
                                has_bias
                                    ? bias[tail_oc]
                                    : int32_t{0};

                            int8_t *output_row =
                                output +
                                (b * out_channels + tail_oc) * output_length;

                            int64_t t = 0;

                            while (t < output_length)
                            {
                                /*
                                 * Original 1 OC × 8 time path.
                                 */
                                if (t + 8 <= output_length &&
                                    can_vectorize_8(
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

                                /*
                                 * Original scalar time tail.
                                 */
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
            });
    }

} // namespace iwhisper
