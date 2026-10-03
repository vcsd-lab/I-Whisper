#include "gelu.h"
#include "../common/fixedpoint.h"

#include <algorithm>
#include <cstdint>

namespace iwhisper
{

void gelu_scalar(
    const int8_t *input,
    const uint8_t *lut,
    const uint8_t *luti_multiplier,
    const int8_t *luti_shift,
    const uint8_t *luto_multiplier,
    const int8_t *luto_shift,
    int8_t *output,
    int64_t numel,
    int64_t channels,
    int64_t luti_param_count,
    int64_t luto_param_count)
{
    constexpr int32_t kLutZero = 8;
    constexpr int32_t kLutUpper = 15;

    constexpr int32_t kActMin = -127;
    constexpr int32_t kActMax = 127;

    for (int64_t i = 0; i < numel; ++i)
    {
        const int64_t channel =
            i % channels;

        const int64_t luti_idx =
            (luti_param_count == 1)
                ? 0
                : channel;

        const int64_t luto_idx =
            (luto_param_count == 1)
                ? 0
                : channel;


        // ====================================================
        // LUT input requant:
        //
        // Torch:
        //
        // index =
        //   x.int() * luti_multiplier.int()
        //
        // luti multiplier is Q7 uint8.
        // Normal INT32 multiply, NOT MULH.
        // ====================================================

        int32_t index =
            static_cast<int32_t>(input[i])
            * static_cast<int32_t>(
                luti_multiplier[luti_idx]);


        // ====================================================
        // Rounded right shift.
        //
        // Must match the NEW frozen Rshift semantics:
        //
        // shift == 0:
        //     identity
        //
        // shift > 0:
        //     bias =
        //       2^(shift-1) - signbit(x)
        //
        //     output =
        //       (x + bias) >> shift
        // ====================================================

        const uint32_t input_shift =
            static_cast<uint32_t>(
                std::clamp<int32_t>(
                    static_cast<int32_t>(
                        luti_shift[luti_idx]),
                    0,
                    32));

        index =
            iwhisper::fixedpoint::rounding_rshift(
                index,
                input_shift);


        // ====================================================
        // Map signed LUT coordinate to [0, 15]
        // ====================================================

        index += kLutZero;

        index =
            std::clamp(
                index,
                int32_t{0},
                kLutUpper);


        // ====================================================
        // uint8 sigmoid LUT
        // ====================================================

        const int32_t gate =
            static_cast<int32_t>(
                lut[index]);


        // ====================================================
        // GeLU:
        //
        // x * sigmoid(1.702x)
        //
        // LUT output is uint8 [0,255].
        //
        // Maximum magnitude before requant:
        //
        // 127 * 255 = 32385
        // ====================================================

        int32_t value =
            static_cast<int32_t>(
                input[i])
            * gate;


        // ====================================================
        // LUT output requant
        //
        // IMPORTANT:
        //
        // This is NOT MULH.
        //
        // Current Torch golden:
        //
        // x.mul(luto_req_multiplier.int())
        //
        // luto multiplier is uint8.
        // ====================================================

        value =
            value
            * static_cast<int32_t>(
                luto_multiplier[luto_idx]);


        const uint32_t output_shift =
            static_cast<uint32_t>(
                std::clamp<int32_t>(
                    static_cast<int32_t>(
                        luto_shift[luto_idx]),
                    0,
                    32));

        value =
            iwhisper::fixedpoint::rounding_rshift(
                value,
                output_shift);


        // ====================================================
        // INT8 symmetric saturation
        // ====================================================

        value =
            std::clamp(
                value,
                kActMin,
                kActMax);

        output[i] =
            static_cast<int8_t>(
                value);
    }
}

} // namespace iwhisper