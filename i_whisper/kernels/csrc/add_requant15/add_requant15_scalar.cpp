#include "add_requant15.h"
#include "../common/fixedpoint.h"

#include <algorithm>
#include <cstdint>

namespace iwhisper
{
namespace
{
inline int32_t s32_from_bits(uint32_t bits)
{
    const int64_t value =
        (bits & uint32_t{0x80000000u})
            ? static_cast<int64_t>(bits) - (int64_t{1} << 32)
            : static_cast<int64_t>(bits);
    return static_cast<int32_t>(value);
}

inline int32_t wrap_mul_s32_u16(int32_t x, uint16_t multiplier)
{
    const uint32_t product =
        static_cast<uint32_t>(x) *
        static_cast<uint32_t>(multiplier);
    return s32_from_bits(product);
}
}

void add_requant15_scalar(
    const int8_t *input,
    const int8_t *addend,
    const uint16_t *multiplier,
    const int8_t *shift,
    int16_t *output,
    int64_t batch,
    int64_t positions,
    int64_t channels,
    int64_t multiplier_count,
    int64_t shift_count)
{
    constexpr int32_t kMin = -16383;
    constexpr int32_t kMax = 16383;

    for (int64_t b = 0; b < batch; ++b)
    {
        for (int64_t t = 0; t < positions; ++t)
        {
            const int64_t in_base = (b * positions + t) * channels;
            const int64_t add_base = t * channels;

            for (int64_t c = 0; c < channels; ++c)
            {
                const int64_t mi = (multiplier_count == 1) ? 0 : c;
                const int64_t si = (shift_count == 1) ? 0 : c;

                int32_t value =
                    static_cast<int32_t>(input[in_base + c]) +
                    static_cast<int32_t>(addend[add_base + c]);

                value = wrap_mul_s32_u16(value, multiplier[mi]);

                const int32_t raw_shift =
                    static_cast<int32_t>(shift[si]);

                const uint32_t requant_shift =
                    static_cast<uint32_t>(
                        std::max<int32_t>(raw_shift, 0));

                value =
                    iwhisper::fixedpoint::rounding_rshift(
                        value,
                        requant_shift);

                value = std::clamp(value, kMin, kMax);

                output[in_base + c] =
                    static_cast<int16_t>(value);
            }
        }
    }
}
} // namespace iwhisper
