#pragma once

#include <cstdint>


namespace iwhisper {
namespace fixedpoint {


inline int64_t arithmetic_rshift(
    int64_t x,
    uint32_t shift)
{
    if (shift == 0)
    {
        return x;
    }

    const int64_t divisor =
        int64_t{1} << shift;

    if (x >= 0)
    {
        return x / divisor;
    }

    // floor(x / 2^shift)
    return -(
        (-x + divisor - 1)
        / divisor
    );
}


inline int32_t mulh(
    int32_t x,
    uint32_t multiplier)
{
    const int64_t product =
        static_cast<int64_t>(x)
        * static_cast<int64_t>(multiplier);

    return static_cast<int32_t>(
        arithmetic_rshift(product, 32)
    );
}


inline int32_t rounding_rshift(
    int32_t x,
    uint32_t shift)
{
    if (shift == 0)
    {
        return x;
    }

    const int64_t rounding =
        (int64_t{1} << (shift - 1))
        - (x < 0 ? 1 : 0);

    const int64_t value =
        static_cast<int64_t>(x)
        + rounding;

    return static_cast<int32_t>(
        arithmetic_rshift(
            value,
            shift
        )
    );
}


inline int32_t requant(
    int32_t x,
    uint32_t multiplier,
    uint32_t shift)
{
    const int32_t high =
        mulh(x, multiplier);

    return rounding_rshift(
        high,
        shift
    );
}


} // namespace fixedpoint
} // namespace iwhisper