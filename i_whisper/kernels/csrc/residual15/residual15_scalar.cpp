#include "residual15.h"
#include "../common/fixedpoint.h"

#include <algorithm>

namespace iwhisper
{

    void residual15_scalar(
        const int32_t *branch,
        const int16_t *residual,
        uint16_t multiplier,
        int16_t *output,
        int64_t numel)
    {
        
        int32_t buffer;
        for (int64_t i = 0; i < numel; ++i)
        {
            buffer = (static_cast<int32_t>(branch[i]) + static_cast<int32_t>(residual[i]) * multiplier);
            buffer = iwhisper::fixedpoint::rounding_rshift(buffer, 15);

            // Clamp the result to the range of int15
            buffer = std::clamp(buffer, int32_t(-16383), int32_t(16383));

            output[i] = static_cast<int16_t>(buffer);
        }
    }

} // namespace iwhisper