#include "linear8.h"

#include <algorithm>
#include <cstdint>


namespace iwhisper {

void linear8_scalar(
    const int8_t* x,
    const int8_t* weight,
    const int32_t* bias,
    const int8_t* weight_zero,
    const uint32_t* multiplier,
    const int8_t* shift,

    int8_t* output,

    int64_t M,
    int64_t K,
    int64_t N,

    bool has_bias,
    bool asymmetric
)
{
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {

            int32_t acc = 0;

            for (int64_t k = 0; k < K; ++k) {
                acc +=
                    static_cast<int32_t>(x[m * K + k]) *
                    static_cast<int32_t>(weight[n * K + k]);
            }

            if (has_bias) {
                acc += bias[n];
            }

            // temporary smoke-test behavior
            acc = std::max<int32_t>(-128, acc);
            acc = std::min<int32_t>(127, acc);

            output[m * N + n] =
                static_cast<int8_t>(acc);
        }
    }
}

}  // namespace iwhisper