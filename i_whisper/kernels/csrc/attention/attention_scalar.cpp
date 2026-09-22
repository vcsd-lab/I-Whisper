#include "attention.h"

#include <cstdint>

namespace iwhisper
{

    void qk_scalar(
        const int8_t *q,
        const int8_t *k,
        int32_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim)
    {
        for (int64_t b = 0; b < batch; ++b)
        {
            for (int64_t h = 0; h < heads; ++h)
            {
                const int64_t q_head_base =
                    ((b * heads + h) * q_length * head_dim);

                const int64_t k_head_base =
                    ((b * heads + h) * k_length * head_dim);

                const int64_t out_head_base =
                    ((b * heads + h) * q_length * k_length);

                for (int64_t qi = 0;
                     qi < q_length;
                     ++qi)
                {
                    const int8_t *q_ptr =
                        q + q_head_base + qi * head_dim;

                    int32_t *out_ptr =
                        output + out_head_base + qi * k_length;

                    for (int64_t ki = 0;
                         ki < k_length;
                         ++ki)
                    {
                        const int8_t *k_ptr =
                            k + k_head_base + ki * head_dim;

                        int32_t acc = 0;

                        for (int64_t d = 0;
                             d < head_dim;
                             ++d)
                        {
                            acc +=
                                static_cast<int32_t>(
                                    q_ptr[d]) *
                                static_cast<int32_t>(
                                    k_ptr[d]);
                        }

                        out_ptr[ki] = acc;
                    }
                }
            }
        }
    }

} // namespace iwhisper