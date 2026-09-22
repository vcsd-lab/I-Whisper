#pragma once

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
    int64_t head_dim);

void qk_simd(
    const int8_t *q,
    const int8_t *k,
    int32_t *output,
    int64_t batch,
    int64_t heads,
    int64_t q_length,
    int64_t k_length,
    int64_t head_dim);

}