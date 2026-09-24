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


void qk_requant_rowmax_scalar(
    const int8_t *q,
    const int8_t *k,
    const bool *mask,
    int16_t *qk_output,
    int16_t *row_max_output,
    int64_t batch,
    int64_t heads,
    int64_t q_length,
    int64_t k_length,
    int64_t head_dim,
    uint8_t multiplier,
    int32_t shift,
    bool has_mask);



void qk_simd(
    const int8_t *q,
    const int8_t *k,
    int32_t *output,
    int64_t batch,
    int64_t heads,
    int64_t q_length,
    int64_t k_length,
    int64_t head_dim);


void shift_exp_scalar(
    const int16_t *input,
    const int16_t *row_max_plus1,
    const bool *mask,
    uint8_t *output,
    int32_t *row_sum_output,
    int64_t batch,
    int64_t heads,
    int64_t q_length,
    int64_t k_length,
    int32_t frac_bits,
    bool has_mask);

}