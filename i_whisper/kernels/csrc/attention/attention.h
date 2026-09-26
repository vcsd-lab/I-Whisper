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

    void qk_requant_rowmax_simd(
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

    void shift_exp_simd(
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

    // ============================================================
    // P @ V scalar accumulation.
    //
    // Input:
    //     p:      [B,H,Tq,Tk] UINT8
    //     v:      [B,H,D,Tk]  INT8
    //
    // Output:
    //     output: [B,H,Tq,D]  INT32
    //
    // This matches:
    //
    //     p.int32 @ v.int32.transpose(-1, -2)
    //
    // Accumulation uses explicit modulo-2^32 behavior to match
    // Torch INT32 matmul even in overflow stress tests.
    // ============================================================
    void pv_scalar(
        const uint8_t *p,
        const int8_t *v,
        int32_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim);

    void pv_simd(
        const uint8_t *p,
        const int8_t *v,
        int32_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim);

    // ============================================================
    // Fused P @ V + softmax normalization + AV requant + layout.
    //
    // Inputs:
    //     p:              [B,H,Tq,Tk] UINT8
    //     v:              [B,H,D,Tk]  INT8
    //     row_sum:        [B,H,Tq,1]  INT32
    //     av_multiplier:  scalar UINT8
    //     av_shift:       scalar INT8
    //
    // Output:
    //     output:          [B,Tq,H*D] INT8
    //
    // For each (b,h,q), rsqrt(row_sum) is computed once and reused
    // for all D output channels.
    // ============================================================
    void pv_norm_requant_scalar(
        const uint8_t *p,
        const int8_t *v,
        const int32_t *row_sum,
        uint8_t av_multiplier,
        int8_t av_shift,
        uint32_t rsqrt_offset,
        int8_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim);

    void pv_norm_requant_simd(
        const uint8_t *p,
        const int8_t *v,
        const int32_t *row_sum,
        uint8_t av_multiplier,
        int8_t av_shift,
        uint32_t rsqrt_offset,
        int8_t *output,
        int64_t batch,
        int64_t heads,
        int64_t q_length,
        int64_t k_length,
        int64_t head_dim);

}