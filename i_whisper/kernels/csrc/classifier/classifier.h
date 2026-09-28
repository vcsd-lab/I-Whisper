#pragma once

#include <cstdint>

namespace iwhisper
{

// Raw terminal classifier accumulation.
//
// input:  [M,K] INT8, where M = B*T
// weight: [N,K] INT8, N = vocabulary size
// output: [M,N] INT32
//
// Arithmetic uses modulo-2^32 accumulation to match
// torch.int32 matmul overflow semantics.
void classifier_accum_scalar(
    const int8_t *input,
    const int8_t *weight,
    int32_t *output,
    int64_t M,
    int64_t K,
    int64_t N);

void classifier_accum_simd(
    const int8_t *input,
    const int8_t *weight,
    int32_t *output,
    int64_t M,
    int64_t K,
    int64_t N);


// Fused terminal classifier.
//
// INT8 x INT8 -> INT32
// -> signed INT32 x UINT32 MULH
// -> rounded Rshift
// -> clamp [-127,127]
// -> INT8
//
// multiplier and shift are per-layer scalars.
void classifier_scalar(
    const int8_t *input,
    const int8_t *weight,
    uint32_t multiplier,
    int8_t shift,
    int8_t *output,
    int64_t M,
    int64_t K,
    int64_t N);

void classifier_simd(
    const int8_t *input,
    const int8_t *weight,
    uint32_t multiplier,
    int8_t shift,
    int8_t *output,
    int64_t M,
    int64_t K,
    int64_t N);

} // namespace iwhisper
