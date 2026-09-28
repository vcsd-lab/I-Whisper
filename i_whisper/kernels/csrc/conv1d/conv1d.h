#pragma once

#include <cstdint>

namespace iwhisper
{

// Raw INT8 Conv1d accumulation.
//
// input:  [B, Cin, Tin]
// weight: [Cout, Cin * K] with flattened order [Cin, K]
// bias:   [Cout] INT32, optional
// output: [B, Cout, Tout] INT32
//
// groups = 1
// dilation = 1
void conv1d_accum_scalar(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    int32_t *output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_length,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool has_bias);

void conv1d_accum_simd(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    int32_t *output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_length,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool has_bias);


// Fused INT8 Conv1d + bias + requant.
//
// Requant matches model_int.py:
//
//   x = (x.long() * multiplier.long() >> 32).int()
//   x = Rshift(x, shift)
//   x = clamp(x, -127, 127).int8()
//
// multiplier / shift are per-layer scalars.
void conv1d_scalar(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    uint32_t multiplier,
    int8_t shift,
    int8_t *output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_length,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool has_bias);

void conv1d_simd(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    uint32_t multiplier,
    int8_t shift,
    int8_t *output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_length,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool has_bias);

} // namespace iwhisper
