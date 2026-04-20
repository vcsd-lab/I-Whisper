/**
 * @file w8a8_ukernel_util_neon.h
 * @brief: NEON utility functions for w8a8 GEMV operations
 * @author: JunAn Chen (chen96013@gmail.com), VCSDL master student
 * @data: 2025-12-30
 * @note: This file is part I-Whisper project, developed on Raspberry Pi 4
 */
#pragma once
#include <cstdint>
#include <arm_neon.h>

#ifndef ROUND_TO_NEAREST_EVEN
#define ROUND_TO_NEAREST
#endif

#ifndef FIXPOINT_ITERATION
#define FIXPOINT_ITERATION
#endif


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn: ukernel_s32_RaRshift_1s_neon
 * @brief:
 * Optimized NEON implementation of Round and Right Shift for int32 array
 * the Right Shift result is Rounded at final zero point
 * this means the flow will be:
 * 1. Round at shmnt bit position (round to nearest even)
 * 2. Right Shift by shmnt bits
 * 
 * @param x: input int32 array
 * @param shmnt: right shift amount (number of bits)
 */
static inline int32x4_t ukernel_s32_RaRshift_s32_neon(
    int32x4_t   x,
    int8_t      shmnt
) {
#ifdef ROUND_TO_NEAREST_EVEN
    // q = x >> shmnt  (variable arithmetic shift: use vshl with negative)
    const int32x4_t vsh = vdupq_n_s32(-shmnt);
    int32x4_t q = vshlq_s32(x, vsh);

    // bias = (1<<(shmnt-1)) - 1 + (q & 1)
    int32_t base = (1 << (shmnt - 1)) - 1;
    int32x4_t vbias = vdupq_n_s32(base);
    vbias = vaddq_s32(vbias, vandq_s32(q, vdupq_n_s32(1)));

    // y = (x + bias) >> shmnt
    int32x4_t tmp = vaddq_s32(x, vbias);
    return vshlq_s32(tmp, vsh);
#else // ROUND_TO_NEAREST
    // simple right shift with NEON build-in round to nearest (no tie to even)
    const int32x4_t vsh = vdupq_n_s32(-shmnt);
    return vrshlq_s32(x, vsh);
#endif
}


/**
 * @fn: ukernel_s32_RaRshift_4s_neon
 * @brief:
 * Optimized NEON implementation of Round and Right Shift for int32 array
 * the Right Shift result is Rounded at final zero point
 * this means the flow will be:
 * 1. Round at shmnt bit position (round to nearest even)
 * 2. Right Shift by shmnt bits
 * 
 * @param x input int32 array
 * @param shmnt right shift amount (number of bits) for 4 elements
 */
static inline int32x4_t ukernel_s32_RaRshift_s32x4_neon(
    int32x4_t       x,
    int32x4_t       shmnt
) {
#ifdef ROUND_TO_NEAREST_EVEN
    // q = x >> shmnt  (variable arithmetic shift: use vshl with negative)
    int32x4_t vsh = vnegq_s32(shmnt);
    int32x4_t q = vshlq_s32(x, vsh);

    // bias = (1<<(shmnt-1)) - 1 + (q & 1)
    int32x4_t vbias = vsubq_s32(vshlq_n_s32(vdupq_n_s32(1), vsubq_s32(shmnt, vdupq_n_s32(1))), vdupq_n_s32(1));
    vbias = vaddq_s32(vbias, vandq_s32(q, vdupq_n_s32(1)));
    // y = (x + bias) >> shmnt
    int32x4_t tmp = vaddq_s32(x, vbias);
    return vshlq_s32(tmp, vsh);
#else // ROUND_TO_NEAREST
    // simple right shift with NEON build-in round to nearest (no tie to even)
    // assume sh in [0,31]
    int32x4_t vsh = vnegq_s32(shmnt);

    // handle sh==0: return x
    uint32x4_t sh_is_zero = vceqq_s32(shmnt, vdupq_n_s32(0));
    // compute half for sh>=1
    int32x4_t ones = vdupq_n_s32(1);
    int32x4_t half = vshlq_s32(ones, vsubq_s32(shmnt, vdupq_n_s32(1)));

    // sb = signbit(x) as 0/1
    uint32x4_t sb = vshrq_n_u32(vreinterpretq_u32_s32(x), 31);

    // bias = half - sb
    int32x4_t bias = vsubq_s32(half, vreinterpretq_s32_u32(sb));

    // tmp = x + bias; y = tmp >> sh
    int32x4_t tmp = vaddq_s32(x, bias);
    int32x4_t y = vshlq_s32(tmp, vsh);

    // select sh==0 lanes: y = x
    return vbslq_s32(sh_is_zero, x, y);
#endif
}


static inline uint32x4_t NetwonRaphson_Iteration(
    uint32x4_t   y,     // current approximation of 1/sqrt(x)
    uint32x4_t   x2,    // x divided by 2
    uint8_t     offset  // fixed-point position of y
) {
    // 1. compute threehalfs in fixed-point
    uint32_t threehalfs_int = (3 << (offset - 1)); // 3/2 in fixed-point
    // 2. compute: t = x2 * y * y
    // note: here we do not need 2 Rshift because var is q32.0 (integer), y2 is q12.20 (if offset=10)
    uint32x4_t t = vshrq_n_u32( vmulq_u32( x2, y ), offset ); // right shift by offset to maintain fixed-point
    t = vmulq_u32( t, y );
    // 3. compute: y = y * (threehalfs - t)
    y = vshrq_n_u32( vmulq_u32( y, vsubq_u32( vdupq_n_u32(threehalfs_int), t ) ), offset );
    return y;
}
/**
 * @fn: ukernel_rsqrt_u32x4_neon
 * @note: this function is designed for 4 lanes NEON processing
 * @brief:
 * Calculate approximate value of Reciprocal Square Root (1/sqrt(x))
 * Implementation by Fast Inverse Square Root method (Quake III Arena)
 * see: https://en.wikipedia.org/wiki/Fast_inverse_square_root
 * difference is:
 * Our result is returned in int32 format with fixed-point ( @param fisr_offset )
 * Also, Newtwon-Raphson iteration is implemented in fixed-point int32 format
 * And when calling function, user can specify number of iterations
 * 0 means no iteration, just use initial approximation
 * this is okay if you need high performance and can tolerate low accuracy
 * @note: more iterations not always lead to better accuracy (due to fixed-point precision limit)
 * all decided by user when calling this function
 * 
 * @param x 4 input uint32 number in uint32x4_t (because apply sqrt on negative number is invalid)
 * @param fisr_offset fixed-point position of result
 * @param overflow_prevent_offset right shift offset to prevent overflow in iteration
 * @param n_iters number of Newton-Raphson iterations to perform
 */
static inline uint32x4_t ukernel_rsqrt_u32x4_neon(
    uint32x4_t   x,
    uint8_t     fisr_offset,
    uint8_t     overflow_prevent_offset,
    uint8_t     n_iters
) {

#define FISR_MAGIC_NUM 0x5f3759df
#define MANTISSA_BITS 23

    /* zero case handling stage */
    const uint32x4_t zero = vdupq_n_u32(0);
    const uint32x4_t one_fixpoint = vdupq_n_u32(1 << fisr_offset);
    uint32x4_t mask = vceqq_u32(x, zero);

    /* initial approximation stage */
    uint32x4_t x_bits = vreinterpretq_u32_f32( vcvtq_f32_u32( x ) ); // convert to float32 then view(reinterpret) as uint32
    x_bits = vsubq_u32( vdupq_n_u32(FISR_MAGIC_NUM), vshrq_n_u32( x_bits, 1 ) ); // x_bits = MAGIC_NUM - (x_bits >> 1)
    // why not using vrsqrtesq_f32
    // after testing, vrsqrtesq_f32 is less accurate than Fast Inverse Square Root initial estimate

#ifdef FIXPOINT_ITERATION

    uint32x4_t exp_offset = vdupq_n_u32(((uint32_t)fisr_offset) << MANTISSA_BITS);
    x_bits = vaddq_u32( x_bits, exp_offset );
    // then, view bits as float32 and convert to int32
    uint32x4_t fisr_approx = vcvtq_u32_f32( vreinterpretq_f32_u32( x_bits ) ); // view(reinterpret) as float32 then convert to uint32

    /* Newton-Raphson iterations in fixed-point int32 stage(Optional) */
    uint32x4_t x2 = vshrq_n_u32( x, 1 ); // x / 2
    for (uint8_t i = 0; i < n_iters; i++) {
        // compute x2 = x / 2^(overflow_prevent_offset + 1)
        // x / 2, here is not fixed-point because x is integer (usually large)
        // if using fixed-point, it may cause overflow in multiplication
        // without using fixed-point, it is okay and we may using less shift in iteration
        fisr_approx = NetwonRaphson_Iteration( fisr_approx, x2, fisr_offset );
    }
    fisr_approx = vbslq_u32(mask, one_fixpoint, fisr_approx);
    return fisr_approx;

#undef FIXPOINT_ITERATION
#else // FLOAT_ITERATION

    float32x4_t estimated = vreinterpretq_f32_u32( x_bits );

    /* iteration stage */
    const float32x4_t x_f = vcvtq_f32_u32(x);
    for (uint8_t i = 0; i < n_iters; i++) {
        estimated = estimated * ( 1.5f - 0.5f * x_f * estimated * estimated );
    }

    /* fixed-point conversion stage */
    uint32x4_t exp_offset = vdupq_n_u32(((uint32_t)fisr_offset) << MANTISSA_BITS);
    uint32x4_t fisr_approx = vreinterpretq_u32_f32( estimated );
    fisr_approx = vaddq_u32( fisr_approx, exp_offset );
    fisr_approx = vcvtq_u32_f32( vrndnq_f32( vreinterpretq_f32_u32( fisr_approx ) ) );

    /* handle zero case */
    fisr_approx = vbslq_u32(mask, one_fixpoint, fisr_approx);
    return fisr_approx;

#undef FLOAT_ITERATION
#endif
}


#ifdef __cplusplus
}
#endif