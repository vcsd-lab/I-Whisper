#include <cstdint>
#include <arm_neon.h>
#include "w8a8_ukernel_gemv_neon.h"


// Optimized NEON implementation of 8x8 GEMV ukernel with packed B matrix
// Micro-kernel(ukernel):
//      computes 8 output channels
//      computes over 8 elements of K dimension per iteration (8 input channel)
//      C[1x8] = A[1xK] * Bp[Kx8]
//      thus, assert K % 8 == 0
// this is first version of neon ukernel computing with manual unroll
// the performance is similar to the version with pragma unroll
// 
// the implementation is longer and less maintainable, but may be easier to understand
// for beginners, reading this code first may help
// this function define the same behavior as ukernel_s8s8s32_gemv_8x8_packed_neon
//
// A: [K], int8
// Bp: packed weights, layout [K][8]
// C: [8], int32 accumulator
void ukernel_s8s8s32_gemv_8x8_packed_neon_manual_unroll(
    const int8_t*   A,
    const int8_t*   Bp,
    int32_t*        C,
    uint32_t        K
) {
    // init 8 int32_t accumulators
    // each accumulator holds 4 int32 values
    // total 2 accumulators holds 8 int32_t accumulators
    int32x4_t acc0 = vdupq_n_s32(0); // C[0..3]
    int32x4_t acc1 = vdupq_n_s32(0); // C[4..7]

    const int8_t* bp = Bp;
    // main loop over K
    for (int32_t k = 0; k + 8 <= K; k += 8) {
        // Load tile A once and broadcast
        int16x8_t a_vec = vmovl_s8(vld1_s8(A + k)); // load A[k..k+7]
        // int8x16_t a_vec = vld1q_s8(A + k); // load A[k..k+15]
        
        // ---- k+0 ----
        int16x4_t a0 = vdup_n_s16(vgetq_lane_s16(a_vec, 0));
        int16x8_t b0 = vmovl_s8(vld1_s8(bp + 0*8));
        acc0 = vmlal_s16(acc0, a0, vget_low_s16(b0));
        acc1 = vmlal_s16(acc1, a0, vget_high_s16(b0));

        // ---- k+1 ----
        int16x4_t a1 = vdup_n_s16(vgetq_lane_s16(a_vec, 1));
        int16x8_t b1 = vmovl_s8(vld1_s8(bp + 1*8));
        acc0 = vmlal_s16(acc0, a1, vget_low_s16(b1));
        acc1 = vmlal_s16(acc1, a1, vget_high_s16(b1));

        // ---- k+2 ----
        int16x4_t a2 = vdup_n_s16(vgetq_lane_s16(a_vec, 2));
        int16x8_t b2 = vmovl_s8(vld1_s8(bp + 2*8));
        acc0 = vmlal_s16(acc0, a2, vget_low_s16(b2));
        acc1 = vmlal_s16(acc1, a2, vget_high_s16(b2));

        // ---- k+3 ----
        int16x4_t a3 = vdup_n_s16(vgetq_lane_s16(a_vec, 3));
        int16x8_t b3 = vmovl_s8(vld1_s8(bp + 3*8));
        acc0 = vmlal_s16(acc0, a3, vget_low_s16(b3));
        acc1 = vmlal_s16(acc1, a3, vget_high_s16(b3));

        // ---- k+4 ----
        int16x4_t a4 = vdup_n_s16(vgetq_lane_s16(a_vec, 4));
        int16x8_t b4 = vmovl_s8(vld1_s8(bp + 4*8));
        acc0 = vmlal_s16(acc0, a4, vget_low_s16(b4));
        acc1 = vmlal_s16(acc1, a4, vget_high_s16(b4));

        // ---- k+5 ----
        int16x4_t a5 = vdup_n_s16(vgetq_lane_s16(a_vec, 5));
        int16x8_t b5 = vmovl_s8(vld1_s8(bp + 5*8));
        acc0 = vmlal_s16(acc0, a5, vget_low_s16(b5));
        acc1 = vmlal_s16(acc1, a5, vget_high_s16(b5));

        // ---- k+6 ----
        int16x4_t a6 = vdup_n_s16(vgetq_lane_s16(a_vec, 6));
        int16x8_t b6 = vmovl_s8(vld1_s8(bp + 6*8));
        acc0 = vmlal_s16(acc0, a6, vget_low_s16(b6));
        acc1 = vmlal_s16(acc1, a6, vget_high_s16(b6));

        // ---- k+7 ----
        int16x4_t a7 = vdup_n_s16(vgetq_lane_s16(a_vec, 7));
        int16x8_t b7 = vmovl_s8(vld1_s8(bp + 7*8));
        acc0 = vmlal_s16(acc0, a7, vget_low_s16(b7));
        acc1 = vmlal_s16(acc1, a7, vget_high_s16(b7));

        bp += 8*8;
    }
    // store back to C
    vst1q_s32(C + 0, acc0);
    vst1q_s32(C + 4, acc1);
}

// Optimized NEON implementation of 8x8 GEMV ukernel with packed B matrix
// Micro-kernel(ukernel):
//      computes 8 output channels
//      computes over 8 elements of K dimension per iteration (8 input channel)
//      C[1x8] = A[1xK] * Bp[Kx8]
//      thus, assert K % 8 == 0
//
// A: [K], int8
// Bp: packed weights, layout [K][8]
// C: [8], int32 accumulator
void ukernel_s8s8s32_gemv_8x8_packed_neon(
    const int8_t*   A,
    const int8_t*   Bp,
    int32_t*        C,
    uint32_t        K
) {
    // init 8 int32_t accumulators
    // each accumulator holds 4 int32 values
    // total 2 accumulators holds 8 int32_t accumulators
    int32x4_t acc0 = vdupq_n_s32(0); // C[0..3]
    int32x4_t acc1 = vdupq_n_s32(0); // C[4..7]

    const int8_t* bp = Bp;
    // main loop over K
    for (int32_t k = 0; k + 8 <= K; k += 8) {
        // Load tile A once and broadcast
        int16x8_t a16 = vmovl_s8(vld1_s8(A + k)); // load A[k..k+7]


        #pragma unroll
        for (int inner = 0; inner < 8; inner++) {
            int16x4_t a0 = vdup_n_s16(vgetq_lane_s16(a16, inner));
            int16x8_t b0 = vmovl_s8(vld1_s8(bp + inner*8));
            acc0 = vmlal_s16(acc0, a0, vget_low_s16(b0));
            acc1 = vmlal_s16(acc1, a0, vget_high_s16(b0));
        }

        bp += 8*8;
    }
    // store back to C
    vst1q_s32(C + 0, acc0);
    vst1q_s32(C + 4, acc1);
}

// Optimized NEON implementation of 16x8 GEMV ukernel with packed B matrix
// Micro-kernel:
//      computes 8 output channels
//      computes over 16 elements of K dimension per iteration (16 input channel)
//      C[1x8] = A[1xK] * Bp[Kx8]
//      thus, assert K % 16 == 0
//
// A: [K], int8
// Bp: packed weights, layout [K][8]
// C: [8], int32 accumulator
void ukernel_s8s8s32_gemv_16x8_packed_neon(
    const int8_t*   A,
    const int8_t*   Bp,
    int32_t*        C,
    uint32_t        K
) {
    // init 8 int32_t accumulators
    // each accumulator holds 4 int32 values
    // total 2 acc holds 8 int32_t accumulators
    int32x4_t acc0 = vdupq_n_s32(0); // C[0..3]
    int32x4_t acc1 = vdupq_n_s32(0); // C[4..7]

    const int8_t* bp = Bp;
    // main loop over K
    for (int32_t k = 0; k + 16 <= K; k += 16) {
        // Load tile A once and broadcast
        int8x16_t a_vec = vld1q_s8(A + k); // load A[k..k+15]
        int16x8_t a_vec_lo = vmovl_s8(vget_low_s8(a_vec));
        int16x8_t a_vec_hi = vmovl_s8(vget_high_s8(a_vec));

        #pragma unroll
        for (uint32_t inner = 0; inner < 16; inner++) {
            int16x4_t a;
            if (inner < 8) {
                a = vdup_n_s16(vgetq_lane_s16(a_vec_lo, inner));
            } else {
                a = vdup_n_s16(vgetq_lane_s16(a_vec_hi, inner - 8));
            }
            int16x8_t b = vmovl_s8(vld1_s8(bp + inner*8));
            acc0 = vmlal_s16(acc0, a, vget_low_s16(b));
            acc1 = vmlal_s16(acc1, a, vget_high_s16(b));
        }
        bp += 16*8;
    }
    // store back to C
    vst1q_s32(C + 0, acc0);
    vst1q_s32(C + 4, acc1);
}



void ukernel_s8s8s32_gemv_8x16_packed_neon(
    const int8_t*   A,
    const int8_t*   Bp,
    int32_t*        C,
    uint32_t        K
) {
    // init 16 int32_t accumulators
    // each acc holds 4 int32 values
    // total 4 acc holds 16 int32_t accumulators
    int32x4_t acc0 = vdupq_n_s32(0); // C[0..3]
    int32x4_t acc1 = vdupq_n_s32(0); // C[4..7]
    int32x4_t acc2 = vdupq_n_s32(0); // C[8..11]
    int32x4_t acc3 = vdupq_n_s32(0); // C[12..15]
    
    const int8_t* bp = Bp;
    // main loop over K
    for (int32_t k = 0; k + 8 <= K; k += 8) {
        // Load tile A once and broadcast
        int16x8_t a_vec = vmovl_s8(vld1_s8(A + k));

        #pragma unroll
        for (int32_t inner = 0; inner < 8; inner++) {
            int16x8_t b_lo = vmovl_s8(vld1_s8(bp + inner*16 + 0));
            int16x8_t b_hi = vmovl_s8(vld1_s8(bp + inner*16 + 8));

            if (inner < 4) {
                acc0 = vmlal_lane_s16(acc0, vget_low_s16(b_lo), vget_low_s16(a_vec), inner);
                acc1 = vmlal_lane_s16(acc1, vget_high_s16(b_lo), vget_low_s16(a_vec), inner);
                acc2 = vmlal_lane_s16(acc2, vget_low_s16(b_hi), vget_low_s16(a_vec), inner);
                acc3 = vmlal_lane_s16(acc3, vget_high_s16(b_hi), vget_low_s16(a_vec), inner);
            } else {
                acc0 = vmlal_lane_s16(acc0, vget_low_s16(b_lo), vget_high_s16(a_vec), inner-4);
                acc1 = vmlal_lane_s16(acc1, vget_high_s16(b_lo), vget_high_s16(a_vec), inner-4);
                acc2 = vmlal_lane_s16(acc2, vget_low_s16(b_hi), vget_high_s16(a_vec), inner-4);
                acc3 = vmlal_lane_s16(acc3, vget_high_s16(b_hi), vget_high_s16(a_vec), inner-4);
            }
        }
        bp += 8*16;
    }
    // store back to C
    vst1q_s32(C + 0, acc0);
    vst1q_s32(C + 4, acc1);
    vst1q_s32(C + 8, acc2);
    vst1q_s32(C + 12, acc3);
}


void ukernel_s8s8s32_gemv_16x16_packed_neon(
    const int8_t*   A,
    const int8_t*   Bp,
    int32_t*        C,
    uint32_t        K
) {
    // init 16 int32_t accumulators
    // each acc holds 4 int32 values
    // total 4 acc holds 16 int32_t accumulators
    int32x4_t acc0 = vdupq_n_s32(0); // C[0..3]
    int32x4_t acc1 = vdupq_n_s32(0); // C[4..7]
    int32x4_t acc2 = vdupq_n_s32(0); // C[8..11]
    int32x4_t acc3 = vdupq_n_s32(0); // C[12..15]
    
    const int8_t* bp = Bp;
    // main loop over K
    for (int32_t k = 0; k + 16 <= K; k += 16) {
        // Load tile A once and broadcast
        // int8x16_t a_vec = vld1q_s8(A + k); // load A[k..k+15]

        #pragma unroll
        for (int32_t inner = 0; inner < 16; inner++) {
            int16x4_t a = vdup_n_s16(static_cast<int16_t>(A[k+inner]));
            int16x8_t b_lo = vmovl_s8(vld1_s8(bp + inner*16 + 0));
            int16x8_t b_hi = vmovl_s8(vld1_s8(bp + inner*16 + 8));

            acc0 = vmlal_s16(acc0, a, vget_low_s16(b_lo));
            acc1 = vmlal_s16(acc1, a, vget_high_s16(b_lo));
            acc2 = vmlal_s16(acc2, a, vget_low_s16(b_hi));
            acc3 = vmlal_s16(acc3, a, vget_high_s16(b_hi));
        }
        bp += 16*16;
    }
    // store back to C
    vst1q_s32(C + 0, acc0);
    vst1q_s32(C + 4, acc1);
    vst1q_s32(C + 8, acc2);
    vst1q_s32(C + 12, acc3);
}


void ukernel_u8s8s32_gemv_8x16_packed_neon(
    const uint8_t*  A,
    const int8_t*   Bp,
    int32_t*        C,
    uint32_t        K
) {
    // init 16 int32_t accumulators
    // each acc holds 4 int32 values
    // total 4 acc holds 16 int32_t accumulators
    int32x4_t acc0 = vdupq_n_s32(0); // C[0..3]
    int32x4_t acc1 = vdupq_n_s32(0); // C[4..7]
    int32x4_t acc2 = vdupq_n_s32(0); // C[8..11]
    int32x4_t acc3 = vdupq_n_s32(0); // C[12..15]
    
    const int8_t* bp = Bp;
    // main loop over K
    for (int32_t k = 0; k + 8 <= K; k += 8) {
        // Load tile A once and broadcast
        int16x8_t a_vec = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(A + k)));

        #pragma unroll
        for (int32_t inner = 0; inner < 8; inner++) {
            int16x8_t b_lo = vmovl_s8(vld1_s8(bp + inner*16 + 0));
            int16x8_t b_hi = vmovl_s8(vld1_s8(bp + inner*16 + 8));

            if (inner < 4) {
                acc0 = vmlal_lane_s16(acc0, vget_low_s16(b_lo), vget_low_s16(a_vec), inner);
                acc1 = vmlal_lane_s16(acc1, vget_high_s16(b_lo), vget_low_s16(a_vec), inner);
                acc2 = vmlal_lane_s16(acc2, vget_low_s16(b_hi), vget_low_s16(a_vec), inner);
                acc3 = vmlal_lane_s16(acc3, vget_high_s16(b_hi), vget_low_s16(a_vec), inner);
            } else {
                acc0 = vmlal_lane_s16(acc0, vget_low_s16(b_lo), vget_high_s16(a_vec), inner-4);
                acc1 = vmlal_lane_s16(acc1, vget_high_s16(b_lo), vget_high_s16(a_vec), inner-4);
                acc2 = vmlal_lane_s16(acc2, vget_low_s16(b_hi), vget_high_s16(a_vec), inner-4);
                acc3 = vmlal_lane_s16(acc3, vget_high_s16(b_hi), vget_high_s16(a_vec), inner-4);
            }
        }
        bp += 8*16;
    }
    // store back to C
    vst1q_s32(C + 0, acc0);
    vst1q_s32(C + 4, acc1);
    vst1q_s32(C + 8, acc2);
    vst1q_s32(C + 12, acc3);
}