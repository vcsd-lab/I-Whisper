/**
 * @file w8a8_ukernel_gemv_neon.h
 * @brief: NEON micro-kernel declarations for w8a8 GEMV operations
 * @author: JunAn Chen (chen96013@gmail.com), VCSDL master student
 * @data: 2025-12-30
 * @note: This file is part I-Whisper project, developed on Raspberry Pi 4
 */
#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ukernel_s8s8s32_gemv_8x8_packed_neon
 * @brief
 * Optimized NEON implementation of 8x8 GEMV ukernel 
 * with packed B matrix
 * in this Micro-kernel(ukernel):
 *  computes 8 output channels
 *  computes over 8 elements of K dimension per iteration (8 input channel)
 *  C[1x8] = A[1xK] * Bp[Kx8]
 * 
 * @param A [K], int8
 * @param Bp packed weights, int8, layout [K][8]
 * @param C [8], int32 accumulator
 * */ 
void ukernel_s8s8s32_gemv_8x8_packed_neon(
    const int8_t* A, const int8_t* Bp, int32_t* C, uint32_t K);


/**
 * @fn ukernel_s8s8s32_gemv_8x16_packed_neon
 * @brief
 * Optimized NEON implementation of 8x16 GEMV ukernel 
 * with packed B matrix
 * in this Micro-kernel(ukernel):
 *  computes 16 output channels
 *  computes over 8 elements of K dimension per iteration (8 input channel)
 *  C[1x16] = A[1xK] * Bp[Kx16]
 * 
 * @param A [K], int8
 * @param Bp packed weights, int8, layout [K][16]
 * @param C [16], int32 accumulator
 * */ 
void ukernel_s8s8s32_gemv_8x16_packed_neon(
    const int8_t* A, const int8_t* Bp, int32_t* C, uint32_t K);


/**
 * @fn ukernel_s8s8s32_gemv_16x8_packed_neon
 * @brief
 * Optimized NEON implementation of 16x8 GEMV ukernel 
 * with packed B matrix
 * in this Micro-kernel(ukernel):
 *  computes 8 output channels
 *  computes over 16 elements of K dimension per iteration (16 input channel)
 *  C[1x8] = A[1xK] * Bp[Kx8]
 * 
 * @param A [K], int8
 * @param Bp packed weights, int8, layout [K][8]
 * @param C [8], int32 accumulator
 * */ 
void ukernel_s8s8s32_gemv_16x8_packed_neon(
    const int8_t* A, const int8_t* Bp, int32_t* C, uint32_t K);


/**
 * @fn ukernel_s8s8s32_gemv_16x16_packed_neon
 * @brief
 * Optimized NEON implementation of 16x16 GEMV ukernel 
 * with packed B matrix
 * in this Micro-kernel(ukernel):
 *  computes 16 output channels
 *  computes over 16 elements of K dimension per iteration (16 input channel)
 *  C[1x16] = A[1xK] * Bp[Kx16]
 * 
 * @param A [K], int8
 * @param Bp packed weights, int8, layout [K][16]
 * @param C [16], int32 accumulator
 * */ 
void ukernel_s8s8s32_gemv_16x16_packed_neon(
    const int8_t* A, const int8_t* Bp, int32_t* C, uint32_t K);


/**
 * @fn ukernel_u8s8s32_gemv_8x16_packed_neon
 * @brief
 * Optimized NEON implementation of 8x16 GEMV ukernel 
 * with packed B matrix
 * in this Micro-kernel(ukernel):
 *  computes 16 output channels
 *  computes over 8 elements of K dimension per iteration (8 input channel)
 *  C[1x16] = A[1xK] * Bp[Kx16]
 * 
 * @param A [K], uint8
 * @param Bp packed weights, layout [K][16]
 * @param C [16], int32 accumulator
 * */ 
void ukernel_u8s8s32_gemv_8x16_packed_neon(
    const uint8_t* A, const int8_t* Bp, int32_t* C, uint32_t K);

/**
 * @fn pack_W_nk_to_Bp_k8
 * @brief
 * assume matrix W is in layout [N][K]
 * Pack a part of weight matrix (layout [8][K], 8 rows from W)
 * to Bp layout [K][8] (transpose a part of W)
 * this is pre-processing for B matrix of above GEMV ukernel
 * this tiling function pack 8 output channels from W
 * and pad zero if N or K is not multiple of 8
 * without this, the GEMV ukernel cannot be used directly
 * thus, this is a helper function to prepare Bp from W
 * 
 * @param W original weight matrix, layout [N][K]
 * @param Bp packed a part weight matrix, layout [K][8]
 * @param K input channel dimension
 * @param N output channel dimension
 * @param n0 starting output channel index in W to pack
 */
static inline void pack_W_nk_to_Bp_k8(
    const int8_t* W,
    int8_t*       Bp,
    uint32_t      K,
    uint32_t      N,
    uint32_t      n0
) {
    for (int k = 0; k < K; k++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            Bp[k*8 + i] = W[(n0 + i)*K + k];
        } // this realize belows code
        // Bp[k*8 + 0] = W[(n0 + 0)*K + k];
        // Bp[k*8 + 1] = W[(n0 + 1)*K + k];
        // Bp[k*8 + 2] = W[(n0 + 2)*K + k];
        // Bp[k*8 + 3] = W[(n0 + 3)*K + k];
        // Bp[k*8 + 4] = W[(n0 + 4)*K + k];
        // Bp[k*8 + 5] = W[(n0 + 5)*K + k];
        // Bp[k*8 + 6] = W[(n0 + 6)*K + k];
        // Bp[k*8 + 7] = W[(n0 + 7)*K + k];
    }
}


/**
 * @fn: pack_W_nk_to_Bp_k16
 * @brief:
 * assume matrix W is in layout [N][K]
 * Pack a part of weight matrix (layout [16][K], 16 rows from W)
 * to Bp layout [K][16] (transpose a part of W)
 * this is pre-processing for B matrix of above GEMV ukernel
 * this tiling function pack 16 output channels from W
 * and pad zero if N or K is not multiple of 16
 * without this, the GEMV ukernel cannot be used directly
 * thus, this is a helper function to prepare Bp from W
 * 
 * @param W original weight matrix, layout [N][K]
 * @param Bp packed a part of weight matrix, layout [K][16]
 * @param K input channel dimension
 * @param N output channel dimension
 * @param n0 starting output channel index in W to pack
 */
static inline void pack_W_nk_to_Bp_k16(
    const int8_t* W,
    int8_t*       Bp,
    uint32_t      K,
    uint32_t      N,
    uint32_t      n0
) {
    for (int k = 0; k < K; k++) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            Bp[k*16 + i] = W[(n0 + i)*K + k];
        }
    }
}




#ifdef __cplusplus
}
#endif
