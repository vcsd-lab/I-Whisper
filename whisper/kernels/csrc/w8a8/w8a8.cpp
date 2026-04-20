#include "w8a8.h"
#include "w8a8_ukernel_gemv_neon.h"
#include "w8a8_ukernel_util_neon.h"
#include <cstring>
#include <cstdlib>
#include <cmath>


#ifndef FIXPOINT_ITERATION
#define FIXPOINT_ITERATION
#endif

/**
 * @fn: pack_B_to_Bp
 * @brief: Pack matrix B [N][K] to packed format Bp 
 *         1. pad zeros
 *         2. transpose & pack
 *         Bp [K*N/16][16] for optimized GEMV ukernel processing
 *         and pad zero if N or K is not multiple of 16
 * @param B: original weight matrix, layout [N][K]
 * @param K: input channel dimension
 * @param N: output channel dimension
 * @return: pointer to packed weight matrix Bp
 */
int8_t* w8a8::pack_B_to_Bp(
    const int8_t* B,
    int32_t       K,
    int32_t       N
) {
    // allocate packed Bp buffer
    // compute padded size
    int32_t N_padded = (N + 15) & ~15; // (N + 15) / 16 * 16
    int32_t K_padded = (K + 7) / 8 * 8;
    int8_t* B_padded = new int8_t[ N_padded * K_padded ];
    // Pad B to B_padded with zeros (no transpose)
    for (int n = 0; n < N_padded; n++) {
        for (int k = 0; k < K_padded; k++) {
            if (n < N && k < K) {
                B_padded[n*K_padded + k] = B[n*K + k];
            } else {
                B_padded[n*K_padded + k] = 0;
            }
        }
    }
    // Pack B_padded to Bp (include transpose)
    int8_t* Bp = new int8_t[ K_padded * N_padded ];
    for (int n0 = 0; n0 < N_padded; n0 += 16) {
        int8_t* Bp_block = Bp + (n0/16) * (K_padded * 16);
        pack_W_nk_to_Bp_k16(B_padded, Bp_block, K_padded, N_padded, n0);
    }

    delete[] B_padded;
    return Bp;
}


/**
 * @fn: s8s8s32_matmul_neon
 * @brief: Optimized NEON implementation of int8 matrix multiplication
 *         C[M][N] = A[M][K] * B[K][N]
 *         all matrices are in row-major layout
 *         assume B is not packed
 *         thus, B is input instead of Bp
 *         and inside this function, pack B to Bp first
 * 
 * @param A: input matrix A, layout [M][K]
 * @param B: input matrix B, layout [K][N]
 * @param C: output matrix C, layout [M][N]
 * @param M: number of rows in A and C
 * @param N: number of columns in B and C
 * @param K: number of columns in A and rows in B
 */
void w8a8::s8s8s32_matmul_neon(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    int32_t  M,
    int32_t  N,
    int32_t  K
) {
    // pack B to Bp first
    // @param B: [N][K], original format
    // @param Bp: [N*K/16][16], packed format
    // @return: pointer to packed weight matrix Bp
    // allocate packed Bp buffer
    // compute padded size
    int32_t Np = (N + 15) & ~15;
    int32_t Kp = (K + 7) & ~7;
    int8_t* B_padded = new int8_t[ Np * Kp ];
    // Pad B to B_padded with zeros (no transpose)
    for (int n = 0; n < Np; n++) {
        for (int k = 0; k < Kp; k++) {
            if (n < N && k < K) {
                B_padded[n*Kp + k] = B[n*K + k];
            } else {
                B_padded[n*Kp + k] = 0;
            }
        }
    }
    // Pack B_padded to Bp (include transpose)
    int8_t* Bp = new int8_t[ Kp * Np ];
    for (int n0 = 0; n0 < Np; n0 += 16) {
        int8_t* Bp_block = Bp + (n0/16) * (Kp * 16);
        pack_W_nk_to_Bp_k16(B_padded, Bp_block, Kp, Np, n0);
    }
    delete[] B_padded;

    // use ukernel of gemv in loop over M rows to realize matmul
    for (int m0 = 0; m0 < M; m0++) {
        int8_t* A_padded = new int8_t[Kp]();
        std::memcpy(A_padded, A + m0*K, K * sizeof(int8_t));
        int32_t* c_row = C + m0*N;

        // Process 16 columns of Bp at a time
        for (int n0 = 0; n0 < Np; n0 += 16) {

            // point to current block of Bp (Kp x 16)
            const int8_t* Bp_block = Bp + (n0/16) * (Kp*16);

            // ukernel always process 16 columns
            // if N is not multiple of 16, some is padded zeros
            // so we need to handle the tail case
            if (n0 + 16 <= N) {
                ukernel_s8s8s32_gemv_8x16_packed_neon(A_padded, Bp_block, c_row + n0, Kp);
            } else {
                // tail case
                int32_t tmp[16] = {0};
                ukernel_s8s8s32_gemv_8x16_packed_neon(A_padded, Bp_block, tmp, Kp);
                for (int i = 0; i < N - n0; i++) c_row[n0+i] = tmp[i];
            }
        }
    }
}


/**
 * @fn u8s8s32_matmul_neon
 * @brief: Optimized NEON implementation of uint8-int8 matrix multiplication
 *         C[M][N] = A[M][K] * B[K][N]
 *         all matrices are in row-major layout
 *         assume B is not packed
 *         thus, B is input instead of Bp
 *         and inside this function, pack B to Bp first
 * 
 * @param A: input matrix A, uint8, layout [M][K]
 * @param B: input matrix B, int8, layout [K][N]
 * @param C: output matrix C, in32, layout [M][N]
 * @param M: number of rows in A and C
 * @param N: number of columns in B and C
 * @param K: number of columns in A and rows in B
 */
void w8a8::u8s8s32_matmul_neon(
    const uint8_t* A,
    const int8_t* B,
    int32_t* C,
    int32_t  M,
    int32_t  N,
    int32_t  K
) {
    // pack B to Bp first
    // @param B: [N][K], original format
    // @param Bp: [N*K/16][16], packed format
    // @return: pointer to packed weight matrix Bp
    // allocate packed Bp buffer
    int32_t Np = (N + 15) & ~15;
    int32_t Kp = (K + 7) & ~7;
    int8_t* B_padded = new int8_t[ Np * Kp ];
    // Pad B to B_padded with zeros (no transpose)
    for (int n = 0; n < Np; n++) {
        for (int k = 0; k < Kp; k++) {
            if (n < N && k < K) {
                B_padded[n*Kp + k] = B[n*K + k];
            } else {
                B_padded[n*Kp + k] = 0;
            }
        }
    }
    // Pack B_padded to Bp (include transpose)
    int8_t* Bp = new int8_t[ Kp * Np ];
    for (int n0 = 0; n0 < Np; n0 += 16) {
        int8_t* Bp_block = Bp + (n0/16) * (Kp * 16);
        pack_W_nk_to_Bp_k16(B_padded, Bp_block, Kp, Np, n0);
    }
    delete[] B_padded;

    // use ukernel of gemv in loop over M rows to realize matmul
    for (int m0 = 0; m0 < M; m0++) {
        uint8_t* A_padded = new uint8_t[Kp]();
        std::memcpy(A_padded, A + m0*K, K * sizeof(uint8_t));
        int32_t* c_row = C + m0*N;
        // Process 16 columns of Bp at a time
        for (int n0 = 0; n0 < Np; n0 += 16) {
            // point to current block of Bp (Kp x 16)
            const int8_t* Bp_block = Bp + (n0/16) * (Kp*16);

            // ukernel always process 16 columns
            // if N is not multiple of 16, some is padded zeros
            // so we need to handle the tail case
            if (n0 + 16 <= N) {
                ukernel_u8s8s32_gemv_8x16_packed_neon(A_padded, Bp_block, c_row + n0, Kp);
            } else {
                // tail case
                int32_t tmp[16] = {0};
                ukernel_u8s8s32_gemv_8x16_packed_neon(A_padded, Bp_block, tmp, Kp);
                for (int i = 0; i < N - n0; i++) c_row[n0+i] = tmp[i];
            }
        }
    }
}

/**
 * @fn: s8s8s32_matmul_neon_packed
 * @brief: Optimized NEON implementation of int8 matrix multiplication
 *         C[M][N] = A[M][K] * B[K][N]
 *         all matrices are in row-major layout
 *         assume B is already packed
 *         thus, Bp is input instead of B
 * 
 * @param A: input matrix A, layout [M][K]
 * @param B: input matrix B, layout [K][N]
 * @param C: output matrix C, layout [M][N]
 * @param M: number of rows in A and C
 * @param N: number of columns in B and C
 * @param K: number of columns in A and rows in B
 */
void w8a8::s8s8s32_matmul_neon_packed(
    const int8_t* A,
    const int8_t* Bp,
    int32_t* C,
    int32_t  M,
    int32_t  N,
    int32_t  K
) {
    // use ukernel of gemv in loop over M rows to realize matmul
    int32_t Kp = (K + 7) & ~7;

    for (int m0 = 0; m0 < M; m0++) {
        int8_t* A_padded = new int8_t[Kp]();
        std::memcpy(A_padded, A + m0*K, K * sizeof(int8_t));
        int32_t* c_row = C + m0*N;

        // Process 16 columns of Bp at a time
        for (int n0 = 0; n0 < N; n0 += 16) {

            // point to current block of Bp (Kp x 16)
            const int8_t* Bp_block = Bp + n0*Kp; // Bp + (n0/16) * (Kp*16);

            // ukernel always process 16 columns
            // if N is not multiple of 16, some is padded zeros
            // so we need to handle the tail case
            if (n0 + 16 <= N) {
                ukernel_s8s8s32_gemv_8x16_packed_neon(A_padded, Bp_block, c_row + n0, Kp);
            } else {
                // tail case
                int32_t tmp[16] = {0};
                ukernel_s8s8s32_gemv_8x16_packed_neon(A_padded, Bp_block, tmp, Kp);
                for (int i = 0; i < N - n0; i++) c_row[n0+i] = tmp[i];
            }
        }
    }
}


/**
 * @fn s32_RaRshift_single_neon
 * @brief Optimized NEON implementation of Round and Right Shift for int32 array
 *        the Right Shift result is Rounded at final zero point
 *        this means the flow will be:
 *          1. Round at shmnt bit position (round to nearest even)
 *          2. Right Shift by shmnt bits
 * 
 * @param x input int32 array
 * @param size number of elements in A and C
 * @param shmnt right shift amount (number of bits)
 * @return pointer to output int32 array (need to be deleted by user)
 */
void w8a8::s32_RaRshift_single_neon(
    const int32_t* x,
    int32_t*       y,
    int32_t        size,
    int8_t         shmnt
) {
    int32_t loop_num = (size + 3) >> 2; // (size + 3) / 4 consider tail not multiple of 4
    int32_t pad_num = (4 - (size & 3)) & 3;  // padding size for last incomplete vector

    for (int32_t i = 0; i < loop_num; i ++) {
        int32x4_t x_vec = vld1q_s32(x + i * 4); // 4 elements in ukernel
        int32x4_t y_vec = ukernel_s32_RaRshift_s32_neon(x_vec, shmnt);
        if (i == loop_num - 1 && pad_num > 0) {
            // tail case with padding
            for (int32_t j = 0; j < 4 - pad_num; j++) {
            y[i * 4 + j] = vgetq_lane_s32(y_vec, j);  // Extract the remaining valid lanes
            }
        } else {
            vst1q_s32(y + i * 4, y_vec);
        }
    }
}


static inline int32_t rshift_rtn_scalar(int32_t x, int8_t n) {
    if (n <= 0) return x;                           // n=0 => no shift
    if (n >= 31) return (x >= 0) ? 0 : -1;          // big shift => all bits shifted out, prevent UB

    const int32_t half = 1 << (n - 1);
    const int32_t bias = (x >= 0) ? half : -half;
    return (x + bias) >> n;
}



/**
 * @fn s32_RaRshift_multiple_neon
 * @brief Optimized NEON implementation of Round and Right Shift for int32 array
 *        the Right Shift result is Rounded at final zero point
 *        this means the flow will be:
 *          1. Round at shmnt bit position (round to nearest even)
 *          2. Right Shift by shmnt bits
 * 
 * @param x input int32 array
 * @param size number of elements in A and C
 * @param shmnt right shift amount (number of bits) of each element
 * @return pointer to output int32 array (need to be deleted by user)
 */
void w8a8::s32_RaRshift_multiple_neon(
    const int32_t* x,
    int32_t*       y,
    int32_t        size,
    const int8_t*  shmnt
) {
    const int32_t full = size & ~7;     // largest multiple of 8 <= size
    int32_t i = 0;

    for (; i < full; i += 8) {
        int32x4_t x0 = vld1q_s32(x + i);
        int32x4_t x1 = vld1q_s32(x + i + 4);

        int8x8_t sh_s8 = vld1_s8(shmnt + i);

        // --- unpack 8x u8 -> two int32x4_t (ZERO-extend version, keep your current behavior) ---
        int16x8_t sh_s16 = vmovl_s8(sh_s8);
        int32x4_t sh0_s32 = vmovl_s16(vget_low_s16(sh_s16));
        int32x4_t sh1_s32 = vmovl_s16(vget_high_s16(sh_s16));

        int32x4_t y0 = ukernel_s32_RaRshift_s32x4_neon(x0, sh0_s32);
        int32x4_t y1 = ukernel_s32_RaRshift_s32x4_neon(x1, sh1_s32);

        vst1q_s32(y + i,     y0);
        vst1q_s32(y + i + 4, y1);
    }

    // tail: i..size-1 
    for (; i < size; i++) {
        y[i] = rshift_rtn_scalar(x[i], shmnt[i]);
    }
}


/**
 * @fn: s32_RaRshift_diff
 * @brief
 * This function performs element-wise Round and Right Shift
 * on an int32 input array, where each element can have a different
 * shift amount specified by the shmnt array.
 * The Right Shift result is Rounded at final zero point
 * this means the flow will be:
 *  1. Round at shmnt bit position (round to nearest even)
 *  2. Right Shift by shmnt bits
 * @note This function does not utilize NEON optimizations due to variable shift amounts.
 * @param x: input int32 array
 * @param y: output int32 array
 * @param size: number of elements in x and y
 * @param shmnt: array of right shift amounts (number of bits) for each element
 */
#ifndef ROUND_TO_NEAREST_EVEN
#define ROUND_TO_NEAREST
#endif
void w8a8::s32_RaRshift_diff(
    const int32_t* x,
    int32_t*       y,
    int32_t        size,
    const int8_t*  shmnt
) {
    for (int32_t i = 0; i < size; ++i) {
        int8_t shmnt_cur = shmnt[i];
        if (shmnt_cur == 0) {
            y[i] = x[i];
            continue;
        }
#ifdef ROUND_TO_NEAREST_EVEN
        int32_t q = x[i] >> shmnt_cur;
        int32_t bias = (1 << (shmnt_cur - 1)) - 1 + (q & 1);
        y[i] = (x[i] + bias) >> shmnt_cur;
#else // ROUND_TO_NEAREST
        // simple right shift with round to nearest (no tie to even)
        int32_t bias = 1 << (shmnt_cur - 1);
        bias -= x[i]>>31; // adjust bias for negative numbers
        y[i] = (x[i] + bias) >> shmnt_cur;
#endif
    }
}

static inline uint32_t cast_f32_to_u32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}
static inline float cast_u32_to_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
/**
 * @fn u32_rsqrt_neon
 * @brief Optimized NEON implementation of Reciprocal Square Root for uint32 array
 *        using Fast Inverse Square Root method with Newton-Raphson iterations
 * @param x input uint32 array
 * @param y output uint32 array
 * @param size number of elements in x and y
 * @param fisr_offset fixed-point position of result
 * @param overflow_prevent_offset right shift offset to prevent overflow in iteration
 * @param n_iters number of Newton-Raphson iterations to perform
 */
#define FISR_MAGIC_NUM 0x5f3759df
#define MANTISSA_BITS 23
void w8a8::u32_rsqrt_neon(
    const uint32_t* x,
    uint32_t*       y,
    int32_t         size,
    uint8_t         fisr_offset,
    uint8_t         overflow_prevent_offset,
    uint8_t         n_iters
) {
    const int32_t full = size & ~3; // largest multiple of 4 <= size
    int32_t i = 0;

    for (; i < full; i += 4) {
        uint32x4_t x_vec = vld1q_u32(x + i); // 4 elements in ukernel
        uint32x4_t y_vec = ukernel_rsqrt_u32x4_neon(
            x_vec,
            fisr_offset,
            overflow_prevent_offset,
            n_iters
        );
        vst1q_u32(y + i, y_vec);
    }

    // tail: i..size-1
    for (; i < size; i++) {

        // ---- 0) handle zero input ----
        const uint32_t one_fixpoint = 1 << (fisr_offset);
        if (x[i] == 0) {
            y[i] = one_fixpoint;
            continue; // skip zero input
        }

        // ---- 1) initial estimate via magic number ----
        uint32_t init_est = cast_f32_to_u32((float32_t)x[i]);
        // initial estimate using magic number
        init_est = FISR_MAGIC_NUM - (init_est >> 1);
        // this result is first approximation if viewed in float32

#ifdef FIXPOINT_ITERATION

        // ---- 2) adjust initial estimate to fixed-point Q(fisr_offset) ----
        init_est += ((uint32_t)(fisr_offset) << MANTISSA_BITS);
        // add fisr_offset to exponent bits
        // this is equivalent to multiplying by 2^(fisr_offset) in float
        uint32_t estimate = static_cast<uint32_t>(cast_u32_to_f32(init_est));

        // ---- 3) Newton-Raphson iterations in fixed-point ----
        const uint32_t x2 = x[i] >> 1;
        const uint32_t threehalfs_fix = (uint32_t)(3 << (fisr_offset - 1)); // Q(fisr_offset)
        for (uint8_t iter = 0; iter < n_iters; iter++) {
            // Newton-Raphson iteration: estimate = estimate * (1.5 - 0.5 * x * estimate * estimate)
            // all operations are in fixed-point
            uint32_t t = (x2 * estimate) >> fisr_offset; // Q(integer)
            t = (t * estimate);                          // Q(fisr_offset)
            // why only shift once here?
            // because x2 is in integer format, not fixed-point
            // thus, only one shift is needed to keep Q(fisr_offset)
            estimate = (estimate * (threehalfs_fix - t)) >> fisr_offset; // Q(fisr_offset)
        }

        y[i] = estimate;

#undef FIXPOINT_ITERATION
#else //FLOAT_RSQRT

        // ---- 2) convert initial estimate to float32 ----
        float32_t estimate = cast_u32_to_f32(init_est); // initial estimate in float32
        // why not using vrsqrtesq_f32
        // after testing, vrsqrtesq_f32 is less accurate than Fast Inverse Square Root initial estimate

        // ---- 3) Newton-Raphson iterations in float32 ----
        float32_t x_f = static_cast<float32_t>(x[i]); // convert integer value to float32 format
        // for (uint8_t iter = 0; iter < n_iters; iter++) {
        //     estimate = estimate * (1.5f - 0.5f * x_f * estimate * estimate);
        // }

        // ---- 4) convert final estimate to fixed-point ----
        uint32_t estimate_bits = cast_f32_to_u32(estimate);
        // add fisr_offset to exponent bits
        estimate_bits += ((uint32_t)(fisr_offset) << MANTISSA_BITS);
        // int like adding in exponent bits leads to multiply 2^(fisr_offset) in float
        y[i] = static_cast<uint32_t>( cast_u32_to_f32(estimate_bits)); // cast u32 back to f32 then convert to u32

#undef FLOAT_ITERATION
#endif
    }
}