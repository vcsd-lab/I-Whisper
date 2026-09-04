#include <cstdint>
#include <arm_neon.h>

namespace w8a8
{
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
    int8_t* pack_B_to_Bp(
        const int8_t* B,
        int32_t       K,
        int32_t       N
    );


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
    void s8s8s32_matmul_neon(
        const int8_t* A,
        const int8_t* B,
        int32_t* C,
        int32_t  M,
        int32_t  N,
        int32_t  K
    );


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
    void u8s8s32_matmul_neon(
        const uint8_t* A,
        const int8_t* B,
        int32_t* C,
        int32_t  M,
        int32_t  N,
        int32_t  K
    );


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
    void s8s8s32_matmul_neon_packed(
        const int8_t* A,
        const int8_t* Bp,
        int32_t* C,
        int32_t  M,
        int32_t  N,
        int32_t  K
    );


    /**
     * @fn s32_RaRshift_single_neon
     * @brief Optimized NEON implementation of Round and Right Shift for int32 array
     *        the Right Shift result is Rounded at final zero point
     *        this means the flow will be:
     *          1. Round at shmnt bit position (round to nearest even)
     *          2. Right Shift by shmnt bits
     * 
     * @param x input int32 array
     * @param y output int32 array
     * @param size number of elements in A and C
     * @param shmnt right shift amount (number of bits)
     */
    void s32_RaRshift_single_neon(
        const int32_t* x,
        int32_t*       y,
        int32_t        size,
        int8_t         shmnt
    );


    /**
     * @fn s32_RaRshift_multiple_neon
     * @brief Optimized NEON implementation of Round and Right Shift for int32 array
     *        the Right Shift result is Rounded at final zero point
     *        this means the flow will be:
     *          1. Round at shmnt bit position (round to nearest even)
     *          2. Right Shift by shmnt bits
     * 
     * @param x input int32 array
     * @param y output int32 array
     * @param size number of elements in A and C
     * @param shmnt right shift amount (number of bits) of each element
     */
    void s32_RaRshift_multiple_neon(
        const int32_t* x,
        int32_t*       y,
        int32_t        size,
        const int8_t*  shmnt
    );

    /**
     * @fn s32_RaRshift_diff
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
    void s32_RaRshift_diff(
        const int32_t* x,
        int32_t*       y,
        int32_t        size,
        const int8_t*  shmnt
    );

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
    void u32_rsqrt_neon(
        const uint32_t* x,
        uint32_t*       y,
        int32_t         size,
        uint8_t         fisr_offset,
        uint8_t         overflow_prevent_offset,
        uint8_t         n_iters
    );
} // namespace w8a8
