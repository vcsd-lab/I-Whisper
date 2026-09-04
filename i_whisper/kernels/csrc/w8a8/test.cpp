#include <iostream>
#include <cstdlib> // 包含 srand 和 rand
#include <cstdint>
#include <arm_neon.h>
#include "w8a8.h"

// Naive implementation of GEMM
// A: [M x K], row-major, layout A[M][K]
// B: [N x K], row-major, layout B[N][K]
// C: [M x N], row-major, layout C[M][N]
void s8s8s32_gemv_naive(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    int32_t M,
    int32_t K,
    int32_t N
)
{
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[m*N + n] += static_cast<int32_t>(A[m*K + k]) * static_cast<int32_t>(B[n*K + k]);
            }
        }
    }
}



// Optimized NEON implementation of GEMV
// realize the below operation in a loop
// C[1xN] = A[1xK] * B[KxN]
//
// A: [1xK], int8, row-major, layout [K]
// B: [NxK], int8, row-major, layout [N][K]
// C: [1xN], int32,row-major, layout [N]
void s8s8s32_gemv_neon(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    int32_t  M,
    int32_t  K,
    int32_t  N
) {
    w8a8::s8s8s32_matmul_neon(A, B, C, M, N, K);
}


static inline uint32_t xorshift32(uint32_t &s){
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

int main(int argc, char** argv) {
    int32_t M = 1500;
    int32_t K = 512;
    int32_t N = 2048;
    alignas(16) int8_t* A = new int8_t[M*K]();    // input vector, A[1 x K], layout [K]
    alignas(16) int8_t* B = new int8_t[N*K]();  // weight matrix, B[N x K], layout [N][K]
    alignas(16) int32_t* C = new int32_t[M*N]();   // output vector, C[1 x N], layout [N]

    uint32_t seed = 123456789;

    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(xorshift32(seed) & 0xFF);
    for (int i = 0; i < N*K; i++) B[i] = (int8_t)(xorshift32(seed) & 0xFF);
    for (int i = 0; i < M*N; i++) C[i] = 0;

    const int iters = 1; // 2e5

    if (argc > 1 && std::string(argv[1]) == "neon") {
        for (int t = 0; t < iters; t++) {
            for (int i = 0; i < M*N; i++) C[i] = 0;

            s8s8s32_gemv_neon(A, B, C, M, K, N);

            asm volatile("" ::: "memory");
        }
    } else if (argc > 1 && std::string(argv[1]) == "naive") {
        for (int t = 0; t < iters; t++) {
            for (int i = 0; i < M*N; i++) C[i] = 0;

            s8s8s32_gemv_naive(A, B, C, M, K, N);

            asm volatile("" ::: "memory");
        }
    } else if (argc > 1 && std::string(argv[1]) == "verify") {
        // srand(static_cast<unsigned int>(time(0)));
        // re-init A, B with random values
        for (int i = 0; i < M*K; i++) A[i] = (int8_t)(rand() & 0xFF);
        for (int i = 0; i < N*K; i++) B[i] = (int8_t)(rand() & 0xFF);
        alignas(16) int32_t* C_ref = new int32_t[M*N]();   // output vector, C[1 x N], layout [N]
        for (int i = 0; i < M*N; i++) C_ref[i] = 0;
        s8s8s32_gemv_naive(A, B, C_ref, M, K, N);
        s8s8s32_gemv_neon(A, B, C, M, K, N);
        bool correct = true;
        for (int32_t i = 0; i < M*N; i++) {
            if (C[i] != C_ref[i]) {
                correct = false;
                std::cout << "Mismatch at index " << i << ": NEON=" << C[i] << ", Ref=" << C_ref[i] << "\n";
            }
        }
        std::cout << "NEON vs Naive verification result\n";
        if (correct) {
            std::cout << "VERIFY:[PASS]\n";
        } else {
            std::cout << "VERIFY:[FAIL]\n";
        }
        // CHECK SUM
        int64_t csum_neon = 0;
        int64_t csum_ref  = 0;
        for (int32_t i = 0; i < N; i++) {
            csum_neon += C[i];
            csum_ref  += C_ref[i];
        }
        std::cout << "CHECK SUM NEON: " << csum_neon << ", REF: " << csum_ref << "\n";
        if (csum_neon == csum_ref) {
            std::cout << "CHECK SUM:[PASS]\n";
        } else {
            std::cout << "CHECK SUM:[FAIL]\n";
        }
    }

    return 0;
}
