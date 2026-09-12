#include <torch/extension.h>
#include <cstring>
#include "w8a8.h"

// ---------- checks ----------
static void check_s8_cpu_config(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(t.dtype() == torch::kInt8, name, " must be int8 (torch.int8)");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_u8_cpu_config(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(t.dtype() == torch::kUInt8, name, " must be uint8 (torch.uint8)");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_bp_layout(const torch::Tensor& Bp, int64_t K) {
    // Bp: [Np/16, Kp*16]
    TORCH_CHECK(Bp.device().is_cpu(), "Bp must be CPU");
    TORCH_CHECK(Bp.dtype() == torch::kChar, "Bp must be int8");
    TORCH_CHECK(Bp.dim() == 2, "Bp must be 2D");
    TORCH_CHECK(Bp.is_contiguous(), "Bp must be contiguous");

    int64_t Kp = (K + 7) & ~7;
    TORCH_CHECK(Bp.size(1) == Kp * 16, "Bp.size(1) must be Kp*16, got ", Bp.size(1),
                " expected ", Kp * 16);
}


// ---------- APIs ----------
torch::Tensor s8s8s32_gemm(torch::Tensor A, torch::Tensor B /* [N,K] */) {
    check_s8_cpu_config(A, "A");
    check_s8_cpu_config(B, "B");

    if (A.dim() == 3 ) {
        TORCH_CHECK(A.size(2) == B.size(1), "A is [Bsz,M,K], B must be [N,K] (row-major W), got K mismatch");
        
        auto Bsz = (uint32_t)A.size(0);
        auto M = (uint32_t)A.size(1);
        auto K = (uint32_t)A.size(2);
        auto N = (uint32_t)B.size(0);

        auto C = torch::empty({Bsz, M, N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

        for (uint32_t b = 0; b < Bsz; b++) {
            w8a8::s8s8s32_matmul_neon(
                A[b].data_ptr<int8_t>(),
                B.data_ptr<int8_t>(),
                C[b].data_ptr<int32_t>(),
                M, N, K
            );
        }
        return C;
    } else if (A.dim() == 2) {
        TORCH_CHECK(A.size(1) == B.size(1), "A is [M,K], B must be [N,K] (row-major W), got K mismatch");

        auto M = (uint32_t)A.size(0);
        auto K = (uint32_t)A.size(1);
        auto N = (uint32_t)B.size(0);

        auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

        w8a8::s8s8s32_matmul_neon(
            A.data_ptr<int8_t>(),
            B.data_ptr<int8_t>(),
            C.data_ptr<int32_t>(),
            M, N, K
        );
        return C;
    } else {
        TORCH_CHECK(false, "A must be 2D or 3D");
        return torch::Tensor();
    }
}


torch::Tensor u8s8s32_gemm(torch::Tensor A, torch::Tensor B /* [N,K] */) {
    check_u8_cpu_config(A, "A");
    check_s8_cpu_config(B, "B");

    if (A.dim() == 3 ) {
        TORCH_CHECK(A.size(2) == B.size(1), "A is [Bsz,M,K], B must be [N,K] (row-major W), got K mismatch");
        
        auto Bsz = (uint32_t)A.size(0);
        auto M = (uint32_t)A.size(1);
        auto K = (uint32_t)A.size(2);
        auto N = (uint32_t)B.size(0);

        auto C = torch::empty({Bsz, M, N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

        for (uint32_t b = 0; b < Bsz; b++) {
            w8a8::u8s8s32_matmul_neon(
                A[b].data_ptr<uint8_t>(),
                B.data_ptr<int8_t>(),
                C[b].data_ptr<int32_t>(),
                M, N, K
            );
        }
        return C;
    } else if (A.dim() == 2) {
        TORCH_CHECK(A.size(1) == B.size(1), "A is [M,K], B must be [N,K] (row-major W), got K mismatch");

        auto M = (uint32_t)A.size(0);
        auto K = (uint32_t)A.size(1);
        auto N = (uint32_t)B.size(0);

        auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

        w8a8::u8s8s32_matmul_neon(
            A.data_ptr<uint8_t>(),
            B.data_ptr<int8_t>(),
            C.data_ptr<int32_t>(),
            M, N, K
        );
        return C;
    } else {
        TORCH_CHECK(false, "A must be 2D or 3D");
        return torch::Tensor();
    }
}


torch::Tensor pack_B_to_Bp(torch::Tensor B /* [N,K] */) {
    check_s8_cpu_config(B, "B");
    int32_t N = (int32_t)B.size(0);
    int32_t K = (int32_t)B.size(1);

    int32_t Np = (N + 15) & ~15;
    int32_t Kp = (K + 7)  & ~7;

    // allocate packed buffer from your C++ packer
    int8_t* Bp_ptr = w8a8::pack_B_to_Bp(B.data_ptr<int8_t>(), K, N);

    // wrap it as a torch tensor (zero-copy) with custom deleter
    auto options = torch::TensorOptions().dtype(torch::kChar).device(torch::kCPU);
    auto sizes = std::vector<int64_t>{Np / 16, (int64_t)Kp * 16};

    auto deleter = [](void* p) { delete[] reinterpret_cast<int8_t*>(p); };
    return torch::from_blob(Bp_ptr, sizes, deleter, options).contiguous();
}

torch::Tensor s8s8s32_gemm_packed(torch::Tensor A /*[M,K]*/, torch::Tensor Bp /*[Np/16, Kp*16]*/, int64_t N) {
    check_s8_cpu_config(A, "A");
    check_s8_cpu_config(Bp, "Bp");
    int32_t N_i32 = (int32_t)N;

    if (A.dim() == 3 ) {
        auto Bsz = (uint32_t)A.size(0);
        auto M = (uint32_t)A.size(1);
        auto K = (uint32_t)A.size(2);
        check_bp_layout(Bp, K);

        auto C = torch::empty({Bsz, M, N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

        for (uint32_t b = 0; b < Bsz; b++) {
            w8a8::s8s8s32_matmul_neon_packed(
                A[b].data_ptr<int8_t>(),
                Bp.data_ptr<int8_t>(),
                C[b].data_ptr<int32_t>(),
                M, N, K
            );
        }
        return C;
    } else if (A.dim() == 2) {
        auto M = (uint32_t)A.size(0);
        auto K = (uint32_t)A.size(1);
        check_bp_layout(Bp, K);

        auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

        w8a8::s8s8s32_matmul_neon_packed(
            A.data_ptr<int8_t>(),
            Bp.data_ptr<int8_t>(),
            C.data_ptr<int32_t>(),
            M, N, K
        );
        return C;
    } else {
        TORCH_CHECK(false, "A must be 2D or 3D");
        return torch::Tensor();
    }
}

torch::Tensor s32_RaRshift(torch::Tensor x, torch::Tensor shmnt) {
    TORCH_CHECK(x.device().is_cpu(), "x must be CPU");
    TORCH_CHECK(x.dtype() == torch::kInt32, "x must be int32");
    TORCH_CHECK(shmnt.device().is_cpu(), "shmnt must be CPU");
    TORCH_CHECK(shmnt.dtype() == torch::kInt8, "shmnt must be int8");
    x = x.contiguous();
    shmnt = shmnt.contiguous();
    int32_t shmnt_num = (int32_t)shmnt.numel();


    if (shmnt_num == 1) {
        auto Y = torch::empty_like(x);
        int8_t shmnt_val = *(shmnt.data_ptr<int8_t>());
        if (shmnt_val == 0) {
            // skip copy if shmnt is zero
            std::memcpy(
                Y.data_ptr<int32_t>(),
                x.data_ptr<int32_t>(),
                x.numel() * sizeof(int32_t)
            );
            return Y;
        } else {
            w8a8::s32_RaRshift_single_neon(
                x.data_ptr<int32_t>(),
                Y.data_ptr<int32_t>(),
                (int32_t)x.numel(),
                shmnt_val
            );
            return Y;
        }

    } else {
        if (x.dim() <2 ) {
            int32_t size = (int32_t)x.size(0);
            TORCH_CHECK(shmnt_num == size, "shmnt numel must be 1 or equal to x's dim 0 (last dim)");
            auto Y = torch::empty_like(x);

            w8a8::s32_RaRshift_multiple_neon(
                x.data_ptr<int32_t>(),
                Y.data_ptr<int32_t>(),
                size,
                shmnt.data_ptr<int8_t>()
            );
            return Y;

        } else {
            int32_t size = (int32_t)x.size(-1);
            TORCH_CHECK(shmnt_num == size, "shmnt numel must be 1 or equal to x's dim (last dim)");
            // and for each row, we can do same shmnt simd Rshift
            auto Y = torch::empty_like(x);

            int32_t vec_size;
            switch (x.dim()) {
                case 2:
                    vec_size = (int32_t)x.size(0);
                    break;
                case 3:
                    vec_size = (int32_t)x.size(0) * (int32_t)x.size(1);
                    break;
                case 4:
                    vec_size = (int32_t)x.size(0) * (int32_t)x.size(1) * (int32_t)x.size(2);
                    break;
                default:
                    TORCH_CHECK(false, "s32_RaRshift not support input dim >4");
                    break;
            }

            for (int32_t i = 0; i < vec_size; i++) {
                w8a8::s32_RaRshift_multiple_neon(
                    x.data_ptr<int32_t>() + i * shmnt_num,
                    Y.data_ptr<int32_t>() + i * shmnt_num,
                    shmnt_num,
                    shmnt.data_ptr<int8_t>()
                );
            }
            return Y;
        }
    }
}

torch::Tensor u32_rsqrt(torch::Tensor x, int fisr_offset, int overflow_prevent_offset, int n_iters) {
    TORCH_CHECK(x.device().is_cpu(), "x must be CPU");
    TORCH_CHECK(x.dtype() == torch::kUInt32, "x must be uint32");
    x = x.contiguous();
    // cast python integers to uint8_t
    uint8_t fisr_offset_u8 = static_cast<uint8_t>(fisr_offset);
    uint8_t overflow_prevent_offset_u8 = static_cast<uint8_t>(overflow_prevent_offset);
    uint8_t n_iters_u8 = static_cast<uint8_t>(n_iters);

    auto Y = torch::empty_like(x);

    w8a8::u32_rsqrt_neon(
        x.data_ptr<uint32_t>(),
        Y.data_ptr<uint32_t>(),
        (int32_t)x.numel(),
        fisr_offset_u8,
        overflow_prevent_offset_u8,
        n_iters_u8
    );
    return Y;
}


PYBIND11_MODULE(w8a8, m) {
    m.def("s8s8s32_gemm", &s8s8s32_gemm, "W8A8->O32 GEMM (CPU), B is [N,K]");
    m.def("u8s8s32_gemm", &u8s8s32_gemm, "U8A8->O32 GEMM (CPU), B is [N,K]");
    m.def("pack_B_to_Bp", &pack_B_to_Bp, "Pack B [N,K] -> Bp [Np/16, Kp*16]");
    m.def("s8s8s32_gemm_packed", &s8s8s32_gemm_packed, "W8A8->O32 GEMM with packed Bp (CPU)");
    m.def("Rshift", &s32_RaRshift, "Int32 Round and Right Shift with different shift amounts (CPU)");
    m.def("u32_rsqrt", &u32_rsqrt, "Uint32 Reciprocal Square Root (CPU)");
}