# demo.py
import torch, os
from torch.utils.cpp_extension import load

ext = load(
    name="w8a8",
    sources=["bindings.cpp", "w8a8_gemm.cpp"],
    extra_cflags=["-march=armv8-a+simd", "-O3", "-std=c++17"],
    verbose=True,
)

def w8a8o32(A, B):
    return ext.w8a8o32_gemm_lt(A, B)

# def w8a8o32_batched(A, B):
#     return ext.w8a8o32_bgemm(A,B)

if __name__ == "__main__":
    M,K,N = 224,512,2048
    A = torch.randint(-128,127,(M,K), dtype=torch.int8, device="cpu")
    B = torch.randint(-128,127,(N,K), dtype=torch.int8, device="cpu")
    C = ext.w8a8o32_gemm_lt(A, B)   # int32

    # 簡單檢驗（注意這不是數學上完全相同的參考）
    C_ref = (A.float() @ B.t().float()).to(torch.int32)
    print("max |diff| =", (C - C_ref).abs().max().item(), C.shape, C.dtype)
    print("C =", C)
    print("C_ref =", C_ref)
    
    # M,K,N = 497,413,451
    # Bsz = 37
    # A = torch.randint(-128,127,(Bsz,M,K), dtype=torch.int8, device="cuda")
    # B = torch.randint(-128,127,(Bsz,K,N), dtype=torch.int8, device="cuda")
    # C = w8a8o32_batched(A,B)   # int32
    # C_ref = torch.matmul(A.float(), B.float()).to(torch.int32)
    # print("batched max |diff| =", (C - C_ref).abs().max().item(), C.shape, C.dtype)
