import os, platform
from torch.utils.cpp_extension import load

_W8A8_DISABLED = os.environ.get("W8A8_DISABLE", "0") == "1"
_IS_ARM = platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l")
kernel_path = os.path.dirname(os.path.realpath(__file__))

w8a8 = None

if (not _W8A8_DISABLED) and _IS_ARM:

    w8a8 = load(
        name="w8a8",
        sources=[
            os.path.join(kernel_path, "csrc/pybind.cpp"),
            os.path.join(kernel_path, "csrc/w8a8/w8a8.cpp"),
            os.path.join(kernel_path, "csrc/w8a8/ukernel/w8a8_ukernel_gemv_neon.cpp"),
        ],
        extra_include_paths=[
            os.path.join(kernel_path, "csrc/w8a8"),
            os.path.join(kernel_path, "csrc/w8a8/ukernel"),
        ],
        extra_cflags=["-O3", "-std=c++17", "-march=armv8-a+simd"],
        verbose=True,
    )
else:
    print("NEON W8A8 kernels are disabled or not supported on this platform.")

__all__ = ["w8a8"]
