import os

simd_impl = os.environ.get(
    "IWHISPER_SIMD_IMPL",
    "auto",
).lower()

import platform
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "i_whisper" / "kernels" / "csrc"

machine = platform.machine().lower()

sources = [
    str(CSRC / "pybind.cpp"),
    str(CSRC / "linear8" / "linear8_torch.cpp"),
    str(CSRC / "linear8" / "linear8_scalar.cpp"),
    str(CSRC / "linear8" / "linear8_simd.cpp"),
    str(CSRC / "linear32" / "linear32_torch.cpp"),
    str(CSRC / "linear32" / "linear32_scalar.cpp"),
    str(CSRC / "linear32" / "linear32_simd.cpp"),
    str(CSRC / "residual15" / "residual15_torch.cpp"),
    str(CSRC / "residual15" / "residual15_scalar.cpp"),
    str(CSRC / "residual15" / "residual15_simd.cpp"),
]

include_dirs = []
define_macros = []


build_type = os.environ.get(
    "IWHISPER_BUILD_TYPE",
    "release",
).lower()

if build_type == "debug":
    compile_args = [
        "-O0",
        "-g",
        "-std=c++20",
    ]
    print(
        "[I-Whisper] Build type: debug"
    )

elif build_type == "release":
    compile_args = [
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
    ]
    print(
        "[I-Whisper] Build type: release"
    )

else:
    raise RuntimeError(
        f"Unknown build type: {build_type}"
    )




def get_cpu_features():
    cpuinfo = Path("/proc/cpuinfo")

    if not cpuinfo.exists():
        return set()

    for line in cpuinfo.read_text().splitlines():
        if line.startswith("Features"):
            _, features = line.split(":", 1)
            return set(features.strip().split())

    return set()

if machine in ("aarch64", "arm64"):
    define_macros.append(
        ("IWHISPER_HAS_NEON", "1")
    )

    features = get_cpu_features()
    has_dotprod = "asimddp" in features

    if simd_impl == "auto":
        use_dotprod = has_dotprod

    elif simd_impl == "dotprod":
        if not has_dotprod:
            raise RuntimeError(
                "DOTPROD requested, but CPU does not report asimddp"
            )
        use_dotprod = True

    elif simd_impl == "baseline":
        use_dotprod = False

    else:
        raise RuntimeError(
            f"Unknown SIMD implementation: {simd_impl}"
        )

    if use_dotprod:
        define_macros.append(
            ("IWHISPER_USE_DOTPROD", "1")
        )

        compile_args.append(
            "-march=armv8.2-a+dotprod"
        )

        print(
            "[I-Whisper] SIMD backend: "
            "native NEON + DOTPROD"
        )

    else:
        compile_args.append(
            "-march=armv8-a"
        )

        print(
            "[I-Whisper] SIMD backend: "
            "native NEON baseline"
        )

elif machine in ("x86_64", "amd64"):

    include_dirs.append(
        str(CSRC / "simde")
    )

    if simd_impl in ("auto", "dotprod"):
        define_macros.append(
            ("IWHISPER_USE_DOTPROD", "1")
        )

    elif simd_impl == "baseline":
        pass

    else:
        raise RuntimeError(
            f"Unknown SIMD implementation: {simd_impl}"
        )


setup(
    name="i-whisper",
    version="0.1.0",

    packages=find_packages(),

    ext_modules=[
        CppExtension(
            name="i_whisper.kernels._C",

            sources=sources,

            include_dirs=include_dirs,

            define_macros=define_macros,

            extra_compile_args=compile_args,
        ),
    ],

    cmdclass={
        "build_ext": BuildExtension,
    },
)
