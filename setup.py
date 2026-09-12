from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="i-whisper",
    version="0.1.0",

    packages=find_packages(),

    ext_modules=[
        CppExtension(
            name="i_whisper.kernels._C",
            sources=[
                "i_whisper/kernels/csrc/pybind.cpp",
                "i_whisper/kernels/csrc/linear8/linear8_torch.cpp",
                "i_whisper/kernels/csrc/linear8/linear8_scalar.cpp",
            ],
            extra_compile_args=[
                "-O0",
                "-g",
                "-std=c++17",
            ],
        ),
    ],

    cmdclass={
        "build_ext": BuildExtension,
    },
)