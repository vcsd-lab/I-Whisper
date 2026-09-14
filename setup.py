from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "i_whisper" / "kernels" / "csrc"


setup(
    name="i-whisper",
    version="0.1.0",

    packages=find_packages(),

    ext_modules=[
        CppExtension(
            name="i_whisper.kernels._C",

            sources=[
                str(CSRC / "pybind.cpp"),
                str(CSRC / "linear8" / "linear8_torch.cpp"),
                str(CSRC / "linear8" / "linear8_scalar.cpp"),
                str(CSRC / "linear8" / "linear8_simde.cpp"),
            ],

            include_dirs=[
                str(CSRC / "simde"),
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