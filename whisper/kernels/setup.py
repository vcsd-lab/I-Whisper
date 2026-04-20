import sys

MIN_PYTHON = (3, 8)

if sys.version_info < MIN_PYTHON:
    sys.exit( "Python {}.{} or later is required.".format( *MIN_PYTHON ) )

from pathlib import Path
from setuptools import setup, find_packages, Extension
from glob import glob
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
import setuptools
import os

__PKGNAME__ = 'compact_llm_inference_engine'
__VERSION__ = '0.1.0'
this_dir = Path(__file__).parent

def has_cuda():
    return CUDA_HOME is not None and os.path.isdir(CUDA_HOME)

class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = []

# Add cabac extension
ext_modules.append(
    CppExtension(
        name=f"{__PKGNAME__}._cabac",   # import compact_llm_inference_engine._cabac
        sources=[
            glob( "csrc/cabac/bindings.cpp*" ) + \
            glob( "csrc/cabac/Lib/CommonLib/*.cpp*" ) + \
            glob( "csrc/cabac/Lib/EncLib/*.cpp*" ) + \
            glob( "csrc/cabac/Lib/DecLib/*.cpp*" )
        ],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            "csrc/cabac",
            "csrc/cabac/Lib",
            "csrc/cabac/Lib/DecLib",
            "csrc/cabac/Lib/EncLib"
            ],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    )
)



setup(
    name=__PKGNAME__,
    version=__VERSION__,
    description='Compact LLM Inference Engine, including CABAC entropy coding and quantization kernels.',
    # long_description='',
    ext_modules=ext_modules,
    # install_requires=['pybind11>=2.3'],
    # setup_requires=['pybind11>=2.3'],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
