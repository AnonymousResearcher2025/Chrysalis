from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "chrysalis._core",
            ["cpp/core.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3", "-march=native", "-fno-fast-math"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
