import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "win32":
    extra_compile_args = ["/Wall", "/arch:AVX"]
else:
    extra_compile_args = ["-flto", "-Wall", "-march=native", "-mavx"]

ext_modules = [
    Pybind11Extension(
        "pyvptree",
        ["src/PythonBindings.cpp"],
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]

with open("README.md", "rt", encoding="utf-8") as fr:
    long_description = fr.read()

setup(
    name="pyvptree",
    version="0.0.1",
    author="Pablo Carneiro Elias",
    author_email="pablo.cael@gmail.com",
    description="An efficient implementation of Vantage Point Tree for Python 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=["numpy>=1.21.2"],
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.6",
    license_files=("LICENSE",),
)
