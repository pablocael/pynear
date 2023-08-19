import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

if sys.platform == "win32":
    extra_compile_args = ["/Wall", "/arch:AVX", "/openmp"]  # /LTCG unrecognized here
    extra_link_args = ["/LTCG"]  # /openmp unrecognized here
elif sys.platform == "darwin":
    extra_compile_args = ["-flto", "-Wall", "-march=native", "-mavx", "-fopenmp"]
    extra_link_args = ["-fopenmp", "-lomp"]
else:
    extra_compile_args = ["-flto", "-Wall", "-march=native", "-mavx", "-fopenmp"]
    extra_link_args = ["-fopenmp", "-lgomp"]

ext_modules = [
    Pybind11Extension(
        "_pynear",
        ["pynear/src/PythonBindings.cpp"],
        include_dirs=["pynear/include"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

with open("README.md", "rt", encoding="utf-8") as fr:
    long_description = fr.read()

exec(open("pynear/_version.py").read())
setup(
    name="pynear",
    version=__version__,
    packages=find_packages(),
    author="Pablo Carneiro Elias",
    author_email="pablo.cael@gmail.com",
    description="An efficient implementation of Vantage Point Tree for Python 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=["numpy>=1.21.2"],
    package_dir={"pynear": "pynear"},
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.6",
    license_files=("LICENSE",),
)
