import os
import platform
import sys

from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages
from setuptools import setup

if sys.platform == "win32":
    extra_compile_args = ["/Wall", "/arch:AVX", "/openmp"]  # /LTCG unrecognized here
    extra_link_args = ["/LTCG"]  # /openmp unrecognized here
elif sys.platform == "darwin":
    # ARCHFLAGS is set by cibuildwheel when cross-compiling (e.g. arm64 host -> x86_64 target).
    # When set, avoid -march=native (which would tune for the host, not the target).
    archflags = os.environ.get("ARCHFLAGS", "")
    is_x86_64 = "x86_64" in archflags or (not archflags and platform.machine() == "x86_64")
    march = [] if archflags else ["-march=native"]
    avx = ["-mavx"] if is_x86_64 else []
    extra_compile_args = ["-flto", "-Wall"] + march + avx + ["-fopenmp"]
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
        define_macros=[("ENABLE_OMP_PARALLEL", "1")],
    ),
]

with open("README.md", "rt", encoding="utf-8") as fr:
    long_description = fr.read()

exec(open("pynear/_version.py").read())
setup(
    name="pynear",
    version=__version__,  # noqa: F821
    packages=find_packages(),
    author="Pablo Carneiro Elias",
    author_email="pablo.cael@gmail.com",
    url="https://github.com/pablocael/pynear",
    description="An efficient implementation of Vantage Point Tree for Python 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=["numpy>=1.21.2"],
    package_dir={"pynear": "pynear"},
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7",
    license_files=("LICENSE",),
)
