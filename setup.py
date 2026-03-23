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
    is_cross_compiling = bool(archflags)
    march = [] if archflags else ["-march=native"]
    # When cross-compiling for x86_64 on Apple Silicon (Rosetta 2), AVX is not supported
    # by Rosetta 2 (only up to SSE4.2), so -mavx would cause "Illegal instruction" at import.
    avx = ["-mavx"] if (is_x86_64 and not is_cross_compiling) else []
    # When cross-compiling (ARCHFLAGS set), Homebrew LLVM is arm64-only:
    #   - its libomp.dylib cannot satisfy x86_64 link requests
    #   - its LTO bitcode (LLVM 22) is incompatible with the Apple linker's LTO reader (LLVM 15)
    # So disable both -flto and OpenMP for cross-compilation builds.
    if is_cross_compiling:
        lto = []
        omp_compile = []
        omp_link = []
    else:
        lto = ["-flto"]
        omp_compile = ["-fopenmp"]
        omp_link = ["-fopenmp", "-lomp"]
    extra_compile_args = lto + ["-Wall"] + march + avx + omp_compile
    extra_link_args = omp_link
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
    description="Fast exact KNN search with Vantage Point Trees — L2, L1, Chebyshev and Hamming, SIMD-accelerated",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=["numpy>=1.21.2"],
    package_dir={"pynear": "pynear"},
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
    license_files=("LICENSE",),
    keywords=[
        "knn",
        "k-nearest-neighbors",
        "nearest-neighbor-search",
        "vptree",
        "vantage-point-tree",
        "metric-tree",
        "spatial-index",
        "similarity-search",
        "vector-search",
        "exact-search",
        "hamming-distance",
        "binary-descriptors",
        "feature-matching",
        "computer-vision",
        "simd",
        "avx2",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
)
