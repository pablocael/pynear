import os
import platform
import subprocess
import sys
import tempfile

from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages
from setuptools import setup


def _tbb_available():
    """Return True if libtbb can be linked on the current platform."""
    src = "int main(){return 0;}"
    try:
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
            f.write(src)
            src_path = f.name
        out_path = src_path.replace(".cpp", ".out")
        result = subprocess.run(
            ["g++", src_path, "-ltbb", "-o", out_path],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        for p in (src_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass

def _target_machine():
    """Best-effort *target* architecture (not necessarily the host's).

    On macOS, cibuildwheel signals cross-compilation through ARCHFLAGS
    (e.g. "-arch x86_64" when building x86_64 wheels on an arm64 host),
    so prefer that over platform.machine().
    """
    if sys.platform == "darwin":
        archflags = os.environ.get("ARCHFLAGS", "")
        if "arm64" in archflags:
            return "arm64"
        if "x86_64" in archflags:
            return "x86_64"
    return platform.machine()


def _simd_flags(machine):
    """Deterministic SIMD flags for gcc/clang builds.

    Default x86-64 baseline is AVX2: `-mavx2 -mfma -mpopcnt` (any
    Haswell-or-newer CPU, 2013+). Wheels built in CI therefore always
    target the same ISA regardless of the build runner's CPU — in
    particular AVX-512 is never emitted, so wheels cannot SIGILL on
    machines without it. DistanceFunctions.hpp picks its kernels at
    compile time from __AVX__/__AVX2__/__AVX512F__/__ARM_NEON, so these
    flags fully determine which SIMD path ships.

    On aarch64/arm64 no flags are needed (and x86 flags like -mavx are
    rejected by the compiler): NEON is implied by the base ISA.

    The PYNEAR_MARCH environment variable replaces the baseline entirely
    with `-march=$PYNEAR_MARCH`:

        PYNEAR_MARCH=native pip install .   # max-perf build for this
                                            # machine (AVX-512 if present)
        PYNEAR_MARCH=x86-64 pip install .   # portable pre-AVX2 build
    """
    march = os.environ.get("PYNEAR_MARCH")
    if march:
        return ["-march=" + march]
    if machine in ("x86_64", "AMD64", "amd64"):
        return ["-mavx2", "-mfma", "-mpopcnt"]
    # aarch64 / arm64 / anything else: no x86 flags.
    return []


if sys.platform == "win32":
    # /arch:AVX2 enables AVX2 + FMA + BMI + F16C — needed for the
    # _mm256_cvtepi8_epi16, _mm256_madd_epi16 (SQ8 kernels) and
    # _mm256_fmadd_ps (L2/dot kernels) intrinsics. /arch:AVX alone is
    # not enough on MSVC; it permits 256-bit float ops but not AVX2 or
    # FMA intrinsics, so the build fails to find those symbols.
    extra_compile_args = ["/Wall", "/arch:AVX2", "/openmp"]  # /LTCG unrecognized here
    extra_link_args = ["/LTCG"]  # /openmp unrecognized here
    extra_macros = [("ENABLE_OMP_PARALLEL", "1")]
elif sys.platform == "darwin":
    # ARCHFLAGS is set by cibuildwheel when cross-compiling (e.g. arm64 host -> x86_64 target).
    # _target_machine() honours it, so x86_64 cross-builds get the same AVX2
    # baseline as native builds (every Intel Mac that can run a supported
    # macOS has AVX2). Note: such wheels cannot be import-tested under
    # Rosetta 2 on macOS <= 14 (Rosetta there stops at SSE4.2) — see
    # CIBW_TEST_SKIP in the CI workflow.
    archflags = os.environ.get("ARCHFLAGS", "")
    is_cross_compiling = bool(archflags)
    simd = _simd_flags(_target_machine())
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
    extra_compile_args = lto + ["-Wall", "-O3", "-fno-math-errno"] + simd + omp_compile
    extra_link_args = omp_link
    # Apple libc++ does not ship std::execution::par_unseq without explicit PSTL;
    # level-parallel OMP build still applies, only the TBB nth_element is disabled.
    extra_macros = [("ENABLE_OMP_PARALLEL", "1")]
else:
    extra_compile_args = ["-flto", "-Wall", "-O3", "-fno-math-errno"] + _simd_flags(_target_machine()) + ["-fopenmp"]
    extra_link_args = ["-fopenmp", "-lgomp"]
    extra_macros = [("ENABLE_OMP_PARALLEL", "1")]
    # Enable TBB parallel nth_element only when libtbb is available (not present
    # in all manylinux images or minimal Linux installs).
    if _tbb_available():
        extra_link_args.append("-ltbb")
        extra_macros.append(("USE_PSTL_NTH_ELEMENT", "1"))

ext_modules = [
    Pybind11Extension(
        "_pynear",
        ["pynear/src/PythonBindings.cpp"],
        include_dirs=["pynear/include"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=extra_macros,
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
    extras_require={
        "test": ["pytest>=6.0"],
        "sklearn": ["scikit-learn"],
    },
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
