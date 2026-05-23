# Contributing to PyNear

Thanks for your interest! PyNear is a small, focused KNN library and welcomes
contributions of every size — from typo fixes to new index types. This guide
covers what you need to get a working development build, run the test suites,
and submit a PR that's likely to land quickly.

If you're unsure whether a change is in scope, **open an issue first** and
we'll talk it through before you spend time on a PR.

---

## Ways to contribute

- **Report a bug** — open an issue with a minimal reproducer (Python snippet
  + the platform / Python version / pynear version). Stack traces and the
  output of `pip show pynear` help a lot.
- **Request a feature** — open an issue describing the use case. New indices
  and new distance metrics are welcome; please link the paper or describe
  the algorithm.
- **Improve docs** — `README.md`, `docs/*.md`, and inline docstrings. Doc-only
  PRs don't need to run the full C++ test suite.
- **Submit a fix or feature** — see the workflow below.

---

## Development setup

You need a C++17 compiler, Python 3.8+, and CMake. On Linux:

```console
sudo apt install build-essential cmake clang-format
```

On macOS:

```console
xcode-select --install
brew install cmake clang-format libomp
```

On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
with the "Desktop development with C++" workload.

Then clone and install in editable mode:

```console
git clone https://github.com/pablocael/pynear.git
cd pynear
pip install -e ".[test]"
make init-repo    # installs black, isort, flake8
```

This builds the C++ extension and links it into the source tree. After any
C++ change you need to re-run `pip install -e .` (or `python setup.py
build_ext --inplace`) to rebuild.

---

## Running tests

Python tests (the primary suite — ~195 tests, runs in under a minute):

```console
make test
# or directly:
pytest pynear/tests
```

C++ tests (covers the VPTree core; HNSW / IVF / MIH are exercised through
the Python suite):

```console
make cpp-test
```

Run a single test file or test:

```console
pytest pynear/tests/test_hnsw.py
pytest pynear/tests/test_hnsw.py::test_hnsw_add_remove_rebuild_l2
```

CI runs the Python suite on Linux, macOS (Intel + Apple Silicon), and
Windows across Python 3.8 → 3.12, plus a compile-only check for the
AVX-512 code paths. Wheels for all three platforms must build before a PR
can merge.

---

## Code style

C++ is formatted with `clang-format` (config in `.clang-format`). Python is
formatted with `black` (line-length 120), import-sorted with `isort`, and
linted with `flake8`. One command does all of it:

```console
make fmt
```

The CI lint step will fail if `make fmt` would produce a diff, so run it
before pushing.

Other conventions:

- Keep public Python API surface in `pynear/__init__.py`. Internal C++
  classes are exported from `_pynear` (the pybind11 module).
- Indices should follow the same shape: `set(data)`, `searchKNN(queries,
  k)`, `search1NN(queries)`, pickle support, and — for graph indices —
  `add(vectors)`, `remove(node_id)`, `rebuild()`.
- Tests live in `pynear/tests/test_<feature>.py`. Mirror the file naming.
- No external runtime deps beyond NumPy. Test-only deps (`scikit-learn`,
  `faiss-cpu`) go behind `try: import` guards.

---

## Adding a new index

If you're adding a new index type, the rough checklist is:

1. C++ implementation in `pynear/include/<YourIndex>.hpp` (header-only is
   fine for small indices; otherwise add a `.cpp` in `pynear/src/`).
2. pybind11 binding in `pynear/src/PythonBindings.cpp` — expose `set`,
   `searchKNN`, `search1NN`, `__getstate__` / `__setstate__` for pickle,
   plus any index-specific methods.
3. Re-export from `pynear/__init__.py`.
4. Tests in `pynear/tests/test_<your_index>.py`. Cover at minimum:
   correctness against brute-force, pickle round-trip, edge cases (empty
   index, k > N, wrong dtype).
5. Documentation in `docs/README.md` (and `docs/hnsw.md` if it's an HNSW
   variant). Update the "Choosing an index" table in `README.md`.
6. Benchmark numbers help — even a short paragraph showing the ms/query and
   recall at one operating point is enough to anchor the claim.

For SIMD code: keep an architecture-neutral scalar fallback. Gate AVX2 /
AVX-512 / NEON paths on compiler macros (`__AVX2__`, `__AVX512F__`,
`__ARM_NEON`) so wheels still build on platforms without the intrinsic.

---

## Submitting a PR

1. Fork the repo and create a branch off `main`. Use a descriptive name:
   `feat/<short-name>` for features, `fix/<short-name>` for bugs, `docs/...`
   for doc-only.
2. Write your change and tests. Run `make fmt && make test` locally.
3. Update `CHANGELOG.md` under the *Unreleased* section.
4. Push and open a PR against `main`. Fill in the PR template — what
   changed, why, and how you tested it.
5. CI runs the full matrix on every push. Address red builds before
   requesting review.
6. A maintainer will review. Small PRs land fastest; if your change is
   large, splitting it into independently reviewable commits helps.

Squash-merge is the default; the PR title becomes the commit message, so
write it as a one-line summary (e.g. `feat(hnsw): parallel add() — 6.6x
speedup at 8 threads`).

---

## Reporting security issues

Please **do not** open a public issue for security reports. Email
`pablo.cael@gmail.com` directly with details and a reproducer.

---

## License

By contributing, you agree that your contributions will be licensed under
the MIT License (see `LICENSE`).
