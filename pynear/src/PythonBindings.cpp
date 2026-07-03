/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <BKTree.hpp>
#include <BinaryIVF.hpp>
#include <BuiltinSerializers.hpp>
#include <DistanceFunctions.hpp>
#include <HNSWIndex.hpp>
#include <ISerializable.hpp>
#include <KMeans.hpp>
#include <MIH.hpp>
#include <SerializableVPTree.hpp>

namespace py = pybind11;

typedef float (*distance_func_f)(const arrayf &, const arrayf &);
typedef int64_t (*distance_func_li)(const arrayli &, const arrayli &);

// ── Shared binding helpers ───────────────────────────────────────────────────

// Copy the payload of a py::bytes into a typed vector without going through
// an intermediate std::string copy.
template <class T> static std::vector<T> bytes_to_vec(const py::bytes& b) {
    char* data = nullptr;
    py::ssize_t size = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &data, &size) != 0)
        throw py::error_already_set();
    std::vector<T> out((size_t)size / sizeof(T));
    if (!out.empty())
        std::memcpy(out.data(), data, out.size() * sizeof(T));
    return out;
}

// Validated view of a 2D float32 array: (n, d) shape plus a pointer to the
// row-major data. `caller` parameterises the error message ("set", "add", ...).
struct Rows2D {
    size_t n;
    size_t d;
    const float* ptr;
};

static Rows2D as_rows_2d(py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                         const char* caller) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error(std::string(caller) + "() expects a 2D float32 array of shape (n, d)");
    return Rows2D{(size_t)buf.shape[0], (size_t)buf.shape[1], static_cast<const float*>(buf.ptr)};
}

static std::vector<arrayf> make_row_spans(const float* ptr, size_t n, size_t d) {
    std::vector<arrayf> spans(n);
    for (size_t i = 0; i < n; i++)
        spans[i] = FlatSpan{ptr + i * d, d};
    return spans;
}

// Validate a (n, d) float32 array and build spans over its rows. The spans
// point into the array's buffer, which the caller keeps alive.
static std::vector<arrayf> rows_to_spans(py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                                         const char* caller) {
    Rows2D r = as_rows_2d(arr, caller);
    return make_row_spans(r.ptr, r.n, r.d);
}

// L2-normalise each row of src into dst. Zero-norm rows are left as zeros and
// behave as orthogonal to every unit vector.
static void normalize_rows(const float* src, float* dst, size_t n, size_t d) {
    for (size_t i = 0; i < n; i++) {
        const float* row = src + i * d;
        float sumsq = 0.0f;
        for (size_t j = 0; j < d; j++) sumsq += row[j] * row[j];
        if (sumsq > 0.0f) {
            float inv = 1.0f / std::sqrt(sumsq);
            for (size_t j = 0; j < d; j++) dst[i * d + j] = row[j] * inv;
        } else {
            for (size_t j = 0; j < d; j++) dst[i * d + j] = 0.0f;
        }
    }
}

// ── searchKNN_arrays shared implementation ───────────────────────────────────
// Allocates the two (n, k) output arrays with the GIL HELD, grabs their raw
// buffers, then releases the GIL and invokes `core(idx, dist)` — a callable
// that runs the existing C++ list-of-lists search core — and copies each row
// straight into the buffers. The `py::gil_scoped_release` scope contains only
// plain C++ (the callable must not touch the Python C API: no py:: calls, no
// casts, no allocations of Python objects).
//
// Output rows are NEAREST-FIRST along axis 1 (faiss-style).
// `core_farthest_first` says whether the core's rows follow pynear's
// farthest-first list convention (VPTree / HNSW / MIH-seeded) and must be
// reversed on copy, or are already ascending (MIH, IVF).
//
// Rows shorter than k are padded at the tail: padded ids are -1 and padded
// distances are `pad_dist` (+inf for float distances, INT64_MAX for integer
// Hamming distances).
template <class DistT, class CoreFn>
static std::pair<py::array_t<int64_t>, py::array_t<DistT>>
knn_to_arrays(size_t n, size_t k, DistT pad_dist, bool core_farthest_first, CoreFn&& core) {
    py::array_t<int64_t> ids({(py::ssize_t)n, (py::ssize_t)k});
    py::array_t<DistT> dists({(py::ssize_t)n, (py::ssize_t)k});
    int64_t* ids_ptr = ids.mutable_data();
    DistT* dists_ptr = dists.mutable_data();
    {
        py::gil_scoped_release release;
        std::vector<std::vector<int64_t>> idx;
        std::vector<std::vector<DistT>> dist;
        core(idx, dist);
        for (size_t i = 0; i < n; i++) {
            size_t m = (i < idx.size()) ? idx[i].size() : (size_t)0;
            if (m > k) m = k;
            for (size_t j = 0; j < m; j++) {
                size_t src = core_farthest_first ? (m - 1 - j) : j;
                ids_ptr[i * k + j] = idx[i][src];
                dists_ptr[i * k + j] = dist[i][src];
            }
            for (size_t j = m; j < k; j++) {
                ids_ptr[i * k + j] = -1;
                dists_ptr[i * k + j] = pad_dist;
            }
        }
    }
    return std::make_pair(std::move(ids), std::move(dists));
}

// Flat-buffer variant of knn_to_arrays for cores that already write
// nearest-first rows (with -1 / pad-distance padding) DIRECTLY into the
// output buffers — HNSWIndex::searchKNN_flat / searchKNN_asym_flat. Skips
// the intermediate vector<vector<>> materialisation and the reversed copy
// entirely. Same GIL discipline as knn_to_arrays: arrays allocated with the
// GIL held, `core(ids_ptr, dists_ptr)` runs with it released and must not
// touch the Python C API.
template <class DistT, class CoreFn>
static std::pair<py::array_t<int64_t>, py::array_t<DistT>>
knn_arrays_direct(size_t n, size_t k, CoreFn&& core) {
    py::array_t<int64_t> ids({(py::ssize_t)n, (py::ssize_t)k});
    py::array_t<DistT> dists({(py::ssize_t)n, (py::ssize_t)k});
    int64_t* ids_ptr = ids.mutable_data();
    DistT* dists_ptr = dists.mutable_data();
    {
        py::gil_scoped_release release;
        core(ids_ptr, dists_ptr);
    }
    return std::make_pair(std::move(ids), std::move(dists));
}

// ── Shared HNSW pickle helpers ───────────────────────────────────────────────
// All four HNSW adapters use the same tuple layout for slots 0–10:
//   (flat, levels, flat_adj, adj_offsets, entry, top_level, dim, seed,
//    M, ef_construction, ef_search, ...)
// followed by optional per-class `extra` slots and finally the tombstone
// bytes, so each class's historical layout is preserved exactly:
//   - Float/Cosine/Binary: deleted at slot 11
//   - SQ8:                 scale (11), alpha (12), beta (13), deleted (14);
//                          legacy pickles had scale (11), deleted (12)

template <class HNSW, class... Extra>
static py::tuple hnsw_get_state(const HNSW& hnsw, Extra... extra) {
    std::vector<uint8_t> flat, deleted;
    std::vector<int32_t> levels, flat_adj, adj_offsets;
    int32_t entry, top_level;
    size_t dim;
    uint64_t seed;
    hnsw.serialize(flat, levels, flat_adj, adj_offsets, entry, top_level, dim, seed, deleted);

    py::bytes flat_bytes(reinterpret_cast<const char*>(flat.data()), flat.size());
    py::bytes lvl_bytes(reinterpret_cast<const char*>(levels.data()),
                        levels.size() * sizeof(int32_t));
    py::bytes adj_bytes(reinterpret_cast<const char*>(flat_adj.data()),
                        flat_adj.size() * sizeof(int32_t));
    py::bytes off_bytes(reinterpret_cast<const char*>(adj_offsets.data()),
                        adj_offsets.size() * sizeof(int32_t));
    py::bytes del_bytes(reinterpret_cast<const char*>(deleted.data()), deleted.size());

    return py::make_tuple(flat_bytes, lvl_bytes, adj_bytes, off_bytes,
                          entry, top_level, (uint64_t)dim, seed,
                          (uint64_t)hnsw.M(),
                          (uint64_t)hnsw.ef_construction(),
                          (uint64_t)hnsw.ef_search(),
                          extra...,
                          del_bytes);
}

struct HNSWPickledState {
    std::vector<uint8_t> flat;
    std::vector<int32_t> levels, flat_adj, adj_offsets;
    int32_t entry, top_level;
    uint64_t dim, seed;
    size_t M, ef_construction, ef_search;
    std::vector<uint8_t> deleted;
};

static HNSWPickledState hnsw_parse_state(const py::tuple& t, size_t deleted_slot) {
    HNSWPickledState s;
    s.flat        = bytes_to_vec<uint8_t>(t[0].cast<py::bytes>());
    s.levels      = bytes_to_vec<int32_t>(t[1].cast<py::bytes>());
    s.flat_adj    = bytes_to_vec<int32_t>(t[2].cast<py::bytes>());
    s.adj_offsets = bytes_to_vec<int32_t>(t[3].cast<py::bytes>());
    s.entry       = t[4].cast<int32_t>();
    s.top_level   = t[5].cast<int32_t>();
    s.dim         = t[6].cast<uint64_t>();
    s.seed        = t[7].cast<uint64_t>();
    s.M           = t[8].cast<uint64_t>();
    s.ef_construction = t[9].cast<uint64_t>();
    s.ef_search   = t[10].cast<uint64_t>();

    // Tombstones — optional tail field for backward-compat with pre-tombstone pickles.
    if (py::len(t) > deleted_slot)
        s.deleted = bytes_to_vec<uint8_t>(t[deleted_slot].cast<py::bytes>());
    return s;
}

template <class HNSW>
static void hnsw_restore(HNSW& hnsw, HNSWPickledState&& s) {
    hnsw.deserialize(std::move(s.flat), std::move(s.levels), s.flat_adj, s.adj_offsets,
                     s.entry, s.top_level, (size_t)s.dim, s.seed,
                     s.M, s.ef_construction, s.ef_search,
                     std::move(s.deleted));
}

template <distance_func_f distance> class VPTreeNumpyAdapter {
public:
    VPTreeNumpyAdapter() = default;

    void set(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        std::vector<arrayf> spans = rows_to_spans(arr, "set");
        {
            py::gil_scoped_release release;
            tree.set(std::move(spans));
        }
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k) {
        std::vector<arrayf> spans = rows_to_spans(queries, "searchKNN");

        std::vector<std::vector<int64_t>> indexes;
        std::vector<std::vector<float>> distances;
        {
            py::gil_scoped_release release;
            std::vector<typename vptree::VPTree<arrayf, float, distance>::VPTreeSearchResultElement> results;
            tree.searchKNN(spans, k, results);

            indexes.resize(results.size());
            distances.resize(results.size());
            for (size_t i = 0; i < results.size(); i++) {
                indexes[i] = std::move(results[i].indexes);
                distances[i] = std::move(results[i].distances);
            }
        }
        return std::make_tuple(std::move(indexes), std::move(distances));
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>>
    searchKNN_arrays(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k) {
        std::vector<arrayf> spans = rows_to_spans(queries, "searchKNN_arrays");
        return knn_to_arrays<float>(spans.size(), k, std::numeric_limits<float>::infinity(), true,
            [&](std::vector<std::vector<int64_t>>& idx, std::vector<std::vector<float>>& dist) {
                std::vector<typename vptree::VPTree<arrayf, float, distance>::VPTreeSearchResultElement> results;
                tree.searchKNN(spans, k, results);
                idx.resize(results.size());
                dist.resize(results.size());
                for (size_t i = 0; i < results.size(); i++) {
                    idx[i] = std::move(results[i].indexes);
                    dist[i] = std::move(results[i].distances);
                }
            });
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries) {
        std::vector<arrayf> spans = rows_to_spans(queries, "search1NN");

        std::vector<int64_t> indices;
        std::vector<float> distances;
        {
            py::gil_scoped_release release;
            tree.search1NN(spans, indices, distances);
        }
        return std::make_tuple(std::move(indices), std::move(distances));
    }

    std::string to_string() {
        std::stringstream stream;
        stream << tree;
        return stream.str();
    }

    static py::tuple get_state(const VPTreeNumpyAdapter<distance>& p) {
        const auto& flat = p.tree.flatBacking();
        size_t dim = p.tree.flatDim();
        const auto& indices = p.tree.indexPermutation();
        const auto& pool = p.tree.partitionPool();
        int32_t root_idx = p.tree.rootPartitionIdx();

        py::bytes flat_bytes(reinterpret_cast<const char*>(flat.data()),
                             flat.size() * sizeof(float));
        py::bytes idx_bytes(reinterpret_cast<const char*>(indices.data()),
                            indices.size() * sizeof(int32_t));
        py::bytes pool_bytes(reinterpret_cast<const char*>(pool.data()),
                             pool.size() * sizeof(pool[0]));

        return py::make_tuple(flat_bytes, (uint64_t)dim, idx_bytes, pool_bytes, root_idx);
    }

    static VPTreeNumpyAdapter<distance> set_state(py::tuple t) {
        VPTreeNumpyAdapter<distance> p;

        auto flat_bytes  = t[0].cast<py::bytes>();
        uint64_t dim     = t[1].cast<uint64_t>();
        auto idx_bytes   = t[2].cast<py::bytes>();
        auto pool_bytes  = t[3].cast<py::bytes>();
        int32_t root_idx = t[4].cast<int32_t>();

        // Flat backing
        std::vector<float> flat = bytes_to_vec<float>(flat_bytes);

        // Indices (int32_t since v2.2)
        std::vector<int32_t> indices = bytes_to_vec<int32_t>(idx_bytes);

        // Node pool
        using NodeT = vptree::VPLevelPartition<float>;
        std::vector<NodeT> pool = bytes_to_vec<NodeT>(pool_bytes);

        p.tree.initFromSerialized(std::move(flat), (size_t)dim,
                                  std::move(indices), std::move(pool), root_idx);
        return p;
    }

    vptree::VPTree<arrayf, float, distance> tree;
};

// VPTreeCosineNumpyAdapter — exact cosine-distance KNN.
//
// Implementation: pre-normalize all input rows and query rows to unit length,
// then use an L2 tree internally. For unit vectors:
//
//     ||u − v||² = 2 − 2 (u·v) = 2 (1 − cos(u,v))
//
// so argmin_L2 ≡ argmax_cos and L2 is a true metric (so VPTree pruning is
// correct). Returned distances are converted back to cosine distance
// d_cos = ||u−v||² / 2 ∈ [0, 2], with 0 = identical direction, 1 = orthogonal,
// 2 = antiparallel. Zero-norm rows are left as zeros and behave as orthogonal
// to every unit vector.
class VPTreeCosineNumpyAdapter {
public:
    VPTreeCosineNumpyAdapter() = default;

    void set(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        Rows2D r = as_rows_2d(arr, "set");
        {
            py::gil_scoped_release release;
            std::vector<float> normalized(r.n * r.d);
            normalize_rows(r.ptr, normalized.data(), r.n, r.d);

            tree.set(make_row_spans(normalized.data(), r.n, r.d));  // VPTree copies into its own _flat_backing
        }
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k) {
        Rows2D r = as_rows_2d(queries, "searchKNN");

        std::vector<std::vector<int64_t>> indexes;
        std::vector<std::vector<float>> distances;
        {
            py::gil_scoped_release release;
            std::vector<float> qnorm(r.n * r.d);
            normalize_rows(r.ptr, qnorm.data(), r.n, r.d);

            std::vector<arrayf> spans = make_row_spans(qnorm.data(), r.n, r.d);

            std::vector<typename vptree::VPTree<arrayf, float, dist_l2_f_avx2>::VPTreeSearchResultElement> results;
            tree.searchKNN(spans, k, results);

            indexes.resize(results.size());
            distances.resize(results.size());
            for (size_t i = 0; i < results.size(); i++) {
                indexes[i] = std::move(results[i].indexes);
                distances[i] = std::move(results[i].distances);
                // dist_l2_f_avx2 returns sqrt(L2²) → square then halve to get cosine distance.
                for (size_t j = 0; j < distances[i].size(); j++) {
                    float l2 = distances[i][j];
                    distances[i][j] = (l2 * l2) * 0.5f;
                }
            }
        }
        return std::make_tuple(std::move(indexes), std::move(distances));
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>>
    searchKNN_arrays(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k) {
        Rows2D r = as_rows_2d(queries, "searchKNN_arrays");
        return knn_to_arrays<float>(r.n, k, std::numeric_limits<float>::infinity(), true,
            [&](std::vector<std::vector<int64_t>>& idx, std::vector<std::vector<float>>& dist) {
                std::vector<float> qnorm(r.n * r.d);
                normalize_rows(r.ptr, qnorm.data(), r.n, r.d);
                std::vector<arrayf> spans = make_row_spans(qnorm.data(), r.n, r.d);

                std::vector<typename vptree::VPTree<arrayf, float, dist_l2_f_avx2>::VPTreeSearchResultElement> results;
                tree.searchKNN(spans, k, results);
                idx.resize(results.size());
                dist.resize(results.size());
                for (size_t i = 0; i < results.size(); i++) {
                    idx[i] = std::move(results[i].indexes);
                    dist[i] = std::move(results[i].distances);
                    // dist_l2_f_avx2 returns sqrt(L2²) → square then halve for cosine distance.
                    for (size_t j = 0; j < dist[i].size(); j++) {
                        float l2 = dist[i][j];
                        dist[i][j] = (l2 * l2) * 0.5f;
                    }
                }
            });
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries) {
        Rows2D r = as_rows_2d(queries, "search1NN");

        std::vector<int64_t> indices;
        std::vector<float> distances;
        {
            py::gil_scoped_release release;
            std::vector<float> qnorm(r.n * r.d);
            normalize_rows(r.ptr, qnorm.data(), r.n, r.d);

            std::vector<arrayf> spans = make_row_spans(qnorm.data(), r.n, r.d);

            tree.search1NN(spans, indices, distances);
            for (size_t j = 0; j < distances.size(); j++) {
                float l2 = distances[j];
                distances[j] = (l2 * l2) * 0.5f;
            }
        }
        return std::make_tuple(std::move(indices), std::move(distances));
    }

    std::string to_string() {
        std::stringstream stream;
        stream << tree;
        return stream.str();
    }

    static py::tuple get_state(const VPTreeCosineNumpyAdapter& p) {
        const auto& flat = p.tree.flatBacking();
        size_t dim = p.tree.flatDim();
        const auto& indices = p.tree.indexPermutation();
        const auto& pool = p.tree.partitionPool();
        int32_t root_idx = p.tree.rootPartitionIdx();

        py::bytes flat_bytes(reinterpret_cast<const char*>(flat.data()),
                             flat.size() * sizeof(float));
        py::bytes idx_bytes(reinterpret_cast<const char*>(indices.data()),
                            indices.size() * sizeof(int32_t));
        py::bytes pool_bytes(reinterpret_cast<const char*>(pool.data()),
                             pool.size() * sizeof(pool[0]));

        return py::make_tuple(flat_bytes, (uint64_t)dim, idx_bytes, pool_bytes, root_idx);
    }

    static VPTreeCosineNumpyAdapter set_state(py::tuple t) {
        VPTreeCosineNumpyAdapter p;

        auto flat_bytes  = t[0].cast<py::bytes>();
        uint64_t dim     = t[1].cast<uint64_t>();
        auto idx_bytes   = t[2].cast<py::bytes>();
        auto pool_bytes  = t[3].cast<py::bytes>();
        int32_t root_idx = t[4].cast<int32_t>();

        std::vector<float> flat = bytes_to_vec<float>(flat_bytes);
        std::vector<int32_t> indices = bytes_to_vec<int32_t>(idx_bytes);

        using NodeT = vptree::VPLevelPartition<float>;
        std::vector<NodeT> pool = bytes_to_vec<NodeT>(pool_bytes);

        p.tree.initFromSerialized(std::move(flat), (size_t)dim,
                                  std::move(indices), std::move(pool), root_idx);
        return p;
    }

    vptree::VPTree<arrayf, float, dist_l2_f_avx2> tree;
};

// Extract a filter mask from a py::object (numpy bool or uint8 array).
// Returns a non-null const uint8_t* if the object is a valid 1D array
// of length == expected_size; otherwise returns nullptr. The caller
// must keep the numpy array alive for the lifetime of the pointer
// (pybind11 keeps it alive while it's in scope of the bound function).
inline const uint8_t* extract_filter_mask(const py::object& filter, size_t expected_size) {
    if (filter.is_none()) return nullptr;
    py::array arr = py::cast<py::array>(filter);
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("filter must be a 1D array");
    if ((size_t)buf.shape[0] != expected_size)
        throw std::runtime_error("filter length must match index size");
    // numpy bool and uint8 are both 1 byte per element.
    if (buf.itemsize != 1)
        throw std::runtime_error("filter must be a bool or uint8 array");
    return static_cast<const uint8_t*>(buf.ptr);
}

// HNSWFloatNumpyAdapter — pybind11 wrapper around hnsw::HNSWIndex for float vectors.
// Templated on the distance function, same pattern as VPTreeNumpyAdapter.
template <distance_func_f distance> class HNSWFloatNumpyAdapter {
public:
    HNSWFloatNumpyAdapter(size_t M = 16,
                          size_t ef_construction = 200,
                          size_t ef_search = 50,
                          uint64_t seed = 42,
                          int n_threads = 1)
        : hnsw(M, ef_construction, ef_search, seed, n_threads) {}

    void set(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        std::vector<arrayf> spans = rows_to_spans(arr, "set");
        {
            py::gil_scoped_release release;
            hnsw.set(spans);
        }
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
              py::object filter = py::none()) {
        std::vector<arrayf> spans = rows_to_spans(queries, "searchKNN");

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<std::vector<int64_t>> idx;
        std::vector<std::vector<float>> dist;
        {
            py::gil_scoped_release release;
            hnsw.searchKNN(spans, k, idx, dist, mask);
            // HNSW pipeline runs on squared L2 internally (skips sqrt in the
            // hot loop). Apply sqrt only to the final returned top-k so the
            // public API still reports L2 distances.
            for (auto& row : dist) {
                for (float& v : row) v = std::sqrt(v);
            }
        }
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>>
    searchKNN_arrays(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
                     py::object filter = py::none()) {
        std::vector<arrayf> spans = rows_to_spans(queries, "searchKNN_arrays");
        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());
        const size_t n = spans.size();
        return knn_arrays_direct<float>(n, k,
            [&](int64_t* ids, float* dists) {
                hnsw.searchKNN_flat(spans, k, ids, dists,
                                    std::numeric_limits<float>::infinity(), mask);
                // Same L2² → L2 conversion as searchKNN. Padded entries are
                // +inf and sqrt(inf) == inf, so a full-buffer pass is safe.
                for (size_t i = 0; i < n * k; i++) dists[i] = std::sqrt(dists[i]);
            });
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
              py::object filter = py::none()) {
        std::vector<arrayf> spans = rows_to_spans(queries, "search1NN");

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<int64_t> idx;
        std::vector<float> dist;
        {
            py::gil_scoped_release release;
            hnsw.search1NN(spans, idx, dist, mask);
            for (float& v : dist) v = std::sqrt(v);
        }
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    void set_ef(size_t ef_search) { hnsw.set_ef(ef_search); }
    size_t ef_search() const { return hnsw.ef_search(); }
    size_t size() const { return hnsw.size(); }
    size_t dim() const { return hnsw.dim(); }

    std::vector<int32_t> add(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        std::vector<arrayf> spans = rows_to_spans(arr, "add");
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            ids = hnsw.add(spans);
        }
        return ids;
    }

    void remove_node(int32_t node_id) { hnsw.remove(node_id); }
    size_t num_deleted() const { return hnsw.num_deleted(); }
    std::vector<int32_t> rebuild() {
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            ids = hnsw.rebuild();
        }
        return ids;
    }

    static py::tuple get_state(const HNSWFloatNumpyAdapter<distance>& p) {
        return hnsw_get_state(p.hnsw);
    }

    static HNSWFloatNumpyAdapter<distance> set_state(py::tuple t) {
        HNSWPickledState s = hnsw_parse_state(t, 11);
        HNSWFloatNumpyAdapter<distance> p(s.M, s.ef_construction, s.ef_search, s.seed);
        hnsw_restore(p.hnsw, std::move(s));
        return p;
    }

    hnsw::HNSWIndex<arrayf, float, distance> hnsw;
};

// HNSWCosineNumpyAdapter — L2-normalises input + queries; reuses the L2 HNSW core.
// Same identity as VPTreeCosineNumpyAdapter: for unit vectors,
// ||u - v||^2 = 2 (1 - cos(u, v)), so argmin L2 == argmin cosine.
// Returned distances are converted L2 -> cosine via d_cos = L2^2 / 2.
class HNSWCosineNumpyAdapter {
public:
    HNSWCosineNumpyAdapter(size_t M = 16,
                           size_t ef_construction = 200,
                           size_t ef_search = 50,
                           uint64_t seed = 42,
                           int n_threads = 1)
        : hnsw(M, ef_construction, ef_search, seed, n_threads) {}

    void set(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        Rows2D r = as_rows_2d(arr, "set");
        {
            py::gil_scoped_release release;
            std::vector<float> normalized(r.n * r.d);
            normalize_rows(r.ptr, normalized.data(), r.n, r.d);

            hnsw.set(make_row_spans(normalized.data(), r.n, r.d));
        }
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
              py::object filter = py::none()) {
        Rows2D r = as_rows_2d(queries, "searchKNN");

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<std::vector<int64_t>> idx;
        std::vector<std::vector<float>> dist;
        {
            py::gil_scoped_release release;
            std::vector<float> qnorm(r.n * r.d);
            normalize_rows(r.ptr, qnorm.data(), r.n, r.d);

            std::vector<arrayf> spans = make_row_spans(qnorm.data(), r.n, r.d);

            hnsw.searchKNN(spans, k, idx, dist, mask);
            // hnsw was built with dist_l2sq_f_avx2 which returns L2² directly.
            // Cosine distance: d_cos = L2² / 2 for unit vectors.
            for (auto& row : dist) {
                for (float& v : row) v = v * 0.5f;
            }
        }
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>>
    searchKNN_arrays(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
                     py::object filter = py::none()) {
        Rows2D r = as_rows_2d(queries, "searchKNN_arrays");
        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());
        return knn_arrays_direct<float>(r.n, k,
            [&](int64_t* ids, float* dists) {
                std::vector<float> qnorm(r.n * r.d);
                normalize_rows(r.ptr, qnorm.data(), r.n, r.d);
                std::vector<arrayf> spans = make_row_spans(qnorm.data(), r.n, r.d);

                hnsw.searchKNN_flat(spans, k, ids, dists,
                                    std::numeric_limits<float>::infinity(), mask);
                // L2² → cosine distance (d_cos = L2² / 2 for unit vectors).
                // Padded entries are +inf; inf * 0.5 == inf, so the
                // full-buffer pass leaves them intact.
                for (size_t i = 0; i < r.n * k; i++) dists[i] *= 0.5f;
            });
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
              py::object filter = py::none()) {
        Rows2D r = as_rows_2d(queries, "search1NN");

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<int64_t> idx;
        std::vector<float> dist;
        {
            py::gil_scoped_release release;
            std::vector<float> qnorm(r.n * r.d);
            normalize_rows(r.ptr, qnorm.data(), r.n, r.d);

            std::vector<arrayf> spans = make_row_spans(qnorm.data(), r.n, r.d);

            hnsw.search1NN(spans, idx, dist, mask);
            for (float& v : dist) v = v * 0.5f;
        }
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    void set_ef(size_t ef_search) { hnsw.set_ef(ef_search); }
    size_t ef_search() const { return hnsw.ef_search(); }
    size_t size() const { return hnsw.size(); }
    size_t dim() const { return hnsw.dim(); }

    std::vector<int32_t> add(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        Rows2D r = as_rows_2d(arr, "add");
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            std::vector<float> normalized(r.n * r.d);
            normalize_rows(r.ptr, normalized.data(), r.n, r.d);
            ids = hnsw.add(make_row_spans(normalized.data(), r.n, r.d));
        }
        return ids;
    }

    void remove_node(int32_t node_id) { hnsw.remove(node_id); }
    size_t num_deleted() const { return hnsw.num_deleted(); }
    std::vector<int32_t> rebuild() {
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            ids = hnsw.rebuild();
        }
        return ids;
    }

    static py::tuple get_state(const HNSWCosineNumpyAdapter& p) {
        return hnsw_get_state(p.hnsw);
    }

    static HNSWCosineNumpyAdapter set_state(py::tuple t) {
        HNSWPickledState s = hnsw_parse_state(t, 11);
        HNSWCosineNumpyAdapter p(s.M, s.ef_construction, s.ef_search, s.seed);
        hnsw_restore(p.hnsw, std::move(s));
        return p;
    }

    hnsw::HNSWIndex<arrayf, float, dist_l2sq_f_avx2> hnsw;
};

// HNSWL2NumpyAdapterSQ8 — HNSW with scalar quantisation (int8 vectors).
//
// Codes are per-dimension affine int8 (faiss QT_8bit-style): each dim d maps
// [vmin_d, vmax_d] onto [-128, 127] via
//     code = round((x − beta[d]) / alpha[d]),  decode = alpha[d]·code + beta[d]
// with alpha[d] = (vmax−vmin)/255 and beta[d] = vmin + 128·alpha[d].
// 4× memory bandwidth reduction vs float. Graph CONSTRUCTION uses the fast
// symmetric int8 code-vs-code distance (topology tolerates the extra noise);
// QUERIES are asymmetric (ADC): the float query is kept unquantised and
// compared against decoded codes, which removes the query-side quantisation
// noise and lifts the recall ceiling to within ~0.005 of faiss's
// IndexHNSWSQ(QT_8bit). Scale handling: the adapter pre-subtracts beta from
// each query row and hands the core per-dim alpha (set_sq8_alpha), so the
// kernel's output is already true squared L2 in the original float space —
// the adapter only applies the final sqrt.
//
// Legacy pickles (global symmetric scale) load as alpha[d] = scale,
// beta[d] = 0 — the identical decode — and search asymmetrically too.
class HNSWL2NumpyAdapterSQ8 {
public:
    HNSWL2NumpyAdapterSQ8(size_t M = 16,
                          size_t ef_construction = 200,
                          size_t ef_search = 50,
                          uint64_t seed = 42,
                          int n_threads = 1)
        : hnsw(M, ef_construction, ef_search, seed, n_threads),
          _scale(0.0f) {}

    void set(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        Rows2D r = as_rows_2d(arr, "set");
        {
            py::gil_scoped_release release;
            train_quantiser(r.ptr, r.n, r.d);

            // Quantise into a temporary buffer — hnsw.set() copies the spans'
            // bytes into its own backing, so nothing here needs to outlive it.
            std::vector<int8_t> quantised = encode_rows(r.ptr, r.n, r.d);

            std::vector<SQ8Span> spans(r.n);
            for (size_t i = 0; i < r.n; i++) {
                spans[i] = SQ8Span{quantised.data() + i * r.d, r.d};
            }
            hnsw.set(spans);
            hnsw.set_sq8_alpha(_alpha);
        }
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
              py::object filter = py::none()) {
        Rows2D r = as_rows_2d(queries, "searchKNN");
        size_t n = r.n;
        size_t d = r.d;
        if (_scale <= 0.0f || hnsw.size() == 0) {
            return std::make_tuple(
                std::vector<std::vector<int64_t>>(n),
                std::vector<std::vector<float>>(n));
        }
        check_query_dim(d, "searchKNN");

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<std::vector<int64_t>> idx;
        std::vector<std::vector<float>> dist_f;
        {
            py::gil_scoped_release release;
            // Asymmetric search: float queries, pre-shifted by beta.
            std::vector<float> q_buf = shift_queries(r.ptr, n, d);

            std::vector<std::vector<float>> dist_sq;
            hnsw.searchKNN_asym(q_buf.data(), n, k, idx, dist_sq, mask);
            // Squared L2 (already in original float units) → L2.
            dist_f.resize(idx.size());
            for (size_t i = 0; i < idx.size(); i++) {
                dist_f[i].reserve(dist_sq[i].size());
                for (float di : dist_sq[i]) {
                    dist_f[i].push_back(std::sqrt(di < 0.0f ? 0.0f : di));
                }
            }
        }
        return std::make_tuple(std::move(idx), std::move(dist_f));
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>>
    searchKNN_arrays(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
                     py::object filter = py::none()) {
        Rows2D r = as_rows_2d(queries, "searchKNN_arrays");
        if (_scale <= 0.0f || hnsw.size() == 0) {
            // Empty index — every row is padding (mirrors searchKNN's early return).
            return knn_arrays_direct<float>(r.n, k,
                [&](int64_t* ids, float* dists) {
                    for (size_t i = 0; i < r.n * k; i++) {
                        ids[i] = -1;
                        dists[i] = std::numeric_limits<float>::infinity();
                    }
                });
        }
        check_query_dim(r.d, "searchKNN_arrays");
        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());
        return knn_arrays_direct<float>(r.n, k,
            [&](int64_t* ids, float* dists) {
                std::vector<float> q_buf = shift_queries(r.ptr, r.n, r.d);

                hnsw.searchKNN_asym_flat(q_buf.data(), r.n, k, ids, dists,
                                         std::numeric_limits<float>::infinity(), mask);
                // Squared L2 → L2 (same as searchKNN). Padded entries are
                // +inf: not < 0, and sqrt(inf) == inf, so they pass through.
                for (size_t i = 0; i < r.n * k; i++) {
                    float di = dists[i];
                    dists[i] = std::sqrt(di < 0.0f ? 0.0f : di);
                }
            });
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
              py::object filter = py::none()) {
        auto [idx, dist] = searchKNN(queries, 1, filter);
        std::vector<int64_t> out_idx(idx.size(), -1);
        std::vector<float> out_dist(idx.size(), 0.0f);
        for (size_t i = 0; i < idx.size(); i++) {
            if (!idx[i].empty()) {
                out_idx[i] = idx[i].back();
                out_dist[i] = dist[i].back();
            }
        }
        return std::make_tuple(std::move(out_idx), std::move(out_dist));
    }

    void set_ef(size_t ef) { hnsw.set_ef(ef); }
    size_t ef_search() const { return hnsw.ef_search(); }
    size_t size() const { return hnsw.size(); }
    float scale() const { return _scale; }

    std::vector<int32_t> add(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        Rows2D r = as_rows_2d(arr, "add");
        size_t n = r.n;
        size_t d = r.d;
        // If the index is empty, train the quantiser via set(). Otherwise we
        // must reuse the existing per-dim parameters (retraining would
        // invalidate prior quantisations); out-of-range values clamp.
        if (size() == 0) {
            set(arr);
            std::vector<int32_t> ids(n);
            for (size_t i = 0; i < n; i++) ids[i] = static_cast<int32_t>(i);
            return ids;
        }
        check_query_dim(d, "add");
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            std::vector<int8_t> quantised = encode_rows(r.ptr, n, d);
            std::vector<SQ8Span> spans(n);
            for (size_t i = 0; i < n; i++) {
                spans[i] = SQ8Span{quantised.data() + i * d, d};
            }
            ids = hnsw.add(spans);
            // Refresh the per-node decoded norms for the appended rows.
            hnsw.set_sq8_alpha(_alpha);
        }
        return ids;
    }

    void remove_node(int32_t node_id) { hnsw.remove(node_id); }
    size_t num_deleted() const { return hnsw.num_deleted(); }
    std::vector<int32_t> rebuild() {
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            ids = hnsw.rebuild();
            // Node ids were compacted — recompute the per-node decoded norms.
            hnsw.set_sq8_alpha(_alpha);
        }
        return ids;
    }

    static py::tuple get_state(const HNSWL2NumpyAdapterSQ8& p) {
        // Tuple layout: [...common fields..., scale (11), alpha (12),
        // beta (13), deleted (14)]. Slots 12-13 (per-dim float32 affine
        // parameters) were appended for the ADC upgrade; the historical
        // layout was [..., scale (11), deleted (12)] and set_state still
        // accepts it (length-guarded) for old pickles.
        py::bytes alpha_bytes(reinterpret_cast<const char*>(p._alpha.data()),
                              p._alpha.size() * sizeof(float));
        py::bytes beta_bytes(reinterpret_cast<const char*>(p._beta.data()),
                             p._beta.size() * sizeof(float));
        return hnsw_get_state(p.hnsw, p._scale, alpha_bytes, beta_bytes);
    }

    static HNSWL2NumpyAdapterSQ8 set_state(py::tuple t) {
        // New format has >= 15 slots (see get_state); legacy formats have
        // 12 (pre-tombstone) or 13 (scale + deleted) and carry only the
        // global symmetric scale.
        const bool per_dim_format = py::len(t) >= 15;
        HNSWPickledState s = hnsw_parse_state(t, per_dim_format ? 14 : 12);
        float scale = t[11].cast<float>();

        HNSWL2NumpyAdapterSQ8 p(s.M, s.ef_construction, s.ef_search, s.seed);
        p._scale = scale;
        if (per_dim_format) {
            p._alpha = bytes_to_vec<float>(t[12].cast<py::bytes>());
            p._beta = bytes_to_vec<float>(t[13].cast<py::bytes>());
        }
        hnsw_restore(p.hnsw, std::move(s));
        if (!per_dim_format) {
            // Legacy global-scale codes decode as scale·code + 0 — express
            // them in the per-dim affine form so asymmetric search works.
            p._alpha.assign(p.hnsw.dim(), scale);
            p._beta.assign(p.hnsw.dim(), 0.0f);
        }
        p.hnsw.set_sq8_alpha(p._alpha);
        return p;
    }

    hnsw::HNSWIndex<SQ8Span, float, dist_l2sq_sq8_f> hnsw;

    // Members are public (not private) so get_state/set_state can read/write
    // them directly without separate accessors.
    // _scale: legacy global symmetric scale (max|x|/127) — kept for the
    // `scale` property, for legacy pickles and as the "index non-empty"
    // sentinel. Quantisation itself uses the per-dim parameters below.
    float _scale;
    std::vector<float> _alpha;   // per-dim decode scale
    std::vector<float> _beta;    // per-dim decode offset

private:
    // Train the per-dim affine quantiser (+ legacy _scale) on the database.
    void train_quantiser(const float* ptr, size_t n, size_t d) {
        _alpha.assign(d, 1.0f);
        _beta.assign(d, 0.0f);
        if (n == 0 || d == 0) {
            _scale = 0.0f;
            return;
        }
        std::vector<float> vmin(ptr, ptr + d);
        std::vector<float> vmax(ptr, ptr + d);
        float max_abs = 0.0f;
        for (size_t i = 0; i < n; i++) {
            const float* row = ptr + i * d;
            for (size_t j = 0; j < d; j++) {
                float v = row[j];
                if (v < vmin[j]) vmin[j] = v;
                if (v > vmax[j]) vmax[j] = v;
                float a = std::fabs(v);
                if (a > max_abs) max_abs = a;
            }
        }
        _scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        for (size_t j = 0; j < d; j++) {
            float span = vmax[j] - vmin[j];
            if (span > 0.0f) {
                _alpha[j] = span / 255.0f;
                _beta[j] = vmin[j] + 128.0f * _alpha[j];
            } else {
                // Degenerate dim (constant value): code 0 decodes exactly.
                _alpha[j] = 1.0f;
                _beta[j] = vmin[j];
            }
        }
    }

    // Encode rows with the trained per-dim parameters (clamped to int8).
    std::vector<int8_t> encode_rows(const float* ptr, size_t n, size_t d) const {
        std::vector<int8_t> out(n * d);
        for (size_t i = 0; i < n; i++) {
            const float* row = ptr + i * d;
            for (size_t j = 0; j < d; j++) {
                int v = static_cast<int>(std::round((row[j] - _beta[j]) / _alpha[j]));
                if (v > 127) v = 127;
                else if (v < -128) v = -128;
                out[i * d + j] = static_cast<int8_t>(v);
            }
        }
        return out;
    }

    // Pre-shift queries by beta for the asymmetric kernel (which computes
    // Σ (q[d] − alpha[d]·code[d])² — see HNSWIndex::searchKNN_asym).
    std::vector<float> shift_queries(const float* ptr, size_t n, size_t d) const {
        std::vector<float> out(n * d);
        for (size_t i = 0; i < n; i++) {
            const float* row = ptr + i * d;
            for (size_t j = 0; j < d; j++) {
                out[i * d + j] = row[j] - _beta[j];
            }
        }
        return out;
    }

    void check_query_dim(size_t d, const char* caller) const {
        if (d != _alpha.size()) {
            throw std::runtime_error(std::string(caller) + "(): vector dimension " +
                                     std::to_string(d) + " does not match index dimension " +
                                     std::to_string(_alpha.size()));
        }
    }
};

// HNSWBinaryNumpyAdapter — HNSW over Hamming distance.
// Templated on the distance function so we can plug in the generic
// dist_hamming or any fixed-width specialisation.
template <distance_func_li distance> class HNSWBinaryNumpyAdapter {
public:
    HNSWBinaryNumpyAdapter(size_t M = 16,
                           size_t ef_construction = 200,
                           size_t ef_search = 50,
                           uint64_t seed = 42,
                           int n_threads = 1)
        : hnsw(M, ef_construction, ef_search, seed, n_threads) {}

    void set(const ndarrayli& data) {
        py::gil_scoped_release release;
        hnsw.set(data);
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k,
              py::object filter = py::none()) {
        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());
        std::vector<std::vector<int64_t>> idx, dist;
        {
            py::gil_scoped_release release;
            hnsw.searchKNN(queries, k, idx, dist, mask);
        }
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    std::pair<py::array_t<int64_t>, py::array_t<int64_t>>
    searchKNN_arrays(const ndarrayli& queries, size_t k,
                     py::object filter = py::none()) {
        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());
        return knn_arrays_direct<int64_t>(queries.size(), k,
            [&](int64_t* ids, int64_t* dists) {
                hnsw.searchKNN_flat(queries, k, ids, dists,
                                    std::numeric_limits<int64_t>::max(), mask);
            });
    }

    std::tuple<std::vector<int64_t>, std::vector<int64_t>>
    search1NN(const ndarrayli& queries,
              py::object filter = py::none()) {
        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());
        std::vector<int64_t> idx, dist;
        {
            py::gil_scoped_release release;
            hnsw.search1NN(queries, idx, dist, mask);
        }
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    void set_ef(size_t ef) { hnsw.set_ef(ef); }
    size_t ef_search() const { return hnsw.ef_search(); }
    size_t size() const { return hnsw.size(); }

    std::vector<int32_t> add(const ndarrayli& data) {
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            ids = hnsw.add(data);
        }
        return ids;
    }
    void remove_node(int32_t node_id) { hnsw.remove(node_id); }
    size_t num_deleted() const { return hnsw.num_deleted(); }
    std::vector<int32_t> rebuild() {
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            ids = hnsw.rebuild();
        }
        return ids;
    }

    static py::tuple get_state(const HNSWBinaryNumpyAdapter<distance>& p) {
        return hnsw_get_state(p.hnsw);
    }

    static HNSWBinaryNumpyAdapter<distance> set_state(py::tuple t) {
        HNSWPickledState s = hnsw_parse_state(t, 11);
        HNSWBinaryNumpyAdapter<distance> p(s.M, s.ef_construction, s.ef_search, s.seed);
        hnsw_restore(p.hnsw, std::move(s));
        return p;
    }

    hnsw::HNSWIndex<arrayli, int64_t, distance> hnsw;
};

// MIHSeededHNSWBinaryAdapter — the novel variant.
//
// Combines an HNSW graph over Hamming distance with a parallel
// MIHBinaryIndex. On every query:
//   1. MIH performs an exact lookup within Hamming radius `mih_radius`
//      and returns up to ef_search candidates (in the small-radius regime
//      this is guaranteed by MIH's pigeonhole property).
//   2. Those candidates are passed as `extra_seeds` to HNSW's
//      `search_one_with_seeds`. The layer-0 beam search runs from
//      (HNSW entry node) ∪ (MIH seeds), so:
//        - near-duplicate queries find their answers immediately via MIH;
//        - larger-radius queries get HNSW graph robustness, but with a
//          higher-quality starting set than the bare entry node.
//
// Reference: see docs/hnsw_design.md "Novel variant".
class MIHSeededHNSWBinaryAdapter {
public:
    MIHSeededHNSWBinaryAdapter(size_t M = 16,
                                size_t ef_construction = 200,
                                size_t ef_search = 50,
                                uint64_t seed = 42,
                                int32_t mih_m = 8,
                                int32_t mih_radius = 8,
                                int n_threads = 1)
        : hnsw(M, ef_construction, ef_search, seed, n_threads),
          mih(mih_m),
          _mih_radius(mih_radius) {}

    void set(const ndarrayli& data) {
        py::gil_scoped_release release;
        hnsw.set(data);
        mih.set(data);
    }

    // GIL-free search core shared by searchKNN and searchKNN_arrays. Fills
    // farthest-first rows (pynear's list convention). Contains no Python API
    // calls, so it is safe to run with the GIL released.
    void searchKNN_core(const ndarrayli& queries, size_t k,
                        std::vector<std::vector<int64_t>>& all_idx,
                        std::vector<std::vector<int64_t>>& all_dist) {
        size_t nq = queries.size();
        all_idx.assign(nq, std::vector<int64_t>());
        all_dist.assign(nq, std::vector<int64_t>());

        for (size_t qi = 0; qi < nq; qi++) {
            // 1. MIH small-radius lookup. We ask for up to ef_search candidates
            //    so the seed set is as informative as possible without
            //    being wasteful.
            ndarrayli one_query{queries[qi]};
            auto mih_res = mih.searchKNN(one_query, hnsw.ef_search(), _mih_radius);
            const auto& mih_indices = std::get<0>(mih_res)[0];

            std::vector<int32_t> seeds;
            seeds.reserve(mih_indices.size());
            for (int64_t s : mih_indices) {
                if (s >= 0) seeds.push_back(static_cast<int32_t>(s));
            }

            // 2. HNSW beam search seeded with MIH candidates.
            auto top = hnsw.search_one_with_seeds(queries[qi], k, seeds);

            // 3. Flip to farthest-first within the top-k for pynear's
            //    standard return convention.
            all_idx[qi].reserve(top.size());
            all_dist[qi].reserve(top.size());
            for (auto it = top.rbegin(); it != top.rend(); ++it) {
                all_idx[qi].push_back(it->node_id);
                all_dist[qi].push_back(it->distance);
            }
        }
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k) {
        std::vector<std::vector<int64_t>> all_idx, all_dist;
        {
            py::gil_scoped_release release;
            searchKNN_core(queries, k, all_idx, all_dist);
        }
        return std::make_tuple(std::move(all_idx), std::move(all_dist));
    }

    std::pair<py::array_t<int64_t>, py::array_t<int64_t>>
    searchKNN_arrays(const ndarrayli& queries, size_t k) {
        return knn_to_arrays<int64_t>(queries.size(), k, std::numeric_limits<int64_t>::max(), true,
            [&](std::vector<std::vector<int64_t>>& idx, std::vector<std::vector<int64_t>>& dist) {
                searchKNN_core(queries, k, idx, dist);
            });
    }

    std::tuple<std::vector<int64_t>, std::vector<int64_t>>
    search1NN(const ndarrayli& queries) {
        auto [idx, dist] = searchKNN(queries, 1);
        std::vector<int64_t> out_idx(queries.size(), -1);
        std::vector<int64_t> out_dist(queries.size(), 0);
        for (size_t i = 0; i < queries.size(); i++) {
            if (!idx[i].empty()) {
                out_idx[i] = idx[i].back();    // back = nearest in farthest-first
                out_dist[i] = dist[i].back();
            }
        }
        return std::make_tuple(std::move(out_idx), std::move(out_dist));
    }

    void set_ef(size_t ef) { hnsw.set_ef(ef); }
    void set_mih_radius(int32_t r) { _mih_radius = r; }
    int32_t mih_radius() const { return _mih_radius; }
    size_t ef_search() const { return hnsw.ef_search(); }
    size_t size() const { return hnsw.size(); }

    hnsw::HNSWIndex<arrayli, int64_t, dist_hamming> hnsw;
    MIHBinaryIndex mih;

private:
    int32_t _mih_radius;
};

template <distance_func_li distance> class VPTreeNumpyAdapterBinary {
public:
    VPTreeNumpyAdapterBinary() = default;

    void set(const ndarrayli &array) {
        py::gil_scoped_release release;
        tree.set(array);
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>> searchKNN(const ndarrayli &queries, size_t k) {

        std::vector<std::vector<int64_t>> indexes;
        std::vector<std::vector<int64_t>> distances;
        {
            py::gil_scoped_release release;
            std::vector<typename vptree::VPTree<arrayli, int64_t, distance>::VPTreeSearchResultElement> results;
            tree.searchKNN(queries, k, results);

            indexes.resize(results.size());
            distances.resize(results.size());
            for (size_t i = 0; i < results.size(); ++i) {
                indexes[i] = std::move(results[i].indexes);
                distances[i] = std::move(results[i].distances);
            }
        }
        return std::make_tuple(std::move(indexes), std::move(distances));
    }

    std::pair<py::array_t<int64_t>, py::array_t<int64_t>>
    searchKNN_arrays(const ndarrayli &queries, size_t k) {
        return knn_to_arrays<int64_t>(queries.size(), k, std::numeric_limits<int64_t>::max(), true,
            [&](std::vector<std::vector<int64_t>>& idx, std::vector<std::vector<int64_t>>& dist) {
                std::vector<typename vptree::VPTree<arrayli, int64_t, distance>::VPTreeSearchResultElement> results;
                tree.searchKNN(queries, k, results);
                idx.resize(results.size());
                dist.resize(results.size());
                for (size_t i = 0; i < results.size(); ++i) {
                    idx[i] = std::move(results[i].indexes);
                    dist[i] = std::move(results[i].distances);
                }
            });
    }

    std::tuple<std::vector<int64_t>, std::vector<int64_t>> search1NN(const ndarrayli &queries) {

        std::vector<int64_t> indices;
        std::vector<int64_t> distances;
        {
            py::gil_scoped_release release;
            tree.search1NN(queries, indices, distances);
        }
        return std::make_tuple(std::move(indices), std::move(distances));
    }

    std::string to_string() {
        std::stringstream stream;
        stream << tree;

        return stream.str();
    }

    static py::tuple get_state(const VPTreeNumpyAdapterBinary<distance> &p) {
        vptree::SerializedStateObject state = p.tree.serialize();
        py::tuple t = py::make_tuple(state.data(), state.checksum());
        return t;
    }

    static VPTreeNumpyAdapterBinary<distance> set_state(py::tuple t) {
        VPTreeNumpyAdapterBinary<distance> p;
        std::vector<uint8_t> state = t[0].cast<std::vector<uint8_t>>();
        uint32_t checksum = t[1].cast<uint32_t>();
        p.tree.deserialize(vptree::SerializedStateObject(state, checksum));
        return p;
    }

    vptree::SerializableVPTree<arrayli, int64_t, distance, vptree::ndarraySerializer<uint8_t>, vptree::ndarrayDeserializer<uint8_t>> tree;
};

template <distance_func_li distance_f> class HammingMetric : Metric<arrayli, int64_t> {
public:
    static int64_t distance(const arrayli &a, const arrayli &b) { return distance_f(a, b); }

    static std::optional<int64_t> threshold_distance(const arrayli &a, const arrayli &b, int64_t threshold) { return distance_f(a, b); }
};

template <distance_func_li distance> class BKTreeBinaryNumpyAdapter {
public:
    typedef arrayli key_t;
    typedef int64_t distance_t;

    BKTree<arrayli, distance_t, HammingMetric<distance>> tree;

    BKTreeBinaryNumpyAdapter() = default;

    void set(const std::vector<key_t> &array) {
        py::gil_scoped_release release;
        tree.update(array);
    }

    std::tuple<std::vector<std::vector<index_t>>, std::vector<std::vector<distance_t>>, std::vector<std::vector<key_t>>>
    find_threshold(const std::vector<key_t> &queries, distance_t threshold) {
        std::tuple<std::vector<std::vector<index_t>>, std::vector<std::vector<distance_t>>, std::vector<std::vector<key_t>>> out;
        {
            py::gil_scoped_release release;
            out = tree.find_batch(queries, threshold);
        }
        return out;
    }

    bool empty() { return tree.empty(); }
    size_t size() { return tree.size(); }
    std::vector<key_t> values() { return tree.values(); }
};

// ── IVFFlatBinaryIndex adapter ────────────────────────────────────────────────
class IVFFlatBinaryNumpyAdapter {
public:
    IVFFlatBinaryNumpyAdapter(int32_t nlist = 256, int32_t nprobe = 8,
                               int32_t max_iter = 20, uint32_t seed = 42)
        : _index(nlist, nprobe, max_iter, seed) {}

    void set(const ndarrayli& data) {
        py::gil_scoped_release release;
        _index.set(data);
    }

    std::tuple<std::vector<std::vector<int64_t>>,
               std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k) {
        std::tuple<std::vector<std::vector<int64_t>>,
                   std::vector<std::vector<int64_t>>> out;
        {
            py::gil_scoped_release release;
            out = _index.searchKNN(queries, k);
        }
        return out;
    }

    std::pair<py::array_t<int64_t>, py::array_t<int64_t>>
    searchKNN_arrays(const ndarrayli& queries, size_t k) {
        // IVF's list rows are already ascending (nearest-first) — no reverse.
        return knn_to_arrays<int64_t>(queries.size(), k, std::numeric_limits<int64_t>::max(), false,
            [&](std::vector<std::vector<int64_t>>& idx, std::vector<std::vector<int64_t>>& dist) {
                auto out = _index.searchKNN(queries, k);
                idx = std::move(std::get<0>(out));
                dist = std::move(std::get<1>(out));
            });
    }

    int32_t nlist()  const { return _index.nlist(); }
    int32_t nprobe() const { return _index.nprobe(); }
    void set_nprobe(int32_t nprobe) { _index.set_nprobe(nprobe); }

private:
    IVFFlatBinaryIndex _index;
};

// ── MIHBinaryIndex adapter ────────────────────────────────────────────────────
class MIHBinaryNumpyAdapter {
public:
    explicit MIHBinaryNumpyAdapter(int32_t m = 8) : _index(m) {}

    void set(const ndarrayli& data) {
        py::gil_scoped_release release;
        _index.set(data);
    }

    std::tuple<std::vector<std::vector<int64_t>>,
               std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k, int32_t radius = 8) {
        std::tuple<std::vector<std::vector<int64_t>>,
                   std::vector<std::vector<int64_t>>> out;
        {
            py::gil_scoped_release release;
            out = _index.searchKNN(queries, k, radius);
        }
        return out;
    }

    std::pair<py::array_t<int64_t>, py::array_t<int64_t>>
    searchKNN_arrays(const ndarrayli& queries, size_t k, int32_t radius = 8) {
        // MIH's list rows are already ascending (nearest-first) — no reverse.
        // Radius-limited rows with fewer than k hits get tail padding.
        return knn_to_arrays<int64_t>(queries.size(), k, std::numeric_limits<int64_t>::max(), false,
            [&](std::vector<std::vector<int64_t>>& idx, std::vector<std::vector<int64_t>>& dist) {
                auto out = _index.searchKNN(queries, k, radius);
                idx = std::move(std::get<0>(out));
                dist = std::move(std::get<1>(out));
            });
    }

    int32_t m()      const { return _index.m(); }
    size_t  n()      const { return _index.n(); }
    size_t  nbytes() const { return _index.nbytes(); }

private:
    MIHBinaryIndex _index;
};

static const char *index_set = "Add vectors to index";
static const char *index_topk = "Batch find top-k vectors in index and return indices and distances";
static const char *index_topk_arrays =
    "Batch top-k search returning dense numpy arrays.\n\n"
    "Returns (ids, distances), each of shape (n_queries, k). Unlike searchKNN\n"
    "(which returns per-query lists ordered farthest-first), rows here are\n"
    "ordered NEAREST-FIRST along axis 1 (faiss-style). ids dtype is int64;\n"
    "distances dtype matches the index distance type: float32 for\n"
    "float/cosine/SQ8 indexes, int64 for hamming indexes. Rows with fewer\n"
    "than k results are padded at the end: entries with id == -1 are padding,\n"
    "and their distances are +inf (float) or INT64_MAX (int64).";
static const char *index_top1 = "Batch find closest vectors in index and return indices and distances";
static const char *index_string = "Return a debug string representation of the tree";
static const char *index_find_threshold = "Batch find all vectors below the distance threshold";
static const char *index_values = "Return all stored vectors in arbitrary order";

static float py_dist_l2(py::array_t<float, py::array::c_style | py::array::forcecast> a,
                        py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    auto ba = a.request(); auto bb = b.request();
    FlatSpan sa{static_cast<const float*>(ba.ptr), (size_t)ba.size};
    FlatSpan sb{static_cast<const float*>(bb.ptr), (size_t)bb.size};
    return dist_l2_f_avx2(sa, sb);
}

static float py_dist_l1(py::array_t<float, py::array::c_style | py::array::forcecast> a,
                        py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    auto ba = a.request(); auto bb = b.request();
    FlatSpan sa{static_cast<const float*>(ba.ptr), (size_t)ba.size};
    FlatSpan sb{static_cast<const float*>(bb.ptr), (size_t)bb.size};
    return dist_l1_f_avx2(sa, sb);
}

static float py_dist_chebyshev(py::array_t<float, py::array::c_style | py::array::forcecast> a,
                               py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    auto ba = a.request(); auto bb = b.request();
    FlatSpan sa{static_cast<const float*>(ba.ptr), (size_t)ba.size};
    FlatSpan sb{static_cast<const float*>(bb.ptr), (size_t)bb.size};
    return dist_chebyshev_f_avx2(sa, sb);
}

static py::tuple py_kmeans_l2(
    py::array_t<float, py::array::c_style | py::array::forcecast> data,
    size_t   k,
    size_t   max_iter,
    uint32_t seed)
{
    auto buf = data.request();
    if (buf.ndim != 2)
        throw std::runtime_error("kmeans_l2: data must be a 2D float32 array (N, D)");
    size_t n = (size_t)buf.shape[0];
    size_t d = (size_t)buf.shape[1];
    if (k == 0 || k > n)
        throw std::runtime_error("kmeans_l2: k must be between 1 and N");

    const float* ptr = static_cast<const float*>(buf.ptr);
    KMeansResult res;
    {
        py::gil_scoped_release release;
        res = kmeans_l2(ptr, n, d, k, max_iter, seed);
    }

    py::array_t<int32_t> labels_out({(py::ssize_t)n});
    std::memcpy(labels_out.mutable_data(), res.labels.data(), n * sizeof(int32_t));

    py::array_t<float> centroids_out({(py::ssize_t)k, (py::ssize_t)d});
    std::memcpy(centroids_out.mutable_data(), res.centroids.data(), k * d * sizeof(float));

    return py::make_tuple(labels_out, centroids_out);
}

// Registers the defs shared by the four HNSW adapters (HNSWL2Index,
// HNSWCosineIndex, HNSWL2IndexSQ8, HNSWBinaryIndex). Per-class extras
// (dist_calls, dim, scale, ...) are chained by the caller on the returned
// class_. MIHSeededHNSWBinaryIndex exposes a different surface (no filter
// args, no add/remove/pickle) and is bound by hand below.
template <class Adapter>
static py::class_<Adapter> bind_hnsw_common(py::module_& m, const char* name,
                                            const char* doc = nullptr) {
    py::class_<Adapter> cls = doc ? py::class_<Adapter>(m, name, doc)
                                  : py::class_<Adapter>(m, name);
    cls.def(py::init<size_t, size_t, size_t, uint64_t, int>(),
            py::arg("M") = 16, py::arg("ef_construction") = 200,
            py::arg("ef_search") = 50, py::arg("seed") = 42,
            py::arg("n_threads") = 1)
        .def("set", &Adapter::set, py::arg("vectors"))
        .def("searchKNN", &Adapter::searchKNN,
             py::arg("vectors"), py::arg("k"), py::arg("filter") = py::none())
        .def("searchKNN_arrays", &Adapter::searchKNN_arrays, index_topk_arrays,
             py::arg("vectors"), py::arg("k"), py::arg("filter") = py::none())
        .def("search1NN", &Adapter::search1NN,
             py::arg("vectors"), py::arg("filter") = py::none())
        .def("set_ef", &Adapter::set_ef, py::arg("ef_search"))
        .def("add", &Adapter::add, py::arg("vectors"))
        .def("remove", &Adapter::remove_node, py::arg("node_id"))
        .def("rebuild", &Adapter::rebuild)
        .def_property_readonly("ef_search", &Adapter::ef_search)
        .def_property_readonly("size", &Adapter::size)
        .def_property_readonly("num_deleted", &Adapter::num_deleted)
        .def(py::pickle(&Adapter::get_state, &Adapter::set_state));
    return cls;
}

PYBIND11_MODULE(_pynear, m) {
    m.def("dist_l2", py_dist_l2);
    m.def("dist_l1", py_dist_l1);
    m.def("dist_chebyshev", py_dist_chebyshev);
    m.def("dist_hamming_64", dist_hamming_64);
    m.def("dist_hamming_128", dist_hamming_128);
    m.def("dist_hamming_256", dist_hamming_256);
    m.def("dist_hamming_512", dist_hamming_512);
    m.def("kmeans_l2", py_kmeans_l2,
          "Lloyd K-Means (K-Means++ init, SIMD L2, OpenMP parallel assignment)\n"
          "Args: data (N,D) float32, k, max_iter, seed  →  (labels int32, centroids float32)",
          py::arg("data"), py::arg("k"), py::arg("max_iter") = 100, py::arg("seed") = 42);

    py::class_<VPTreeNumpyAdapter<dist_l2_f_avx2>>(m, "VPTreeL2Index")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter<dist_l2_f_avx2>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapter<dist_l2_f_avx2>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapter<dist_l2_f_avx2>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeNumpyAdapter<dist_l2_f_avx2>::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapter<dist_l2_f_avx2>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapter<dist_l2_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_l2_f_avx2>::set_state));

    py::class_<VPTreeNumpyAdapter<dist_l1_f_avx2>>(m, "VPTreeL1Index")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter<dist_l1_f_avx2>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapter<dist_l1_f_avx2>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapter<dist_l1_f_avx2>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeNumpyAdapter<dist_l1_f_avx2>::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapter<dist_l1_f_avx2>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapter<dist_l1_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_l1_f_avx2>::set_state));

    py::class_<VPTreeNumpyAdapter<dist_chebyshev_f_avx2>>(m, "VPTreeChebyshevIndex")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::set_state));

    py::class_<VPTreeCosineNumpyAdapter>(m, "VPTreeCosineIndex")
        .def(py::init<>())
        .def("set", &VPTreeCosineNumpyAdapter::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeCosineNumpyAdapter::to_string, index_string)
        .def("searchKNN", &VPTreeCosineNumpyAdapter::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeCosineNumpyAdapter::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeCosineNumpyAdapter::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeCosineNumpyAdapter::get_state, &VPTreeCosineNumpyAdapter::set_state));

    bind_hnsw_common<HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>>(m, "HNSWL2Index")
        .def("dist_calls",
             [](const HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>& a){ return a.hnsw.dist_calls(); })
        .def("reset_dist_calls",
             [](HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>& a){ a.hnsw.reset_dist_calls(); })
        .def_property_readonly("dim", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::dim);

    bind_hnsw_common<HNSWCosineNumpyAdapter>(m, "HNSWCosineIndex")
        .def_property_readonly("dim", &HNSWCosineNumpyAdapter::dim);

    bind_hnsw_common<HNSWL2NumpyAdapterSQ8>(m, "HNSWL2IndexSQ8",
        "HNSW with per-dimension affine int8 scalar quantisation and "
        "asymmetric (float-query vs decoded-code) search. ~4x less memory "
        "and faster queries at large N, at a small recall cost vs "
        "HNSWL2Index. Public API mirrors HNSWL2Index; distances returned "
        "are L2 in the original float units.")
        .def_property_readonly("scale", &HNSWL2NumpyAdapterSQ8::scale);

    bind_hnsw_common<HNSWBinaryNumpyAdapter<dist_hamming>>(m, "HNSWBinaryIndex");

    py::class_<MIHSeededHNSWBinaryAdapter>(m, "MIHSeededHNSWBinaryIndex",
        "Novel variant: HNSW over Hamming distance, layer-0 beam search seeded with "
        "exact MIH lookups within a Hamming radius. Gives exact small-radius retrieval "
        "AND HNSW robustness for larger queries in one index. See docs/hnsw_design.md.")
        .def(py::init<size_t, size_t, size_t, uint64_t, int32_t, int32_t, int>(),
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("ef_search") = 50, py::arg("seed") = 42,
             py::arg("mih_m") = 8, py::arg("mih_radius") = 8,
             py::arg("n_threads") = 1)
        .def("set", &MIHSeededHNSWBinaryAdapter::set, py::arg("vectors"))
        .def("searchKNN", &MIHSeededHNSWBinaryAdapter::searchKNN,
             py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &MIHSeededHNSWBinaryAdapter::searchKNN_arrays, index_topk_arrays,
             py::arg("vectors"), py::arg("k"))
        .def("search1NN", &MIHSeededHNSWBinaryAdapter::search1NN, py::arg("vectors"))
        .def("set_ef", &MIHSeededHNSWBinaryAdapter::set_ef, py::arg("ef_search"))
        .def("set_mih_radius", &MIHSeededHNSWBinaryAdapter::set_mih_radius,
             py::arg("radius"))
        .def_property_readonly("mih_radius", &MIHSeededHNSWBinaryAdapter::mih_radius)
        .def_property_readonly("ef_search", &MIHSeededHNSWBinaryAdapter::ef_search)
        .def_property_readonly("size", &MIHSeededHNSWBinaryAdapter::size);

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_512>>(m, "VPTreeBinaryIndex512")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_512>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_512>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_512>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeNumpyAdapterBinary<dist_hamming_512>::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_512>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_512>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_512>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_256>>(m, "VPTreeBinaryIndex256")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_256>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_256>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_256>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeNumpyAdapterBinary<dist_hamming_256>::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_256>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_256>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_256>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_128>>(m, "VPTreeBinaryIndex128")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_128>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_128>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_128>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeNumpyAdapterBinary<dist_hamming_128>::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_128>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_128>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_128>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_64>>(m, "VPTreeBinaryIndex64")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_64>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_64>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_64>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeNumpyAdapterBinary<dist_hamming_64>::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_64>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_64>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_64>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming>>(m, "VPTreeBinaryIndex")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &VPTreeNumpyAdapterBinary<dist_hamming>::searchKNN_arrays, index_topk_arrays, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming>::set_state));

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming_512>>(m, "BKTreeBinaryIndex512")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::set, index_set, py::arg("vectors"))
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::find_threshold, index_find_threshold, py::arg("vectors"),
             py::arg("threshold"))
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::empty)
        .def("size", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::size)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::values, index_values);

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming_256>>(m, "BKTreeBinaryIndex256")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::set, index_set, py::arg("vectors"))
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::find_threshold, index_find_threshold, py::arg("vectors"),
             py::arg("threshold"))
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::empty)
        .def("size", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::size)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::values, index_values);

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming_128>>(m, "BKTreeBinaryIndex128")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::set, index_set, py::arg("vectors"))
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::find_threshold, index_find_threshold, py::arg("vectors"),
             py::arg("threshold"))
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::empty)
        .def("size", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::size)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::values, index_values);

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming_64>>(m, "BKTreeBinaryIndex64")
        .def(py::init<>(), "hi")
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::set, index_set, py::arg("vectors"))
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::find_threshold, index_find_threshold, py::arg("vectors"),
             py::arg("threshold"))
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::empty)
        .def("size", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::size)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::values, index_values);

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming>>(m, "BKTreeBinaryIndex")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming>::set, index_set, py::arg("vectors"))
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming>::find_threshold, index_find_threshold, py::arg("vectors"),
             py::arg("threshold"))
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming>::empty)
        .def("size", &BKTreeBinaryNumpyAdapter<dist_hamming>::size)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming>::values, index_values);

    // ── IVFFlatBinaryIndex ────────────────────────────────────────────────────
    py::class_<IVFFlatBinaryNumpyAdapter>(m, "IVFFlatBinaryIndex")
        .def(py::init<int32_t, int32_t, int32_t, uint32_t>(),
             "Inverted File Index for binary descriptors (approximate Hamming KNN).\n"
             "Args: nlist (clusters), nprobe (clusters scanned per query), "
             "max_iter, seed",
             py::arg("nlist") = 256, py::arg("nprobe") = 8,
             py::arg("max_iter") = 20, py::arg("seed") = 42)
        .def("set", &IVFFlatBinaryNumpyAdapter::set, index_set, py::arg("vectors"))
        .def("searchKNN", &IVFFlatBinaryNumpyAdapter::searchKNN, index_topk,
             py::arg("vectors"), py::arg("k"))
        .def("searchKNN_arrays", &IVFFlatBinaryNumpyAdapter::searchKNN_arrays, index_topk_arrays,
             py::arg("vectors"), py::arg("k"))
        .def("nlist",      &IVFFlatBinaryNumpyAdapter::nlist)
        .def("nprobe",     &IVFFlatBinaryNumpyAdapter::nprobe)
        .def("set_nprobe", &IVFFlatBinaryNumpyAdapter::set_nprobe, py::arg("nprobe"));

    // ── MIHBinaryIndex ────────────────────────────────────────────────────────
    py::class_<MIHBinaryNumpyAdapter>(m, "MIHBinaryIndex")
        .def(py::init<int32_t>(),
             "Multi-Index Hashing for binary descriptors (approximate Hamming KNN).\n"
             "Args: m — number of sub-strings; descriptor byte width must be "
             "divisible by m and nbytes/m ≤ 8.",
             py::arg("m") = 8)
        .def("set", &MIHBinaryNumpyAdapter::set, index_set, py::arg("vectors"))
        .def("searchKNN", &MIHBinaryNumpyAdapter::searchKNN, index_topk,
             py::arg("vectors"), py::arg("k"), py::arg("radius") = 8)
        .def("searchKNN_arrays", &MIHBinaryNumpyAdapter::searchKNN_arrays, index_topk_arrays,
             py::arg("vectors"), py::arg("k"), py::arg("radius") = 8)
        .def("m",      &MIHBinaryNumpyAdapter::m)
        .def("n",      &MIHBinaryNumpyAdapter::n)
        .def("nbytes", &MIHBinaryNumpyAdapter::nbytes);
};
