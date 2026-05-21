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
#include <omp.h>
#include <sstream>
#include <stdexcept>
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

template <distance_func_f distance> class VPTreeNumpyAdapter {
public:
    VPTreeNumpyAdapter() = default;

    void set(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("set() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{ptr + i * d, d};
        tree.set(spans);
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("searchKNN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{ptr + i * d, d};

        std::vector<typename vptree::VPTree<arrayf, float, distance>::VPTreeSearchResultElement> results;
        tree.searchKNN(spans, k, results);

        std::vector<std::vector<int64_t>> indexes(results.size());
        std::vector<std::vector<float>> distances(results.size());
        for (size_t i = 0; i < results.size(); i++) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
        }
        return std::make_tuple(indexes, distances);
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("search1NN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{ptr + i * d, d};

        std::vector<int64_t> indices;
        std::vector<float> distances;
        tree.search1NN(spans, indices, distances);
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
        std::string flat_str(flat_bytes);
        std::vector<float> flat(flat_str.size() / sizeof(float));
        std::memcpy(flat.data(), flat_str.data(), flat_str.size());

        // Indices (int32_t since v2.2)
        std::string idx_str(idx_bytes);
        std::vector<int32_t> indices(idx_str.size() / sizeof(int32_t));
        std::memcpy(indices.data(), idx_str.data(), idx_str.size());

        // Node pool
        using NodeT = vptree::VPLevelPartition<float>;
        std::string pool_str(pool_bytes);
        std::vector<NodeT> pool(pool_str.size() / sizeof(NodeT));
        std::memcpy(pool.data(), pool_str.data(), pool_str.size());

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

    void set(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("set() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);

        std::vector<float> normalized(n * d);
        normalize_rows(ptr, normalized.data(), n, d);

        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{normalized.data() + i * d, d};
        tree.set(spans);  // VPTree copies into its own _flat_backing
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("searchKNN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);

        std::vector<float> qnorm(n * d);
        normalize_rows(ptr, qnorm.data(), n, d);

        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{qnorm.data() + i * d, d};

        std::vector<typename vptree::VPTree<arrayf, float, dist_l2_f_avx2>::VPTreeSearchResultElement> results;
        tree.searchKNN(spans, k, results);

        std::vector<std::vector<int64_t>> indexes(results.size());
        std::vector<std::vector<float>> distances(results.size());
        for (size_t i = 0; i < results.size(); i++) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
            // dist_l2_f_avx2 returns sqrt(L2²) → square then halve to get cosine distance.
            for (size_t j = 0; j < distances[i].size(); j++) {
                float l2 = distances[i][j];
                distances[i][j] = (l2 * l2) * 0.5f;
            }
        }
        return std::make_tuple(indexes, distances);
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("search1NN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);

        std::vector<float> qnorm(n * d);
        normalize_rows(ptr, qnorm.data(), n, d);

        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{qnorm.data() + i * d, d};

        std::vector<int64_t> indices;
        std::vector<float> distances;
        tree.search1NN(spans, indices, distances);
        for (size_t j = 0; j < distances.size(); j++) {
            float l2 = distances[j];
            distances[j] = (l2 * l2) * 0.5f;
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

        std::string flat_str(flat_bytes);
        std::vector<float> flat(flat_str.size() / sizeof(float));
        std::memcpy(flat.data(), flat_str.data(), flat_str.size());

        std::string idx_str(idx_bytes);
        std::vector<int32_t> indices(idx_str.size() / sizeof(int32_t));
        std::memcpy(indices.data(), idx_str.data(), idx_str.size());

        using NodeT = vptree::VPLevelPartition<float>;
        std::string pool_str(pool_bytes);
        std::vector<NodeT> pool(pool_str.size() / sizeof(NodeT));
        std::memcpy(pool.data(), pool_str.data(), pool_str.size());

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
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("set() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{ptr + i * d, d};
        hnsw.set(spans);
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
              py::object filter = py::none()) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("searchKNN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{ptr + i * d, d};

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<std::vector<int64_t>> idx;
        std::vector<std::vector<float>> dist;
        hnsw.searchKNN(spans, k, idx, dist, mask);
        // HNSW pipeline runs on squared L2 internally (skips sqrt in the
        // hot loop). Apply sqrt only to the final returned top-k so the
        // public API still reports L2 distances.
        for (auto& row : dist) {
            for (float& v : row) v = std::sqrt(v);
        }
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
              py::object filter = py::none()) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("search1NN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{ptr + i * d, d};

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<int64_t> idx;
        std::vector<float> dist;
        hnsw.search1NN(spans, idx, dist, mask);
        for (float& v : dist) v = std::sqrt(v);
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    void set_ef(size_t ef_search) { hnsw.set_ef(ef_search); }
    size_t ef_search() const { return hnsw.ef_search(); }
    size_t size() const { return hnsw.size(); }
    size_t dim() const { return hnsw.dim(); }

    std::vector<int32_t> add(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("add() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{ptr + i * d, d};
        return hnsw.add(spans);
    }

    void remove_node(int32_t node_id) { hnsw.remove(node_id); }
    size_t num_deleted() const { return hnsw.num_deleted(); }
    std::vector<int32_t> rebuild() { return hnsw.rebuild(); }

    static py::tuple get_state(const HNSWFloatNumpyAdapter<distance>& p) {
        std::vector<uint8_t> flat, deleted;
        std::vector<int32_t> levels, flat_adj, adj_offsets;
        int32_t entry, top_level;
        size_t dim;
        uint64_t seed;
        p.hnsw.serialize(flat, levels, flat_adj, adj_offsets, entry, top_level, dim, seed, deleted);

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
                              (uint64_t)p.hnsw.M(),
                              (uint64_t)p.hnsw.ef_construction(),
                              (uint64_t)p.hnsw.ef_search(),
                              del_bytes);
    }

    static HNSWFloatNumpyAdapter<distance> set_state(py::tuple t) {
        auto flat_bytes  = t[0].cast<py::bytes>();
        auto lvl_bytes   = t[1].cast<py::bytes>();
        auto adj_bytes   = t[2].cast<py::bytes>();
        auto off_bytes   = t[3].cast<py::bytes>();
        int32_t entry    = t[4].cast<int32_t>();
        int32_t top_lvl  = t[5].cast<int32_t>();
        uint64_t dim     = t[6].cast<uint64_t>();
        uint64_t seed    = t[7].cast<uint64_t>();
        size_t M         = t[8].cast<uint64_t>();
        size_t ef_con    = t[9].cast<uint64_t>();
        size_t ef_search = t[10].cast<uint64_t>();

        std::string flat_str(flat_bytes);
        std::vector<uint8_t> flat(flat_str.size());
        std::memcpy(flat.data(), flat_str.data(), flat_str.size());

        std::string lvl_str(lvl_bytes);
        std::vector<int32_t> levels(lvl_str.size() / sizeof(int32_t));
        std::memcpy(levels.data(), lvl_str.data(), lvl_str.size());

        std::string adj_str(adj_bytes);
        std::vector<int32_t> flat_adj(adj_str.size() / sizeof(int32_t));
        std::memcpy(flat_adj.data(), adj_str.data(), adj_str.size());

        std::string off_str(off_bytes);
        std::vector<int32_t> adj_offsets(off_str.size() / sizeof(int32_t));
        std::memcpy(adj_offsets.data(), off_str.data(), off_str.size());

        // Tombstones — optional tail field for backward-compat with pre-tombstone pickles.
        std::vector<uint8_t> deleted;
        if (py::len(t) > 11) {
            std::string del_str(t[11].cast<py::bytes>());
            deleted.assign(del_str.begin(), del_str.end());
        }

        HNSWFloatNumpyAdapter<distance> p(M, ef_con, ef_search, seed);
        p.hnsw.deserialize(std::move(flat), std::move(levels), flat_adj, adj_offsets,
                           entry, top_lvl, (size_t)dim, seed, M, ef_con, ef_search,
                           std::move(deleted));
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

    void set(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("set() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<float> normalized(n * d);
        normalize_rows(ptr, normalized.data(), n, d);

        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{normalized.data() + i * d, d};
        hnsw.set(spans);
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
              py::object filter = py::none()) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("searchKNN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<float> qnorm(n * d);
        normalize_rows(ptr, qnorm.data(), n, d);

        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{qnorm.data() + i * d, d};

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<std::vector<int64_t>> idx;
        std::vector<std::vector<float>> dist;
        hnsw.searchKNN(spans, k, idx, dist, mask);
        // hnsw was built with dist_l2sq_f_avx2 which returns L2² directly.
        // Cosine distance: d_cos = L2² / 2 for unit vectors.
        for (auto& row : dist) {
            for (float& v : row) v = v * 0.5f;
        }
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    std::tuple<std::vector<int64_t>, std::vector<float>>
    search1NN(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
              py::object filter = py::none()) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("search1NN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<float> qnorm(n * d);
        normalize_rows(ptr, qnorm.data(), n, d);

        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{qnorm.data() + i * d, d};

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<int64_t> idx;
        std::vector<float> dist;
        hnsw.search1NN(spans, idx, dist, mask);
        for (float& v : dist) v = v * 0.5f;
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    void set_ef(size_t ef_search) { hnsw.set_ef(ef_search); }
    size_t ef_search() const { return hnsw.ef_search(); }
    size_t size() const { return hnsw.size(); }
    size_t dim() const { return hnsw.dim(); }

    std::vector<int32_t> add(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("add() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<float> normalized(n * d);
        normalize_rows(ptr, normalized.data(), n, d);
        std::vector<arrayf> spans(n);
        for (size_t i = 0; i < n; i++)
            spans[i] = FlatSpan{normalized.data() + i * d, d};
        return hnsw.add(spans);
    }

    void remove_node(int32_t node_id) { hnsw.remove(node_id); }
    size_t num_deleted() const { return hnsw.num_deleted(); }
    std::vector<int32_t> rebuild() { return hnsw.rebuild(); }

    static py::tuple get_state(const HNSWCosineNumpyAdapter& p) {
        std::vector<uint8_t> flat, deleted;
        std::vector<int32_t> levels, flat_adj, adj_offsets;
        int32_t entry, top_level;
        size_t dim;
        uint64_t seed;
        p.hnsw.serialize(flat, levels, flat_adj, adj_offsets, entry, top_level, dim, seed, deleted);

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
                              (uint64_t)p.hnsw.M(),
                              (uint64_t)p.hnsw.ef_construction(),
                              (uint64_t)p.hnsw.ef_search(),
                              del_bytes);
    }

    static HNSWCosineNumpyAdapter set_state(py::tuple t) {
        auto flat_bytes  = t[0].cast<py::bytes>();
        auto lvl_bytes   = t[1].cast<py::bytes>();
        auto adj_bytes   = t[2].cast<py::bytes>();
        auto off_bytes   = t[3].cast<py::bytes>();
        int32_t entry    = t[4].cast<int32_t>();
        int32_t top_lvl  = t[5].cast<int32_t>();
        uint64_t dim     = t[6].cast<uint64_t>();
        uint64_t seed    = t[7].cast<uint64_t>();
        size_t M         = t[8].cast<uint64_t>();
        size_t ef_con    = t[9].cast<uint64_t>();
        size_t ef_search = t[10].cast<uint64_t>();

        std::string flat_str(flat_bytes);
        std::vector<uint8_t> flat(flat_str.size());
        std::memcpy(flat.data(), flat_str.data(), flat_str.size());

        std::string lvl_str(lvl_bytes);
        std::vector<int32_t> levels(lvl_str.size() / sizeof(int32_t));
        std::memcpy(levels.data(), lvl_str.data(), lvl_str.size());

        std::string adj_str(adj_bytes);
        std::vector<int32_t> flat_adj(adj_str.size() / sizeof(int32_t));
        std::memcpy(flat_adj.data(), adj_str.data(), adj_str.size());

        std::string off_str(off_bytes);
        std::vector<int32_t> adj_offsets(off_str.size() / sizeof(int32_t));
        std::memcpy(adj_offsets.data(), off_str.data(), off_str.size());

        std::vector<uint8_t> deleted;
        if (py::len(t) > 11) {
            std::string del_str(t[11].cast<py::bytes>());
            deleted.assign(del_str.begin(), del_str.end());
        }

        HNSWCosineNumpyAdapter p(M, ef_con, ef_search, seed);
        p.hnsw.deserialize(std::move(flat), std::move(levels), flat_adj, adj_offsets,
                           entry, top_lvl, (size_t)dim, seed, M, ef_con, ef_search,
                           std::move(deleted));
        return p;
    }

    hnsw::HNSWIndex<arrayf, float, dist_l2sq_f_avx2> hnsw;
};

// HNSWL2NumpyAdapterSQ8 — HNSW with scalar quantisation (int8 vectors).
//
// Stores each vector as int8 using a single global scale: int8_i = round(f_i / scale).
// 4× memory bandwidth reduction vs float, plus AVX2 processes 32 int8 lanes
// per op (vs 8 floats). Distance ordering is preserved (we multiply by
// scale² to get true L2², but that's a constant so HNSW ordering is correct
// without it). On exit we apply scale to recover float L2 distances.
//
// Recall typically drops 1-3 % vs full-float at the same params on Gaussian
// data; users wanting maximum recall stick with HNSWL2Index.
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
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("set() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);

        // Compute global max-abs for symmetric int8 quantisation.
        float max_abs = 0.0f;
        for (size_t i = 0; i < n * d; i++) {
            float v = std::fabs(ptr[i]);
            if (v > max_abs) max_abs = v;
        }
        _scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        const float inv_scale = 1.0f / _scale;

        _sq8_backing.assign(n * d, 0);
        for (size_t i = 0; i < n * d; i++) {
            int v = static_cast<int>(std::round(ptr[i] * inv_scale));
            if (v > 127) v = 127;
            else if (v < -128) v = -128;
            _sq8_backing[i] = static_cast<int8_t>(v);
        }

        std::vector<SQ8Span> spans(n);
        for (size_t i = 0; i < n; i++) {
            spans[i] = SQ8Span{_sq8_backing.data() + i * d, d};
        }
        hnsw.set(spans);
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>>
    searchKNN(py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
              py::object filter = py::none()) {
        auto buf = queries.request();
        if (buf.ndim != 2)
            throw std::runtime_error("searchKNN() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        if (_scale <= 0.0f) {
            return std::make_tuple(
                std::vector<std::vector<int64_t>>(n),
                std::vector<std::vector<float>>(n));
        }
        const float* ptr = static_cast<const float*>(buf.ptr);
        const float inv_scale = 1.0f / _scale;

        // Quantise queries into a temporary int8 buffer.
        std::vector<int8_t> q_buf(n * d);
        for (size_t i = 0; i < n * d; i++) {
            int v = static_cast<int>(std::round(ptr[i] * inv_scale));
            if (v > 127) v = 127;
            else if (v < -128) v = -128;
            q_buf[i] = static_cast<int8_t>(v);
        }
        std::vector<SQ8Span> qspans(n);
        for (size_t i = 0; i < n; i++) {
            qspans[i] = SQ8Span{q_buf.data() + i * d, d};
        }

        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());

        std::vector<std::vector<int64_t>> idx;
        std::vector<std::vector<int32_t>> dist_i;
        hnsw.searchKNN(qspans, k, idx, dist_i, mask);
        // Convert int32 squared-quantised distance → float L2 distance.
        // L2 = scale * sqrt(d_int)   (scale² for L2², then sqrt)
        std::vector<std::vector<float>> dist_f(idx.size());
        for (size_t i = 0; i < idx.size(); i++) {
            dist_f[i].reserve(dist_i[i].size());
            for (int32_t di : dist_i[i]) {
                dist_f[i].push_back(_scale * std::sqrt(static_cast<float>(di)));
            }
        }
        return std::make_tuple(std::move(idx), std::move(dist_f));
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
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("add() expects a 2D float32 array of shape (n, d)");
        size_t n = (size_t)buf.shape[0];
        size_t d = (size_t)buf.shape[1];
        const float* ptr = static_cast<const float*>(buf.ptr);
        // If the index is empty, seed the scale via set(). Otherwise we
        // must reuse the existing scale (rescaling would invalidate prior
        // quantisations).
        if (size() == 0) {
            set(arr);
            std::vector<int32_t> ids(n);
            for (size_t i = 0; i < n; i++) ids[i] = static_cast<int32_t>(i);
            return ids;
        }
        const float inv_scale = 1.0f / _scale;
        std::vector<int8_t> quantised(n * d);
        for (size_t i = 0; i < n * d; i++) {
            int v = static_cast<int>(std::round(ptr[i] * inv_scale));
            if (v > 127) v = 127;
            else if (v < -128) v = -128;
            quantised[i] = static_cast<int8_t>(v);
        }
        std::vector<SQ8Span> spans(n);
        for (size_t i = 0; i < n; i++) {
            spans[i] = SQ8Span{quantised.data() + i * d, d};
        }
        return hnsw.add(spans);
    }

    void remove_node(int32_t node_id) { hnsw.remove(node_id); }
    size_t num_deleted() const { return hnsw.num_deleted(); }
    std::vector<int32_t> rebuild() { return hnsw.rebuild(); }

    static py::tuple get_state(const HNSWL2NumpyAdapterSQ8& p) {
        std::vector<uint8_t> flat, deleted;
        std::vector<int32_t> levels, flat_adj, adj_offsets;
        int32_t entry, top_level;
        size_t dim;
        uint64_t seed;
        p.hnsw.serialize(flat, levels, flat_adj, adj_offsets, entry, top_level, dim, seed, deleted);

        py::bytes flat_bytes(reinterpret_cast<const char*>(flat.data()), flat.size());
        py::bytes lvl_bytes(reinterpret_cast<const char*>(levels.data()),
                            levels.size() * sizeof(int32_t));
        py::bytes adj_bytes(reinterpret_cast<const char*>(flat_adj.data()),
                            flat_adj.size() * sizeof(int32_t));
        py::bytes off_bytes(reinterpret_cast<const char*>(adj_offsets.data()),
                            adj_offsets.size() * sizeof(int32_t));
        py::bytes del_bytes(reinterpret_cast<const char*>(deleted.data()), deleted.size());

        // Tuple layout: [...common fields..., scale (11), deleted (12)].
        return py::make_tuple(flat_bytes, lvl_bytes, adj_bytes, off_bytes,
                              entry, top_level, (uint64_t)dim, seed,
                              (uint64_t)p.hnsw.M(),
                              (uint64_t)p.hnsw.ef_construction(),
                              (uint64_t)p.hnsw.ef_search(),
                              p._scale,
                              del_bytes);
    }

    static HNSWL2NumpyAdapterSQ8 set_state(py::tuple t) {
        auto flat_bytes  = t[0].cast<py::bytes>();
        auto lvl_bytes   = t[1].cast<py::bytes>();
        auto adj_bytes   = t[2].cast<py::bytes>();
        auto off_bytes   = t[3].cast<py::bytes>();
        int32_t entry    = t[4].cast<int32_t>();
        int32_t top_lvl  = t[5].cast<int32_t>();
        uint64_t dim     = t[6].cast<uint64_t>();
        uint64_t seed    = t[7].cast<uint64_t>();
        size_t M         = t[8].cast<uint64_t>();
        size_t ef_con    = t[9].cast<uint64_t>();
        size_t ef_search = t[10].cast<uint64_t>();
        float scale      = t[11].cast<float>();

        std::string flat_str(flat_bytes);
        std::vector<uint8_t> flat(flat_str.size());
        std::memcpy(flat.data(), flat_str.data(), flat_str.size());

        std::string lvl_str(lvl_bytes);
        std::vector<int32_t> levels(lvl_str.size() / sizeof(int32_t));
        std::memcpy(levels.data(), lvl_str.data(), lvl_str.size());

        std::string adj_str(adj_bytes);
        std::vector<int32_t> flat_adj(adj_str.size() / sizeof(int32_t));
        std::memcpy(flat_adj.data(), adj_str.data(), adj_str.size());

        std::string off_str(off_bytes);
        std::vector<int32_t> adj_offsets(off_str.size() / sizeof(int32_t));
        std::memcpy(adj_offsets.data(), off_str.data(), off_str.size());

        std::vector<uint8_t> deleted;
        if (py::len(t) > 12) {
            std::string del_str(t[12].cast<py::bytes>());
            deleted.assign(del_str.begin(), del_str.end());
        }

        HNSWL2NumpyAdapterSQ8 p(M, ef_con, ef_search, seed);
        p._scale = scale;
        p.hnsw.deserialize(std::move(flat), std::move(levels), flat_adj, adj_offsets,
                           entry, top_lvl, (size_t)dim, seed, M, ef_con, ef_search,
                           std::move(deleted));
        return p;
    }

    hnsw::HNSWIndex<SQ8Span, int32_t, dist_l2sq_sq8> hnsw;

    // _scale is left as a member (not private) so get_state/set_state can
    // read/write it directly without a separate accessor.
    std::vector<int8_t> _sq8_backing;
    float _scale;
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

    void set(const ndarrayli& data) { hnsw.set(data); }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k,
              py::object filter = py::none()) {
        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());
        std::vector<std::vector<int64_t>> idx, dist;
        hnsw.searchKNN(queries, k, idx, dist, mask);
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    std::tuple<std::vector<int64_t>, std::vector<int64_t>>
    search1NN(const ndarrayli& queries,
              py::object filter = py::none()) {
        const uint8_t* mask = extract_filter_mask(filter, hnsw.size());
        std::vector<int64_t> idx, dist;
        hnsw.search1NN(queries, idx, dist, mask);
        return std::make_tuple(std::move(idx), std::move(dist));
    }

    void set_ef(size_t ef) { hnsw.set_ef(ef); }
    size_t ef_search() const { return hnsw.ef_search(); }
    size_t size() const { return hnsw.size(); }

    std::vector<int32_t> add(const ndarrayli& data) { return hnsw.add(data); }
    void remove_node(int32_t node_id) { hnsw.remove(node_id); }
    size_t num_deleted() const { return hnsw.num_deleted(); }
    std::vector<int32_t> rebuild() { return hnsw.rebuild(); }

    static py::tuple get_state(const HNSWBinaryNumpyAdapter<distance>& p) {
        std::vector<uint8_t> flat, deleted;
        std::vector<int32_t> levels, flat_adj, adj_offsets;
        int32_t entry, top_level;
        size_t dim;
        uint64_t seed;
        p.hnsw.serialize(flat, levels, flat_adj, adj_offsets, entry, top_level, dim, seed, deleted);

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
                              (uint64_t)p.hnsw.M(),
                              (uint64_t)p.hnsw.ef_construction(),
                              (uint64_t)p.hnsw.ef_search(),
                              del_bytes);
    }

    static HNSWBinaryNumpyAdapter<distance> set_state(py::tuple t) {
        auto flat_bytes  = t[0].cast<py::bytes>();
        auto lvl_bytes   = t[1].cast<py::bytes>();
        auto adj_bytes   = t[2].cast<py::bytes>();
        auto off_bytes   = t[3].cast<py::bytes>();
        int32_t entry    = t[4].cast<int32_t>();
        int32_t top_lvl  = t[5].cast<int32_t>();
        uint64_t dim     = t[6].cast<uint64_t>();
        uint64_t seed    = t[7].cast<uint64_t>();
        size_t M         = t[8].cast<uint64_t>();
        size_t ef_con    = t[9].cast<uint64_t>();
        size_t ef_search = t[10].cast<uint64_t>();

        std::string flat_str(flat_bytes);
        std::vector<uint8_t> flat(flat_str.size());
        std::memcpy(flat.data(), flat_str.data(), flat_str.size());

        std::string lvl_str(lvl_bytes);
        std::vector<int32_t> levels(lvl_str.size() / sizeof(int32_t));
        std::memcpy(levels.data(), lvl_str.data(), lvl_str.size());

        std::string adj_str(adj_bytes);
        std::vector<int32_t> flat_adj(adj_str.size() / sizeof(int32_t));
        std::memcpy(flat_adj.data(), adj_str.data(), adj_str.size());

        std::string off_str(off_bytes);
        std::vector<int32_t> adj_offsets(off_str.size() / sizeof(int32_t));
        std::memcpy(adj_offsets.data(), off_str.data(), off_str.size());

        std::vector<uint8_t> deleted;
        if (py::len(t) > 11) {
            std::string del_str(t[11].cast<py::bytes>());
            deleted.assign(del_str.begin(), del_str.end());
        }

        HNSWBinaryNumpyAdapter<distance> p(M, ef_con, ef_search, seed);
        p.hnsw.deserialize(std::move(flat), std::move(levels), flat_adj, adj_offsets,
                           entry, top_lvl, (size_t)dim, seed, M, ef_con, ef_search,
                           std::move(deleted));
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
        hnsw.set(data);
        mih.set(data);
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k) {
        size_t nq = queries.size();
        std::vector<std::vector<int64_t>> all_idx(nq), all_dist(nq);

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
        return std::make_tuple(std::move(all_idx), std::move(all_dist));
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

    void set(const ndarrayli &array) { tree.set(array); }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>> searchKNN(const ndarrayli &queries, size_t k) {

        std::vector<typename vptree::VPTree<arrayli, int64_t, distance>::VPTreeSearchResultElement> results;
        tree.searchKNN(queries, k, results);

        std::vector<std::vector<int64_t>> indexes;
        std::vector<std::vector<int64_t>> distances;
        indexes.resize(results.size());
        distances.resize(results.size());
        for (size_t i = 0; i < results.size(); ++i) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
        }
        return std::make_tuple(indexes, distances);
    }

    std::tuple<std::vector<int64_t>, std::vector<int64_t>> search1NN(const ndarrayli &queries) {

        std::vector<int64_t> indices;
        std::vector<int64_t> distances;
        tree.search1NN(queries, indices, distances);

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

    void set(const std::vector<key_t> &array) { tree.update(array); }

    std::tuple<std::vector<std::vector<index_t>>, std::vector<std::vector<distance_t>>, std::vector<std::vector<key_t>>>
    find_threshold(const std::vector<key_t> &queries, distance_t threshold) {
        return tree.find_batch(queries, threshold);
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

    void set(const ndarrayli& data) { _index.set(data); }

    std::tuple<std::vector<std::vector<int64_t>>,
               std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k) {
        return _index.searchKNN(queries, k);
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

    void set(const ndarrayli& data) { _index.set(data); }

    std::tuple<std::vector<std::vector<int64_t>>,
               std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k, int32_t radius = 8) {
        return _index.searchKNN(queries, k, radius);
    }

    int32_t m()      const { return _index.m(); }
    size_t  n()      const { return _index.n(); }
    size_t  nbytes() const { return _index.nbytes(); }

private:
    MIHBinaryIndex _index;
};

static const char *index_set = "Add vectors to index";
static const char *index_topk = "Batch find top-k vectors in index and return indices and distances";
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
    KMeansResult res  = kmeans_l2(ptr, n, d, k, max_iter, seed);

    py::array_t<int32_t> labels_out({(py::ssize_t)n});
    std::memcpy(labels_out.mutable_data(), res.labels.data(), n * sizeof(int32_t));

    py::array_t<float> centroids_out({(py::ssize_t)k, (py::ssize_t)d});
    std::memcpy(centroids_out.mutable_data(), res.centroids.data(), k * d * sizeof(float));

    return py::make_tuple(labels_out, centroids_out);
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
        .def("search1NN", &VPTreeNumpyAdapter<dist_l2_f_avx2>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapter<dist_l2_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_l2_f_avx2>::set_state));

    py::class_<VPTreeNumpyAdapter<dist_l1_f_avx2>>(m, "VPTreeL1Index")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter<dist_l1_f_avx2>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapter<dist_l1_f_avx2>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapter<dist_l1_f_avx2>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapter<dist_l1_f_avx2>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapter<dist_l1_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_l1_f_avx2>::set_state));

    py::class_<VPTreeNumpyAdapter<dist_chebyshev_f_avx2>>(m, "VPTreeChebyshevIndex")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::set_state));

    py::class_<VPTreeCosineNumpyAdapter>(m, "VPTreeCosineIndex")
        .def(py::init<>())
        .def("set", &VPTreeCosineNumpyAdapter::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeCosineNumpyAdapter::to_string, index_string)
        .def("searchKNN", &VPTreeCosineNumpyAdapter::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeCosineNumpyAdapter::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeCosineNumpyAdapter::get_state, &VPTreeCosineNumpyAdapter::set_state));

    py::class_<HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>>(m, "HNSWL2Index")
        .def(py::init<size_t, size_t, size_t, uint64_t, int>(),
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("ef_search") = 50, py::arg("seed") = 42,
             py::arg("n_threads") = 1)
        .def("set", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::set, py::arg("vectors"))
        .def("searchKNN", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::searchKNN,
             py::arg("vectors"), py::arg("k"), py::arg("filter") = py::none())
        .def("search1NN", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::search1NN,
             py::arg("vectors"), py::arg("filter") = py::none())
        .def("set_ef", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::set_ef, py::arg("ef_search"))
        .def("dist_calls",
             [](const HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>& a){ return a.hnsw.dist_calls(); })
        .def("reset_dist_calls",
             [](HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>& a){ a.hnsw.reset_dist_calls(); })
        .def("add", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::add, py::arg("vectors"))
        .def("remove", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::remove_node, py::arg("node_id"))
        .def("rebuild", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::rebuild)
        .def_property_readonly("ef_search", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::ef_search)
        .def_property_readonly("size", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::size)
        .def_property_readonly("dim", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::dim)
        .def_property_readonly("num_deleted", &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::num_deleted)
        .def(py::pickle(&HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::get_state,
                        &HNSWFloatNumpyAdapter<dist_l2sq_f_avx2>::set_state));

    py::class_<HNSWCosineNumpyAdapter>(m, "HNSWCosineIndex")
        .def(py::init<size_t, size_t, size_t, uint64_t, int>(),
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("ef_search") = 50, py::arg("seed") = 42,
             py::arg("n_threads") = 1)
        .def("set", &HNSWCosineNumpyAdapter::set, py::arg("vectors"))
        .def("searchKNN", &HNSWCosineNumpyAdapter::searchKNN,
             py::arg("vectors"), py::arg("k"), py::arg("filter") = py::none())
        .def("search1NN", &HNSWCosineNumpyAdapter::search1NN,
             py::arg("vectors"), py::arg("filter") = py::none())
        .def("set_ef", &HNSWCosineNumpyAdapter::set_ef, py::arg("ef_search"))
        .def("add", &HNSWCosineNumpyAdapter::add, py::arg("vectors"))
        .def("remove", &HNSWCosineNumpyAdapter::remove_node, py::arg("node_id"))
        .def("rebuild", &HNSWCosineNumpyAdapter::rebuild)
        .def_property_readonly("ef_search", &HNSWCosineNumpyAdapter::ef_search)
        .def_property_readonly("size", &HNSWCosineNumpyAdapter::size)
        .def_property_readonly("dim", &HNSWCosineNumpyAdapter::dim)
        .def_property_readonly("num_deleted", &HNSWCosineNumpyAdapter::num_deleted)
        .def(py::pickle(&HNSWCosineNumpyAdapter::get_state,
                        &HNSWCosineNumpyAdapter::set_state));

    py::class_<HNSWL2NumpyAdapterSQ8>(m, "HNSWL2IndexSQ8",
        "HNSW with int8 scalar quantisation. ~4x less memory and faster "
        "queries at large N, at the cost of ~1-3% recall vs HNSWL2Index. "
        "Public API mirrors HNSWL2Index; distances returned are L2 (scaled).")
        .def(py::init<size_t, size_t, size_t, uint64_t, int>(),
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("ef_search") = 50, py::arg("seed") = 42,
             py::arg("n_threads") = 1)
        .def("set", &HNSWL2NumpyAdapterSQ8::set, py::arg("vectors"))
        .def("searchKNN", &HNSWL2NumpyAdapterSQ8::searchKNN,
             py::arg("vectors"), py::arg("k"), py::arg("filter") = py::none())
        .def("search1NN", &HNSWL2NumpyAdapterSQ8::search1NN,
             py::arg("vectors"), py::arg("filter") = py::none())
        .def("set_ef", &HNSWL2NumpyAdapterSQ8::set_ef, py::arg("ef_search"))
        .def("add", &HNSWL2NumpyAdapterSQ8::add, py::arg("vectors"))
        .def("remove", &HNSWL2NumpyAdapterSQ8::remove_node, py::arg("node_id"))
        .def("rebuild", &HNSWL2NumpyAdapterSQ8::rebuild)
        .def_property_readonly("ef_search", &HNSWL2NumpyAdapterSQ8::ef_search)
        .def_property_readonly("size", &HNSWL2NumpyAdapterSQ8::size)
        .def_property_readonly("scale", &HNSWL2NumpyAdapterSQ8::scale)
        .def_property_readonly("num_deleted", &HNSWL2NumpyAdapterSQ8::num_deleted)
        .def(py::pickle(&HNSWL2NumpyAdapterSQ8::get_state,
                        &HNSWL2NumpyAdapterSQ8::set_state));

    py::class_<HNSWBinaryNumpyAdapter<dist_hamming>>(m, "HNSWBinaryIndex")
        .def(py::init<size_t, size_t, size_t, uint64_t, int>(),
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("ef_search") = 50, py::arg("seed") = 42,
             py::arg("n_threads") = 1)
        .def("set", &HNSWBinaryNumpyAdapter<dist_hamming>::set, py::arg("vectors"))
        .def("searchKNN", &HNSWBinaryNumpyAdapter<dist_hamming>::searchKNN,
             py::arg("vectors"), py::arg("k"), py::arg("filter") = py::none())
        .def("search1NN", &HNSWBinaryNumpyAdapter<dist_hamming>::search1NN,
             py::arg("vectors"), py::arg("filter") = py::none())
        .def("set_ef", &HNSWBinaryNumpyAdapter<dist_hamming>::set_ef, py::arg("ef_search"))
        .def("add", &HNSWBinaryNumpyAdapter<dist_hamming>::add, py::arg("vectors"))
        .def("remove", &HNSWBinaryNumpyAdapter<dist_hamming>::remove_node, py::arg("node_id"))
        .def("rebuild", &HNSWBinaryNumpyAdapter<dist_hamming>::rebuild)
        .def_property_readonly("ef_search", &HNSWBinaryNumpyAdapter<dist_hamming>::ef_search)
        .def_property_readonly("size", &HNSWBinaryNumpyAdapter<dist_hamming>::size)
        .def_property_readonly("num_deleted", &HNSWBinaryNumpyAdapter<dist_hamming>::num_deleted)
        .def(py::pickle(&HNSWBinaryNumpyAdapter<dist_hamming>::get_state,
                        &HNSWBinaryNumpyAdapter<dist_hamming>::set_state));

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
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_512>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_512>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_512>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_256>>(m, "VPTreeBinaryIndex256")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_256>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_256>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_256>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_256>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_256>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_256>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_128>>(m, "VPTreeBinaryIndex128")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_128>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_128>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_128>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_128>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_128>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_128>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_64>>(m, "VPTreeBinaryIndex64")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_64>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_64>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_64>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_64>::search1NN, index_top1, py::arg("vectors"))
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_64>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_64>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming>>(m, "VPTreeBinaryIndex")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming>::set, index_set, py::arg("vectors"))
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming>::to_string, index_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming>::searchKNN, index_topk, py::arg("vectors"), py::arg("k"))
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
        .def("m",      &MIHBinaryNumpyAdapter::m)
        .def("n",      &MIHBinaryNumpyAdapter::n)
        .def("nbytes", &MIHBinaryNumpyAdapter::nbytes);
};
