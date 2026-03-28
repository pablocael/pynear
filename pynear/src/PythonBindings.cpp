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
#include <BuiltinSerializers.hpp>
#include <DistanceFunctions.hpp>
#include <ISerializable.hpp>
#include <KMeans.hpp>
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
                            indices.size() * sizeof(int64_t));
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

        // Indices
        std::string idx_str(idx_bytes);
        std::vector<int64_t> indices(idx_str.size() / sizeof(int64_t));
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
};
