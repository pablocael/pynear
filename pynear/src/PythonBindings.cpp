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
#include <DistanceFunctions.hpp>
#include <ISerializable.hpp>
#include <SerializableVPTree.hpp>

namespace py = pybind11;

typedef float (*distance_func_f)(const arrayf &, const arrayf &);
typedef int64_t (*distance_func_li)(const arrayli &, const arrayli &);

template <distance_func_f distance> class VPTreeNumpyAdapter {
public:
    VPTreeNumpyAdapter() = default;

    void set(const ndarrayf &array) { tree.set(array); }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<float>>> searchKNN(const ndarrayf &queries, size_t k) {

        std::vector<typename vptree::VPTree<arrayf, float, distance>::VPTreeSearchResultElement> results;
        tree.searchKNN(queries, k, results);

        std::vector<std::vector<int64_t>> indexes;
        std::vector<std::vector<float>> distances;
        indexes.resize(results.size());
        distances.resize(results.size());
        for (size_t i = 0; i < results.size(); ++i) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
        }

        return std::make_tuple(indexes, distances);
    }

    std::tuple<std::vector<int64_t>, std::vector<float>> search1NN(const ndarrayf &queries) {

        std::vector<int64_t> indices;
        std::vector<float> distances;
        tree.search1NN(queries, indices, distances);

        return std::make_tuple(std::move(indices), std::move(distances));
    }

    std::string to_string() {
        std::stringstream stream;
        stream << tree;

        return stream.str();
    }

    static py::tuple get_state(const VPTreeNumpyAdapter<distance> &p) {
        vptree::SerializedStateObject state = p.tree.serialize();
        py::tuple t = py::make_tuple(state.data(), state.checksum());
        return t;
    }

    static VPTreeNumpyAdapter<distance> set_state(py::tuple t) {
        VPTreeNumpyAdapter<distance> p;
        std::vector<uint8_t> state = t[0].cast<std::vector<uint8_t>>();
        uint8_t checksum = t[1].cast<uint8_t>();
        p.tree.deserialize(vptree::SerializedStateObject(state, checksum));
        return p;
    }

    vptree::SerializableVPTree<arrayf, float, distance, vptree::ndarraySerializer<float>, vptree::ndarrayDeserializer<float>> tree;
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
        uint8_t checksum = t[1].cast<uint8_t>();
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
    BKTree<arrayli, int64_t, HammingMetric<distance>> tree;

    BKTreeBinaryNumpyAdapter() = default;

    void set(const ndarrayli &array) { tree.update(array); }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<ndarrayli>> find_threshold(const ndarrayli &queries, int64_t threshold) {
        return tree.find_batch(queries, threshold);
    }

    bool empty() { return tree.empty(); }
    ndarrayli values() { return tree.values(); }
};

PYBIND11_MODULE(_pynear, m) {
    py::class_<VPTreeNumpyAdapter<dist_l2_f_avx2>>(m, "VPTreeL2Index")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter<dist_l2_f_avx2>::set)
        .def("to_string", &VPTreeNumpyAdapter<dist_l2_f_avx2>::to_string)
        .def("searchKNN", &VPTreeNumpyAdapter<dist_l2_f_avx2>::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapter<dist_l2_f_avx2>::search1NN)
        .def(py::pickle(&VPTreeNumpyAdapter<dist_l2_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_l2_f_avx2>::set_state));

    py::class_<VPTreeNumpyAdapter<dist_l1_f_avx2>>(m, "VPTreeL1Index")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter<dist_l1_f_avx2>::set)
        .def("to_string", &VPTreeNumpyAdapter<dist_l1_f_avx2>::to_string)
        .def("searchKNN", &VPTreeNumpyAdapter<dist_l1_f_avx2>::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapter<dist_l1_f_avx2>::search1NN)
        .def(py::pickle(&VPTreeNumpyAdapter<dist_l1_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_l1_f_avx2>::set_state));

    py::class_<VPTreeNumpyAdapter<dist_chebyshev_f_avx2>>(m, "VPTreeChebyshevIndex")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::set)
        .def("to_string", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::to_string)
        .def("searchKNN", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::search1NN)
        .def(py::pickle(&VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::get_state, &VPTreeNumpyAdapter<dist_chebyshev_f_avx2>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_512>>(m, "VPTreeBinaryIndex512")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_512>::set)
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_512>::to_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_512>::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_512>::search1NN)
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_512>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_512>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_256>>(m, "VPTreeBinaryIndex256")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_256>::set)
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_256>::to_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_256>::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_256>::search1NN)
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_256>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_256>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_128>>(m, "VPTreeBinaryIndex128")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_128>::set)
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_128>::to_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_128>::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_128>::search1NN)
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_128>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_128>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming_64>>(m, "VPTreeBinaryIndex64")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming_64>::set)
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming_64>::to_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming_64>::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming_64>::search1NN)
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming_64>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming_64>::set_state));

    py::class_<VPTreeNumpyAdapterBinary<dist_hamming>>(m, "VPTreeBinaryIndex")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapterBinary<dist_hamming>::set)
        .def("to_string", &VPTreeNumpyAdapterBinary<dist_hamming>::to_string)
        .def("searchKNN", &VPTreeNumpyAdapterBinary<dist_hamming>::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapterBinary<dist_hamming>::search1NN)
        .def(py::pickle(&VPTreeNumpyAdapterBinary<dist_hamming>::get_state, &VPTreeNumpyAdapterBinary<dist_hamming>::set_state));

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming_512>>(m, "BKTreeBinaryIndex512")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::set)
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::find_threshold)
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::empty)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming_512>::values);

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming_256>>(m, "BKTreeBinaryIndex256")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::set)
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::find_threshold)
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::empty)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming_256>::values);

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming_128>>(m, "BKTreeBinaryIndex128")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::set)
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::find_threshold)
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::empty)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming_128>::values);

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming_64>>(m, "BKTreeBinaryIndex64")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::set)
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::find_threshold)
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::empty)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming_64>::values);

    py::class_<BKTreeBinaryNumpyAdapter<dist_hamming>>(m, "BKTreeBinaryIndex")
        .def(py::init<>())
        .def("set", &BKTreeBinaryNumpyAdapter<dist_hamming>::set)
        .def("find_threshold", &BKTreeBinaryNumpyAdapter<dist_hamming>::find_threshold)
        .def("empty", &BKTreeBinaryNumpyAdapter<dist_hamming>::empty)
        .def("values", &BKTreeBinaryNumpyAdapter<dist_hamming>::values);
};
