/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#include <DistanceFunctions.hpp>
#include <VPTree.hpp>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <omp.h>

namespace py = pybind11;

class VPTreeNumpyAdapter {
    public:
    VPTreeNumpyAdapter() = default;

    void set(const ndarrayf &array) { _tree = vptree::VPTree<arrayf, dist_optimized_float>(array); }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<float>>> searchKNN(const ndarrayf &queries, unsigned int k) {

        std::vector<vptree::VPTree<arrayf, dist_optimized_float>::VPTreeSearchResultElement> results;
        _tree.searchKNN(queries, k, results);

        std::vector<std::vector<unsigned int>> indexes;
        std::vector<std::vector<float>> distances;
        indexes.resize(results.size());
        distances.resize(results.size());
        for (int i = 0; i < results.size(); ++i) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
        }

        return std::make_tuple(indexes, distances);
    }

    std::tuple<std::vector<unsigned int>, std::vector<float>> search1NN(const ndarrayf &queries) {

        std::vector<unsigned int> indices;
        std::vector<float> distances;
        _tree.search1NN(queries, indices, distances);

        return std::make_tuple(std::move(indices), std::move(distances));
    }

    void serialize(std::vector<char> &data) { return _tree.serialize(data); }

    void deserialize(const std::vector<char> &data) { _tree.deserialize(data); }

    private:
    vptree::VPTree<arrayf, dist_optimized_float> _tree;
};

class VPTreeBinaryNumpyAdapter {
    public:
    VPTreeBinaryNumpyAdapter() = default;

    void set(const ndarrayli &array) { _tree = vptree::VPTree<arrayli, distHamming>(array); }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<float>>> searchKNN(const ndarrayli &queries, unsigned int k) {

        std::vector<vptree::VPTree<arrayli, distHamming>::VPTreeSearchResultElement> results;
        _tree.searchKNN(queries, k, results);

        std::vector<std::vector<unsigned int>> indexes;
        std::vector<std::vector<float>> distances;
        indexes.resize(results.size());
        distances.resize(results.size());
        for (int i = 0; i < results.size(); ++i) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
        }
        return std::make_tuple(indexes, distances);
    }

    void serialize(std::vector<char> &data) { return _tree.serialize(data); }

    void deserialize(const std::vector<char> &data) { _tree.deserialize(data); }

    std::tuple<std::vector<unsigned int>, std::vector<float>> search1NN(const ndarrayli &queries) {

        std::vector<unsigned int> indices;
        std::vector<float> distances;
        _tree.search1NN(queries, indices, distances);

        return std::make_tuple(std::move(indices), std::move(distances));
    }

    private:
    vptree::VPTree<arrayli, distHamming> _tree;
};

PYBIND11_MODULE(_pyvptree, m) {
    py::class_<VPTreeNumpyAdapter>(m, "VPTreeL2Index")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter::set)
        .def("searchKNN", &VPTreeNumpyAdapter::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapter::search1NN)
        .def(py::pickle(
            [](const VPTreeNumpyAdapter &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                std::vector<char> state;
                const_cast<VPTreeNumpyAdapter &>(p).serialize(state);
                py::tuple t = py::make_tuple(state);

                return t;
            },
            [](py::tuple t) { // __setstate__
                /* Create a new C++ instance */
                VPTreeNumpyAdapter p;
                std::vector<char> vec = t[0].cast<std::vector<char>>();
                p.deserialize(vec);

                return p;
            }));

    py::class_<VPTreeBinaryNumpyAdapter>(m, "VPTreeBinaryIndex")
        .def(py::init<>())
        .def("set", &VPTreeBinaryNumpyAdapter::set)
        .def("searchKNN", &VPTreeBinaryNumpyAdapter::searchKNN)
        .def("search1NN", &VPTreeBinaryNumpyAdapter::search1NN)
        .def(py::pickle(
            [](const VPTreeBinaryNumpyAdapter &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple();
            },
            [](py::tuple t) { // __setstate__
                /* Create a new C++ instance */
                VPTreeBinaryNumpyAdapter p;

                return p;
            }));
}
