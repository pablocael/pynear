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

#include <ISerializable.hpp>

namespace py = pybind11;

class VPTreeNumpyAdapter {
public:
    VPTreeNumpyAdapter() = default;

    void set(const ndarrayf &array) { tree = vptree::VPTree<arrayf, dist_optimized_float>(array); }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<float>>> searchKNN(const ndarrayf &queries, unsigned int k) {

        std::vector<vptree::VPTree<arrayf, dist_optimized_float>::VPTreeSearchResultElement> results;
        tree.searchKNN(queries, k, results);

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
        tree.search1NN(queries, indices, distances);

        return std::make_tuple(std::move(indices), std::move(distances));
    }

    vptree::VPTree<arrayf, dist_optimized_float> tree;
};

class VPTreeBinaryNumpyAdapter {
public:
    VPTreeBinaryNumpyAdapter() = default;

    void set(const ndarrayli &array) { tree = vptree::VPTree<arrayli, distHamming>(array); }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<float>>> searchKNN(const ndarrayli &queries, unsigned int k) {

        std::vector<vptree::VPTree<arrayli, distHamming>::VPTreeSearchResultElement> results;
        tree.searchKNN(queries, k, results);

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

    std::tuple<std::vector<unsigned int>, std::vector<float>> search1NN(const ndarrayli &queries) {

        std::vector<unsigned int> indices;
        std::vector<float> distances;
        tree.search1NN(queries, indices, distances);

        return std::make_tuple(std::move(indices), std::move(distances));
    }

    vptree::VPTree<arrayli, distHamming> tree;
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
                vptree::SerializedState state = p.tree.serialize();
                py::tuple t = py::make_tuple(state.data, state.checksum);

                return t;
            },
            [](py::tuple t) { // __setstate__
                /* Create a new C++ instance */
                VPTreeNumpyAdapter p;
                std::vector<uint8_t> data = t[0].cast<std::vector<uint8_t>>();
                uint8_t checksum = t[1].cast<uint8_t>();
                p.tree.deserialize(vptree::SerializedState(data, checksum));

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
                vptree::SerializedState state = p.tree.serialize();
                py::tuple t = py::make_tuple(state.data, state.checksum);

                return t;
            },
            [](py::tuple t) { // __setstate__
                /* Create a new C++ instance */
                VPTreeBinaryNumpyAdapter p;
                std::vector<uint8_t> data = t[0].cast<std::vector<uint8_t>>();
                uint8_t checksum = t[1].cast<uint8_t>();
                p.tree.deserialize(vptree::SerializedState(data, checksum));

                return p;
            }));
    };
