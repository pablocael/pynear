/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <VPTree.hpp>

namespace py = pybind11;

using arrayd = std::vector<double>;
using ndarrayd = std::vector<arrayd>;

double dist(const arrayd& p1, const arrayd& p2) {

    double result = 0;
    for(int i = 0; i < p1.size(); ++i) {
        double d = (p1[i] - p2[i]);
        result += d * d; 
    }

    return std::sqrt(result);
}

class VPTreeNumpyAdapter {
public:
    VPTreeNumpyAdapter() = default;

    void set(const ndarrayd& array) {
        _tree = vptree::VPTree<arrayd, dist>(array);
    }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<double>>> searchKNN(const ndarrayd& queries, unsigned int k) {


        std::vector<vptree::VPTree<arrayd,dist>::VPTreeSearchResultElement> results;
        _tree.searchKNN(queries, k, results);

        std::vector<std::vector<unsigned int>> indexes;
        std::vector<std::vector<double>> distances;
        indexes.resize(results.size());
        distances.resize(results.size());
        for(int i = 0; i < results.size(); ++i) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
        }
        return std::make_tuple(indexes, distances);
    }

    std::tuple<std::vector<unsigned int>, std::vector<double>> search1NN(const ndarrayd& queries) {


        std::vector<unsigned int> indices; std::vector<double> distances;
        _tree.search1NN(queries, indices, distances);

        return std::make_tuple(std::move(indices), std::move(distances));
    }

private:
    vptree::VPTree<arrayd, dist> _tree;

};

PYBIND11_MODULE(pyvptree, m) {
    py::class_<VPTreeNumpyAdapter>(m, "VPTree")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter::set)
        .def("searchKNN", &VPTreeNumpyAdapter::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapter::search1NN);
}
