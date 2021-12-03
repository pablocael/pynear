/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <VPTree.hpp>

#include <nmmintrin.h>
#include <stdint.h>
#include <omp.h>

namespace py = pybind11;

using arrayd = std::vector<double>;
using arrayli = std::vector<uint64_t>;
using ndarrayd = std::vector<arrayd>;
using ndarrayli = std::vector<arrayli>;

typedef int32_t hamdis_t;

inline int popcount64(uint64_t x) {
    return __builtin_popcountl(x);
}

/* Hamming distances for multiples of 64 bits */
template <size_t nbits>
hamdis_t hamming(const uint64_t* bs1, const uint64_t* bs2) {
    const size_t nwords = nbits / 64;
    size_t i;
    hamdis_t h = 0;
    for (i = 0; i < nwords; i++)
        h += popcount64(bs1[i] ^ bs2[i]);
    return h;
}

/* specialized (optimized) functions */
template <>
hamdis_t hamming<64>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]);
}

template <>
hamdis_t hamming<128>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]);
}

template <>
hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]) +
            popcount64(pa[2] ^ pb[2]) + popcount64(pa[3] ^ pb[3]);
}

double distL2(const arrayd& p1, const arrayd& p2) {

    double result = 0;
    for(int i = 0; i < p1.size(); ++i) {
        double d = (p1[i] - p2[i]);
        result += d * d;
    }

    return std::sqrt(result);
}

double distHamming(const arrayli& p1, const arrayli& p2) {

    return hamming<128>(&p1[0], &p2[0]);
}

inline int pop_count(uint64_t x, uint64_t y) {
    return __builtin_popcountll(x ^ y);
}

/* double distHamming(const std::vector<unsigned char>& p1, const std::vector<unsigned char>& p2) { */

/*     // assume v1 and v2 sizes are multiple of 8 */
/*     // assume 32 bytes for now */
/*     double result = 0; */
/*     const uint64_t* a = (reinterpret_cast<const uint64_t*>(&p1[0])); */
/*     const uint64_t* b = (reinterpret_cast<const uint64_t*>(&p2[0])); */
/*     for(int i = 0; i < p1.size()/sizeof(uint64_t); i++) { */
/*         result += pop_count(a[i], b[i]); */
/*     } */
/*     return result; */
/* } */

class VPTreeNumpyAdapter {
public:
    VPTreeNumpyAdapter() = default;

    void set(const ndarrayd& array) {
        _tree = vptree::VPTree<arrayd, distL2>(array);
    }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<double>>> searchKNN(const ndarrayd& queries, unsigned int k) {


        std::vector<vptree::VPTree<arrayd,distL2>::VPTreeSearchResultElement> results;
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
    vptree::VPTree<arrayd, distL2> _tree;

};

class VPTreeBinaryNumpyAdapter {
public:
    VPTreeBinaryNumpyAdapter() = default;

    void set(const ndarrayli& array) {
        _tree = vptree::VPTree<arrayli, distHamming>(array);
    }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<double>>> searchKNN(const ndarrayli& queries, unsigned int k) {


        std::vector<vptree::VPTree<arrayli,distHamming>::VPTreeSearchResultElement> results;
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

    std::tuple<std::vector<unsigned int>, std::vector<double>> search1NN(const ndarrayli& queries) {


        std::vector<unsigned int> indices; std::vector<double> distances;
        _tree.search1NN(queries, indices, distances);

        return std::make_tuple(std::move(indices), std::move(distances));
    }

private:
    vptree::VPTree<arrayli, distHamming> _tree;

};
PYBIND11_MODULE(pyvptree, m) {
    py::class_<VPTreeNumpyAdapter>(m, "VPTreeL2Index")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter::set)
        .def("searchKNN", &VPTreeNumpyAdapter::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapter::search1NN);

    py::class_<VPTreeBinaryNumpyAdapter>(m, "VPTreeBinaryIndex")
        .def(py::init<>())
        .def("set", &VPTreeBinaryNumpyAdapter::set)
        .def("searchKNN", &VPTreeBinaryNumpyAdapter::searchKNN)
        .def("search1NN", &VPTreeBinaryNumpyAdapter::search1NN);
}
