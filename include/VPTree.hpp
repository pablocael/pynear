#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class VPTree {
public:
    VPTree();
    py::array_t<double> execute();
    void test(py::array_t<double> array);
};
