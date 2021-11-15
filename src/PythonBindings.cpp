#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <VPTree.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pyvptree, m) {
    py::class_<VPTree>(m, "VPTree")
        .def(py::init<>())
        .def("execute", &VPTree::execute)
        .def("test", &VPTree::test);
}
