#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <VPTree.hpp>

namespace py = pybind11;

class VPTreeNumpyAdapter {
public:
    VPTreeNumpyAdapter();

private:
    vptree::VPTree<double> _tree;

};

PYBIND11_MODULE(pyvptree, m) {
    py::class_<VPTreeNumpyAdapter>(m, "VPTree")
        .def(py::init<>());
        /* .def("test", &VPTree::test); */
}
