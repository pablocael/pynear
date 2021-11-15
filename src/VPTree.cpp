#include "VPTree.hpp"
#include "BindingUtils.hpp"

#include <iostream>

VPTree::VPTree() {
    // TODO implement
}

py::array_t<double> VPTree::execute() {
    double *foo = new double[100];
    for (size_t i = 0; i < 100; i++) {
        foo[i] = (double)i;
    }

    return BindingUtils::bufferToNumpy2d(foo, 10, 10);
}

