/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#include <iostream>

#include "VPTree.hpp"
#include "BindingUtils.hpp"


VPTree::VPTree() {
    // TODO(pablo.cael) implement
}

py::array_t<double> VPTree::execute() {
    double *foo = new double[100];
    for (size_t i = 0; i < 100; i++) {
        foo[i] = static_cast<double>(i);
    }

    return BindingUtils::bufferToNumpyNdArray<double, 10, 10>(foo);
}

void VPTree::test(py::array_t<double> array)
{
    auto buff = array.request();
    for(auto e: buff.shape)
    {
        std::cout << e << std::endl;
    }

}
