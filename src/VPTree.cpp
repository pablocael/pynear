#include "VPTree.hpp"
#include "BindingUtils.hpp"

#include <iostream>

VPTree::VPTree() {
    // TODO implement
}

py::array_t<double> VPTree::execute()
{
    double *foo = new double[100];
    for (size_t i = 0; i < 100; i++)
    {
        foo[i] = (double)i;
    }

    return BindingUtils::bufferToNumpyNdArray<double, 10, 10>(foo);
}

void VPTree::test(py::array_t<double> array)
{
    auto buff = array.request();
    std::cout << buff.shape[0] << ", " << buff.shape[1] << "," << buff.shape[2]  << std::endl;

}
