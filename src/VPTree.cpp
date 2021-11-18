/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#include <vector>
#include <utility>
#include <cstdlib>
#include <iostream>
#include <functional>

#include <VPTree.hpp>
#include <BindingUtils.hpp>

#include <MathUtils.hpp>

namespace vptree {

template <typename T, double(*distance)(const T&, const T&)>
VPTree<T, distance>::VPTree(const ::std::vector<T>& array) {

    _examples = ::std::move(array);
}

template <typename T, double(*distance)(const T&, const T&)>
void VPTree<T,distance>::build(const std::vector<T>& array) {

    // Select vantage point
    std::vector<VPLevelPartition<T>*> _toSplit;

    auto* root = new VPLevelPartition<T>(-1, 0, array.size() - 1);
    _toSplit.push_back(root);

    while(!_toSplit.empty()) {

        VPLevelPartition<T>* current = _toSplit.back();
        _toSplit.pop_back();

        unsigned int start =  current->start();
        unsigned int end = current->end();

        if(end - start < MIN_POINTS_PER_PARTITION) {
            // we have just a few points, end division
            continue;
        }

        unsigned vpIndex = selectVantagePoint(start, end);

        // put vantage point as the first element within the examples list
        std::swap<T>(array.begin() + vpIndex, array.end() + start);

        unsigned int median = (end - start) / 2;

        // partition in order to keep all elements smaller than median in the left and larger in the right
         std::nth_element(array.begin() + start + 1, array.begin() + median, array.begin() + end);

         // distance from vantage point (which is at start index) and the median element
         double medianDistance = distance(array[start], array[median]);
         current->setRadius(medianDistance);

         // Schedule to build next levels
         //
         // Left is every one within the median distance radius
         auto* left = new VPLevelPartition<T>(-1, start + 1, median);
        _toSplit.push_back(left);

         auto* right = new VPLevelPartition<T>(-1, start + 1, median);
        _toSplit.push_back(right);

        current->setChild(left, right);
    }
}


template <typename T, double(*distance)(const T&, const T&)>
unsigned int VPTree<T, distance>::selectVantagePoint(unsigned int fromIndex, unsigned int toIndex) {

    // for now, simple random point selection as basic strategy: TODO: better vantage point selection
    unsigned int range = (toIndex-fromIndex) + 1;
    return fromIndex + (rand() % range);
}

/* py::array_t<double> VPTree::execute() { */
/*     double *foo = new double[100]; */
/*     for (size_t i = 0; i < 100; i++) { */
/*         foo[i] = static_cast<double>(i); */
/*     } */

/*     return BindingUtils::bufferToNumpyNdArray<double, 10, 10>(foo); */
/* } */

/* void VPTree::test(py::array_t<double> array) */
/* { */
/*     auto buff = array.request(); */
/*     for(auto e: buff.shape) */
/*     { */
/*         std::cout << e << std::endl; */
/*     } */

/* } */

}; // namespace vptree
