/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#include <cstdlib>
#include <iostream>
#include <functional>

#include <VPTree.hpp>
#include <BindingUtils.hpp>

#include <MathUtils.hpp>

namespace vptree {

template <typename T>
void swapElements(const std::vector<T>& array, unsigned int from, unsigned int to, unsigned int stride) {
    
    for(int i = 0; i < stride; ++i) {
        std::swap(array[from * stride + i],  array[to * dim + i]);
    }
}

template <typename T>
VPTree<T>::VPTree(const std::vector<T>& array, unsigned int dimension) {

    _metric = std::bind(vptree::math::distance<T>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, dimension);
}

template <typename T>
void VPTree<T>::build(const std::vector<T>& array, unsigned int dimension) {

    // Select vantage point
    std::vector<VPLevelPartition*> _toSplit;

    auto* root = new VPLevelPartition(-1, 0, (array.size() / dimension) - 1);
    _toSplit.push_back(root);

    while(!_toSplit.empty()) {

        VPLevelPartition* current = _toSplit.back();
        _toSplit.pop_back();

        unsigned int start =  current->start();
        unsigned int end = current->end();

        if(start == end) {
            // we have just one point, end division
            continue;
        }

        unsigned vpIndex = selectVantagePoint(start, end);

        // put vantage point as the first element within the examples list
        swapElements<T>(array, vpIndex, start);

        std::vector<double> distances;
        vptree::math::distancesToRef(array, start+1, end, start, distances);

         /* std::nth_element( */ 
         /*        _items.begin() + lower + 1, */ 
         /*        _items.begin() + median, */
         /*        _items.begin() + upper, */
         /*        DistanceComparator( _items[lower] )); */

         /*    // what was the median? */
         /*    node->threshold = distance( _items[lower], _items[median] ); */

        // Calculate the median of all points to the vantage point
        /* _rootPartition = VPLevelPartition() */

    }
}


template <typename T>
unsigned int VPTree<T>::selectVantagePoint(unsigned int from, unsigned int to) {

    // for now, simple random point selection as basic strategy: TODO: better vantage point selection
    unsigned int range = (to-from) + 1;
    return from + (rand() % range);

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
