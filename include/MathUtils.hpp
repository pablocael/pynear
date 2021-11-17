#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

namespace vptree::math {

/*
 *  Calculates the squared L2 distance between two elements within a coordinate buffer.
 *
 * :param coordinates:
 * :param p0Index:
 * :param p1Index:
 * :param dim:
 *
 */
template <typename T>
inline double distance(const std::vector<T>& coordinates, unsigned int p0Index, unsigned int p1Index, unsigned int dim) {

    double sum = 0;
    for(int i = 0; i < dim; ++i) {
        double d = coordinates[p0Index * dim + i] - coordinates[p1Index * dim + i];
        sum += d*d;
    }

    return sum;
}

/*
 *  Calculates hamming distance between two elements within a coordinate buffer of binary examples.
 *
 * :param coordinates:
 * :param p0Index:
 * :param p1Index:
 * :param dim:
 *
 */
template <>
inline double distance(const std::vector<unsigned char>& coordinates, unsigned int p0Index, unsigned int p1Index, unsigned int dim) {

    return 0;
}

template <typename T>
inline void distancesToRef(const std::vector<T>& coordinates, unsigned int startIndex, unsigned int endIndex, unsigned int referencePointIndex, unsigned int dim, std::vector<double>& distances)
    // here we should find the median of distances to all elements between startIndex and endIndex to the referencePointIndex.
    // we first calculate those distances
    //
{

    unsigned int range = endIndex - startIndex + 1;
    for(int i = 0; i < range; ++i) {

        unsigned int pointIndex = startIndex + i;
        double d = distance<T>(coordinates, pointIndex, referencePointIndex, dim);
        distances.push_back(d);
    }
};
