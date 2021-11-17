#pragma once

#include <vector>
#include <limits>

namespace vptree {

class VPLevelPartition {
public:
    VPLevelPartition(double radius, unsigned int start, unsigned int end) {
        // For each partition, the vantage point is the first point within the partition (pointed by indexStart)

        _radius = radius;
        _indexStart = start;
        _indexEnd = end;
    }

    VPLevelPartition() {
        // Constructs an empty level

        _radius = 0;
        _indexStart = -1;
        _indexEnd = -1;
    }

    ~VPLevelPartition() {
        if( _left != nullptr )
            delete _left;

        if( _right != nullptr )
            delete _right;
    }

    bool isEmpty() { return _indexStart == -1 || _indexStart == -1; }
    unsigned int start() { return _indexStart; }
    unsigned int end() { return _indexEnd; }

private:
    double _radius;

    //
    // _indexStart and _indexEnd are index pointers to examples within the examples list, not index of coordinates within the coordinate buffer.
    // For instance, _indexEnd pointing to last element of a coordinate buffer of 9 entries (3 examples of 3 dimensions each) would be pointing to 2, which is the index of the 3rd element.
    // If _indexStart == _indexEnd then the level contains only one element.
    unsigned int _indexStart; // points to the first of the example in which this level starts
    unsigned int _indexEnd;

    VPLevelPartition* _left = nullptr;
    VPLevelPartition* _right = nullptr;
};


template <typename T>
class VPTree {
public:
    VPTree(const std::vector<T>& array, unsigned int dimension);
    /*
     *  Builds a Vantage Point tree using each element of the given array as one coordinate buffer
     *  using L2 Metric Distance.
     *
     *  :param array: an array containing N * D elements, where N is the number of examples (e.g: points in D dimensions) and D is the size of the dimensional space.
     *  :param dimension: the number of coordinates of each example within the array. Must be dimension >= 1.
     */

protected:
    void build(const std::vector<T>& array, unsigned int dimension);
    unsigned int selectVantagePoint(unsigned int from, unsigned int to);

protected:

    VPLevelPartition _rootPartition;

    unsigned int _dimension;
    std::vector<T> _coordinatesBuffer;
    std::function<double(const std::vector<T>&, unsigned int, unsigned int)> _metric;
};

} // namespace vptree
