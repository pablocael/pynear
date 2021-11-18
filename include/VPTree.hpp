#pragma once

#include <vector>
#include <limits>

namespace vptree {

template <typename T>
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
    void setRadius(double radius) { _radius = radius; }

    void setChild(VPLevelPartition<T>* left, VPLevelPartition<T>* right) {
        _left = left;
        _right = right;
    }

private:
    double _radius;


    //
    // _indexStart and _indexEnd are index pointers to examples within the examples list, not index of coordinates within the coordinate buffer.
    // For instance, _indexEnd pointing to last element of a coordinate buffer of 9 entries (3 examples of 3 dimensions each) would be pointing to 2, which is the index of the 3rd element.
    // If _indexStart == _indexEnd then the level contains only one element.
    unsigned int _indexStart; // points to the first of the example in which this level starts
    unsigned int _indexEnd;

    VPLevelPartition<T>* _left = nullptr;
    VPLevelPartition<T>* _right = nullptr;
};


template <typename T, double(*distance)(const T&, const T&)>
class VPTree {
public:
    // TODO: create and Item element containing original index and comparator. We need to reorder elements keeping original index
    // TODO: create another constructor for IVF with custom ids
    VPTree(const std::vector<T>& array);

protected:
    /*
     *  Builds a Vantage Point tree using each element of the given array as one coordinate buffer
     *  using the given metric distance.
     */
    void build(const std::vector<T>& array);
    unsigned int selectVantagePoint(unsigned int fromIndex, unsigned int toIndex);

    /*
     * A vantage point distance comparator. Will check which from two points are closer to the reference vantage point.
     * This is used to find the median distance from vantage point in order to split the VPLevelPartition into two sets.
     */
    struct VPDistanceComparator {

        const T& item;
        VPDistanceComparator( const T& item ) : item(item) {}
        bool operator()(const T& a, const T& b) {
            return distance( item, a ) < distance( item, b );
            
        }
        
    };

protected:

    std::vector<T> _examples;
    VPLevelPartition<T>* _rootPartition = nullptr;
    const unsigned int MIN_POINTS_PER_PARTITION = 20;
};

} // namespace vptree
