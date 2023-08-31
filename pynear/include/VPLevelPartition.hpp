
#pragma once

#include "ISerializable.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <utility>
#include <vector>

namespace vptree {

template <typename distance_type> class VPLevelPartition;

template <typename distance_type> void rec_print_state(std::ostream &os, VPLevelPartition<distance_type> *partition, int level);

template <typename distance_type> class VPLevelPartition {
public:
    VPLevelPartition(distance_type radius, int64_t start, int64_t end) {
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

    VPLevelPartition(const VPLevelPartition &other) { *this = other; }

    virtual ~VPLevelPartition() { clear(); }


    bool isEmpty() const { return _indexStart == -1; }
    int64_t start() const { return _indexStart; }
    int64_t end() const { return _indexEnd; }
    int64_t size() const { return _indexEnd - _indexStart + 1; }
    void setRadius(distance_type radius) { _radius = radius; }
    distance_type radius() const { return _radius; }

    int height() { return rec_height(this, 0); }

    int numSubnodes() { return rec_num_subnodes(this); }

    void setChild(VPLevelPartition<distance_type> *left, VPLevelPartition<distance_type> *right) {
        _left = left;
        _right = right;
    }

    VPLevelPartition *deepcopy() { return rec_deepcopy(this); }

    VPLevelPartition *left() const { return _left; }
    VPLevelPartition *right() const { return _right; }

    friend std::ostream &operator<<(std::ostream &os, const VPLevelPartition<distance_type> &partition) {
        rec_print_state<distance_type>(os, &const_cast<VPLevelPartition<distance_type> &>(partition), 0);
        return os;
    }

private:
    void clear() {
        if (_left != nullptr)
            delete _left;

        if (_right != nullptr)
            delete _right;

        _left = nullptr;
        _right = nullptr;
    }


    VPLevelPartition *rec_deepcopy(VPLevelPartition *root) {
        if (root == nullptr) {
            return nullptr;
        }

        VPLevelPartition *result = new VPLevelPartition(*root);

        result->_left = rec_deepcopy(root->_left);
        result->_right = rec_deepcopy(root->_right);

        return result;
    }

    int rec_height(VPLevelPartition *root, int level = 0) {

        if (root == nullptr) {
            return level;
        }
        int l_l = rec_height(root->_left, level + 1);
        int l_r = rec_height(root->_right, level + 1);
        return std::max(l_l, l_r) + 1;
    }

    int rec_num_subnodes(VPLevelPartition *root) {

        if (root == nullptr) {
            return 0;
        }
        int l_l = rec_num_subnodes(root->_left);
        int l_r = rec_num_subnodes(root->_right);
        return l_l + l_r + 1;
    }

    distance_type _radius;

    // _indexStart and _indexEnd are index pointers to examples within the examples list of the VPTre, not index of coordinates
    // within the coordinate buffer. For instance, if _indexEnd pointing to last element of a coordinate buffer of 9 entries
    // (3 examples of 3 dimensions each), then it would be pointing to 2, which is the index of the 3rd element.
    // If _indexStart == _indexEnd then the level contains only one element.
    // If _indexStart == -1 then the level is empty.
    int64_t _indexStart; // points to the first of the example in which this level starts
    int64_t _indexEnd;

    VPLevelPartition<distance_type> *_left = nullptr;
    VPLevelPartition<distance_type> *_right = nullptr;
};

template <typename distance_type> void rec_print_state(std::ostream &os, VPLevelPartition<distance_type> *partition, int level) {
    if (partition == nullptr) {
        return;
    }

    std::string pad;
    for (int i = 0; i < 4 * level; ++i) {
        pad.push_back('.');
    }

    os << pad << " Depth: " << level << std::endl;
    os << pad << " Height: " << partition->height() << std::endl;
    os << pad << " Num Sub Nodes: " << partition->numSubnodes() << std::endl;
    os << pad << " Index Start: " << partition->start() << std::endl;
    os << pad << " Index End:   " << partition->end() << std::endl;

    int64_t lsize = partition->left() != nullptr ? partition->left()->height() : 0;
    int64_t rsize = partition->right() != nullptr ? partition->right()->height() : 0;
    os << pad << " Left Subtree Height: " << lsize << std::endl;
    os << pad << " Right Subtree Height: " << rsize << std::endl;

    if (partition->left()) {
        os << pad << " [+] Left children:" << std::endl;
    }
    rec_print_state(os, partition->left(), level + 1);
    if (partition->right()) {
        os << pad << " [+] Right children:" << std::endl;
    }
    rec_print_state(os, partition->right(), level + 1);
}

}; // namespace vptree
