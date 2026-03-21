
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

template <typename distance_type> class VPLevelPartition {
public:
    VPLevelPartition(distance_type radius, int64_t start, int64_t end)
        : _radius(radius), _indexStart(start), _indexEnd(end) {}

    VPLevelPartition() : _radius(0), _indexStart(-1), _indexEnd(-1) {}

    bool isEmpty() const { return _indexStart == -1; }
    int64_t start() const { return _indexStart; }
    int64_t end() const { return _indexEnd; }
    int64_t size() const { return _indexEnd - _indexStart + 1; }
    void setRadius(distance_type radius) { _radius = radius; }
    distance_type radius() const { return _radius; }

    int32_t left_idx() const { return _left_idx; }
    int32_t right_idx() const { return _right_idx; }
    void setChildIdx(int32_t left, int32_t right) { _left_idx = left; _right_idx = right; }

    int height(const std::vector<VPLevelPartition<distance_type>> &pool) const {
        return rec_height(pool, *this, 0);
    }
    int numSubnodes(const std::vector<VPLevelPartition<distance_type>> &pool) const {
        return rec_num_subnodes(pool, *this);
    }

private:
    static int rec_height(const std::vector<VPLevelPartition<distance_type>> &pool,
                          const VPLevelPartition<distance_type> &node, int level) {
        int l = (node._left_idx >= 0) ? rec_height(pool, pool[node._left_idx], level + 1) : level;
        int r = (node._right_idx >= 0) ? rec_height(pool, pool[node._right_idx], level + 1) : level;
        return std::max(l, r) + 1;
    }
    static int rec_num_subnodes(const std::vector<VPLevelPartition<distance_type>> &pool,
                                const VPLevelPartition<distance_type> &node) {
        int l = (node._left_idx >= 0) ? rec_num_subnodes(pool, pool[node._left_idx]) : 0;
        int r = (node._right_idx >= 0) ? rec_num_subnodes(pool, pool[node._right_idx]) : 0;
        return l + r + 1;
    }

    distance_type _radius = 0;
    int64_t _indexStart = -1;
    int64_t _indexEnd = -1;
    int32_t _left_idx = -1;
    int32_t _right_idx = -1;
};

template <typename distance_type>
void rec_print_state(std::ostream &os, const std::vector<VPLevelPartition<distance_type>> &pool, int32_t idx, int level) {
    if (idx < 0) {
        return;
    }

    const VPLevelPartition<distance_type> &partition = pool[idx];

    std::string pad;
    for (int i = 0; i < 4 * level; ++i) {
        pad.push_back('.');
    }

    os << pad << " Depth: " << level << std::endl;
    os << pad << " Height: " << partition.height(pool) << std::endl;
    os << pad << " Num Sub Nodes: " << partition.numSubnodes(pool) << std::endl;
    os << pad << " Index Start: " << partition.start() << std::endl;
    os << pad << " Index End:   " << partition.end() << std::endl;

    int64_t lsize = (partition.left_idx() >= 0) ? pool[partition.left_idx()].height(pool) : 0;
    int64_t rsize = (partition.right_idx() >= 0) ? pool[partition.right_idx()].height(pool) : 0;
    os << pad << " Left Subtree Height: " << lsize << std::endl;
    os << pad << " Right Subtree Height: " << rsize << std::endl;

    if (partition.left_idx() >= 0) {
        os << pad << " [+] Left children:" << std::endl;
    }
    rec_print_state(os, pool, partition.left_idx(), level + 1);
    if (partition.right_idx() >= 0) {
        os << pad << " [+] Right children:" << std::endl;
    }
    rec_print_state(os, pool, partition.right_idx(), level + 1);
}

}; // namespace vptree
