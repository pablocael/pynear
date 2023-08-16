
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

template <typename distance_type> class VPLevelPartition : public ISerializable {
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

    virtual ~VPLevelPartition() { clear(); }

    SerializedState serialize() const {

        SerializedState state;
        std::vector<const VPLevelPartition *> flatten_tree_state;

        flatten_tree(this, flatten_tree_state);
        // we need to reverse since we will pop elements in reverse order when deserializing
        std::reverse(flatten_tree_state.begin(), flatten_tree_state.end());

        size_t total_size = flatten_tree_state.size() * (2 * sizeof(int64_t) + sizeof(float));
        state.reserve(total_size);

        // reverse the tree state since we will push it in a stack for serializing
        for (const VPLevelPartition *elem : flatten_tree_state) {
            if (elem == nullptr) {
                state.push((float)(0));
                state.push((int64_t)(-1));
                state.push((int64_t)(-1));
                continue;
            }

            state.push((float)(elem->_radius));
            state.push((int64_t)(elem->_indexStart));
            state.push((int64_t)(elem->_indexEnd));
        }

        if (state.size() != total_size) {
            throw new std::out_of_range("invalid serialization state, offsets dont match!");
        }

        state.buildChecksum();

        return state;
    }

    void deserialize(const SerializedState &state) {
        clear();
        SerializedState state_copy(state);

        VPLevelPartition *recovered = rebuild_from_state(state_copy);
        if (recovered == nullptr) {
            return;
        }

        _left = recovered->_left;
        _right = recovered->_right;
        _radius = recovered->_radius;
        _indexStart = recovered->_indexStart;
        _indexEnd = recovered->_indexEnd;
    }

    bool isEmpty() { return _indexStart == -1; }
    int64_t start() { return _indexStart; }
    int64_t end() { return _indexEnd; }
    int64_t size() { return _indexEnd - _indexStart + 1; }
    void setRadius(distance_type radius) { _radius = radius; }
    distance_type radius() { return _radius; }

    int height() { return rec_height(this, 0); }

    int numSubnodes() { return rec_num_subnodes(this); }

    void setChild(VPLevelPartition<distance_type> *left, VPLevelPartition<distance_type> *right) {
        _left = left;
        _right = right;
    }

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

    void flatten_tree(const VPLevelPartition *root, std::vector<const VPLevelPartition *> &flatten_tree_state) const {
        // visit partitions tree in preorder push all values.
        // implement pre order using a vector as a stack
        flatten_tree_state.push_back(root);
        if (root != nullptr) {
            flatten_tree(root->_left, flatten_tree_state);
            flatten_tree(root->_right, flatten_tree_state);
        }
    }

    VPLevelPartition *rebuild_from_state(SerializedState &state) {
        if (state.empty()) {
            return nullptr;
        }

        int64_t indexEnd = state.pop<int64_t>();
        int64_t indexStart = state.pop<int64_t>();
        float radius = state.pop<float>();
        if (indexEnd == -1) {
            return nullptr;
        }

        VPLevelPartition *root = new VPLevelPartition(radius, indexStart, indexEnd);
        VPLevelPartition *left = rebuild_from_state(state);
        VPLevelPartition *right = rebuild_from_state(state);
        root->setChild(left, right);
        return root;
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
