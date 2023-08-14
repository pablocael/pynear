
#pragma once

#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <utility>
#include <vector>

#include "ISerializable.hpp"

#define ENABLE_OMP_PARALLEL 1
namespace vptree {

template <typename distance_type> class VPLevelPartition {
    public:
    VPLevelPartition(distance_type radius, unsigned int start, unsigned int end) {
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

        size_t total_size = flatten_tree_state.size() * (2 * sizeof(int64_t) + sizeof(float));
        state.data.resize(total_size);

        uint8_t *buffer = &state.data[0];
        uint8_t *p_buffer = buffer;

        // reverse the tree state since we will push it in a stack for serializing
        for (const VPLevelPartition *elem : flatten_tree_state) {
            if (elem == nullptr) {
                (*(int64_t *)(p_buffer)) = (int64_t)(-1);
                p_buffer += sizeof(int64_t);
                (*(int64_t *)(p_buffer)) = (int64_t)(-1);
                p_buffer += sizeof(int64_t);
                (*(float *)(p_buffer)) = (float)(-1);
                p_buffer += sizeof(float);
                continue;
            }
            (*(int64_t *)(p_buffer)) = (int64_t)(elem->_indexEnd);
            p_buffer += sizeof(int64_t);
            (*(int64_t *)(p_buffer)) = (int64_t)(elem->_indexStart);
            p_buffer += sizeof(int64_t);
            (*(float *)(p_buffer)) = (float)(elem->_radius);
            p_buffer += sizeof(float);
        }

        if ((size_t)(p_buffer - buffer) != total_size) {
            throw new std::out_of_range("invalid serialization state, offsets dont match!");
        }

        return state;
    }

    void deserialize(const SerializedState &state, uint32_t offset) {
        clear();

        uint8_t *p_buffer = (&const_cast<std::vector<uint8_t> &>(state.data)[0]) + offset;
        VPLevelPartition *recovered = rebuild_from_state(&p_buffer);
        if (recovered == nullptr) {
            return;
        }

        _left = recovered->_left;
        _right = recovered->_right;
        _radius = recovered->_radius;
        _indexStart = recovered->_indexStart;
        _indexEnd = recovered->_indexEnd;
    }

    bool isEmpty() { return _radius == 0; }
    unsigned int start() { return _indexStart; }
    unsigned int end() { return _indexEnd; }
    unsigned int size() { return _indexEnd - _indexStart + 1; }
    void setRadius(distance_type radius) { _radius = radius; }
    distance_type radius() { return _radius; }

    void setChild(VPLevelPartition<distance_type> *left, VPLevelPartition<distance_type> *right) {
        _left = left;
        _right = right;
    }

    VPLevelPartition *left() const { return _left; }
    VPLevelPartition *right() const { return _right; }

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
        flatten_tree_state.push_back(root);
        if (root != nullptr) {
            flatten_tree(root->left(), flatten_tree_state);
            flatten_tree(root->right(), flatten_tree_state);
        }
    }

    VPLevelPartition *rebuild_from_state(uint8_t **p_buffer) {
        int64_t indexEnd = (*(int64_t *)(*p_buffer));
        *p_buffer += sizeof(int64_t);
        int64_t indexStart = (*(int64_t *)(*p_buffer));
        *p_buffer += sizeof(int64_t);
        float radius = (*(float *)(*p_buffer));
        *p_buffer += sizeof(float);

        if (indexEnd == -1) {
            return nullptr;
        }

        VPLevelPartition *root = new VPLevelPartition(radius, (unsigned int)indexStart, (unsigned int)indexEnd);
        VPLevelPartition *left = rebuild_from_state(p_buffer);
        VPLevelPartition *right = rebuild_from_state(p_buffer);
        root->setChild(left, right);
        return root;
    }

    distance_type _radius;

    // _indexStart and _indexEnd are index pointers to examples within the examples list, not index of coordinates
    // within the coordinate buffer.For instance, _indexEnd pointing to last element of a coordinate buffer of 9 entries
    // (3 examples of 3 dimensions each) would be pointing to 2, which is the index of the 3rd element.
    // If _indexStart == _indexEnd then the level contains only one element.
    unsigned int _indexStart; // points to the first of the example in which this level starts
    unsigned int _indexEnd;

    VPLevelPartition<distance_type> *_left = nullptr;
    VPLevelPartition<distance_type> *_right = nullptr;
};

};
