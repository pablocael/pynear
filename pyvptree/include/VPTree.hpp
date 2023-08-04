/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#pragma once

#include <algorithm>
#include <bits/stdc++.h>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#define ENABLE_OMP_PARALLEL 1

namespace vptree {

class Serializable {
    public:
    virtual std::vector<char> serialize() = 0;
    virtual void deserialize(const std::vector<char> &data) = 0;
};

class VPLevelPartition : public Serializable {
    public:
    VPLevelPartition(float radius, unsigned int start, unsigned int end) {
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

    // this function contains the serialization of pyvptree
    std::vector<char> serialize() override {
        std::vector<VPLevelPartition *> flatten_tree_state;
        flatten_tree(this, flatten_tree_state);

        size_t total_size = flatten_tree_state.size() * (2 * sizeof(int64_t) + sizeof(float));
        char *buffer = new char[total_size];
        char *p_buffer = buffer;

        // reverse the tree state since we will push it in a stack for serializing
        std::reverse(flatten_tree_state.begin(), flatten_tree_state.end());
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

        std::vector<char> result;
        return result;
    }

    // this function contains the unserialization process of your class
    void deserialize(const std::vector<char> &data) override {
        VPLevelPartition *recovered = rebuild_from_state(&const_cast<std::vector<char> &>(data)[0]);
        if (recovered == nullptr) {
            return;
        }

        _left = recovered->_left;
        _right = recovered->_right;
        _radius = recovered->_radius;
        _indexStart = recovered->_indexStart;
        _indexEnd = recovered->_indexEnd;
    }

    void flatten_tree(VPLevelPartition *root, std::vector<VPLevelPartition *> &flatten_tree_state) {
        // visit partitions tree in preorder push all values.
        flatten_tree_state.push_back(root);
        if (root != nullptr) {
            flatten_tree(root->right(), flatten_tree_state);
            flatten_tree(root->left(), flatten_tree_state);
        }
    }

    VPLevelPartition *rebuild_from_state(char *p_buffer) {
        float radius = (*(float *)(p_buffer));
        p_buffer += sizeof(float);
        if (radius == -1) {
            return nullptr;
        }
        int64_t indexStart = (*(int64_t *)(p_buffer));
        p_buffer += sizeof(int64_t);
        int64_t indexEnd = (*(int64_t *)(p_buffer));
        p_buffer += sizeof(int64_t);
        VPLevelPartition *root = new VPLevelPartition(radius, (unsigned int)indexStart, (unsigned int)indexEnd);
        VPLevelPartition *left = rebuild_from_state(p_buffer);
        VPLevelPartition *right = rebuild_from_state(p_buffer);
        root->setChild(left, right);
        return root;
    }

    bool isEmpty() { return _indexStart == -1 || _indexStart == -1; }
    unsigned int start() { return _indexStart; }
    unsigned int end() { return _indexEnd; }
    unsigned int size() { return _indexEnd - _indexStart + 1; }
    void setRadius(float radius) { _radius = radius; }
    float radius() { return _radius; }

    void setChild(VPLevelPartition *left, VPLevelPartition *right) {
        _left = left;
        _right = right;
    }

    VPLevelPartition *left() { return _left; }
    VPLevelPartition *right() { return _right; }

    private:
    void clear() {
        if (_left != nullptr)
            delete _left;

        if (_right != nullptr)
            delete _right;

        _left = nullptr;
        _right = nullptr;
    }

    float _radius;

    // _indexStart and _indexEnd are index pointers to examples within the examples list, not index of coordinates
    // within the coordinate buffer.For instance, _indexEnd pointing to last element of a coordinate buffer of 9 entries
    // (3 examples of 3 dimensions each) would be pointing to 2, which is the index of the 3rd element.
    // If _indexStart == _indexEnd then the level contains only one element.
    unsigned int _indexStart; // points to the first of the example in which this level starts
    unsigned int _indexEnd;

    VPLevelPartition *_left = nullptr;
    VPLevelPartition *_right = nullptr;
};

template <typename T, float (*distance)(const T &, const T &)> class VPTree : public Serializable {
    public:
    struct VPTreeElement {

        VPTreeElement() = default;
        VPTreeElement(unsigned int index, const T &value) {
            originalIndex = index;
            val = value;
        }

        unsigned int originalIndex;
        T val;
    };

    struct VPTreeSearchResultElement {
        std::vector<unsigned int> indexes;
        std::vector<float> distances;
    };

    VPTree() = default;

    VPTree(const std::vector<T> &array) {

        _examples.reserve(array.size());
        _examples.resize(array.size());
        for (unsigned int i = 0; i < array.size(); ++i) {
            _examples[i] = VPTreeElement(i, array[i]);
        }

        build(_examples);
    }

    // this function contains the serialization of pyvptree
    std::vector<char> serialize() override {
        if (_rootPartition == nullptr) {
            return std::vector<char>();
        }

        std::vector<char> data = _rootPartition->serialize();

        size_t element_size = 0;
        size_t total_size = data.size() + 2 * sizeof(size_t);

        if (_examples.size() > 0) {

            // total size is the _examples total size + the examples array size plus element size
            // _examples[0] is an array of some type (variable)
            element_size = _examples[0].val.size() * sizeof(_examples[0].val[0]);
            total_size += _examples.size() * (sizeof(unsigned int) + element_size);
        }

        char *buffer = new char[total_size];
        char *p_buffer = buffer;

        for (const VPTreeElement &elem : _examples) {
            *((unsigned int *)p_buffer) = elem.originalIndex;
            p_buffer += sizeof(unsigned int);

            for (auto v : elem.val) {
                // since we dont know the sub element type of T (T is an array of something)
                // we need to copy using memcopy and memory size
                std::memcpy(p_buffer, &v, sizeof(v));
                p_buffer += sizeof(v);
            }
        }

        // finally set the total size of array and size of an element
        *((size_t *)p_buffer) = element_size;
        p_buffer += sizeof(size_t);

        *((size_t *)p_buffer) = _examples.size();
        p_buffer += sizeof(size_t);

        std::vector<char> new_data;
        new_data.assign(buffer, buffer + total_size);
        data.insert(data.begin(), new_data.begin(), new_data.end());
        return data;
    }

    // this function contains the unserialization process of your class
    void deserialize(const std::vector<char> &data) override {
        size_t num_examples;
        ss >> num_examples;
        std::cout << "** DESER: num examples is " << num_examples << std::endl << std::flush;
        /* _examples.clear(); */
        /* _examples.reserve(num_examples); */
        /* _examples.resize(num_examples); */
        /* for(size_t i = 0; i < num_examples; ++i) { */

        /*     size_t val_size; */
        /*     ss >> val_size; */
        /*     T val; */
        /*     val.resize(val_size); */
        /*     val.reserve(val_size); */
        /*     for(size_t i = 0; i < val_size; ++i) { */
        /*         ss >> val[i]; */
        /*     } */

        /*     unsigned int index;; */
        /*     ss >> index; */
        /*     _examples[i].val = val; */
        /*     _examples[i].originalIndex = index; */
        /* } */

        /* if(_rootPartition != nullptr) { */
        /*     delete _rootPartition; */
        /* } */
        /* _rootPartition = new VPLevelPartition(); */
        /* std::string str = ss.str(); */
        /* _rootPartition->deserialize(std::vector<char>(str.begin(), str.end())); */
    }

    void searchKNN(const std::vector<T> &queries, unsigned int k, std::vector<VPTree::VPTreeSearchResultElement> &results) {

        if (_rootPartition == nullptr) {
            return;
        }

        // we must return one result per queries
        results.resize(queries.size());

#if (ENABLE_OMP_PARALLEL)
#pragma omp parallel for schedule(static, 1) num_threads(8)
#endif
        for (int i = 0; i < queries.size(); ++i) {
            const T &query = queries[i];
            std::priority_queue<VPTreeSearchElement> knnQueue;
            searchKNN(_rootPartition, query, k, knnQueue);

            // we must always return k elements for each search unless there is no k elements
            assert(static_cast<unsigned int>(knnQueue.size()) == std::min<unsigned int>(_examples.size(), k));

            fillSearchResult(knnQueue, results[i]);
        }
    }

    // An optimized version for 1 NN search
    void search1NN(const std::vector<T> &queries, std::vector<unsigned int> &indices, std::vector<float> &distances) {

        if (_rootPartition == nullptr) {
            return;
        }

        // we must return one result per queries
        indices.resize(queries.size());
        distances.resize(queries.size());

#if (ENABLE_OMP_PARALLEL)
#pragma omp parallel for schedule(static, 1) num_threads(8)
#endif
        for (int i = 0; i < queries.size(); ++i) {
            const T &query = queries[i];
            float dist = 0;
            unsigned int index = -1;
            search1NN(_rootPartition, query, index, dist);
            distances[i] = dist;
            indices[i] = index;
        }
    }

    protected:
    /*
     *  Builds a Vantage Point tree using each element of the given array as one coordinate buffer
     *  using the given metric distance.
     */
    void build(const std::vector<VPTreeElement> &array) {

        // Select vantage point
        std::vector<VPLevelPartition *> _toSplit;

        auto *root = new VPLevelPartition(0, 0, _examples.size() - 1);
        _toSplit.push_back(root);
        _rootPartition = root;

        while (!_toSplit.empty()) {

            VPLevelPartition *current = _toSplit.back();
            _toSplit.pop_back();

            unsigned int start = current->start();
            unsigned int end = current->end();

            if (start == end) {
                // stop dividing if there is only one point inside
                continue;
            }

            unsigned vpIndex = selectVantagePoint(start, end);

            // put vantage point as the first element within the examples list
            std::swap(_examples[vpIndex], _examples[start]);

            unsigned int median = (end + start) / 2;

            // partition in order to keep all elements smaller than median in the left and larger in the right
            std::nth_element(_examples.begin() + start + 1, _examples.begin() + median, _examples.begin() + end + 1,
                             VPDistanceComparator(_examples[start]));

            /* // distance from vantage point (which is at start index) and the median element */
            float medianDistance = distance(_examples[start].val, _examples[median].val);
            current->setRadius(medianDistance);

            // Schedule to build next levels
            // Left is every one within the median distance radius
            VPLevelPartition *left = nullptr;
            if (start + 1 <= median) {
                left = new VPLevelPartition(-1, start + 1, median);
                _toSplit.push_back(left);
            }

            VPLevelPartition *right = nullptr;
            if (median + 1 <= end) {
                right = new VPLevelPartition(-1, median + 1, end);
                _toSplit.push_back(right);
            }

            current->setChild(left, right);
        }
    }

    // Internal temporary struct to organize K closest elements in a priorty queue
    struct VPTreeSearchElement {
        VPTreeSearchElement(int index, float dist) : index(index), dist(dist) {}
        int index;
        float dist;
        bool operator<(const VPTreeSearchElement &v) const { return dist < v.dist; }
    };

    void exaustivePartitionSearch(VPLevelPartition *partition, const T &val, unsigned int k, std::priority_queue<VPTreeSearchElement> &knnQueue,
                                  float tau) {
        for (int i = partition->start(); i <= partition->end(); ++i) {

            float dist = distance(val, _examples[i].val);
            if (dist < tau || knnQueue.size() < k) {

                if (knnQueue.size() == k) {
                    knnQueue.pop();
                }
                unsigned int indexToAdd = _examples[i].originalIndex;
                knnQueue.push(VPTreeSearchElement(indexToAdd, dist));

                tau = knnQueue.top().dist;
            }
        }
    }

    void searchKNN(VPLevelPartition *partition, const T &val, unsigned int k, std::priority_queue<VPTreeSearchElement> &knnQueue) {

        float tau = std::numeric_limits<float>::max();

        // stores the distance to the partition border at the time of the storage. Since tau value will change
        // whiling performing the DFS search from on level, the storage distance will be checked again when about
        // to dive into that partition. It might not be necessary to dig into the partition anymore if tau decreased.
        std::vector<std::tuple<float, VPLevelPartition *>> toSearch = {{-1, partition}};

        while (!toSearch.empty()) {
            auto [distToBorder, current] = toSearch.back();
            toSearch.pop_back();

            float dist = distance(val, _examples[current->start()].val);
            if (dist < tau || knnQueue.size() < k) {

                if (knnQueue.size() == k) {
                    knnQueue.pop();
                }
                unsigned int indexToAdd = _examples[current->start()].originalIndex;
                knnQueue.push(VPTreeSearchElement(indexToAdd, dist));

                tau = knnQueue.top().dist;
            }

            if (distToBorder >= 0 && distToBorder > tau) {

                // distance to this partition border change and its not necessary to search within it anymore
                continue;
            }

            unsigned int neighborsSoFar = knnQueue.size();
            if (dist > current->radius()) {
                // must search outside

                /*
                    We may need to search inside as well. We will schedule this partition to be queried later.
                    We add it first (if needed) and the outside partition after: since we are doing DFS, we do LIFO
                    By the time this partition is accessed later, it might be rejected since tau might decrease
                    during the search of the outside partition (which will be searched for sure)

                    We store current toBorder distance to use later to compare to latest value of tau, so the partition
                   might not even need to be searched, depending on the tau resulting from the DFS search in outside
                   partition.

                    The exact same logic is applied to the inside case in the else statement.

                */
                if (current->left() != nullptr) {

                    unsigned int rightPartitionSize = (current->right() != nullptr) ? current->right()->size() : 0;
                    bool notEnoughPointsOutside = rightPartitionSize < (k - neighborsSoFar);
                    float toBorder = dist - current->radius();

                    // we might not have enough point outside to reject the inside partition, so we might need to search
                    // for both
                    if (notEnoughPointsOutside) {
                        toSearch.push_back({-1, current->left()});
                    } else if (toBorder <= tau) {
                        toSearch.push_back({toBorder, current->left()});
                    }
                }

                // now schedule outside
                if (current->right() != nullptr) {
                    toSearch.push_back({-1, current->right()});
                }
            } else {
                // must search inside
                // logic is analogous to the outside case

                if (current->right() != nullptr) {

                    unsigned int leftPartitionSize = (current->left() != nullptr) ? current->left()->size() : 0;
                    bool notEnoughPointsInside = leftPartitionSize < (k - neighborsSoFar);
                    float toBorder = current->radius() - dist;

                    if (notEnoughPointsInside) {
                        toSearch.push_back({-1, current->right()});
                    } else if (toBorder <= tau) {
                        toSearch.push_back({toBorder, current->right()});
                    }
                }

                // now schedule inside
                if (current->left() != nullptr) {
                    toSearch.push_back({-1, current->left()});
                }
            }
        }
    }

    void search1NN(VPLevelPartition *partition, const T &val, unsigned int &resultIndex, float &resultDist) {

        resultDist = std::numeric_limits<float>::max();
        resultIndex = -1;

        std::vector<std::tuple<float, VPLevelPartition *>> toSearch = {{-1, partition}};

        while (!toSearch.empty()) {

            auto [distToBorder, current] = toSearch.back();
            toSearch.pop_back();

            float dist = distance(val, _examples[current->start()].val);
            if (dist < resultDist) {
                resultDist = dist;
                resultIndex = _examples[current->start()].originalIndex;
            }

            if (distToBorder >= 0 && distToBorder > resultDist) {

                // distance to this partition border change and its not necessary to search within it anymore
                continue;
            }

            if (dist > current->radius()) {
                // may need to search inside as well
                float toBorder = dist - current->radius();
                if (toBorder < resultDist && current->left() != nullptr) {
                    toSearch.push_back({toBorder, current->left()});
                }

                // must search outside
                if (current->right() != nullptr) {
                    toSearch.push_back({-1, current->right()});
                }
            } else {
                float toBorder = current->radius() - dist;
                // may need to search outside as well
                if (toBorder < resultDist && current->right() != nullptr) {
                    toSearch.push_back({toBorder, current->right()});
                }

                // must search inside
                if (current->left() != nullptr) {
                    toSearch.push_back({-1, current->left()});
                }
            }
        }
    }

    unsigned int selectVantagePoint(unsigned int fromIndex, unsigned int toIndex) {

        // for now, simple random point selection as basic strategy: TODO: better vantage point selection
        // considering length of active region border (as in Yianilos (1993) paper)
        //
        assert(fromIndex >= 0 && fromIndex < _examples.size() && toIndex >= 0 && toIndex < _examples.size() && fromIndex <= toIndex &&
               "fromIndex and toIndex must be in a valid range");

        unsigned int range = (toIndex - fromIndex) + 1;
        return fromIndex + (rand() % range);
    }

    // Fill result element from serach element internal structure
    // After a call to that function, knnQueue gets invalidated!
    void fillSearchResult(std::priority_queue<VPTreeSearchElement> &knnQueue, VPTreeSearchResultElement &element) {
        element.distances.reserve(knnQueue.size());
        element.indexes.reserve(knnQueue.size());

        while (!knnQueue.empty()) {
            const VPTreeSearchElement &top = knnQueue.top();
            element.distances.push_back(top.dist);
            element.indexes.push_back(top.index);
            knnQueue.pop();
        }
    }
    /*
     * A vantage point distance comparator. Will check which from two points are closer to the reference vantage point.
     * This is used to find the median distance from vantage point in order to split the VPLevelPartition into two sets.
     */
    struct VPDistanceComparator {

        const T &item;
        VPDistanceComparator(const VPTreeElement &item) : item(item.val) {}
        bool operator()(const VPTreeElement &a, const VPTreeElement &b) { return distance(item, a.val) < distance(item, b.val); }
    };

    protected:
    std::vector<VPTreeElement> _examples;
    VPLevelPartition *_rootPartition = nullptr;
};

} // namespace vptree
