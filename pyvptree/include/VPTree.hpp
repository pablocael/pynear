/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ISerializable.hpp"
#include "VPLevelPartition.hpp"

#define ENABLE_OMP_PARALLEL 1

namespace vptree {

template <typename T, typename distance_type, distance_type (*distance)(const T &, const T &)> class VPTree : public ISerializable {
    public:
    struct VPTreeElement {

        VPTreeElement() = default;
        VPTreeElement(int64_t index, const T &value) {
            originalIndex = index;
            val = value;
        }

        int64_t originalIndex;
        T val;
    };

    struct VPTreeSearchResultElement {
        std::vector<int64_t> indexes;
        std::vector<distance_type> distances;
    };

    VPTree() {
        _rootPartition = nullptr;
        _examples.clear();
    }

    VPTree(const VPTree<T, distance_type, distance> &other) {
        auto other_state = other.serialize();
        deserialize(other_state);
    }

    const VPTree<T, distance_type, distance> &operator=(const VPTree<T, distance_type, distance> &other) { this->deserialize(other.serialize()); }

    ~VPTree() { clear(); };

    void clear() {
        if (_rootPartition != nullptr) {
            delete _rootPartition;
        }
        _rootPartition = nullptr;
        _examples.clear();
    }

    VPTree(const std::vector<T> &array) { set(array); }

    void set(const std::vector<T> &array) {
        clear();

        if(array.empty()) {
            return;
        }

        _examples.reserve(array.size());
        _examples.resize(array.size());
        for (size_t i = 0; i < array.size(); ++i) {
            _examples[i] = VPTreeElement(i, array[i]);
        }

        build(_examples);
    }

    bool isEmpty() { return _rootPartition == nullptr; }

    void print_state() {
        if (_rootPartition == nullptr) {
            return;
        }

        _rootPartition->print_state();
    }

    SerializedState serialize() const override {
        if (_rootPartition == nullptr) {
            return SerializedState();
        }

        size_t element_size = 0;
        size_t num_elements_per_example = 0;
        size_t total_size = (size_t)(3 * sizeof(size_t));

        if (_examples.size() > 0) {

            // total size is the _examples total size + the examples array size plus element size
            // _examples[0] is an array of some type (variable)
            num_elements_per_example = _examples[0].val.size();
            element_size = sizeof(_examples[0].val[0]);
            int64_t total_elements_size = num_elements_per_example * element_size;
            total_size += _examples.size() * (sizeof(int64_t) + total_elements_size);
        }

        SerializedState state;
        state.reserve(total_size);

        for (const VPTreeElement &elem : _examples) {
            state.push(elem.originalIndex);

            for (size_t i = 0; i < num_elements_per_example; ++i) {
                // since we dont know the sub element type of T (T is an array of something)
                // we need to copy using memcopy and memory size
                state.push_by_size(&elem.val[i], element_size);
            }
        }

        state.push(_examples.size());
        state.push(num_elements_per_example);
        state.push(element_size);

        if (state.size() != total_size) {
            throw new std::out_of_range("invalid serialization state, offsets dont match!");
        }

        SerializedState partition_state = _rootPartition->serialize();
        partition_state += state;
        partition_state.buildChecksum();
        return partition_state;
    }

    void deserialize(const SerializedState &state) override {
        clear();
        if (state.data.empty()) {
            return;
        }

        if (!state.isValid()) {
            throw new std::invalid_argument("invalid state - checksum mismatch");
        }

        SerializedState copy(state);

        _rootPartition = new VPLevelPartition<distance_type>();

        size_t elem_size = copy.pop<size_t>();
        size_t num_elements_per_example = copy.pop<size_t>();
        size_t num_examples = copy.pop<size_t>();

        _examples.reserve(num_examples);
        _examples.resize(num_examples);
        for (int64_t i = num_examples - 1; i >= 0; --i) {

            auto &example = _examples[i];
            auto &val = example.val;
            val.resize(num_elements_per_example);

            for (int64_t j = num_elements_per_example - 1; j >= 0; --j) {
                copy.pop_by_size(&val[j], elem_size);
            }
            int64_t originalIndex = copy.pop<int64_t>();
            example.originalIndex = originalIndex;
        }

        _rootPartition->deserialize(copy);
    }

    void searchKNN(const std::vector<T> &queries, unsigned int k, std::vector<VPTreeSearchResultElement> &results) {

        if (isEmpty()) {
            throw std::runtime_error("index must be first initialized with .set() function and non empty dataset");
        }

        // we must return one result per queries
        results.resize(queries.size());

#if (ENABLE_OMP_PARALLEL)
#pragma omp parallel for schedule(static, 1)
#endif
        // i should be size_t, however msvc requires signed integral loop variables (except with -openmp:llvm)
        for (int i = 0; i < queries.size(); ++i) {
            const T &query = queries[i];
            std::priority_queue<VPTreeSearchElement> knnQueue;
            searchKNN(_rootPartition, query, k, knnQueue);

            // we must always return k elements for each search unless there is no k elements
            assert(knnQueue.size() == std::min<size_t>(_examples.size(), k));

            fillSearchResult(knnQueue, results[i]);
        }
    }

    // An optimized version for 1 NN search
    void search1NN(const std::vector<T> &queries, std::vector<int64_t> &indices, std::vector<distance_type> &distances) {

        if (isEmpty()) {
            throw std::runtime_error("index must be first initialized with .set() function and non empty dataset");
        }

        // we must return one result per queries
        indices.resize(queries.size());
        distances.resize(queries.size());

#if (ENABLE_OMP_PARALLEL)
#pragma omp parallel for schedule(static, 1)
#endif
        // i should be size_t, see above
        for (int i = 0; i < queries.size(); ++i) {
            const T &query = queries[i];
            distance_type dist = 0;
            int64_t index = -1;
            search1NN(_rootPartition, query, index, dist);
            distances[i] = dist;
            indices[i] = index;
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const VPTree<T, distance_type, distance> &vptree) {
        os << "####################" << std::endl;
        os << "# [VPTree state]" << std::endl;
        os << "Num Data Points: " << vptree._examples.size() << std::endl;

        int64_t total_memory = 0;
        if (vptree._rootPartition != nullptr) {
            total_memory =
                vptree._rootPartition->numSubnodes() * sizeof(VPLevelPartition<distance_type>) + vptree._examples.size() * sizeof(VPTreeElement);
        }
        os << "Total Memory: " << total_memory << " bytes" << std::endl;
        os << "####################" << std::endl;
        os << "[+] Root Level:" << std::endl;
        if (vptree._rootPartition != nullptr) {
            total_memory =
                vptree._rootPartition->numSubnodes() * sizeof(VPLevelPartition<distance_type>) + vptree._examples.size() * sizeof(VPTreeElement);
            os << *vptree._rootPartition << std::endl;
        } else {
            os << "<empty>" << std::endl;
        }

        return os;
    }

    protected:
    /*
     *  Builds a Vantage Point tree using each element of the given array as one coordinate buffer
     *  using the given metric distance.
     */
    void build(const std::vector<VPTreeElement> &array) {

        // Select vantage point
        std::vector<VPLevelPartition<distance_type> *> _toSplit;

        auto *root = new VPLevelPartition<distance_type>(0, 0, _examples.size() - 1);
        _toSplit.push_back(root);
        _rootPartition = root;

        while (!_toSplit.empty()) {

            VPLevelPartition<distance_type> *current = _toSplit.back();
            _toSplit.pop_back();

            int64_t start = current->start();
            int64_t end = current->end();

            if (start == end) {
                // stop dividing if there is only one point inside
                continue;
            }

            unsigned vpIndex = selectVantagePoint(start, end);

            // put vantage point as the first element within the examples list
            std::swap(_examples[vpIndex], _examples[start]);

            int64_t median = (end + start) / 2;

            // partition in order to keep all elements smaller than median in the left and larger in the right
            std::nth_element(_examples.begin() + start + 1, _examples.begin() + median, _examples.begin() + end + 1,
                             VPDistanceComparator(_examples[start]));

            /* // distance from vantage point (which is at start index) and the median element */
            auto medianDistance = distance(_examples[start].val, _examples[median].val);
            current->setRadius(medianDistance);

            // Schedule to build next levels
            // Left is every one within the median distance radius
            VPLevelPartition<distance_type> *left = nullptr;
            if (start + 1 <= median) {
                left = new VPLevelPartition<distance_type>(0, start + 1, median);
                _toSplit.push_back(left);
            }

            VPLevelPartition<distance_type> *right = nullptr;
            if (median + 1 <= end) {
                right = new VPLevelPartition<distance_type>(0, median + 1, end);
                _toSplit.push_back(right);
            }

            current->setChild(left, right);
        }
    }

    // Internal temporary struct to organize K closest elements in a priorty queue
    struct VPTreeSearchElement {
        VPTreeSearchElement(int index, distance_type dist) : index(index), dist(dist) {}
        int index;
        distance_type dist;
        bool operator<(const VPTreeSearchElement &v) const { return dist < v.dist; }
    };

    void exaustivePartitionSearch(VPLevelPartition<distance_type> *partition, const T &val, unsigned int k,
                                  std::priority_queue<VPTreeSearchElement> &knnQueue, distance_type tau) {
        for (int i = partition->start(); i <= partition->end(); ++i) {

            auto dist = distance(val, _examples[i].val);
            if (dist < tau || knnQueue.size() < k) {

                if (knnQueue.size() == k) {
                    knnQueue.pop();
                }
                int64_t indexToAdd = _examples[i].originalIndex;
                knnQueue.push(VPTreeSearchElement(indexToAdd, dist));

                tau = knnQueue.top().dist;
            }
        }
    }

    void searchKNN(VPLevelPartition<distance_type> *partition, const T &val, unsigned int k, std::priority_queue<VPTreeSearchElement> &knnQueue) {

        auto tau = std::numeric_limits<distance_type>::max();

        // stores the distance to the partition border at the time of the storage. Since tau value will change
        // whiling performing the DFS search from on level, the storage distance will be checked again when about
        // to dive into that partition. It might not be necessary to dig into the partition anymore if tau decreased.
        std::vector<std::tuple<distance_type, VPLevelPartition<distance_type> *>> toSearch = {{-1, partition}};

        while (!toSearch.empty()) {
            auto [distToBorder, current] = toSearch.back();
            toSearch.pop_back();

            auto dist = distance(val, _examples[current->start()].val);
            if (dist < tau || knnQueue.size() < k) {

                if (knnQueue.size() == k) {
                    knnQueue.pop();
                }
                int64_t indexToAdd = _examples[current->start()].originalIndex;
                knnQueue.push(VPTreeSearchElement(indexToAdd, dist));

                tau = knnQueue.top().dist;
            }

            if (distToBorder >= 0 && distToBorder > tau) {

                // distance to this partition border change and its not necessary to search within it anymore
                continue;
            }

            size_t neighborsSoFar = knnQueue.size();
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

                    size_t rightPartitionSize = (current->right() != nullptr) ? current->right()->size() : 0;
                    bool notEnoughPointsOutside = rightPartitionSize < (k - neighborsSoFar);
                    auto toBorder = dist - current->radius();

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

                    size_t leftPartitionSize = (current->left() != nullptr) ? current->left()->size() : 0;
                    bool notEnoughPointsInside = leftPartitionSize < (k - neighborsSoFar);
                    auto toBorder = current->radius() - dist;

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

    void search1NN(VPLevelPartition<distance_type> *partition, const T &val, int64_t &resultIndex, distance_type &resultDist) {

        resultDist = std::numeric_limits<distance_type>::max();
        resultIndex = -1;

        std::vector<std::tuple<distance_type, VPLevelPartition<distance_type> *>> toSearch = {{-1, partition}};

        while (!toSearch.empty()) {

            auto [distToBorder, current] = toSearch.back();
            toSearch.pop_back();

            auto dist = distance(val, _examples[current->start()].val);
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
                auto toBorder = dist - current->radius();
                if (toBorder < resultDist && current->left() != nullptr) {
                    toSearch.push_back({toBorder, current->left()});
                }

                // must search outside
                if (current->right() != nullptr) {
                    toSearch.push_back({-1, current->right()});
                }
            } else {
                auto toBorder = current->radius() - dist;
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

    int64_t selectVantagePoint(int64_t fromIndex, int64_t toIndex) {

        // for now, simple random point selection as basic strategy: TODO: better vantage point selection
        // considering length of active region border (as in Yianilos (1993) paper)
        //
        assert(fromIndex >= 0 && fromIndex < _examples.size() && toIndex >= 0 && toIndex < _examples.size() && fromIndex <= toIndex &&
               "fromIndex and toIndex must be in a valid range");

        int64_t range = (toIndex - fromIndex) + 1;
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
    VPLevelPartition<distance_type> *_rootPartition = nullptr;
};

} // namespace vptree
