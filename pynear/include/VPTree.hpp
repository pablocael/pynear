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
#include <numeric>
#include <omp.h>
#include <queue>
#include <sstream>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DistanceFunctions.hpp"
#include "VPLevelPartition.hpp"

namespace vptree {

template <typename T, typename distance_type, distance_type (*distance)(const T &, const T &)> class VPTree {
    /*
     * Template arguments:
     * - T: a custom user type that will compose the VPTree element. Distance function must input this type.
     * - distance_type: the numeric type for the distance value retrieved by distance function when
     * - distance: a function pointer of a distance operator that will measure the distance between two objects
     *   of type T.
     *   measuring ddistances between two objects of type T.
     */
public:
    struct VPTreeSearchResultElement {
        std::vector<int64_t> indexes;
        std::vector<distance_type> distances;
    };

    VPTree() {}

    VPTree(const VPTree<T, distance_type, distance> &other) {
        _indices = other._indices;
        _nodePool = other._nodePool;
        _rootIdx = other._rootIdx;
        _dim = other._dim;
        if constexpr (std::is_same_v<T, FlatSpan>) {
            _flat_backing = other._flat_backing;
            _examples.resize(other._examples.size());
            for (size_t i = 0; i < _examples.size(); i++)
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
        } else {
            _examples = other._examples;
        }
    }

    const VPTree<T, distance_type, distance> &operator=(const VPTree<T, distance_type, distance> &other) {
        if (this == &other) return *this;
        _indices = other._indices;
        _nodePool = other._nodePool;
        _rootIdx = other._rootIdx;
        _dim = other._dim;
        if constexpr (std::is_same_v<T, FlatSpan>) {
            _flat_backing = other._flat_backing;
            _examples.resize(other._examples.size());
            for (size_t i = 0; i < _examples.size(); i++)
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
        } else {
            _examples = other._examples;
        }
        return *this;
    }

    virtual ~VPTree() { clear(); };

    void clear() {
        _rootIdx = -1;
        _nodePool.clear();
        _examples.clear();
        _flat_backing.clear();
        _dim = 0;
    }

    VPTree(const std::vector<T> &array) { set(array); }

    void set(const std::vector<T> &array) {
        clear();
        if (array.empty()) return;

        if constexpr (std::is_same_v<T, FlatSpan>) {
            _dim = array[0].sz;
            size_t n = array.size();
            _flat_backing.resize(n * _dim);
            for (size_t i = 0; i < n; i++)
                std::memcpy(_flat_backing.data() + i * _dim, array[i].ptr, _dim * sizeof(float));
            _examples.resize(n);
            for (size_t i = 0; i < n; i++)
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
        } else {
            _examples = array;
        }
        build(_examples);
    }

    void set(std::vector<T> &&array) {
        clear();
        if (array.empty()) return;

        if constexpr (std::is_same_v<T, FlatSpan>) {
            // For FlatSpan, data is owned externally — copy into flat backing
            _dim = array[0].sz;
            size_t n = array.size();
            _flat_backing.resize(n * _dim);
            for (size_t i = 0; i < n; i++)
                std::memcpy(_flat_backing.data() + i * _dim, array[i].ptr, _dim * sizeof(float));
            _examples.resize(n);
            for (size_t i = 0; i < n; i++)
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
        } else {
            _examples = std::move(array);
        }
        build(_examples);
    }

    bool isEmpty() { return _rootIdx == -1; }

    void print_state() {
        if (_rootIdx == -1) {
            return;
        }

        rec_print_state<distance_type>(std::cout, _nodePool, _rootIdx, 0);
    }

    void searchKNN(const std::vector<T> &queries, size_t k, std::vector<VPTreeSearchResultElement> &results) {

        if (isEmpty()) {
            throw std::runtime_error("index must be first initialized with .set() function and non empty dataset");
        }

        // we must return one result per queries
        results.resize(queries.size());

#if (ENABLE_OMP_PARALLEL)
#pragma omp parallel for schedule(dynamic) if (queries.size() > 1)
#endif
        // i should be size_t, however msvc requires signed integral loop variables (except with -openmp:llvm)
        for (int i = 0; i < static_cast<int>(queries.size()); ++i) {
            const T &query = queries[i];
            std::priority_queue<VPTreeSearchElement> knnQueue;
            searchKNN(_rootIdx, query, k, knnQueue);

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
#pragma omp parallel for schedule(dynamic) if (queries.size() > 1)
#endif
        // i should be size_t, see above
        for (int i = 0; i < static_cast<int>(queries.size()); ++i) {
            const T &query = queries[i];
            distance_type dist = 0;
            int64_t index = -1;
            search1NN(_rootIdx, query, index, dist);
            distances[i] = dist;
            indices[i] = index;
        }
    }

    const std::vector<float>& flatBacking() const { return _flat_backing; }
    size_t flatDim() const { return _dim; }
    const std::vector<int64_t>& indexPermutation() const { return _indices; }
    const std::vector<VPLevelPartition<distance_type>>& partitionPool() const { return _nodePool; }
    int32_t rootPartitionIdx() const { return _rootIdx; }

    void initFromSerialized(std::vector<float> flat, size_t dim,
                            std::vector<int64_t> indices,
                            std::vector<VPLevelPartition<distance_type>> pool,
                            int32_t root_idx) {
        clear();
        _flat_backing = std::move(flat);
        _dim = dim;
        size_t n = (dim > 0) ? _flat_backing.size() / dim : 0;
        _examples.resize(n);
        if constexpr (std::is_same_v<T, FlatSpan>) {
            for (size_t i = 0; i < n; i++)
                _examples[i] = FlatSpan{_flat_backing.data() + i * dim, dim};
        }
        _indices = std::move(indices);
        _nodePool = std::move(pool);
        _rootIdx = root_idx;
    }

    friend std::ostream &operator<<(std::ostream &os, const VPTree<T, distance_type, distance> &vptree) {
        os << "####################" << std::endl;
        os << "# [VPTree state]" << std::endl;
        os << "Num Data Points: " << vptree._examples.size() << std::endl;

        int64_t total_memory = 0;
        if (vptree._rootIdx != -1) {
            total_memory = vptree._nodePool[vptree._rootIdx].numSubnodes(vptree._nodePool) * sizeof(VPLevelPartition<distance_type>) + vptree._examples.size() * sizeof(T);
        }
        os << "Total Memory: " << total_memory << " bytes" << std::endl;
        os << "####################" << std::endl;
        os << "[+] Root Level:" << std::endl;
        if (vptree._rootIdx != -1) {
            total_memory = vptree._nodePool[vptree._rootIdx].numSubnodes(vptree._nodePool) * sizeof(VPLevelPartition<distance_type>) + vptree._examples.size() * sizeof(T);
            rec_print_state<distance_type>(os, vptree._nodePool, vptree._rootIdx, 0);
            os << std::endl;
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
    void build(const std::vector<T> &array) {

        _indices.resize(_examples.size());

        // initialize indices sequentially
        std::iota(_indices.begin(), _indices.end(), 0);

        // Select vantage point
        std::vector<int32_t> _toSplit;
        // Pre-allocated scratch buffer: (distance, example_index) pairs for nth_element.
        // Reused across all build iterations to avoid repeated heap allocations.
        std::vector<std::pair<distance_type, int64_t>> distPairs;
        distPairs.reserve(_examples.size());

        _nodePool.clear();
        _nodePool.reserve(_examples.size());
        _nodePool.push_back(VPLevelPartition<distance_type>(0, 0, _examples.size() - 1));
        _rootIdx = 0;
        _toSplit.push_back(_rootIdx);

        while (!_toSplit.empty()) {

            int32_t currentIdx = _toSplit.back();
            _toSplit.pop_back();

            int64_t start = _nodePool[currentIdx].start();
            int64_t end = _nodePool[currentIdx].end();

            if (start == end) {
                // stop dividing if there is only one point inside
                continue;
            }

            unsigned vpIndex = selectVantagePoint(start, end);

            // put vantage point as the first element within the examples list
            std::swap(_indices[vpIndex], _indices[start]);

            int64_t median = (end + start) / 2;

            // Pre-compute all distances from the VP once, storing (dist, example_idx) pairs.
            // nth_element then sorts the pairs by distance (O(1) comparisons, no hashing).
            // This reduces distance calls from O(n log n) to O(n) per build level.
            const int64_t range_size = end - start;
            const int64_t medianInPairs = median - start - 1;

            // When end == start+1 there is exactly one non-VP element and median == start,
            // giving medianInPairs == -1. No partitioning is needed: radius = 0 and the
            // single element goes straight to the right child.
            distance_type medianDistance = 0;
            if (medianInPairs >= 0) {
                distPairs.resize(range_size);
                const auto &vp = _examples[_indices[start]];
                for (int64_t ci = 0; ci < range_size; ci++) {
                    const int64_t exIdx = _indices[start + 1 + ci];
                    distPairs[ci] = {distance(vp, _examples[exIdx]), exIdx};
                }

                std::nth_element(distPairs.begin(), distPairs.begin() + medianInPairs, distPairs.begin() + range_size,
                                 [](const auto &a, const auto &b) { return a.first < b.first; });

                // Write the partitioned example indices back to _indices
                for (int64_t ci = 0; ci < range_size; ci++)
                    _indices[start + 1 + ci] = distPairs[ci].second;

                medianDistance = distPairs[medianInPairs].first;
            }
            _nodePool[currentIdx].setRadius(medianDistance);

            // Schedule to build next levels
            // Left is every one within the median distance radius
            int32_t left_idx = -1, right_idx = -1;
            if (start + 1 <= median) {
                _nodePool.push_back(VPLevelPartition<distance_type>(0, start + 1, median));
                left_idx = _nodePool.size() - 1;
                _toSplit.push_back(left_idx);
            }

            if (median + 1 <= end) {
                _nodePool.push_back(VPLevelPartition<distance_type>(0, median + 1, end));
                right_idx = _nodePool.size() - 1;
                _toSplit.push_back(right_idx);
            }

            _nodePool[currentIdx].setChildIdx(left_idx, right_idx);
        }
    }

    // Internal temporary struct to organize K closest elements in a priorty queue
    struct VPTreeSearchElement {
        VPTreeSearchElement(int index, distance_type dist) : index(index), dist(dist) {}
        int index;
        distance_type dist;
        bool operator<(const VPTreeSearchElement &v) const { return dist < v.dist; }
    };

    void searchKNN(int32_t partitionIdx, const T &val, unsigned int k, std::priority_queue<VPTreeSearchElement> &knnQueue) {

        auto tau = std::numeric_limits<distance_type>::max();

        // stores the distance to the partition border at the time of the storage. Since tau value will change
        // whiling performing the DFS search from on level, the storage distance will be checked again when about
        // to dive into that partition. It might not be necessary to dig into the partition anymore if tau decreased.
        thread_local std::vector<std::tuple<distance_type, int32_t>> toSearch;
        toSearch.clear();
        toSearch.push_back({(distance_type)-1, partitionIdx});

        while (!toSearch.empty()) {
            auto [distToBorder, currentIdx] = toSearch.back();
            toSearch.pop_back();

            const VPLevelPartition<distance_type> &current = _nodePool[currentIdx];

            auto dist = distance(val, _examples[_indices[current.start()]]);
            if (dist < tau || knnQueue.size() < k) {

                if (knnQueue.size() == k) {
                    knnQueue.pop();
                }
                int64_t indexToAdd = _indices[current.start()];
                knnQueue.push(VPTreeSearchElement(indexToAdd, dist));

                tau = knnQueue.top().dist;
            }

            if (distToBorder >= 0 && distToBorder > tau && knnQueue.size() >= k) {

                // distance to this partition border changed and its not necessary to search within it anymore
                continue;
            }

            size_t neighborsSoFar = knnQueue.size();
            int32_t left_idx = current.left_idx();
            int32_t right_idx = current.right_idx();

            if (dist > current.radius()) {
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
                if (left_idx >= 0) {

                    size_t rightPartitionSize = (right_idx >= 0) ? _nodePool[right_idx].size() : 0;
                    bool notEnoughPointsOutside = rightPartitionSize < (k - neighborsSoFar);
                    auto toBorder = dist - current.radius();

                    // we might not have enough point outside to reject the inside partition, so we might need to search
                    // for both
                    if (notEnoughPointsOutside) {
                        toSearch.push_back({(distance_type)-1, left_idx});
                    } else if (knnQueue.size() < k || toBorder <= tau) {
                        toSearch.push_back({toBorder, left_idx});
                    }
                }

                // now schedule outside
                if (right_idx >= 0) {
                    toSearch.push_back({(distance_type)-1, right_idx});
                }
            } else {
                // must search inside
                // logic is analogous to the outside case

                if (right_idx >= 0) {

                    size_t leftPartitionSize = (left_idx >= 0) ? _nodePool[left_idx].size() : 0;
                    bool notEnoughPointsInside = leftPartitionSize < (k - neighborsSoFar);
                    auto toBorder = current.radius() - dist;

                    if (notEnoughPointsInside) {
                        toSearch.push_back({(distance_type)-1, right_idx});
                    } else if (knnQueue.size() < k || toBorder <= tau) {
                        toSearch.push_back({toBorder, right_idx});
                    }
                }

                // now schedule inside
                if (left_idx >= 0) {
                    toSearch.push_back({(distance_type)-1, left_idx});
                }
            }
        }
    }

    void search1NN(int32_t partitionIdx, const T &val, int64_t &resultIndex, distance_type &resultDist) {

        resultDist = std::numeric_limits<distance_type>::max();
        resultIndex = -1;

        thread_local std::vector<std::tuple<distance_type, int32_t>> toSearch;
        toSearch.clear();
        toSearch.push_back({(distance_type)-1, partitionIdx});

        while (!toSearch.empty()) {

            auto [distToBorder, currentIdx] = toSearch.back();
            toSearch.pop_back();

            const VPLevelPartition<distance_type> &current = _nodePool[currentIdx];

            auto dist = distance(val, _examples[_indices[current.start()]]);
            if (dist < resultDist) {
                resultDist = dist;
                resultIndex = _indices[current.start()];
            }

            if (distToBorder >= 0 && distToBorder > resultDist) {

                // distance to this partition border change and its not necessary to search within it anymore
                continue;
            }

            int32_t left_idx = current.left_idx();
            int32_t right_idx = current.right_idx();

            if (dist > current.radius()) {
                // may need to search inside as well
                auto toBorder = dist - current.radius();
                if (toBorder < resultDist && left_idx >= 0) {
                    toSearch.push_back({toBorder, left_idx});
                }

                // must search outside
                if (right_idx >= 0) {
                    toSearch.push_back({(distance_type)-1, right_idx});
                }
            } else {
                auto toBorder = current.radius() - dist;
                // may need to search outside as well
                if (toBorder < resultDist && right_idx >= 0) {
                    toSearch.push_back({toBorder, right_idx});
                }

                // must search inside
                if (left_idx >= 0) {
                    toSearch.push_back({(distance_type)-1, left_idx});
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

protected:
    std::vector<T> _examples;
    std::vector<int64_t> _indices;
    std::vector<VPLevelPartition<distance_type>> _nodePool;
    int32_t _rootIdx = -1;
    std::vector<float> _flat_backing;
    size_t _dim = 0;
};

} // namespace vptree
