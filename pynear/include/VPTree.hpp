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
#include <random>
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
        reorderForCache();
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
        reorderForCache();
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
    const std::vector<int32_t>& indexPermutation() const { return _indices; }
    const std::vector<VPLevelPartition<distance_type>>& partitionPool() const { return _nodePool; }
    int32_t rootPartitionIdx() const { return _rootIdx; }

    void initFromSerialized(std::vector<float> flat, size_t dim,
                            std::vector<int32_t> indices,
                            std::vector<VPLevelPartition<distance_type>> pool,
                            int32_t root_idx) {
        clear();
        _flat_backing = std::move(flat);
        _dim = dim;
        size_t n = (dim > 0) ? _flat_backing.size() / dim : 0;
        _examples.resize(n);
        if constexpr (std::is_same_v<T, FlatSpan>) {
            // After serialization, _flat_backing is already in tree-traversal order.
            // _examples[i] points directly to position i in the tree.
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
        std::vector<std::pair<distance_type, int32_t>> distPairs;
        distPairs.reserve(_examples.size());

        _nodePool.clear();
        _nodePool.reserve(_examples.size());
        _nodePool.push_back(VPLevelPartition<distance_type>(0, 0, (int32_t)_examples.size() - 1));
        _rootIdx = 0;
        _toSplit.push_back(_rootIdx);

        // Minimum partition size to justify OpenMP parallelism in distance loop
        constexpr int32_t MIN_PARALLEL_DIST = 4096;

        while (!_toSplit.empty()) {

            int32_t currentIdx = _toSplit.back();
            _toSplit.pop_back();

            int32_t start = _nodePool[currentIdx].start();
            int32_t end = _nodePool[currentIdx].end();

            if (start == end) {
                // stop dividing if there is only one point inside
                continue;
            }

            int32_t vpIndex = selectVantagePoint(start, end);

            // put vantage point as the first element within the examples list
            std::swap(_indices[vpIndex], _indices[start]);

            int32_t median = (end + start) / 2;

            // Pre-compute all distances from the VP once, storing (dist, example_idx) pairs.
            // nth_element then sorts the pairs by distance (O(1) comparisons, no hashing).
            // This reduces distance calls from O(n log n) to O(n) per build level.
            const int32_t range_size = end - start;
            const int32_t medianInPairs = median - start - 1;

            // When end == start+1 there is exactly one non-VP element and median == start,
            // giving medianInPairs == -1. No partitioning is needed: radius = 0 and the
            // single element goes straight to the right child.
            distance_type medianDistance = 0;
            if (medianInPairs >= 0) {
                distPairs.resize(range_size);
                const auto &vp = _examples[_indices[start]];

                // Parallelize the distance loop for large partitions (top few levels).
                // The _indices and distPairs ranges are disjoint across partitions,
                // so writes are race-free.
                if (range_size >= MIN_PARALLEL_DIST) {
                    #pragma omp parallel for schedule(static)
                    for (int32_t ci = 0; ci < range_size; ci++) {
                        const int32_t exIdx = _indices[start + 1 + ci];
                        distPairs[ci] = {distance(vp, _examples[exIdx]), exIdx};
                    }
                } else {
                    for (int32_t ci = 0; ci < range_size; ci++) {
                        const int32_t exIdx = _indices[start + 1 + ci];
                        distPairs[ci] = {distance(vp, _examples[exIdx]), exIdx};
                    }
                }

                std::nth_element(distPairs.begin(), distPairs.begin() + medianInPairs, distPairs.begin() + range_size,
                                 [](const auto &a, const auto &b) { return a.first < b.first; });

                // Write the partitioned example indices back to _indices
                for (int32_t ci = 0; ci < range_size; ci++)
                    _indices[start + 1 + ci] = distPairs[ci].second;

                medianDistance = distPairs[medianInPairs].first;
            }
            _nodePool[currentIdx].setRadius(medianDistance);

            // Schedule to build next levels
            // Left is every one within the median distance radius
            int32_t left_idx = -1, right_idx = -1;
            if (start + 1 <= median) {
                _nodePool.push_back(VPLevelPartition<distance_type>(0, start + 1, median));
                left_idx = (int32_t)_nodePool.size() - 1;
                _toSplit.push_back(left_idx);
            }

            if (median + 1 <= end) {
                _nodePool.push_back(VPLevelPartition<distance_type>(0, median + 1, end));
                right_idx = (int32_t)_nodePool.size() - 1;
                _toSplit.push_back(right_idx);
            }

            _nodePool[currentIdx].setChildIdx(left_idx, right_idx);
        }
    }

    /*
     * Reorder _flat_backing so that tree position i holds the data for _indices[i].
     * After this, _examples[i] can be accessed directly in search (no _indices indirection
     * for data lookups), improving cache locality during tree traversal.
     * _indices[i] retains the original row index for result reporting.
     *
     * Only applicable for FlatSpan (float32 data). No-op for other types.
     */
    void reorderForCache() {
        if constexpr (std::is_same_v<T, FlatSpan>) {
            size_t n = _examples.size();
            if (n == 0) return;

            std::vector<float> ordered(n * _dim);
            for (size_t i = 0; i < n; i++) {
                std::memcpy(ordered.data() + i * _dim,
                            _flat_backing.data() + (size_t)_indices[i] * _dim,
                            _dim * sizeof(float));
            }
            _flat_backing = std::move(ordered);

            // Fix up FlatSpan pointers into the new contiguous buffer
            for (size_t i = 0; i < n; i++)
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
            // _indices[i] still holds the original row index — used only for result reporting
        }
    }

    // Internal temporary struct to organize K closest elements in a priority queue
    struct VPTreeSearchElement {
        VPTreeSearchElement(int64_t index, distance_type dist) : index(index), dist(dist) {}
        int64_t index;
        distance_type dist;
        bool operator<(const VPTreeSearchElement &v) const { return dist < v.dist; }
    };

    /*
     * Best-first KNN search.
     *
     * Uses a min-heap sorted by distToBorder (lower bound on distance to any point
     * in the partition) so that tau shrinks as fast as possible.  Every decrease in
     * tau prunes more pending heap entries → dramatically fewer distance evaluations
     * than the original DFS traversal at moderate-to-large N.
     *
     * For FlatSpan data (after reorderForCache), data is accessed directly as
     * _examples[pos] (sequential memory) — no _indices lookup for data.
     */
    void searchKNN(int32_t partitionIdx, const T &val, size_t k,
                   std::priority_queue<VPTreeSearchElement> &knnQueue) {

        auto tau = std::numeric_limits<distance_type>::max();

        // Thread-local backing vector — reused across calls (avoids per-query allocation)
        thread_local std::vector<std::pair<distance_type, int32_t>> tl_heap;
        tl_heap.clear();

        static constexpr auto heap_cmp = std::greater<std::pair<distance_type, int32_t>>{};

        tl_heap.push_back({(distance_type)0, partitionIdx});
        // single-element "heap" is trivially valid

        while (!tl_heap.empty()) {
            std::pop_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
            auto [distToBorder, currentIdx] = tl_heap.back();
            tl_heap.pop_back();

            // Prune: lower bound on this partition > current search radius
            if (distToBorder > tau && knnQueue.size() >= k) continue;

            const VPLevelPartition<distance_type> &current = _nodePool[currentIdx];

            // Access point data — for FlatSpan, _examples[pos] is direct after reorderForCache()
            distance_type dist;
            if constexpr (std::is_same_v<T, FlatSpan>) {
                dist = distance(val, _examples[current.start()]);
            } else {
                dist = distance(val, _examples[_indices[current.start()]]);
            }

            if (dist < tau || knnQueue.size() < k) {
                if (knnQueue.size() == k) knnQueue.pop();
                knnQueue.push(VPTreeSearchElement((int64_t)_indices[current.start()], dist));
                tau = knnQueue.top().dist;
            }

            int32_t left_idx  = current.left_idx();
            int32_t right_idx = current.right_idx();

            if (dist > current.radius()) {
                // Mandatory: right (outside sphere)
                if (right_idx >= 0) {
                    tl_heap.push_back({(distance_type)0, right_idx});
                    std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                }
                // Optional: left (inside sphere) — lower bound = dist - radius
                if (left_idx >= 0) {
                    auto toBorder = dist - current.radius();
                    if (knnQueue.size() < k || toBorder <= tau) {
                        tl_heap.push_back({toBorder, left_idx});
                        std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                    }
                }
            } else {
                // Mandatory: left (inside sphere)
                if (left_idx >= 0) {
                    tl_heap.push_back({(distance_type)0, left_idx});
                    std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                }
                // Optional: right (outside sphere) — lower bound = radius - dist
                if (right_idx >= 0) {
                    auto toBorder = current.radius() - dist;
                    if (knnQueue.size() < k || toBorder <= tau) {
                        tl_heap.push_back({toBorder, right_idx});
                        std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                    }
                }
            }
        }
    }

    void search1NN(int32_t partitionIdx, const T &val, int64_t &resultIndex, distance_type &resultDist) {

        resultDist  = std::numeric_limits<distance_type>::max();
        resultIndex = -1;

        thread_local std::vector<std::pair<distance_type, int32_t>> tl_heap;
        tl_heap.clear();

        static constexpr auto heap_cmp = std::greater<std::pair<distance_type, int32_t>>{};

        tl_heap.push_back({(distance_type)0, partitionIdx});

        while (!tl_heap.empty()) {
            std::pop_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
            auto [distToBorder, currentIdx] = tl_heap.back();
            tl_heap.pop_back();

            if (distToBorder > resultDist) continue;

            const VPLevelPartition<distance_type> &current = _nodePool[currentIdx];

            distance_type dist;
            if constexpr (std::is_same_v<T, FlatSpan>) {
                dist = distance(val, _examples[current.start()]);
            } else {
                dist = distance(val, _examples[_indices[current.start()]]);
            }

            if (dist < resultDist) {
                resultDist  = dist;
                resultIndex = (int64_t)_indices[current.start()];
            }

            int32_t left_idx  = current.left_idx();
            int32_t right_idx = current.right_idx();

            if (dist > current.radius()) {
                // Must search outside (right)
                if (right_idx >= 0) {
                    tl_heap.push_back({(distance_type)0, right_idx});
                    std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                }
                // May search inside (left)
                if (left_idx >= 0) {
                    auto toBorder = dist - current.radius();
                    if (toBorder < resultDist) {
                        tl_heap.push_back({toBorder, left_idx});
                        std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                    }
                }
            } else {
                // Must search inside (left)
                if (left_idx >= 0) {
                    tl_heap.push_back({(distance_type)0, left_idx});
                    std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                }
                // May search outside (right)
                if (right_idx >= 0) {
                    auto toBorder = current.radius() - dist;
                    if (toBorder < resultDist) {
                        tl_heap.push_back({toBorder, right_idx});
                        std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                    }
                }
            }
        }
    }

    /*
     * Sample S candidate vantage points and select the one with the highest
     * variance of distances to a random probe set.  Higher variance → a more
     * balanced median split → shallower tree → better search pruning.
     *
     * Cost: O(S²) distance evaluations per partition level — negligible compared
     * to the O(N) distance loop in build().
     */
    int32_t selectVantagePoint(int32_t fromIndex, int32_t toIndex) {
        int32_t range = (toIndex - fromIndex) + 1;
        if (range <= 2) return fromIndex;

        constexpr int S = 5;
        int nSample = std::min(range, (int32_t)S);

        int32_t best_vp     = fromIndex;
        float   best_spread = -1.f;

        std::uniform_int_distribution<int32_t> uni(fromIndex, toIndex);

        for (int s = 0; s < nSample; ++s) {
            int32_t cand_pos = uni(_rng);
            const auto &cand = _examples[_indices[cand_pos]];

            float sum = 0.f, sum2 = 0.f;
            for (int p = 0; p < nSample; ++p) {
                int32_t probe_pos = uni(_rng);
                float d = (float)distance(cand, _examples[_indices[probe_pos]]);
                sum += d; sum2 += d * d;
            }
            float var = sum2 / nSample - (sum / nSample) * (sum / nSample);
            if (var > best_spread) {
                best_spread = var;
                best_vp = cand_pos;
            }
        }
        return best_vp;
    }

    // Fill result element from search element internal structure
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
    std::vector<int32_t> _indices;   // tree-position → original row index (for result reporting)
    std::vector<VPLevelPartition<distance_type>> _nodePool;
    int32_t _rootIdx = -1;
    std::vector<float> _flat_backing;
    size_t _dim = 0;
    std::mt19937 _rng{42};
};

} // namespace vptree
