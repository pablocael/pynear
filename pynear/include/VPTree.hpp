/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <execution>
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
            // Thread-local vector-backed max-heap — reused across queries (no per-query allocation).
            // std::priority_queue is specified as push_back+push_heap / pop_heap+pop_back over its
            // container, so this preserves the exact admission/ordering semantics.
            thread_local std::vector<VPTreeSearchElement> tl_knnHeap;
            tl_knnHeap.clear();
            tl_knnHeap.reserve(std::min<size_t>(_examples.size(), k));
            searchKNN(_rootIdx, query, k, tl_knnHeap);

            // we must always return k elements for each search unless there is no k elements
            assert(tl_knnHeap.size() == std::min<size_t>(_examples.size(), k));

            fillSearchResult(tl_knnHeap, results[i]);
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
    // Leaf bucketing: stop splitting once a node's range holds this many points
    // or fewer. Such a node keeps its full [start, end] range, gets no children,
    // and its radius stays 0 / vantage point unused (search never reads either
    // for childless nodes — it scans the whole range linearly instead).
    // Trees deserialized from older pickles simply have single-point leaves,
    // which the same leaf-scan path handles (range size 1).
    static constexpr int32_t kLeafSize = 32;

    /*
     *  Builds a Vantage Point tree using a level-parallel BFS strategy.
     *
     *  All nodes at the same tree depth are processed concurrently:
     *  - Each worker thread owns its distPairs scratch buffer (thread_local)
     *    so nth_element calls across sibling partitions run in parallel.
     *  - Child nodePool slots are pre-assigned via prefix-sum before the
     *    parallel region, so _nodePool is never mutated during parallel work.
     *  - _indices ranges are disjoint per partition → no write races.
     *  - selectVantagePoint uses a thread_local RNG → race-free.
     */
    void build(const std::vector<T> &array) {

        const int32_t n = (int32_t)_examples.size();
        if (n == 0) return;

        _indices.resize(n);
        std::iota(_indices.begin(), _indices.end(), 0);

        _nodePool.clear();
        _nodePool.reserve(n);
        _nodePool.push_back(VPLevelPartition<distance_type>(0, 0, n - 1));
        _rootIdx = 0;

        struct WorkItem { int32_t nodeIdx, start, end; };
        std::vector<WorkItem> current;
        current.push_back({0, 0, n - 1});

        while (!current.empty()) {
            const int32_t nItems = (int32_t)current.size();

            // --- Pre-assign child nodePool slots (serial, O(nItems)) ---
            // Ranges are determined by (start,end) alone — independent of
            // nth_element outcome — so we can assign before the parallel work.
            std::vector<int32_t> leftSlot(nItems, -1), rightSlot(nItems, -1);
            {
                int32_t nextSlot = (int32_t)_nodePool.size();
                for (int32_t i = 0; i < nItems; i++) {
                    const int32_t s = current[i].start, e = current[i].end;
                    if (e - s + 1 <= kLeafSize) continue; // leaf bucket: no children
                    const int32_t median = (s + e) / 2;
                    if (s + 1   <= median) leftSlot[i]  = nextSlot++;
                    if (median + 1 <= e)   rightSlot[i] = nextSlot++;
                }
                _nodePool.resize(nextSlot); // default-init; written in parallel below
            }

            // Pre-build the next-level work list (ranges known before nth_element).
            std::vector<WorkItem> next;
            next.reserve(2 * nItems);
            for (int32_t i = 0; i < nItems; i++) {
                const int32_t s = current[i].start, e = current[i].end;
                if (e - s + 1 <= kLeafSize) continue; // leaf bucket: no children
                const int32_t median = (s + e) / 2;
                if (leftSlot[i]  >= 0) next.push_back({leftSlot[i],  s + 1,      median});
                if (rightSlot[i] >= 0) next.push_back({rightSlot[i], median + 1, e});
            }

            // --- Process all items at this level in parallel ---
            // Each item writes to a disjoint slice of _indices and to
            // pre-assigned, non-overlapping slots in _nodePool → no races.
#if ENABLE_OMP_PARALLEL
            #pragma omp parallel for schedule(dynamic) if(nItems > 1)
#endif
            for (int32_t i = 0; i < nItems; i++) {
                const int32_t nodeIdx = current[i].nodeIdx;
                const int32_t start   = current[i].start;
                const int32_t end_    = current[i].end;

                // Leaf bucket: keep the whole [start, end_] range, no vantage
                // point, radius 0, no children (defaults from construction).
                if (end_ - start + 1 <= kLeafSize) continue;

                const int32_t vpIndex = selectVantagePoint(start, end_);
                std::swap(_indices[vpIndex], _indices[start]);

                const int32_t median        = (start + end_) / 2;
                const int32_t range_size    = end_ - start;
                const int32_t medianInPairs = median - start - 1;

                distance_type medianDistance = 0;
                if (medianInPairs >= 0) {
                    const auto &vp = _examples[_indices[start]];

#if ENABLE_OMP_PARALLEL && USE_PSTL_NTH_ELEMENT
                    // Linux only: TBB provides std::execution::par_unseq.
                    // At level 0 (not yet inside a parallel region) and for large
                    // partitions, use a local shared buffer so OMP threads can all
                    // write into it, then use TBB par_unseq for nth_element.
                    // For inner nodes (already inside OMP parallel for), use a
                    // thread_local buffer — each thread processes its own partition
                    // entirely, so no sharing is needed.
                    constexpr int32_t PAR_THRESHOLD = 4096;
                    if (!omp_in_parallel() && range_size >= PAR_THRESHOLD) {
                        std::vector<std::pair<distance_type, int32_t>> distPairs(range_size);

                        #pragma omp parallel for schedule(static)
                        for (int32_t ci = 0; ci < range_size; ci++) {
                            const int32_t exIdx = _indices[start + 1 + ci];
                            distPairs[ci] = {distance(vp, _examples[exIdx]), exIdx};
                        }

                        std::nth_element(std::execution::par_unseq,
                                         distPairs.begin(),
                                         distPairs.begin() + medianInPairs,
                                         distPairs.begin() + range_size,
                                         [](const auto &a, const auto &b) { return a.first < b.first; });

                        for (int32_t ci = 0; ci < range_size; ci++)
                            _indices[start + 1 + ci] = distPairs[ci].second;

                        medianDistance = distPairs[medianInPairs].first;
                    } else
#endif
                    {
                        // Thread-local scratch: each thread owns its own full partition,
                        // so there are no write conflicts between concurrent threads.
                        thread_local std::vector<std::pair<distance_type, int32_t>> tl_distPairs;
                        tl_distPairs.resize(range_size);

                        for (int32_t ci = 0; ci < range_size; ci++) {
                            const int32_t exIdx = _indices[start + 1 + ci];
                            tl_distPairs[ci] = {distance(vp, _examples[exIdx]), exIdx};
                        }

                        std::nth_element(tl_distPairs.begin(),
                                         tl_distPairs.begin() + medianInPairs,
                                         tl_distPairs.begin() + range_size,
                                         [](const auto &a, const auto &b) { return a.first < b.first; });

                        for (int32_t ci = 0; ci < range_size; ci++)
                            _indices[start + 1 + ci] = tl_distPairs[ci].second;

                        medianDistance = tl_distPairs[medianInPairs].first;
                    }
                }

                _nodePool[nodeIdx].setRadius(medianDistance);
                if (leftSlot[i]  >= 0)
                    _nodePool[leftSlot[i]]  = VPLevelPartition<distance_type>(0, start + 1, median);
                if (rightSlot[i] >= 0)
                    _nodePool[rightSlot[i]] = VPLevelPartition<distance_type>(0, median + 1, end_);
                _nodePool[nodeIdx].setChildIdx(leftSlot[i], rightSlot[i]);
            }

            current = std::move(next);
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
                   std::vector<VPTreeSearchElement> &knnHeap) {

        auto tau = std::numeric_limits<distance_type>::max();

        // Thread-local backing vector — reused across calls (avoids per-query allocation)
        thread_local std::vector<std::pair<distance_type, int32_t>> tl_heap;
        tl_heap.clear();

        static constexpr auto heap_cmp = std::greater<std::pair<distance_type, int32_t>>{};

        // Descend directly into the mandatory child (border bound 0) instead of
        // push_heap+pop_heap cycling it through the frontier heap each level.
        // A (0, idx) entry can only lose the pop against another 0-bound entry with a
        // smaller partition index (pairs compare lexicographically), so we descend
        // directly unless the heap minimum beats (0, mandatoryIdx) — this replicates
        // the exact pop sequence (and hence distance-evaluation/tau-update order,
        // including tie behavior) of the previous implementation.
        int32_t currentIdx = partitionIdx;

        for (;;) {
            const VPLevelPartition<distance_type> &current = _nodePool[currentIdx];

            if (current.left_idx() < 0 && current.right_idx() < 0) {
                // Leaf bucket: childless node — scan every point in [start, end]
                // linearly, feeding the k-NN heap with the same admission logic as
                // individual points. After reorderForCache()/initFromSerialized()
                // the FlatSpan rows are contiguous, so this is a sequential SIMD
                // sweep. Handles any leaf range size, including the single-point
                // leaves of trees deserialized from older versions.
                const int32_t leafEnd = current.end();
                for (int32_t pos = current.start(); pos <= leafEnd; ++pos) {
                    distance_type dist;
                    if constexpr (std::is_same_v<T, FlatSpan>) {
                        dist = distance(val, FlatSpan{_flat_backing.data() + (size_t)pos * _dim, _dim});
                    } else {
                        dist = distance(val, _examples[_indices[pos]]);
                    }

                    if (dist < tau || knnHeap.size() < k) {
                        if (knnHeap.size() == k) {
                            std::pop_heap(knnHeap.begin(), knnHeap.end());
                            knnHeap.pop_back();
                        }
                        knnHeap.emplace_back((int64_t)_indices[pos], dist);
                        std::push_heap(knnHeap.begin(), knnHeap.end());
                        tau = knnHeap.front().dist;
                    }
                }
            } else {
                // Internal node: evaluate the vantage point at start(), then descend.
                // Access point data — for FlatSpan, after reorderForCache()/initFromSerialized()
                // tree position i always maps to _flat_backing[i*_dim .. ), so build the span
                // inline (avoids the dependent load through the _examples array).
                distance_type dist;
                if constexpr (std::is_same_v<T, FlatSpan>) {
                    dist = distance(val, FlatSpan{_flat_backing.data() + (size_t)current.start() * _dim, _dim});
                } else {
                    dist = distance(val, _examples[_indices[current.start()]]);
                }

                if (dist < tau || knnHeap.size() < k) {
                    if (knnHeap.size() == k) {
                        std::pop_heap(knnHeap.begin(), knnHeap.end());
                        knnHeap.pop_back();
                    }
                    knnHeap.emplace_back((int64_t)_indices[current.start()], dist);
                    std::push_heap(knnHeap.begin(), knnHeap.end());
                    tau = knnHeap.front().dist;
                }

                int32_t mandatoryIdx, optionalIdx;
                distance_type toBorder;
                if (dist > current.radius()) {
                    // Mandatory: right (outside sphere); optional: left — bound = dist - radius
                    mandatoryIdx = current.right_idx();
                    optionalIdx  = current.left_idx();
                    toBorder     = dist - current.radius();
                } else {
                    // Mandatory: left (inside sphere); optional: right — bound = radius - dist
                    mandatoryIdx = current.left_idx();
                    optionalIdx  = current.right_idx();
                    toBorder     = current.radius() - dist;
                }

                if (optionalIdx >= 0 && (knnHeap.size() < k || toBorder <= tau)) {
                    tl_heap.emplace_back(toBorder, optionalIdx);
                    std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                }

                if (mandatoryIdx >= 0) {
                    if (tl_heap.empty() ||
                        std::pair<distance_type, int32_t>((distance_type)0, mandatoryIdx) < tl_heap.front()) {
                        // A 0-bound entry is never pruned (tau >= 0), so no prune check needed.
                        currentIdx = mandatoryIdx;
                        continue;
                    }
                    tl_heap.emplace_back((distance_type)0, mandatoryIdx);
                    std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                }
            }

            // Pull the next non-pruned partition from the frontier heap
            bool found = false;
            while (!tl_heap.empty()) {
                std::pop_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                auto [distToBorder, nextIdx] = tl_heap.back();
                tl_heap.pop_back();

                // Prune: lower bound on this partition > current search radius
                if (distToBorder > tau && knnHeap.size() >= k) continue;

                currentIdx = nextIdx;
                found = true;
                break;
            }
            if (!found) break;
        }
    }

    void search1NN(int32_t partitionIdx, const T &val, int64_t &resultIndex, distance_type &resultDist) {

        resultDist  = std::numeric_limits<distance_type>::max();
        resultIndex = -1;

        thread_local std::vector<std::pair<distance_type, int32_t>> tl_heap;
        tl_heap.clear();

        static constexpr auto heap_cmp = std::greater<std::pair<distance_type, int32_t>>{};

        // Same direct-descent strategy as searchKNN — see comment there.
        int32_t currentIdx = partitionIdx;

        for (;;) {
            const VPLevelPartition<distance_type> &current = _nodePool[currentIdx];

            if (current.left_idx() < 0 && current.right_idx() < 0) {
                // Leaf bucket: childless node — scan every point in [start, end]
                // linearly (see searchKNN for details; also covers single-point
                // leaves of trees deserialized from older versions).
                const int32_t leafEnd = current.end();
                for (int32_t pos = current.start(); pos <= leafEnd; ++pos) {
                    distance_type dist;
                    if constexpr (std::is_same_v<T, FlatSpan>) {
                        dist = distance(val, FlatSpan{_flat_backing.data() + (size_t)pos * _dim, _dim});
                    } else {
                        dist = distance(val, _examples[_indices[pos]]);
                    }

                    if (dist < resultDist) {
                        resultDist  = dist;
                        resultIndex = (int64_t)_indices[pos];
                    }
                }
            } else {
                distance_type dist;
                if constexpr (std::is_same_v<T, FlatSpan>) {
                    dist = distance(val, FlatSpan{_flat_backing.data() + (size_t)current.start() * _dim, _dim});
                } else {
                    dist = distance(val, _examples[_indices[current.start()]]);
                }

                if (dist < resultDist) {
                    resultDist  = dist;
                    resultIndex = (int64_t)_indices[current.start()];
                }

                int32_t mandatoryIdx, optionalIdx;
                distance_type toBorder;
                if (dist > current.radius()) {
                    // Must search outside (right); may search inside (left)
                    mandatoryIdx = current.right_idx();
                    optionalIdx  = current.left_idx();
                    toBorder     = dist - current.radius();
                } else {
                    // Must search inside (left); may search outside (right)
                    mandatoryIdx = current.left_idx();
                    optionalIdx  = current.right_idx();
                    toBorder     = current.radius() - dist;
                }

                if (optionalIdx >= 0 && toBorder < resultDist) {
                    tl_heap.emplace_back(toBorder, optionalIdx);
                    std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                }

                if (mandatoryIdx >= 0) {
                    if (tl_heap.empty() ||
                        std::pair<distance_type, int32_t>((distance_type)0, mandatoryIdx) < tl_heap.front()) {
                        // 0-bound entries are never pruned (resultDist >= 0), descend directly.
                        currentIdx = mandatoryIdx;
                        continue;
                    }
                    tl_heap.emplace_back((distance_type)0, mandatoryIdx);
                    std::push_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                }
            }

            bool found = false;
            while (!tl_heap.empty()) {
                std::pop_heap(tl_heap.begin(), tl_heap.end(), heap_cmp);
                auto [distToBorder, nextIdx] = tl_heap.back();
                tl_heap.pop_back();

                if (distToBorder > resultDist) continue;

                currentIdx = nextIdx;
                found = true;
                break;
            }
            if (!found) break;
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
        // For small partitions the ~S*S sampled distance evaluations cost more than
        // any split-quality benefit they buy — just take the first element.
        if (range < 32) return fromIndex;

        constexpr int S = 5;
        int nSample = std::min(range, (int32_t)S);

        int32_t best_vp     = fromIndex;
        float   best_spread = -1.f;

        // Thread-local RNG so selectVantagePoint is safe in parallel build.
        thread_local std::mt19937 tl_rng{std::random_device{}()};
        std::uniform_int_distribution<int32_t> uni(fromIndex, toIndex);

        for (int s = 0; s < nSample; ++s) {
            int32_t cand_pos = uni(tl_rng);
            const auto &cand = _examples[_indices[cand_pos]];

            float sum = 0.f, sum2 = 0.f;
            int   nProbes = 0;
            for (int p = 0; p < nSample; ++p) {
                int32_t probe_pos = uni(tl_rng);
                if (probe_pos == cand_pos) continue; // skip self-probe (d == 0 would bias variance)
                float d = (float)distance(cand, _examples[_indices[probe_pos]]);
                sum += d; sum2 += d * d;
                ++nProbes;
            }
            if (nProbes == 0) continue;
            float var = sum2 / nProbes - (sum / nProbes) * (sum / nProbes);
            if (var > best_spread) {
                best_spread = var;
                best_vp = cand_pos;
            }
        }
        return best_vp;
    }

    // Fill result element from the vector-backed max-heap of search elements
    // After a call to that function, knnHeap gets invalidated (emptied)!
    // Emits elements in the same (descending-distance) order as the previous
    // std::priority_queue-based implementation.
    void fillSearchResult(std::vector<VPTreeSearchElement> &knnHeap, VPTreeSearchResultElement &element) {
        element.distances.reserve(knnHeap.size());
        element.indexes.reserve(knnHeap.size());

        while (!knnHeap.empty()) {
            std::pop_heap(knnHeap.begin(), knnHeap.end());
            const VPTreeSearchElement &top = knnHeap.back();
            element.distances.push_back(top.dist);
            element.indexes.push_back(top.index);
            knnHeap.pop_back();
        }
    }

protected:
    std::vector<T> _examples;
    std::vector<int32_t> _indices;   // tree-position → original row index (for result reporting)
    std::vector<VPLevelPartition<distance_type>> _nodePool;
    int32_t _rootIdx = -1;
    std::vector<float> _flat_backing;
    size_t _dim = 0;
};

} // namespace vptree
