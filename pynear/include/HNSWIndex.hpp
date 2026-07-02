// HNSWIndex — Hierarchical Navigable Small World graph for ANN search.
//
// Paper:    Malkov & Yashunin 2016, "Efficient and robust approximate nearest
//           neighbor search using Hierarchical Navigable Small World graphs",
//           arXiv:1603.09320.
// Mirrors:  the in-memory layout of hnswlib (github.com/nmslib/hnswlib) —
//           flat per-node adjacency, geometric layer assignment, α-heuristic
//           neighbour selection. Single-threaded build/search in v1.
//
// Distance is plugged in via the same function-pointer template parameter
// pynear's VPTree uses, so all the existing SIMD kernels (dist_l2_f_avx2,
// dist_l1_f_avx2, ...) work without modification.

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
#endif

#include "DistanceFunctions.hpp"

namespace hnsw {

// Pair sortable by distance (min-heap by .first if used directly in priority_queue<,,greater>)
template <typename distT> struct DistNode {
    distT distance;
    int32_t node_id;
};

template <typename distT> struct DistNodeMin {
    bool operator()(const DistNode<distT>& a, const DistNode<distT>& b) const {
        return a.distance > b.distance;  // priority_queue is a max-heap, so > gives a min-heap
    }
};

template <typename distT> struct DistNodeMax {
    bool operator()(const DistNode<distT>& a, const DistNode<distT>& b) const {
        return a.distance < b.distance;  // max-heap by distance
    }
};

// Thread-safe pool of versioned visited-stamp arrays (hnswlib pattern).
// A search checks a list out (RAII VisitedGuard below), bumps its epoch,
// stamps nodes it visits, and returns the list on scope exit. The pool
// grows on demand, so it works for any number of concurrent searches —
// including searches running on non-OpenMP threads — and is just a
// vector pop/push per search when uncontended.
class VisitedListPool {
public:
    struct VisitedList {
        std::vector<uint32_t> stamps;
        uint32_t version = 0;
        // Start a fresh visit epoch and make sure `stamps` covers n nodes.
        // Same versioning scheme as before: a node is "visited" iff its
        // stamp equals the current epoch; on wrap-around we zero the array.
        uint32_t advance(size_t n) {
            if (stamps.size() < n) stamps.resize(n, 0u);
            ++version;
            if (version == 0) {
                std::fill(stamps.begin(), stamps.end(), 0u);
                version = 1;
            }
            return version;
        }
    };

    std::unique_ptr<VisitedList> acquire() {
        {
            std::lock_guard<std::mutex> lock(_mtx);
            if (!_free.empty()) {
                std::unique_ptr<VisitedList> vl = std::move(_free.back());
                _free.pop_back();
                return vl;
            }
        }
        return std::make_unique<VisitedList>();
    }

    void release(std::unique_ptr<VisitedList> vl) {
        std::lock_guard<std::mutex> lock(_mtx);
        _free.push_back(std::move(vl));
    }

private:
    std::mutex _mtx;
    std::vector<std::unique_ptr<VisitedList>> _free;
};

// RAII checkout of a VisitedList from a VisitedListPool.
class VisitedGuard {
public:
    explicit VisitedGuard(VisitedListPool& pool) : _pool(&pool), _vl(pool.acquire()) {}
    ~VisitedGuard() { _pool->release(std::move(_vl)); }
    VisitedGuard(const VisitedGuard&) = delete;
    VisitedGuard& operator=(const VisitedGuard&) = delete;
    VisitedListPool::VisitedList& operator*() { return *_vl; }

private:
    VisitedListPool* _pool;
    std::unique_ptr<VisitedListPool::VisitedList> _vl;
};

template <typename T, typename distT, distT (*distance_fn)(const T&, const T&)>
class HNSWIndex {
public:
    HNSWIndex(size_t M = 16, size_t ef_construction = 200, size_t ef_search = 50,
              uint64_t seed = 42, int n_threads = 1)
        : _M(M),
          _M_max0(2 * M),
          _ef_construction(ef_construction),
          _ef_search(ef_search),
          _mL(1.0 / std::log(static_cast<double>(M))),
          _entry_point(-1),
          _top_level(-1),
          _dim(0),
          _seed_used(seed),
          _n_threads(n_threads) {
        if (M < 2) throw std::invalid_argument("M must be >= 2");
        if (ef_construction < 1) throw std::invalid_argument("ef_construction must be >= 1");
    }

    // Profiling: total distance calls since last reset. Useful for comparing
    // beam-search efficiency against Faiss's hnsw_stats.ndis.
    //
    // Wrapped in a copyable relaxed-atomic so parallel batch queries can
    // bump it without a data race — a bare std::atomic member would delete
    // the index's move constructor (adapters are returned by value for
    // pybind11 pickle).
    struct RelaxedCounter {
        std::atomic<uint64_t> value{0};
        RelaxedCounter() = default;
        RelaxedCounter(const RelaxedCounter& o) noexcept
            : value(o.value.load(std::memory_order_relaxed)) {}
        RelaxedCounter& operator=(const RelaxedCounter& o) noexcept {
            value.store(o.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
        void operator+=(uint64_t n) noexcept { value.fetch_add(n, std::memory_order_relaxed); }
    };
    mutable RelaxedCounter _dist_calls;
    uint64_t dist_calls() const { return _dist_calls.value.load(std::memory_order_relaxed); }
    void reset_dist_calls() { _dist_calls.value.store(0, std::memory_order_relaxed); }

    void set_ef(size_t ef_search) { _ef_search = ef_search; }
    size_t ef_search() const { return _ef_search; }
    size_t size() const { return _examples.size(); }
    size_t dim() const { return _dim; }

    // Build the index from a batch of input vectors (rebuilds from scratch).
    void set(const std::vector<T>& data) {
        clear();
        if (data.empty()) return;

        // For FlatSpan-backed input, copy all vectors into _flat_backing and
        // rebuild _examples to point into our owned storage.
        if constexpr (std::is_same_v<T, FlatSpan>) {
            _dim = data[0].sz;
            size_t n = data.size();
            _flat_backing.resize(n * _dim);
            for (size_t i = 0; i < n; i++) {
                std::memcpy(_flat_backing.data() + i * _dim,
                            data[i].ptr,
                            _dim * sizeof(float));
            }
            _examples.resize(n);
            for (size_t i = 0; i < n; i++) {
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
            }
        } else if constexpr (std::is_same_v<T, SQ8Span>) {
            // SQ8Span: copy the int8 data into our own backing so the
            // index is self-contained (caller can free its buffer).
            _dim = data[0].sz;
            size_t n = data.size();
            _sq8_backing.resize(n * _dim);
            for (size_t i = 0; i < n; i++) {
                std::memcpy(_sq8_backing.data() + i * _dim, data[i].ptr, _dim);
            }
            _examples.resize(n);
            for (size_t i = 0; i < n; i++) {
                _examples[i] = SQ8Span{_sq8_backing.data() + i * _dim, _dim};
            }
        } else {
            _examples = data;
            _dim = (data.empty() ? 0 : data[0].size());
        }

        _levels.assign(_examples.size(), 0);
        _adjacency.assign(_examples.size(), std::vector<std::vector<int32_t>>{});
        _layer0_adj.assign(_examples.size() * _M_max0, -1);
        _layer0_count.assign(_examples.size(), 0);
        _deleted.assign(_examples.size(), 0u);
        _num_deleted = 0;
        if constexpr (std::is_same_v<T, FlatSpan>) {
            // Precompute ||v_i||² for the dot-product distance trick.
            _norms_sq.assign(_examples.size(), 0.0f);
            for (size_t i = 0; i < _examples.size(); i++) {
                _norms_sq[i] = vec_l2sq_avx2(_examples[i].ptr, _examples[i].sz);
            }
        }
        init_thread_resources(_examples.size());

        if (_n_threads <= 1) {
            // Sequential build — fully deterministic given the seed.
            // _during_build stays false so read_neighbours() reads the
            // adjacency in place, with no locking or copying.
            for (size_t i = 0; i < _examples.size(); i++) {
                add_point(static_cast<int32_t>(i));
            }
        } else {
            // Parallel build. Per-node shared_mutexes protect adjacency
            // modifications; reads in search_layer take shared locks via
            // read_neighbours(). Graph topology is non-deterministic across
            // runs but search quality is comparable to the sequential build.
            _during_build = true;
#pragma omp parallel for schedule(dynamic, 64) num_threads(_n_threads)
            for (int64_t i = 0; i < (int64_t)_examples.size(); i++) {
                add_point(static_cast<int32_t>(i));
            }
            _during_build = false;
        }
    }

    // ─── Incremental mutation API ──────────────────────────────────────────
    // add() / remove() / rebuild() are single-threaded — callers must not
    // invoke them concurrently with searches or each other.

    // Append vectors to an existing index. Returns the new node IDs.
    // For the very first call (empty index), behaves like set(). Each new
    // point is woven into the graph via add_point() so subsequent searches
    // can reach it.
    std::vector<int32_t> add(const std::vector<T>& data) {
        if (data.empty()) return {};

        size_t old_n;
        const bool was_empty = _examples.empty();
        if (was_empty) {
            // Use set() to seed dim, backings, thread resources etc.
            // Then return the trivial id mapping.
            set(data);
            std::vector<int32_t> ids(data.size());
            for (size_t i = 0; i < data.size(); i++) ids[i] = static_cast<int32_t>(i);
            return ids;
        }

        old_n = _examples.size();
        const size_t added = data.size();
        const size_t new_n = old_n + added;

        // Extend storage in lockstep with set()'s allocation shape.
        if constexpr (std::is_same_v<T, FlatSpan>) {
            const float* old_base = _flat_backing.data();
            _flat_backing.resize(new_n * _dim);
            for (size_t i = 0; i < added; i++) {
                std::memcpy(_flat_backing.data() + (old_n + i) * _dim,
                            data[i].ptr, _dim * sizeof(float));
            }
            _examples.resize(new_n);
            // Repoint spans only if _flat_backing actually reallocated;
            // otherwise only the freshly appended rows need pointers.
            const size_t repoint_from = (_flat_backing.data() == old_base) ? old_n : 0;
            for (size_t i = repoint_from; i < new_n; i++) {
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
            }
            _norms_sq.resize(new_n);
            for (size_t i = 0; i < added; i++) {
                _norms_sq[old_n + i] = vec_l2sq_avx2(_examples[old_n + i].ptr, _dim);
            }
        } else if constexpr (std::is_same_v<T, SQ8Span>) {
            const int8_t* old_base = _sq8_backing.data();
            _sq8_backing.resize(new_n * _dim);
            for (size_t i = 0; i < added; i++) {
                std::memcpy(_sq8_backing.data() + (old_n + i) * _dim, data[i].ptr, _dim);
            }
            _examples.resize(new_n);
            const size_t repoint_from = (_sq8_backing.data() == old_base) ? old_n : 0;
            for (size_t i = repoint_from; i < new_n; i++) {
                _examples[i] = SQ8Span{_sq8_backing.data() + i * _dim, _dim};
            }
        } else {
            // Per-row T (e.g. arrayli) — owns its own data; just append.
            _examples.reserve(new_n);
            for (size_t i = 0; i < added; i++) _examples.push_back(data[i]);
        }

        _levels.resize(new_n, 0);
        _adjacency.resize(new_n);
        _layer0_adj.resize(new_n * _M_max0, -1);
        _layer0_count.resize(new_n, 0);
        _deleted.resize(new_n, 0u);
        // Visited-stamp arrays live in the pool and grow lazily on checkout
        // (VisitedList::advance), so nothing to resize here.
        //
        // Grow the per-node lock array geometrically. std::shared_mutex is
        // not movable so growth means re-creating the whole array — doing
        // that only on capacity exhaustion (doubling) keeps add() amortised
        // O(added) instead of O(N) per call. Locks are only ever taken
        // during a build/weave; add() is the single mutator here and
        // concurrent searches are disallowed (caller's responsibility), so
        // the realloc is safe.
        if (new_n > _num_locks) {
            size_t new_cap = std::max(new_n, _num_locks * 2);
            _node_locks = std::unique_ptr<std::shared_mutex[]>(new std::shared_mutex[new_cap]);
            _num_locks = new_cap;
        }

        if (_n_threads <= 1) {
            // Sequential weave — _during_build stays false so
            // read_neighbours() reads the adjacency in place, lock-free.
            for (size_t i = 0; i < added; i++) {
                add_point(static_cast<int32_t>(old_n + i));
            }
        } else {
            // Parallel weave just like set(). Each add_point() takes
            // per-node shared_mutex locks via read_neighbours() / its
            // own write paths, so multiple inserters don't corrupt each
            // other's adjacency lists. The order in which new ids get
            // inserted is non-deterministic across threads, so the
            // resulting graph differs slightly run-to-run — same property
            // as parallel set().
            _during_build = true;
#pragma omp parallel for num_threads(static_cast<int>(_n_threads)) schedule(dynamic, 16)
            for (int64_t i = 0; i < static_cast<int64_t>(added); i++) {
                add_point(static_cast<int32_t>(old_n + i));
            }
            _during_build = false;
        }

        std::vector<int32_t> ids(added);
        for (size_t i = 0; i < added; i++) ids[i] = static_cast<int32_t>(old_n + i);
        return ids;
    }

    // Mark a node deleted (lazy). Out-of-range / already-deleted is a no-op.
    void remove(int32_t node_id) {
        if (node_id < 0 || (size_t)node_id >= _deleted.size()) return;
        if (!_deleted[node_id]) {
            _deleted[node_id] = 1u;
            _num_deleted++;
        }
    }

    // Compact away tombstones. Builds a fresh graph from the live vectors
    // and returns a mapping: result[old_id] = new_id, or -1 if old_id was
    // deleted. The mapping has size == old _examples.size() so callers can
    // translate any external bookkeeping.
    std::vector<int32_t> rebuild() {
        size_t old_n = _examples.size();
        std::vector<int32_t> mapping(old_n, -1);
        std::vector<T> live;
        live.reserve(old_n);
        for (size_t i = 0; i < old_n; i++) {
            if (_deleted[i]) continue;
            mapping[i] = static_cast<int32_t>(live.size());
            live.push_back(_examples[i]);
        }
        // Snapshot the data we need before clear() invalidates the spans.
        // For FlatSpan / SQ8Span the spans point into our owned backing —
        // copy the raw bytes out so live's spans stay valid after clear().
        std::vector<float>   live_flat;
        std::vector<int8_t>  live_sq8;
        if constexpr (std::is_same_v<T, FlatSpan>) {
            live_flat.resize(live.size() * _dim);
            for (size_t i = 0; i < live.size(); i++) {
                std::memcpy(live_flat.data() + i * _dim, live[i].ptr, _dim * sizeof(float));
                live[i] = FlatSpan{live_flat.data() + i * _dim, _dim};
            }
        } else if constexpr (std::is_same_v<T, SQ8Span>) {
            live_sq8.resize(live.size() * _dim);
            for (size_t i = 0; i < live.size(); i++) {
                std::memcpy(live_sq8.data() + i * _dim, live[i].ptr, _dim);
                live[i] = SQ8Span{live_sq8.data() + i * _dim, _dim};
            }
        }
        set(live);
        return mapping;
    }

    size_t num_deleted() const { return _num_deleted; }

    // Query: returns top-k indices and distances per query.
    // Convention matches VPTreeNumpyAdapter: distances within the top-k are
    // returned farthest-first (caller reverses if they need nearest-first).
    // Optional `filter_mask`: byte-per-node array of length `size()`. When
    // non-null, only nodes with mask[id] != 0 are eligible for the top-k.
    // Implemented as a post-filter with inflation in search_one() — works
    // best when filter selectivity ≥ 10 %; very selective filters fall
    // back to scanning more candidates (capped at 8×k).
    void searchKNN(const std::vector<T>& queries,
                   size_t k,
                   std::vector<std::vector<int64_t>>& indices_out,
                   std::vector<std::vector<distT>>& distances_out,
                   const uint8_t* filter_mask = nullptr) {
        size_t nq = queries.size();
        indices_out.assign(nq, {});
        distances_out.assign(nq, {});

        if (_entry_point < 0 || _examples.empty()) return;

        auto process_query = [&](size_t qi) {
            std::vector<DistNode<distT>> top = search_one(queries[qi], k, filter_mask);
            // top is sorted nearest → farthest. Reverse for pynear's convention.
            indices_out[qi].reserve(top.size());
            distances_out[qi].reserve(top.size());
            for (auto it = top.rbegin(); it != top.rend(); ++it) {
                indices_out[qi].push_back(it->node_id);
                distances_out[qi].push_back(it->distance);
            }
        };

#ifdef _OPENMP
        if (_n_threads > 1 && nq > 1) {
            // Queries are independent and the graph is read-only during
            // search (visited state comes from the thread-safe pool, and
            // _dist_calls is a relaxed atomic), so each query produces
            // exactly the same result as in the serial loop below.
#pragma omp parallel for schedule(dynamic) num_threads(_n_threads)
            for (int64_t qi = 0; qi < static_cast<int64_t>(nq); qi++) {
                process_query(static_cast<size_t>(qi));
            }
            return;
        }
#endif
        for (size_t qi = 0; qi < nq; qi++) {
            process_query(qi);
        }
    }

    // Convenience wrapper: k=1 searchKNN unwrapped to flat outputs.
    // Queries with no eligible result (empty index / fully filtered)
    // keep the -1 / zero-distance placeholders.
    void search1NN(const std::vector<T>& queries,
                   std::vector<int64_t>& indices_out,
                   std::vector<distT>& distances_out,
                   const uint8_t* filter_mask = nullptr) {
        size_t nq = queries.size();
        indices_out.assign(nq, -1);
        distances_out.assign(nq, distT{});

        std::vector<std::vector<int64_t>> indices;
        std::vector<std::vector<distT>> distances;
        searchKNN(queries, 1, indices, distances, filter_mask);
        for (size_t qi = 0; qi < nq; qi++) {
            if (!indices[qi].empty()) {
                indices_out[qi] = indices[qi].front();
                distances_out[qi] = distances[qi].front();
            }
        }
    }

    // Serialisation helpers — caller (Python binding) assembles bytes.
    // `flat_bytes` carries the raw vector data with no per-element type
    // — for T=FlatSpan it's float-as-bytes, for T=arrayli it's uint8_t
    // bytes. This way the same serialize/deserialize works for the
    // float L2/cosine pipelines and the binary Hamming pipeline.
    void serialize(std::vector<uint8_t>& flat_bytes,
                   std::vector<int32_t>& levels,
                   std::vector<int32_t>& flat_adj,
                   std::vector<int32_t>& adj_offsets,
                   int32_t& entry,
                   int32_t& top_level,
                   size_t& dim_out,
                   uint64_t& seed_out,
                   std::vector<uint8_t>& deleted_out) const {
        deleted_out = _deleted;
        if constexpr (std::is_same_v<T, FlatSpan>) {
            flat_bytes.resize(_flat_backing.size() * sizeof(float));
            std::memcpy(flat_bytes.data(), _flat_backing.data(), flat_bytes.size());
        } else if constexpr (std::is_same_v<T, SQ8Span>) {
            // SQ8: bytes come straight from our owned int8 backing.
            // memcpy because int8_t↔uint8_t need an explicit bitcast.
            flat_bytes.resize(_sq8_backing.size());
            std::memcpy(flat_bytes.data(), _sq8_backing.data(), flat_bytes.size());
        } else if constexpr (std::is_same_v<T, arrayli>) {
            // Binary vectors live in _examples (each is a std::vector<uint8_t>).
            // Flatten row-major into the single byte buffer.
            size_t n = _examples.size();
            flat_bytes.resize(n * _dim);
            for (size_t i = 0; i < n; i++) {
                std::memcpy(flat_bytes.data() + i * _dim,
                            _examples[i].data(),
                            _dim);
            }
        } else {
            flat_bytes.clear();
        }
        levels = _levels;
        entry = _entry_point;
        top_level = _top_level;
        dim_out = _dim;
        seed_out = _seed_used;

        // Compute total size and offsets. For node i:
        //   offsets[2*i]     = start index of its adjacency chunk in flat_adj
        //   offsets[2*i + 1] = total number of int32s for that node
        // The chunk is laid out as:
        //   layer_size_0, layer_size_1, ..., layer_size_levels[i],
        //   edge_list_layer_0, edge_list_layer_1, ...
        flat_adj.clear();
        adj_offsets.clear();
        adj_offsets.reserve(2 * _adjacency.size());

        // Reconstitute the per-node "layer 0 + upper" view for pickle.
        // Layer 0 lives in the flat buffer; upper layers in the nested vector.
        //
        // Layer count comes from _levels[i] + 1, NOT _adjacency[i].size():
        // level-0 nodes keep their _adjacency entry empty (memory saving),
        // but they still serialise the same "1 layer" record as before —
        // for nodes with upper layers _adjacency[i].size() == _levels[i]+1,
        // so the emitted bytes are identical to the historical format.
        for (size_t i = 0; i < _adjacency.size(); i++) {
            adj_offsets.push_back(static_cast<int32_t>(flat_adj.size()));
            int32_t before = static_cast<int32_t>(flat_adj.size());
            int32_t nlayers = _levels[i] + 1;
            // sizes header — layer 0 size from flat buffer, upper from nested
            for (int32_t l = 0; l < nlayers; l++) {
                if (l == 0) {
                    flat_adj.push_back(_layer0_count[i]);
                } else {
                    flat_adj.push_back(
                        l < (int32_t)_adjacency[i].size()
                            ? static_cast<int32_t>(_adjacency[i][l].size())
                            : 0);
                }
            }
            // edge lists — same source as the sizes header above
            for (int32_t l = 0; l < nlayers; l++) {
                if (l == 0) {
                    int32_t cnt = _layer0_count[i];
                    for (int32_t k = 0; k < cnt; k++) {
                        flat_adj.push_back(_layer0_adj[(size_t)i * _M_max0 + k]);
                    }
                } else if (l < (int32_t)_adjacency[i].size()) {
                    for (int32_t e : _adjacency[i][l]) flat_adj.push_back(e);
                }
            }
            adj_offsets.push_back(static_cast<int32_t>(flat_adj.size()) - before);
        }
    }

    void deserialize(std::vector<uint8_t>&& flat_bytes,
                     std::vector<int32_t>&& levels,
                     const std::vector<int32_t>& flat_adj,
                     const std::vector<int32_t>& adj_offsets,
                     int32_t entry,
                     int32_t top_level,
                     size_t dim,
                     uint64_t seed,
                     size_t M,
                     size_t ef_construction,
                     size_t ef_search,
                     std::vector<uint8_t>&& deleted_in = {}) {
        clear();
        _M = M;
        _M_max0 = 2 * M;
        _ef_construction = ef_construction;
        _ef_search = ef_search;
        _mL = 1.0 / std::log(static_cast<double>(M));
        _seed_used = seed;

        _levels = std::move(levels);
        _entry_point = entry;
        _top_level = top_level;
        _dim = dim;

        size_t n = _levels.size();
        _examples.resize(n);
        if constexpr (std::is_same_v<T, FlatSpan>) {
            // Bytes → float backing.
            _flat_backing.resize(flat_bytes.size() / sizeof(float));
            std::memcpy(_flat_backing.data(), flat_bytes.data(), flat_bytes.size());
            for (size_t i = 0; i < n; i++) {
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
            }
        } else if constexpr (std::is_same_v<T, SQ8Span>) {
            // Bytes → int8 backing → SQ8Span pointers. Bitcast via memcpy
            // (int8_t and uint8_t aren't assignable container-to-container).
            _sq8_backing.resize(flat_bytes.size());
            std::memcpy(_sq8_backing.data(), flat_bytes.data(), flat_bytes.size());
            for (size_t i = 0; i < n; i++) {
                _examples[i] = SQ8Span{_sq8_backing.data() + i * _dim, _dim};
            }
        } else if constexpr (std::is_same_v<T, arrayli>) {
            // Bytes → per-vector arrayli rows.
            for (size_t i = 0; i < n; i++) {
                _examples[i].assign(
                    flat_bytes.data() + i * _dim,
                    flat_bytes.data() + (i + 1) * _dim);
            }
        }

        _adjacency.assign(n, std::vector<std::vector<int32_t>>{});
        _layer0_adj.assign(n * _M_max0, -1);
        _layer0_count.assign(n, 0);
        // Tombstones from pickle (backward compat: missing/short = all live).
        if (deleted_in.size() == n) {
            _deleted = std::move(deleted_in);
        } else {
            _deleted.assign(n, 0u);
        }
        // The tombstone count is derived state — recompute it once here.
        _num_deleted = 0;
        for (uint8_t b : _deleted) if (b) _num_deleted++;
        if constexpr (std::is_same_v<T, FlatSpan>) {
            // Re-populate precomputed norms after deserialise — they aren't
            // serialised because they are a function of the vectors.
            _norms_sq.assign(n, 0.0f);
            for (size_t i = 0; i < n; i++) {
                _norms_sq[i] = vec_l2sq_avx2(_examples[i].ptr, _examples[i].sz);
            }
        }
        init_thread_resources(n);
        for (size_t i = 0; i < n; i++) {
            int32_t start = adj_offsets[2 * i];
            int32_t nlayers = _levels[i] + 1;
            // Level-0 nodes keep _adjacency[i] empty — their single layer
            // lives in the flat layer-0 buffer. Works for both old pickles
            // (which stored "1 layer, empty" for them) and new ones.
            if (nlayers > 1) _adjacency[i].resize(nlayers);
            int32_t cursor = start;
            std::vector<int32_t> layer_sizes(nlayers);
            for (int32_t l = 0; l < nlayers; l++) {
                layer_sizes[l] = flat_adj[cursor++];
            }
            for (int32_t l = 0; l < nlayers; l++) {
                if (l == 0) {
                    _layer0_count[i] = layer_sizes[0];
                    for (int32_t e = 0; e < layer_sizes[0]; e++) {
                        _layer0_adj[i * _M_max0 + e] = flat_adj[cursor++];
                    }
                } else {
                    _adjacency[i][l].reserve(layer_sizes[l]);
                    for (int32_t e = 0; e < layer_sizes[l]; e++) {
                        _adjacency[i][l].push_back(flat_adj[cursor++]);
                    }
                }
            }
        }
    }

    size_t M() const { return _M; }
    size_t ef_construction() const { return _ef_construction; }

    void clear() {
        _examples.clear();
        _flat_backing.clear();
        _sq8_backing.clear();
        _levels.clear();
        _deleted.clear();
        _num_deleted = 0;
        _adjacency.clear();
        _layer0_adj.clear();
        _layer0_count.clear();
        _visited_pool.reset();
        _rng_per_thread.clear();
        _node_locks.reset();
        _num_locks = 0;
        _entry_lock.reset();
        _entry_point = -1;
        _top_level = -1;
        _dim = 0;
    }

private:
    // ─── Algorithm core ─────────────────────────────────────────────────────

    int32_t random_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        // _rng_per_thread is always populated before add_point() can run —
        // every build path (set / add / deserialize) calls
        // init_thread_resources() first.
        auto& rng = _rng_per_thread[omp_get_thread_num()];
        double r;
        do { r = u(rng); } while (r <= 0.0);  // avoid log(0)
        return static_cast<int32_t>(-std::log(r) * _mL);
    }

    // Greedy single-step search at upper layers: walk neighbours until no
    // closer one is found. Returns the closest node.
    int32_t greedy_descent(const T& query, int32_t entry, int32_t layer) const {
        int32_t current = entry;
        distT current_d = distance_fn(query, _examples[current]);
        bool changed = true;
        while (changed) {
            changed = false;
            NeighbourView nv = read_neighbours(current, layer);
            for (size_t i = 0; i < nv.count; i++) {
                int32_t n = nv.ptr[i];
                distT d = distance_fn(query, _examples[n]);
                if (d < current_d) {
                    current_d = d;
                    current = n;
                    changed = true;
                }
            }
        }
        return current;
    }

    // A neighbour view that's either:
    //   - a borrowed pointer into _adjacency (post-build / non-locking path)
    //   - an owned copy taken under a shared lock (build-time path)
    // The two-mode design avoids copying neighbour lists on every visit at
    // query time (saving an alloc + memcpy per node visited).
    struct NeighbourView {
        const int32_t* ptr = nullptr;
        size_t count = 0;
        std::vector<int32_t> owned;
    };

    NeighbourView read_neighbours(int32_t node, int32_t layer) const {
        NeighbourView v;
        // Layer 0 — flat buffer, cache-friendly, the hot path for queries.
        if (layer == 0) {
            if (_layer0_adj.empty()) return v;
            if (_during_build && _num_locks > 0) {
                std::shared_lock<std::shared_mutex> rlock(_node_locks[node]);
                int32_t cnt = _layer0_count[node];
                v.owned.assign(
                    _layer0_adj.data() + (size_t)node * _M_max0,
                    _layer0_adj.data() + (size_t)node * _M_max0 + cnt);
                v.ptr = v.owned.data();
                v.count = v.owned.size();
                return v;
            }
            v.ptr = _layer0_adj.data() + (size_t)node * _M_max0;
            v.count = static_cast<size_t>(_layer0_count[node]);
            return v;
        }

        // Upper layers — nested vector path.
        if (_during_build && _num_locks > 0) {
            std::shared_lock<std::shared_mutex> rlock(_node_locks[node]);
            if (layer >= (int32_t)_adjacency[node].size()) return v;
            v.owned = _adjacency[node][layer];
            v.ptr = v.owned.data();
            v.count = v.owned.size();
            return v;
        }
        if (layer >= (int32_t)_adjacency[node].size()) return v;
        v.ptr = _adjacency[node][layer].data();
        v.count = _adjacency[node][layer].size();
        return v;
    }

    // Beam search at a given layer. Uses raw std::vector + std::push_heap /
    // std::pop_heap so we control the underlying allocation (reserve to
    // avoid reallocations in the inner loop). std::priority_queue hides its
    // container and reallocates on growth.
    //
    // Optional `extra_seeds`: additional entry nodes (e.g. from an MIH
    // lookup) pre-inserted into both the candidate queue and the result set
    // with their true distances, de-duplicated via the visited buffer, so
    // they participate fully in the beam.
    // Optional `known_query_norm_sq`: caller-supplied ||q||² (FlatSpan path
    // only). During build the query IS the freshly inserted point, whose
    // norm already sits in _norms_sq — passing it avoids recomputing it per
    // layer. The stored value comes from the exact same vec_l2sq_avx2 call
    // on the same data, so distances are bit-identical either way.
    std::vector<DistNode<distT>>
    search_layer(const T& query, int32_t entry, size_t ef, int32_t layer,
                 const std::vector<int32_t>* extra_seeds = nullptr,
                 const float* known_query_norm_sq = nullptr) const {
        // Check a visited-stamp array out of the pool for the duration of
        // this beam search (returned by the guard's destructor).
        VisitedGuard vguard(*_visited_pool);
        auto& vlist = *vguard;
        const uint32_t ver = vlist.advance(_examples.size());

        DistNodeMin<distT> min_cmp;
        DistNodeMax<distT> max_cmp;

        // Precompute query norm squared (for the dot-product distance trick;
        // only meaningful for the FlatSpan/float-L2 path).
        float query_norm_sq = 0.0f;
        if constexpr (std::is_same_v<T, FlatSpan>) {
            query_norm_sq = known_query_norm_sq
                                ? *known_query_norm_sq
                                : vec_l2sq_avx2(query.ptr, query.sz);
        }

        // Candidates: min-heap by distance (explore nearest first).
        // Results:    max-heap by distance (drop the farthest when full).
        std::vector<DistNode<distT>> candidates;
        std::vector<DistNode<distT>> results;
        candidates.reserve(ef * 2 + (extra_seeds ? extra_seeds->size() : 0) + 8);
        results.reserve(ef + 1);

        auto& vis = vlist.stamps;

        // Admit a scored node into the beam: push into candidates + results,
        // pop the farthest result once the beam exceeds ef. Shared by every
        // distance path below.
        auto try_add = [&](distT d, int32_t n) {
            if (results.size() < ef || d < results.front().distance) {
                candidates.push_back({d, n});
                std::push_heap(candidates.begin(), candidates.end(), min_cmp);
                results.push_back({d, n});
                std::push_heap(results.begin(), results.end(), max_cmp);
                if (results.size() > ef) {
                    std::pop_heap(results.begin(), results.end(), max_cmp);
                    results.pop_back();
                }
            }
        };

        // Seed the beam. Unlike try_add, seeds always enter the candidate
        // queue (the beam must explore from them) even when they don't make
        // the result set.
        auto seed = [&](int32_t node) {
            if (node < 0 || node >= (int32_t)_examples.size()) return;
            if (vis[node] == ver) return;
            vis[node] = ver;
            distT d = distance_fn(query, _examples[node]);
            candidates.push_back({d, node});
            std::push_heap(candidates.begin(), candidates.end(), min_cmp);
            if (results.size() < ef || d < results.front().distance) {
                results.push_back({d, node});
                std::push_heap(results.begin(), results.end(), max_cmp);
                if (results.size() > ef) {
                    std::pop_heap(results.begin(), results.end(), max_cmp);
                    results.pop_back();
                }
            }
        };
        seed(entry);
        if (extra_seeds) {
            for (int32_t s : *extra_seeds) seed(s);
        }

        // Shared scaffolding for the SQ8/FlatSpan batched fast paths. Three
        // layered optimisations, following Faiss's HNSW.cpp pattern:
        //
        //   (a) Pre-pass over the neighbour list that prefetches
        //       each visited-array entry into L1. This is what
        //       Faiss does (`vt.prefetch(v1)`) — for typical
        //       neighbour counts (M_max0 = 32) all visited entries
        //       fit in a few cache lines; prefetching them ahead
        //       hides the L2/L3 visited-check latency.
        //   (b) Filter to unvisited and prefetch each unvisited
        //       vector's first cache line.
        //   (c) Single batched SIMD distance call — supplied by the
        //       caller as `compute(vptr, ids, n, dists)`; `get_ptr`
        //       extracts the raw vector pointer for a node id.
        auto process_batch = [&](const NeighbourView& nv,
                                 auto&& get_ptr, auto&& compute) {
            const size_t cap = std::min(nv.count, kMaxBatch);
#if defined(__AVX__) || defined(__AVX2__)
            // (a) prefetch visited-array entries.
            for (size_t ni = 0; ni < cap; ni++) {
                _mm_prefetch(
                    reinterpret_cast<const char*>(&vis[nv.ptr[ni]]),
                    _MM_HINT_T0);
            }
#endif
            int32_t unvis[kMaxBatch];
            decltype(get_ptr(int32_t{0})) vptr[kMaxBatch];
            size_t n_unvis = 0;
            for (size_t ni = 0; ni < cap; ni++) {
                int32_t n = nv.ptr[ni];
                if (vis[n] == ver) continue;
                vis[n] = ver;
                unvis[n_unvis] = n;
                vptr[n_unvis] = get_ptr(n);
#if defined(__AVX__) || defined(__AVX2__)
                // (b) prefetch this unvisited vector's first cache line
                // — the rest will stream in via HW prefetcher once the
                // load pattern is established.
                _mm_prefetch(reinterpret_cast<const char*>(vptr[n_unvis]),
                             _MM_HINT_T0);
#endif
                n_unvis++;
            }
            if (n_unvis == 0) return;

            // (c) batched distances, then admit into the beam.
            distT dists[kMaxBatch];
            compute(vptr, unvis, n_unvis, dists);
            _dist_calls += n_unvis;
            for (size_t i = 0; i < n_unvis; i++) {
                try_add(dists[i], unvis[i]);
            }
        };
        (void)process_batch;  // unused for the generic (non-span) T path

        while (!candidates.empty()) {
            DistNode<distT> cur = candidates.front();
            std::pop_heap(candidates.begin(), candidates.end(), min_cmp);
            candidates.pop_back();

            if (results.size() >= ef && cur.distance > results.front().distance) {
                break;
            }

            NeighbourView nv = read_neighbours(cur.node_id, layer);
            if (nv.count == 0) continue;

            if constexpr (std::is_same_v<T, SQ8Span>) {
                // SQ8 fast path: int8 storage, int32 distance via
                // batch_l2sq_sq8_avx2 (32 lanes per AVX2 op, 4× wider than
                // float). Per-vector data is 4× smaller too — much friendlier
                // to L2/L3 cache.
                process_batch(
                    nv,
                    [&](int32_t n) -> const int8_t* { return _examples[n].ptr; },
                    [&](auto vptr, const int32_t*, size_t n, distT* dists) {
#if defined(__AVX512F__)
                        // AVX-512 SQ8: 64 int8 lanes/op (vs AVX2's 32), ~2×
                        // throughput on supporting hardware. We dispatch
                        // one-at-a-time because the single-distance AVX-512
                        // kernel is already very wide.
                        for (size_t bi = 0; bi < n; bi++) {
                            dists[bi] = static_cast<distT>(
                                dist_l2sq_sq8_avx512(query.ptr, vptr[bi], query.sz));
                        }
#else
                        int32_t dists_i32[kMaxBatch];
                        batch_l2sq_sq8_avx2(query.ptr, query.sz, vptr,
                                            n, dists_i32);
                        for (size_t i = 0; i < n; i++) {
                            dists[i] = static_cast<distT>(dists_i32[i]);
                        }
#endif
                    });
            } else if constexpr (std::is_same_v<T, FlatSpan>) {
                // Float-vector fast path. Dot-product batch: compute q·v for
                // each unvisited neighbour, then reconstruct ||q-v||² via
                //   ||q-v||² = ||q||² + ||v||² − 2 q·v
                // using precomputed _norms_sq[v]. One FMA per chunk (dot)
                // instead of two (sub + square), so per-distance FLOPs halve.
                process_batch(
                    nv,
                    [&](int32_t n) -> const float* { return _examples[n].ptr; },
                    [&](auto vptr, const int32_t* ids, size_t n, distT* dists) {
                        float dots[kMaxBatch];
#if defined(__AVX512F__)
                        // AVX-512 path: 16 floats/op, 8 accumulators saturate
                        // FMA throughput on Zen 4 / Sapphire Rapids / Xeon Gold.
                        batch_dot_f_avx512_8(query.ptr, query.sz, vptr, n, dots);
#else
                        batch_dot_f_avx2_8(query.ptr, query.sz, vptr, n, dots);
#endif
                        for (size_t i = 0; i < n; i++) {
                            float d_sq = query_norm_sq + _norms_sq[ids[i]] - 2.0f * dots[i];
                            dists[i] = d_sq < 0.0f ? 0.0f : d_sq;  // guard against fp noise
                        }
                    });
            } else {
                // Generic path — used for binary/Hamming and any other T.
                for (size_t ni = 0; ni < nv.count; ni++) {
                    int32_t n = nv.ptr[ni];
                    if (vis[n] == ver) continue;
                    vis[n] = ver;
                    distT d = distance_fn(query, _examples[n]);
                    try_add(d, n);
                }
            }
        }

        // Results are in heap order; caller does the final sort.
        return results;
    }

    // α-heuristic from §4 of the HNSW paper. Selects up to M items from
    // `candidates` such that for every selected pair (p, q), the inserted
    // point is closer to both than they are to each other — biases toward
    // long-range, "navigable" edges that give the graph its small-world
    // property.
    //
    // Two variants from algorithm 4 of the paper:
    //  - extendCandidates: also consider neighbours of candidates. We skip
    //    this; it's expensive and Faiss/hnswlib don't enable it by default.
    //  - keepPrunedConnections: after the heuristic, if |selected| < M,
    //    pad up to M from the rejected pool (nearest-first). Without this,
    //    nodes can end up with fewer than M edges and recall drops measurably
    //    at every ef. We enable it — matches Faiss/hnswlib defaults.
    std::vector<int32_t>
    select_neighbours_heuristic(const T& query,
                                std::vector<DistNode<distT>> candidates,
                                size_t M) const {
        if (candidates.size() <= M) {
            std::vector<int32_t> ids;
            ids.reserve(candidates.size());
            std::sort(candidates.begin(), candidates.end(),
                      [](const DistNode<distT>& a, const DistNode<distT>& b) {
                          return a.distance < b.distance;
                      });
            for (auto& c : candidates) ids.push_back(c.node_id);
            return ids;
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const DistNode<distT>& a, const DistNode<distT>& b) {
                      return a.distance < b.distance;
                  });

        std::vector<int32_t> selected;
        std::vector<DistNode<distT>> rejected;
        selected.reserve(M);
        rejected.reserve(candidates.size());

        for (const auto& c : candidates) {
            if (selected.size() >= M) break;
            bool keep = true;
            for (int32_t s : selected) {
                distT d_sc = distance_fn(_examples[c.node_id], _examples[s]);
                if (d_sc < c.distance) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected.push_back(c.node_id);
            } else {
                rejected.push_back(c);
            }
        }
        // keepPrunedConnections — pad to M from the nearest rejected candidates.
        for (const auto& c : rejected) {
            if (selected.size() >= M) break;
            selected.push_back(c.node_id);
        }
        return selected;
    }

    // Insert one point at random level, link it via the α-heuristic.
    //
    // Thread-safety: callable concurrently from OpenMP threads. Writes to
    // adjacency lists are protected by per-node std::shared_mutex; the
    // entry-point/top-level globals are guarded by _entry_lock; reads inside
    // search_layer also take shared locks via read_neighbours().
    void add_point(int32_t pidx) {
        int32_t level = random_level();

        // Own adjacency setup. No lock needed because no other thread can
        // reach pidx yet (pidx is unreachable from the graph until we add
        // reverse edges below).
        _levels[pidx] = level;
        // Upper layers go into the nested vector. Layer 0 lives in
        // _layer0_adj (flat) and is initialised to count=0 by set().
        // Level-0 nodes (~94 % of them) keep _adjacency[pidx] EMPTY — their
        // only layer is layer 0, which never touches the nested vector.
        // Readers handle this: read_neighbours() and the reverse-edge path
        // both bounds-check the layer against _adjacency[node].size().
        if (level > 0) {
            _adjacency[pidx].assign(level + 1, std::vector<int32_t>{});
            for (int32_t l = 1; l <= level; l++) {
                _adjacency[pidx][l].reserve(_M);
            }
        }

        // Snapshot entry point + top level under the entry lock. The greedy
        // descent below uses this snapshot; if another thread updates the
        // entry point afterwards, we still produce a valid (if slightly
        // suboptimal) graph.
        int32_t snap_entry;
        int32_t snap_top;
        {
            std::lock_guard<std::mutex> elock(*_entry_lock);
            if (_entry_point < 0) {
                _entry_point = pidx;
                _top_level = level;
                return;
            }
            snap_entry = _entry_point;
            snap_top = _top_level;
        }

        // Greedy descent from the top of the index down to level + 1.
        int32_t current = snap_entry;
        for (int32_t l = snap_top; l > level; l--) {
            current = greedy_descent(_examples[pidx], current, l);
        }

        // The new point's ||·||² is already stored (FlatSpan path) — reuse it
        // in every search_layer call below instead of recomputing per layer.
        const float* known_norm_sq = nullptr;
        if constexpr (std::is_same_v<T, FlatSpan>) {
            known_norm_sq = &_norms_sq[pidx];
        }

        // From min(level, top_level) down to layer 0, build candidates and connect.
        int32_t start_layer = std::min(level, snap_top);
        for (int32_t l = start_layer; l >= 0; l--) {
            auto candidates = search_layer(_examples[pidx], current, _ef_construction, l,
                                           nullptr, known_norm_sq);

            std::vector<int32_t> chosen = select_neighbours_heuristic(_examples[pidx],
                                                                     candidates,
                                                                     _M);

            // Link new node → chosen. No lock: this is our own node.
            if (l == 0) {
                for (size_t i = 0; i < chosen.size(); i++) {
                    _layer0_adj[(size_t)pidx * _M_max0 + i] = chosen[i];
                }
                _layer0_count[pidx] = static_cast<int32_t>(chosen.size());
            } else {
                _adjacency[pidx][l] = chosen;
            }

            // Link reverse edges; prune neighbour's adjacency if it now
            // exceeds M_max. Per-node exclusive lock protects the modification.
            for (int32_t c : chosen) {
                std::unique_lock<std::shared_mutex> wlock(_node_locks[c]);
                if (l == 0) {
                    int32_t& cnt = _layer0_count[c];
                    if (cnt < (int32_t)_M_max0) {
                        _layer0_adj[(size_t)c * _M_max0 + cnt] = pidx;
                        cnt++;
                    } else {
                        // Slot full — apply α-heuristic over existing edges +
                        // new one. Distance computations dominate here, so
                        // run them OUTSIDE the exclusive lock: snapshot the
                        // neighbour list, unlock, score + select, re-lock and
                        // install only if the list is unchanged. If another
                        // inserter modified it meanwhile, redo the whole
                        // prune under the lock (the pre-computed result may
                        // be based on stale edges) — correctness over speed.
                        // Vector data (_examples) is immutable during build,
                        // so the unlocked distance reads are safe.
                        std::vector<int32_t> snap(
                            _layer0_adj.data() + (size_t)c * _M_max0,
                            _layer0_adj.data() + (size_t)c * _M_max0 + cnt);
                        wlock.unlock();

                        std::vector<DistNode<distT>> nc;
                        nc.reserve(snap.size() + 1);
                        for (int32_t n : snap) {
                            nc.push_back({distance_fn(_examples[c], _examples[n]), n});
                        }
                        nc.push_back({distance_fn(_examples[c], _examples[pidx]), pidx});
                        auto kept = select_neighbours_heuristic(_examples[c], nc, _M_max0);

                        wlock.lock();
                        const int32_t* cur_ptr = _layer0_adj.data() + (size_t)c * _M_max0;
                        const bool unchanged =
                            cnt == (int32_t)snap.size() &&
                            std::equal(snap.begin(), snap.end(), cur_ptr);
                        if (unchanged) {
                            for (size_t i = 0; i < kept.size(); i++) {
                                _layer0_adj[(size_t)c * _M_max0 + i] = kept[i];
                            }
                            cnt = static_cast<int32_t>(kept.size());
                        } else if (cnt < (int32_t)_M_max0) {
                            // List shrank while unlocked — plain append,
                            // exactly what the fast path would have done.
                            _layer0_adj[(size_t)c * _M_max0 + cnt] = pidx;
                            cnt++;
                        } else {
                            // List changed and is still full — redo the
                            // prune under the lock on the current edges.
                            nc.clear();
                            for (int32_t i = 0; i < cnt; i++) {
                                int32_t n = _layer0_adj[(size_t)c * _M_max0 + i];
                                nc.push_back({distance_fn(_examples[c], _examples[n]), n});
                            }
                            nc.push_back({distance_fn(_examples[c], _examples[pidx]), pidx});
                            kept = select_neighbours_heuristic(_examples[c], nc, _M_max0);
                            for (size_t i = 0; i < kept.size(); i++) {
                                _layer0_adj[(size_t)c * _M_max0 + i] = kept[i];
                            }
                            cnt = static_cast<int32_t>(kept.size());
                        }
                    }
                } else {
                    if (l >= (int32_t)_adjacency[c].size()) continue;
                    auto& neigh_adj = _adjacency[c][l];
                    neigh_adj.push_back(pidx);
                    if (neigh_adj.size() > _M) {
                        std::vector<DistNode<distT>> nc;
                        nc.reserve(neigh_adj.size());
                        for (int32_t n : neigh_adj) {
                            nc.push_back({distance_fn(_examples[c], _examples[n]), n});
                        }
                        _adjacency[c][l] = select_neighbours_heuristic(_examples[c], nc, _M);
                    }
                }
            }

            // Update current for next (lower) layer's descent.
            if (!candidates.empty()) {
                distT best = candidates.front().distance;
                int32_t best_id = candidates.front().node_id;
                for (auto& cand : candidates) {
                    if (cand.distance < best) {
                        best = cand.distance;
                        best_id = cand.node_id;
                    }
                }
                current = best_id;
            }
        }

        if (level > snap_top) {
            std::lock_guard<std::mutex> elock(*_entry_lock);
            // Re-check under lock — another thread may have raised top_level too.
            if (level > _top_level) {
                _top_level = level;
                _entry_point = pidx;
            }
        }
    }

    // Query path: greedy descent to layer 0, then full beam search.
    // Returns up to k results sorted nearest → farthest. Optional
    // `extra_seeds` are forwarded to the layer-0 beam (see search_layer).
    std::vector<DistNode<distT>> search_one(const T& query, size_t k,
                                             const uint8_t* filter_mask = nullptr,
                                             const std::vector<int32_t>* extra_seeds = nullptr) const {
        int32_t current = _entry_point;
        for (int32_t l = _top_level; l > 0; l--) {
            current = greedy_descent(query, current, l);
        }
        // If there are tombstones OR a filter, fetch more raw candidates
        // than k so we still return k *eligible* results after filtering.
        // Cap inflation at 8× to avoid pathological blow-up for very
        // selective filters (caller should pre-partition / shard).
        const size_t n_deleted = num_deleted();
        const bool has_tombstones = n_deleted > 0;
        const bool has_filter = filter_mask != nullptr;
        size_t fetch_k = k;
        if (has_tombstones || has_filter) {
            fetch_k = std::min(k * 8, std::max<size_t>(k * 2, k + n_deleted));
        }
        size_t ef = std::max(_ef_search, fetch_k);
        auto layer0 = search_layer(query, current, ef, 0, extra_seeds);

        std::sort(layer0.begin(), layer0.end(),
                  [](const DistNode<distT>& a, const DistNode<distT>& b) {
                      return a.distance < b.distance;
                  });

        // Single filter pass — combines tombstone + user filter checks.
        if (has_tombstones || has_filter) {
            std::vector<DistNode<distT>> live;
            live.reserve(std::min(k, layer0.size()));
            for (const auto& dn : layer0) {
                if (dn.node_id < 0 || (size_t)dn.node_id >= _examples.size()) continue;
                if (has_tombstones && _deleted[dn.node_id]) continue;
                if (has_filter && !filter_mask[dn.node_id]) continue;
                live.push_back(dn);
                if (live.size() >= k) break;
            }
            return live;
        }
        if (layer0.size() > k) layer0.resize(k);
        return layer0;
    }

public:
    // Multi-seed variant — runs layer-0 beam search seeded with the
    // standard HNSW entry plus a list of externally-supplied seed nodes
    // (e.g. from an MIH lookup). The seed nodes are pre-inserted into both
    // the candidate min-heap and the result max-heap with their true
    // distances, so they participate fully in the beam.
    //
    // Usage in this codebase: the `MIHSeededHNSWBinaryAdapter` calls this
    // with the top-K MIH candidates as `extra_seeds` so that exact
    // small-radius neighbours are always reachable, while HNSW handles
    // larger-radius queries via normal graph traversal.
    std::vector<DistNode<distT>>
    search_one_with_seeds(const T& query, size_t k,
                          const std::vector<int32_t>& extra_seeds) const {
        if (_entry_point < 0 || _examples.empty()) return {};
        return search_one(query, k, nullptr, &extra_seeds);
    }

private:
    // ─── State ──────────────────────────────────────────────────────────────

    size_t _M;
    size_t _M_max0;
    size_t _ef_construction;
    size_t _ef_search;
    double _mL;

    // Upper bound on a node's neighbour count processed per batched-SIMD
    // sweep in search_layer. Class-scope (not function-local) so it is a
    // valid constant expression for stack-array bounds inside the nested
    // lambdas on MSVC as well as gcc/clang.
    static constexpr size_t kMaxBatch = 512;

    int32_t _entry_point;
    int32_t _top_level;
    size_t _dim;

    std::vector<T> _examples;
    std::vector<float> _flat_backing;            // owns the raw vectors when T = FlatSpan
    std::vector<int8_t> _sq8_backing;            // owns the raw vectors when T = SQ8Span
    // Precomputed ||v_i||² for the dot-product distance trick:
    // ||q − v||² = ||q||² + ||v||² − 2 q·v. Halves the FMA count per
    // distance vs sub-then-square. Only populated for the float-vector
    // template instantiation (FlatSpan); empty otherwise.
    std::vector<float> _norms_sq;
    std::vector<int32_t> _levels;                // _levels[i] = max layer for node i

    // Tombstone bitmap for lazy deletion. 1 byte per node so we can extend
    // it cheaply on add(). Search traverses through deleted nodes (preserves
    // graph reachability) but excludes them from results. Compact away with
    // rebuild().
    std::vector<uint8_t> _deleted;
    // Number of set bits in _deleted, kept in sync by remove()/set()/
    // deserialize() so num_deleted() is O(1). Derived state — never
    // serialised.
    size_t _num_deleted = 0;

    // Layer 0 lives in a contiguous flat buffer for cache locality on
    // queries. Each node owns a fixed-size slot of _M_max0 int32 IDs at
    // offset `node * _M_max0`; the actual edge count is in _layer0_count.
    // This is the single biggest cache-locality win on the query path.
    std::vector<int32_t> _layer0_adj;            // size = N * _M_max0
    std::vector<int32_t> _layer0_count;          // size = N

    // Upper layers (layer >= 1) stay as nested vectors — sparse, less critical.
    // _adjacency[node].size() == _levels[node] + 1; index 0 is unused.
    std::vector<std::vector<std::vector<int32_t>>> _adjacency;
        // _adjacency[node][layer] = neighbour IDs at that layer (only layer >= 1)

    uint64_t _seed_used = 42;
    int _n_threads = 1;

    // Pool of versioned visited lists — each search (build-time or query)
    // checks one out for its duration, so any number of concurrent searches
    // work regardless of which thread runs them (OpenMP or not).
    mutable std::unique_ptr<VisitedListPool> _visited_pool;

    // Per-thread RNGs for parallel random_level(). Each is seeded from
    // (_seed_used + thread_id) to keep per-thread sequences distinct and
    // a build deterministic for a given thread count.
    mutable std::vector<std::mt19937> _rng_per_thread;

    // Per-node read/write locks. Allocated as a raw array via unique_ptr
    // because std::shared_mutex is non-movable (so std::vector won't work
    // and the enclosing adapter class needs to stay movable for pybind11
    // pickle). Only held during the parallel build phase; post-build search
    // runs lock-free thanks to the `_during_build` flag below.
    mutable std::unique_ptr<std::shared_mutex[]> _node_locks;
    mutable size_t _num_locks = 0;
    mutable std::unique_ptr<std::mutex> _entry_lock;
    mutable bool _during_build = false;

    void init_thread_resources(size_t n) {
        // RNGs are indexed by omp_get_thread_num() inside parallel regions
        // that request num_threads(_n_threads) — size for whichever is
        // larger so the index can never run past the end.
        int max_threads = std::max(1, omp_get_max_threads());
        max_threads = std::max(max_threads, _n_threads);
        _visited_pool = std::make_unique<VisitedListPool>();
        _rng_per_thread.clear();
        _rng_per_thread.reserve(max_threads);
        for (int t = 0; t < max_threads; t++) {
            _rng_per_thread.emplace_back(static_cast<uint32_t>(_seed_used + (uint64_t)t * 7919));
        }
        // Allocate a raw array of shared_mutexes — they're non-movable so
        // we can't store them by value in std::vector.
        _node_locks = std::unique_ptr<std::shared_mutex[]>(new std::shared_mutex[n]);
        _num_locks = n;
        _entry_lock = std::make_unique<std::mutex>();
    }
};

}  // namespace hnsw
