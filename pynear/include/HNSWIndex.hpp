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
          _rng(seed),
          _seed_used(seed),
          _n_threads(n_threads) {
        if (M < 2) throw std::invalid_argument("M must be >= 2");
        if (ef_construction < 1) throw std::invalid_argument("ef_construction must be >= 1");
    }

    int n_threads() const { return _n_threads; }
    void set_n_threads(int n) { _n_threads = n; }

    // Profiling: total distance calls since last reset. Useful for comparing
    // beam-search efficiency against Faiss's hnsw_stats.ndis.
    mutable uint64_t _dist_calls = 0;
    uint64_t dist_calls() const { return _dist_calls; }
    void reset_dist_calls() { _dist_calls = 0; }

    void set_ef(size_t ef_search) { _ef_search = ef_search; }
    size_t ef_search() const { return _ef_search; }
    size_t size() const { return _examples.size(); }
    size_t dim() const { return _dim; }
    int32_t entry_point() const { return _entry_point; }
    int32_t top_level() const { return _top_level; }
    const std::vector<float>& flat_backing() const { return _flat_backing; }
    const std::vector<int32_t>& levels() const { return _levels; }
    const std::vector<std::vector<std::vector<int32_t>>>& adjacency() const { return _adjacency; }

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
        } else {
            _examples = data;
            _dim = (data.empty() ? 0 : data[0].size());
        }

        _levels.assign(_examples.size(), 0);
        _adjacency.assign(_examples.size(), std::vector<std::vector<int32_t>>{});
        _layer0_adj.assign(_examples.size() * _M_max0, -1);
        _layer0_count.assign(_examples.size(), 0);
        init_thread_resources(_examples.size());

        _during_build = true;
        if (_n_threads <= 1) {
            // Sequential build — fully deterministic given the seed.
            for (size_t i = 0; i < _examples.size(); i++) {
                add_point(static_cast<int32_t>(i));
            }
        } else {
            // Parallel build. Per-node shared_mutexes protect adjacency
            // modifications; reads in search_layer take shared locks via
            // read_neighbours(). Graph topology is non-deterministic across
            // runs but search quality is comparable to the sequential build.
#pragma omp parallel for schedule(dynamic, 64) num_threads(_n_threads)
            for (int64_t i = 0; i < (int64_t)_examples.size(); i++) {
                add_point(static_cast<int32_t>(i));
            }
        }
        _during_build = false;
    }

    // Query: returns top-k indices and distances per query.
    // Convention matches VPTreeNumpyAdapter: distances within the top-k are
    // returned farthest-first (caller reverses if they need nearest-first).
    void searchKNN(const std::vector<T>& queries,
                   size_t k,
                   std::vector<std::vector<int64_t>>& indices_out,
                   std::vector<std::vector<distT>>& distances_out) {
        size_t nq = queries.size();
        indices_out.assign(nq, {});
        distances_out.assign(nq, {});

        if (_entry_point < 0 || _examples.empty()) return;

        for (size_t qi = 0; qi < nq; qi++) {
            std::vector<DistNode<distT>> top = search_one(queries[qi], k);
            // top is sorted nearest → farthest. Reverse for pynear's convention.
            indices_out[qi].reserve(top.size());
            distances_out[qi].reserve(top.size());
            for (auto it = top.rbegin(); it != top.rend(); ++it) {
                indices_out[qi].push_back(it->node_id);
                distances_out[qi].push_back(it->distance);
            }
        }
    }

    void search1NN(const std::vector<T>& queries,
                   std::vector<int64_t>& indices_out,
                   std::vector<distT>& distances_out) {
        size_t nq = queries.size();
        indices_out.assign(nq, -1);
        distances_out.assign(nq, distT{});

        if (_entry_point < 0 || _examples.empty()) return;

        for (size_t qi = 0; qi < nq; qi++) {
            std::vector<DistNode<distT>> top = search_one(queries[qi], 1);
            if (!top.empty()) {
                indices_out[qi] = top.front().node_id;
                distances_out[qi] = top.front().distance;
            }
        }
    }

    // Serialisation helpers — caller (Python binding) assembles bytes.
    // Flatten adjacency into one int32 stream + an offset table.
    void serialize(std::vector<float>& flat,
                   std::vector<int32_t>& levels,
                   std::vector<int32_t>& flat_adj,
                   std::vector<int32_t>& adj_offsets,
                   int32_t& entry,
                   int32_t& top_level,
                   size_t& dim_out,
                   uint64_t& seed_out) const {
        flat = _flat_backing;
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
        for (size_t i = 0; i < _adjacency.size(); i++) {
            adj_offsets.push_back(static_cast<int32_t>(flat_adj.size()));
            int32_t before = static_cast<int32_t>(flat_adj.size());
            int32_t nlayers = static_cast<int32_t>(_adjacency[i].size());
            // sizes header — layer 0 size from flat buffer, upper from nested
            for (int32_t l = 0; l < nlayers; l++) {
                if (l == 0) {
                    flat_adj.push_back(_layer0_count[i]);
                } else {
                    flat_adj.push_back(static_cast<int32_t>(_adjacency[i][l].size()));
                }
            }
            // edge lists — same source as the sizes header above
            for (int32_t l = 0; l < nlayers; l++) {
                if (l == 0) {
                    int32_t cnt = _layer0_count[i];
                    for (int32_t k = 0; k < cnt; k++) {
                        flat_adj.push_back(_layer0_adj[(size_t)i * _M_max0 + k]);
                    }
                } else {
                    for (int32_t e : _adjacency[i][l]) flat_adj.push_back(e);
                }
            }
            adj_offsets.push_back(static_cast<int32_t>(flat_adj.size()) - before);
        }
    }

    void deserialize(std::vector<float>&& flat,
                     std::vector<int32_t>&& levels,
                     const std::vector<int32_t>& flat_adj,
                     const std::vector<int32_t>& adj_offsets,
                     int32_t entry,
                     int32_t top_level,
                     size_t dim,
                     uint64_t seed,
                     size_t M,
                     size_t ef_construction,
                     size_t ef_search) {
        clear();
        _M = M;
        _M_max0 = 2 * M;
        _ef_construction = ef_construction;
        _ef_search = ef_search;
        _mL = 1.0 / std::log(static_cast<double>(M));
        _seed_used = seed;
        _rng.seed(seed);

        _flat_backing = std::move(flat);
        _levels = std::move(levels);
        _entry_point = entry;
        _top_level = top_level;
        _dim = dim;

        size_t n = _levels.size();
        _examples.resize(n);
        if constexpr (std::is_same_v<T, FlatSpan>) {
            for (size_t i = 0; i < n; i++) {
                _examples[i] = FlatSpan{_flat_backing.data() + i * _dim, _dim};
            }
        }

        _adjacency.assign(n, std::vector<std::vector<int32_t>>{});
        _layer0_adj.assign(n * _M_max0, -1);
        _layer0_count.assign(n, 0);
        init_thread_resources(n);
        for (size_t i = 0; i < n; i++) {
            int32_t start = adj_offsets[2 * i];
            int32_t nlayers = _levels[i] + 1;
            _adjacency[i].resize(nlayers);
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
    size_t M_max0() const { return _M_max0; }
    size_t ef_construction() const { return _ef_construction; }

    void clear() {
        _examples.clear();
        _flat_backing.clear();
        _levels.clear();
        _adjacency.clear();
        _layer0_adj.clear();
        _layer0_count.clear();
        _visited_per_thread.clear();
        _visited_version_per_thread.clear();
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
        auto& rng = _rng_per_thread.empty() ? _rng : _rng_per_thread[omp_get_thread_num()];
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
    std::vector<DistNode<distT>>
    search_layer(const T& query, int32_t entry, size_t ef, int32_t layer) const {
        const uint32_t ver = next_visited_version();

        DistNodeMin<distT> min_cmp;
        DistNodeMax<distT> max_cmp;

        // Candidates: min-heap by distance (explore nearest first).
        // Results:    max-heap by distance (drop the farthest when full).
        std::vector<DistNode<distT>> candidates;
        std::vector<DistNode<distT>> results;
        candidates.reserve(ef * 2 + 8);
        results.reserve(ef + 1);

        distT d_entry = distance_fn(query, _examples[entry]);
        candidates.push_back({d_entry, entry});
        std::push_heap(candidates.begin(), candidates.end(), min_cmp);
        results.push_back({d_entry, entry});
        std::push_heap(results.begin(), results.end(), max_cmp);
        visited_buf()[entry] = ver;

        while (!candidates.empty()) {
            DistNode<distT> cur = candidates.front();
            std::pop_heap(candidates.begin(), candidates.end(), min_cmp);
            candidates.pop_back();

            if (results.size() >= ef && cur.distance > results.front().distance) {
                break;
            }

            NeighbourView nv = read_neighbours(cur.node_id, layer);
            if (nv.count == 0) continue;

            auto& vis = visited_buf();

            if constexpr (std::is_same_v<T, FlatSpan>) {
                // Float-vector fast path: filter to unvisited, then batch
                // distance with 4-way SIMD. Saves ~30 % of query time vs
                // calling dist_l2_f_avx2 once per neighbour — the batch
                // exposes ILP that the per-call version cannot.
                // Buffer size covers M up to 256 (M_max0 = 512). Larger M
                // is exotic; we fall back to the generic path if exceeded.
                constexpr size_t MAX_BATCH = 512;
                int32_t unvis[MAX_BATCH];
                const float* vptr[MAX_BATCH];
                size_t n_unvis = 0;
                const size_t cap = std::min(nv.count, MAX_BATCH);
                for (size_t ni = 0; ni < cap; ni++) {
                    int32_t n = nv.ptr[ni];
                    if (vis[n] == ver) continue;
                    vis[n] = ver;
                    unvis[n_unvis] = n;
                    vptr[n_unvis] = _examples[n].ptr;
                    n_unvis++;
                }
                if (n_unvis == 0) continue;

                float dists[MAX_BATCH];
                // HNSW pipelines use the squared-L2 internal kernel (no sqrt
                // in the hot loop) and the Python adapter applies sqrt only
                // to the final top-k. Cosine adapter consumes squared L2
                // directly (d_cos = L2_sq / 2).
                batch_l2sq_f_avx2(query.ptr, query.sz, vptr,
                                  n_unvis, dists);
                _dist_calls += n_unvis;

                for (int i = 0; i < n_unvis; i++) {
                    distT d = dists[i];
                    int32_t n = unvis[i];
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
                }
            } else {
                // Generic path — used for binary/Hamming and any other T.
                for (size_t ni = 0; ni < nv.count; ni++) {
                    int32_t n = nv.ptr[ni];
                    if (vis[n] == ver) continue;
                    vis[n] = ver;
                    distT d = distance_fn(query, _examples[n]);
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
        _adjacency[pidx].assign(level + 1, std::vector<int32_t>{});
        for (int32_t l = 1; l <= level; l++) {
            _adjacency[pidx][l].reserve(_M);
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

        // From min(level, top_level) down to layer 0, build candidates and connect.
        int32_t start_layer = std::min(level, snap_top);
        for (int32_t l = start_layer; l >= 0; l--) {
            auto candidates = search_layer(_examples[pidx], current, _ef_construction, l);
            size_t M_max = (l == 0) ? _M_max0 : _M;

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
                        // Slot full — apply α-heuristic over existing edges + new one.
                        std::vector<DistNode<distT>> nc;
                        nc.reserve(_M_max0 + 1);
                        for (int32_t i = 0; i < cnt; i++) {
                            int32_t n = _layer0_adj[(size_t)c * _M_max0 + i];
                            nc.push_back({distance_fn(_examples[c], _examples[n]), n});
                        }
                        nc.push_back({distance_fn(_examples[c], _examples[pidx]), pidx});
                        auto kept = select_neighbours_heuristic(_examples[c], nc, _M_max0);
                        for (size_t i = 0; i < kept.size(); i++) {
                            _layer0_adj[(size_t)c * _M_max0 + i] = kept[i];
                        }
                        cnt = static_cast<int32_t>(kept.size());
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
    // Returns up to k results sorted nearest → farthest.
    std::vector<DistNode<distT>> search_one(const T& query, size_t k) const {
        int32_t current = _entry_point;
        for (int32_t l = _top_level; l > 0; l--) {
            current = greedy_descent(query, current, l);
        }
        size_t ef = std::max(_ef_search, k);
        auto layer0 = search_layer(query, current, ef, 0);

        std::sort(layer0.begin(), layer0.end(),
                  [](const DistNode<distT>& a, const DistNode<distT>& b) {
                      return a.distance < b.distance;
                  });
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

        int32_t current = _entry_point;
        for (int32_t l = _top_level; l > 0; l--) {
            current = greedy_descent(query, current, l);
        }
        size_t ef = std::max(_ef_search, k);
        auto layer0 = search_layer_with_seeds(query, current, extra_seeds, ef, 0);

        std::sort(layer0.begin(), layer0.end(),
                  [](const DistNode<distT>& a, const DistNode<distT>& b) {
                      return a.distance < b.distance;
                  });
        if (layer0.size() > k) layer0.resize(k);
        return layer0;
    }

private:
    // Like search_layer, but pre-seeds both the candidate queue and the
    // result set with `primary_entry` and every node in `extra_seeds`,
    // de-duplicated.
    std::vector<DistNode<distT>>
    search_layer_with_seeds(const T& query, int32_t primary_entry,
                            const std::vector<int32_t>& extra_seeds,
                            size_t ef, int32_t layer) const {
        const uint32_t ver = next_visited_version();

        DistNodeMin<distT> min_cmp;
        DistNodeMax<distT> max_cmp;

        std::vector<DistNode<distT>> candidates;
        std::vector<DistNode<distT>> results;
        candidates.reserve(ef * 2 + extra_seeds.size() + 8);
        results.reserve(ef + 1);

        auto& vis = visited_buf();
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

        seed(primary_entry);
        for (int32_t s : extra_seeds) seed(s);

        while (!candidates.empty()) {
            DistNode<distT> cur = candidates.front();
            std::pop_heap(candidates.begin(), candidates.end(), min_cmp);
            candidates.pop_back();
            if (results.size() >= ef && cur.distance > results.front().distance) break;
            NeighbourView nv = read_neighbours(cur.node_id, layer);
            if (nv.count == 0) continue;
            for (size_t ni = 0; ni < nv.count; ni++) {
                int32_t n = nv.ptr[ni];
                if (vis[n] == ver) continue;
                vis[n] = ver;
                distT d = distance_fn(query, _examples[n]);
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
            }
        }
        return results;
    }

    // ─── State ──────────────────────────────────────────────────────────────

    size_t _M;
    size_t _M_max0;
    size_t _ef_construction;
    size_t _ef_search;
    double _mL;

    int32_t _entry_point;
    int32_t _top_level;
    size_t _dim;

    std::vector<T> _examples;
    std::vector<float> _flat_backing;            // owns the raw vectors when T = FlatSpan
    std::vector<int32_t> _levels;                // _levels[i] = max layer for node i

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

    std::mt19937 _rng;
    uint64_t _seed_used = 42;
    int _n_threads = 1;

    // Versioned visited lists — one per thread so parallel build can run
    // search_layer concurrently without trampling each other's state.
    // For single-threaded search (post-build) we use slot 0.
    mutable std::vector<std::vector<uint32_t>> _visited_per_thread;
    mutable std::vector<uint32_t> _visited_version_per_thread;

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

    uint32_t next_visited_version() const {
        int tid = omp_get_thread_num();
        auto& ver = _visited_version_per_thread[tid];
        auto& vec = _visited_per_thread[tid];
        ++ver;
        if (ver == 0) {
            std::fill(vec.begin(), vec.end(), 0u);
            ver = 1;
        }
        return ver;
    }

    std::vector<uint32_t>& visited_buf() const {
        return _visited_per_thread[omp_get_thread_num()];
    }

    void init_thread_resources(size_t n) {
        int max_threads = std::max(1, omp_get_max_threads());
        _visited_per_thread.assign(max_threads, std::vector<uint32_t>(n, 0u));
        _visited_version_per_thread.assign(max_threads, 0u);
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
