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
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

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
    HNSWIndex(size_t M = 16, size_t ef_construction = 200, size_t ef_search = 50, uint64_t seed = 42)
        : _M(M),
          _M_max0(2 * M),
          _ef_construction(ef_construction),
          _ef_search(ef_search),
          _mL(1.0 / std::log(static_cast<double>(M))),
          _entry_point(-1),
          _top_level(-1),
          _dim(0),
          _rng(seed) {
        if (M < 2) throw std::invalid_argument("M must be >= 2");
        if (ef_construction < 1) throw std::invalid_argument("ef_construction must be >= 1");
    }

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

        for (size_t i = 0; i < _examples.size(); i++) {
            add_point(static_cast<int32_t>(i));
        }
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

        for (size_t i = 0; i < _adjacency.size(); i++) {
            adj_offsets.push_back(static_cast<int32_t>(flat_adj.size()));
            int32_t before = static_cast<int32_t>(flat_adj.size());
            int32_t nlayers = static_cast<int32_t>(_adjacency[i].size());
            for (int32_t l = 0; l < nlayers; l++) {
                flat_adj.push_back(static_cast<int32_t>(_adjacency[i][l].size()));
            }
            for (int32_t l = 0; l < nlayers; l++) {
                for (int32_t e : _adjacency[i][l]) flat_adj.push_back(e);
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
                _adjacency[i][l].reserve(layer_sizes[l]);
                for (int32_t e = 0; e < layer_sizes[l]; e++) {
                    _adjacency[i][l].push_back(flat_adj[cursor++]);
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
        _entry_point = -1;
        _top_level = -1;
        _dim = 0;
    }

private:
    // ─── Algorithm core ─────────────────────────────────────────────────────

    int32_t random_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r;
        do { r = u(_rng); } while (r <= 0.0);  // avoid log(0)
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
            if (layer >= static_cast<int32_t>(_adjacency[current].size())) break;
            const auto& neighbours = _adjacency[current][layer];
            for (int32_t n : neighbours) {
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

    // Beam search at a given layer. Returns up to ef candidates ordered by
    // distance (any order in the heap; caller sorts).
    std::vector<DistNode<distT>>
    search_layer(const T& query, int32_t entry, size_t ef, int32_t layer) const {
        std::unordered_set<int32_t> visited;
        visited.reserve(ef * 4);

        // Candidates min-heap (explore nearest first)
        std::priority_queue<DistNode<distT>,
                            std::vector<DistNode<distT>>,
                            DistNodeMin<distT>> candidates;
        // Results max-heap (drop farthest when we exceed ef)
        std::priority_queue<DistNode<distT>,
                            std::vector<DistNode<distT>>,
                            DistNodeMax<distT>> results;

        distT d_entry = distance_fn(query, _examples[entry]);
        candidates.push({d_entry, entry});
        results.push({d_entry, entry});
        visited.insert(entry);

        while (!candidates.empty()) {
            DistNode<distT> cur = candidates.top();
            candidates.pop();
            // Stopping condition: if current candidate is farther than the worst
            // result and we already have ef results, no further exploration helps.
            if (results.size() >= ef && cur.distance > results.top().distance) {
                break;
            }

            if (layer >= static_cast<int32_t>(_adjacency[cur.node_id].size())) continue;
            const auto& neighbours = _adjacency[cur.node_id][layer];

            // Prefetch the next few neighbours' vectors to reduce L3 misses.
#if defined(__AVX__) || defined(__AVX2__)
            for (size_t pi = 0; pi < neighbours.size() && pi < 4; pi++) {
                if constexpr (std::is_same_v<T, FlatSpan>) {
                    _mm_prefetch(reinterpret_cast<const char*>(_examples[neighbours[pi]].ptr),
                                 _MM_HINT_T0);
                }
            }
#endif

            for (size_t ni = 0; ni < neighbours.size(); ni++) {
                int32_t n = neighbours[ni];
                if (!visited.insert(n).second) continue;
#if defined(__AVX__) || defined(__AVX2__)
                if (ni + 4 < neighbours.size()) {
                    if constexpr (std::is_same_v<T, FlatSpan>) {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(_examples[neighbours[ni + 4]].ptr),
                            _MM_HINT_T0);
                    }
                }
#endif
                distT d = distance_fn(query, _examples[n]);
                if (results.size() < ef || d < results.top().distance) {
                    candidates.push({d, n});
                    results.push({d, n});
                    if (results.size() > ef) results.pop();
                }
            }
        }

        std::vector<DistNode<distT>> out;
        out.reserve(results.size());
        while (!results.empty()) {
            out.push_back(results.top());
            results.pop();
        }
        return out;  // farthest-first (popped from max-heap)
    }

    // α-heuristic from §4 of the HNSW paper. Selects up to M items from
    // candidates such that for every selected pair (p, q), the inserted point
    // is closer to both than they are to each other — biases toward
    // long-range, "navigable" edges.
    std::vector<int32_t>
    select_neighbours_heuristic(const T& query,
                                std::vector<DistNode<distT>> candidates,
                                size_t M) const {
        if (candidates.size() <= M) {
            std::vector<int32_t> ids;
            ids.reserve(candidates.size());
            // Sort nearest-first
            std::sort(candidates.begin(), candidates.end(),
                      [](const DistNode<distT>& a, const DistNode<distT>& b) {
                          return a.distance < b.distance;
                      });
            for (auto& c : candidates) ids.push_back(c.node_id);
            return ids;
        }

        // Sort nearest-first and apply the α-heuristic.
        std::sort(candidates.begin(), candidates.end(),
                  [](const DistNode<distT>& a, const DistNode<distT>& b) {
                      return a.distance < b.distance;
                  });

        std::vector<int32_t> selected;
        selected.reserve(M);
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
            if (keep) selected.push_back(c.node_id);
        }
        return selected;
    }

    // Insert one point at random level, link it via the α-heuristic.
    void add_point(int32_t pidx) {
        int32_t level = random_level();
        _levels[pidx] = level;
        _adjacency[pidx].assign(level + 1, std::vector<int32_t>{});

        if (_entry_point < 0) {
            _entry_point = pidx;
            _top_level = level;
            return;
        }

        // Greedy descent from the top of the index down to level + 1.
        int32_t current = _entry_point;
        for (int32_t l = _top_level; l > level; l--) {
            current = greedy_descent(_examples[pidx], current, l);
        }

        // From min(level, top_level) down to layer 0, build candidates and connect.
        int32_t start_layer = std::min(level, _top_level);
        for (int32_t l = start_layer; l >= 0; l--) {
            auto candidates = search_layer(_examples[pidx], current, _ef_construction, l);
            size_t M_max = (l == 0) ? _M_max0 : _M;

            std::vector<int32_t> chosen = select_neighbours_heuristic(_examples[pidx],
                                                                     candidates,
                                                                     _M);

            // Link new node → chosen.
            _adjacency[pidx][l] = chosen;

            // Link reverse edges; prune neighbour's adjacency if it now exceeds M_max.
            for (int32_t c : chosen) {
                auto& neigh_adj = _adjacency[c][l];
                neigh_adj.push_back(pidx);
                if (neigh_adj.size() > M_max) {
                    std::vector<DistNode<distT>> nc;
                    nc.reserve(neigh_adj.size());
                    for (int32_t n : neigh_adj) {
                        nc.push_back({distance_fn(_examples[c], _examples[n]), n});
                    }
                    _adjacency[c][l] = select_neighbours_heuristic(_examples[c], nc, M_max);
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

        if (level > _top_level) {
            _top_level = level;
            _entry_point = pidx;
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
        std::unordered_set<int32_t> visited;
        visited.reserve(ef * 4 + extra_seeds.size() + 1);

        std::priority_queue<DistNode<distT>,
                            std::vector<DistNode<distT>>,
                            DistNodeMin<distT>> candidates;
        std::priority_queue<DistNode<distT>,
                            std::vector<DistNode<distT>>,
                            DistNodeMax<distT>> results;

        auto seed = [&](int32_t node) {
            if (node < 0 || node >= (int32_t)_examples.size()) return;
            if (!visited.insert(node).second) return;
            distT d = distance_fn(query, _examples[node]);
            candidates.push({d, node});
            if (results.size() < ef || d < results.top().distance) {
                results.push({d, node});
                if (results.size() > ef) results.pop();
            }
        };

        seed(primary_entry);
        for (int32_t s : extra_seeds) seed(s);

        while (!candidates.empty()) {
            DistNode<distT> cur = candidates.top();
            candidates.pop();
            if (results.size() >= ef && cur.distance > results.top().distance) break;
            if (layer >= static_cast<int32_t>(_adjacency[cur.node_id].size())) continue;
            const auto& neighbours = _adjacency[cur.node_id][layer];
            for (int32_t n : neighbours) {
                if (!visited.insert(n).second) continue;
                distT d = distance_fn(query, _examples[n]);
                if (results.size() < ef || d < results.top().distance) {
                    candidates.push({d, n});
                    results.push({d, n});
                    if (results.size() > ef) results.pop();
                }
            }
        }

        std::vector<DistNode<distT>> out;
        out.reserve(results.size());
        while (!results.empty()) {
            out.push_back(results.top());
            results.pop();
        }
        return out;
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
    std::vector<std::vector<std::vector<int32_t>>> _adjacency;
        // _adjacency[node][layer] = neighbour IDs at that layer

    std::mt19937 _rng;
    uint64_t _seed_used = 42;
};

}  // namespace hnsw
