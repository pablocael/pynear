#pragma once
/*
 * IVFFlatBinaryIndex — Inverted File Index for binary (Hamming) descriptors.
 *
 * Build
 * ─────
 *   1. Binary k-means with K-Means++ initialisation.
 *      Centroids are majority-vote bit-strings.
 *   2. Assign every descriptor to its nearest centroid.
 *   3. Build per-cluster inverted lists: cluster → [point indices].
 *
 * Search
 * ──────
 *   1. Compute Hamming distance from query to every centroid.
 *   2. Probe the nprobe nearest clusters.
 *   3. Linear scan those clusters with POPCNT; collect top-k via max-heap.
 *
 * Complexity
 * ──────────
 *   Build   O(iter × N × k × d/64)  Hamming distance evaluations
 *   Query   O(k + nprobe × cluster_size × d/64)
 *
 * where d = descriptor width in bits, N = database size, k = nlist.
 */

#include <DistanceFunctions.hpp>
#include <algorithm>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

class IVFFlatBinaryIndex {
public:
    /*
     * nlist    – number of clusters (Voronoi cells)
     * nprobe   – clusters scanned per query (accuracy ↑ as nprobe ↑)
     * max_iter – maximum k-means iterations
     * seed     – RNG seed for k-means++ initialisation
     */
    explicit IVFFlatBinaryIndex(int32_t nlist   = 256,
                                int32_t nprobe  = 8,
                                int32_t max_iter = 20,
                                uint32_t seed   = 42)
        : _nlist(nlist), _nprobe(nprobe), _max_iter(max_iter), _seed(seed) {}

    /* Add vectors to the index (replaces any existing content). */
    void set(const ndarrayli& data) {
        if (data.empty()) {
            _db.clear();
            _centroids.clear();
            _invlists.clear();
            _nbytes = 0;
            return;
        }
        _nbytes = data[0].size();
        _db     = data;
        _build();
    }

    /* Batch top-k search.  Returns (indices, distances). */
    std::tuple<std::vector<std::vector<int64_t>>,
               std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k) const {
        size_t nq = queries.size();
        std::vector<std::vector<int64_t>> all_idx(nq), all_dist(nq);

        int32_t nprobe = std::min(_nprobe, (int32_t)_centroids.size());

        for (size_t qi = 0; qi < nq; ++qi) {
            // ── Find nprobe nearest centroids ────────────────────────────────
            int32_t nc = (int32_t)_centroids.size();
            std::vector<std::pair<int64_t, int32_t>> cdists(nc);
            for (int32_t c = 0; c < nc; ++c)
                cdists[c] = {dist_hamming(queries[qi], _centroids[c]), c};
            std::partial_sort(cdists.begin(), cdists.begin() + nprobe,
                              cdists.end());

            // ── Scan chosen clusters, keep top-k in a max-heap ───────────────
            using Elem = std::pair<int64_t, int64_t>; // (distance, original_idx)
            std::priority_queue<Elem> heap;

            for (int32_t p = 0; p < nprobe; ++p) {
                int32_t c = cdists[p].second;
                for (int32_t idx : _invlists[c]) {
                    int64_t d = dist_hamming(queries[qi], _db[idx]);
                    if ((int64_t)heap.size() < (int64_t)k ||
                        d < heap.top().first) {
                        heap.push({d, (int64_t)idx});
                        if (heap.size() > k) heap.pop();
                    }
                }
            }

            // ── Extract results in ascending distance order ──────────────────
            std::vector<int64_t> idxs, dists;
            idxs.reserve(heap.size());
            dists.reserve(heap.size());
            while (!heap.empty()) {
                idxs.push_back(heap.top().second);
                dists.push_back(heap.top().first);
                heap.pop();
            }
            std::reverse(idxs.begin(), idxs.end());
            std::reverse(dists.begin(), dists.end());
            all_idx[qi]  = std::move(idxs);
            all_dist[qi] = std::move(dists);
        }
        return {std::move(all_idx), std::move(all_dist)};
    }

    int32_t nlist()  const { return _nlist; }
    int32_t nprobe() const { return _nprobe; }
    void set_nprobe(int32_t nprobe) { _nprobe = nprobe; }

private:
    int32_t  _nlist, _nprobe, _max_iter;
    uint32_t _seed;
    size_t   _nbytes = 0;

    ndarrayli _db;
    ndarrayli _centroids;
    std::vector<std::vector<int32_t>> _invlists;

    // ── Binary k-means with K-Means++ initialisation ─────────────────────────
    void _build() {
        size_t n = _db.size();
        int32_t k = std::min(_nlist, (int32_t)n);

        std::mt19937 rng(_seed);
        std::uniform_int_distribution<size_t> pick(0, n - 1);

        // ── K-Means++ init ───────────────────────────────────────────────────
        _centroids.resize(k);
        _centroids[0] = _db[pick(rng)];

        std::vector<int64_t> min_d(n, std::numeric_limits<int64_t>::max());

        for (int32_t ci = 1; ci < k; ++ci) {
            const arrayli& prev = _centroids[ci - 1];
            for (size_t i = 0; i < n; ++i) {
                int64_t d = dist_hamming(_db[i], prev);
                if (d < min_d[i]) min_d[i] = d;
            }
            int64_t total = 0;
            for (int64_t v : min_d) total += v;

            size_t chosen = pick(rng);
            if (total > 0) {
                std::uniform_int_distribution<int64_t> wsel(0, total - 1);
                int64_t r = wsel(rng), cum = 0;
                for (size_t i = 0; i < n; ++i) {
                    cum += min_d[i];
                    if (cum > r) { chosen = i; break; }
                }
            }
            _centroids[ci] = _db[chosen];
        }

        // ── Lloyd iterations ─────────────────────────────────────────────────
        std::vector<int32_t> labels(n, 0);
        // bit_counts[c][b*8 + bit] = number of assigned points with that bit = 1
        std::vector<std::vector<int32_t>> bit_counts(
            k, std::vector<int32_t>(_nbytes * 8, 0));
        std::vector<int32_t> cluster_counts(k, 0);

        for (int32_t iter = 0; iter < _max_iter; ++iter) {
            int64_t n_changed = 0;

            // Assignment step (parallelised when OMP is available)
#ifdef ENABLE_OMP_PARALLEL
            #pragma omp parallel for schedule(static) reduction(+:n_changed)
#endif
            for (int64_t i = 0; i < (int64_t)n; ++i) {
                int64_t best_d = std::numeric_limits<int64_t>::max();
                int32_t best_c = 0;
                for (int32_t c = 0; c < k; ++c) {
                    int64_t d = dist_hamming(_db[i], _centroids[c]);
                    if (d < best_d) { best_d = d; best_c = c; }
                }
                if (labels[i] != best_c) {
                    labels[i] = best_c;
                    ++n_changed;
                }
            }
            if (n_changed == 0) break;

            // Update step: majority vote per bit
            for (auto& bc : bit_counts)
                std::fill(bc.begin(), bc.end(), 0);
            std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

            for (size_t i = 0; i < n; ++i) {
                int32_t c = labels[i];
                ++cluster_counts[c];
                for (size_t b = 0; b < _nbytes; ++b) {
                    uint8_t byte = _db[i][b];
                    for (int bit = 0; bit < 8; ++bit)
                        if ((byte >> bit) & 1)
                            ++bit_counts[c][b * 8 + bit];
                }
            }

            for (int32_t c = 0; c < k; ++c) {
                arrayli& cent = _centroids[c];
                cent.assign(_nbytes, 0);
                if (cluster_counts[c] == 0) {
                    cent = _db[pick(rng)];
                    continue;
                }
                int32_t half = cluster_counts[c] / 2;
                for (size_t b = 0; b < _nbytes; ++b) {
                    uint8_t byte = 0;
                    for (int bit = 0; bit < 8; ++bit)
                        if (bit_counts[c][b * 8 + bit] > half)
                            byte |= (uint8_t)(1 << bit);
                    cent[b] = byte;
                }
            }
        }

        // ── Build inverted lists ─────────────────────────────────────────────
        _invlists.assign(k, {});
        for (size_t i = 0; i < n; ++i)
            _invlists[labels[i]].push_back((int32_t)i);
    }
};
