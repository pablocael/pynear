#pragma once
/*
 * MIHBinaryIndex — Multi-Index Hashing for binary (Hamming) descriptors.
 *
 * Algorithm
 * ─────────
 *   Each d-bit descriptor is split into m sub-strings of (d/m) bits.
 *   m hash tables are built: sub-string key → list of point indices.
 *
 * Build
 * ─────
 *   For each descriptor, insert its m sub-strings into the m tables.
 *   Complexity: O(N × m)
 *
 * Query (Hamming radius r)
 * ────────────────────────
 *   Pigeonhole principle: any true neighbour at distance ≤ r must have at
 *   least one sub-string at distance ≤ floor(r / m) from the query.
 *
 *   For each sub-table t:
 *     1. Enumerate all sub-string keys within Hamming distance r_sub = floor(r/m).
 *     2. Look them up in table t; collect matching point indices as candidates.
 *   Union all candidate sets.
 *   Verify each candidate's full Hamming distance; return top-k.
 *
 * Constraints
 * ───────────
 *   sub-string width = nbytes / m  must be ≤ 8 (fits in a uint64_t key).
 *   For typical image descriptors: d=512 → nbytes=64, m=8 → sub_nbytes=8.
 *
 * Complexity
 * ──────────
 *   Candidate collection: O(m × C(sub_nbits, r_sub))  hash lookups per query
 *   Verification:         O(|candidates| × d/64)       POPCNT evaluations
 *   vs. brute force:      O(N × d/64)
 *
 * For d=512, m=8, r=8 (r_sub=1): 8×65 = 520 hash lookups per sub-table pass.
 */

#include <DistanceFunctions.hpp>
#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class MIHBinaryIndex {
public:
    /*
     * m – number of sub-strings (must evenly divide the descriptor byte width).
     *     For best performance, choose m such that nbytes/m ≤ 8 (fits uint64_t).
     *     Recommended: m=8 for 512-bit, m=4 for 256-bit, m=4 for 128-bit.
     */
    explicit MIHBinaryIndex(int32_t m = 8) : _m(m) {}

    /* Add vectors to the index (replaces any existing content). */
    void set(const ndarrayli& data) {
        _db.clear();
        _tables.clear();
        if (data.empty()) return;

        _nbytes = data[0].size();
        if ((int32_t)_nbytes < _m || _nbytes % (size_t)_m != 0)
            throw std::invalid_argument(
                "MIH: descriptor byte width must be divisible by m");
        _sub_nbytes = _nbytes / (size_t)_m;
        if (_sub_nbytes > 8)
            throw std::invalid_argument(
                "MIH: sub-string width exceeds 8 bytes (uint64_t key capacity). "
                "Increase m so that nbytes/m ≤ 8.");
        _sub_nbits = _sub_nbytes * 8;
        _n = data.size();

        _db = data;
        _tables.assign((size_t)_m, {});

        for (size_t i = 0; i < _n; ++i)
            for (int32_t t = 0; t < _m; ++t)
                _tables[(size_t)t][_extract_key(data[i], t)].push_back((int32_t)i);
    }

    /*
     * Batch approximate top-k search.
     *
     * radius – Hamming radius for candidate enumeration.
     *          Any true neighbour at distance ≤ radius is retrieved with
     *          probability 1 (exact guarantee via pigeonhole).
     *          Larger radius → higher recall, more candidates, slower.
     *
     * Returns (indices, distances).  Distances are Hamming (integer).
     * May return fewer than k results when fewer candidates pass the radius.
     */
    std::tuple<std::vector<std::vector<int64_t>>,
               std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k, int32_t radius = 8) const {
        size_t nq = queries.size();
        std::vector<std::vector<int64_t>> all_idx(nq), all_dist(nq);

        int32_t r_sub = radius / _m; // pigeonhole radius per sub-table

        for (size_t qi = 0; qi < nq; ++qi) {
            // ── Collect candidates from all sub-tables ───────────────────────
            std::unordered_set<int32_t> candidates;

            std::vector<uint64_t> neighbor_keys;
            for (int32_t t = 0; t < _m; ++t) {
                uint64_t qkey = _extract_key(queries[qi], t);
                neighbor_keys.clear();
                _enumerate_neighbors(qkey, (int)_sub_nbits, r_sub,
                                     neighbor_keys, 0);

                const auto& table = _tables[(size_t)t];
                for (uint64_t nk : neighbor_keys) {
                    auto it = table.find(nk);
                    if (it != table.end())
                        for (int32_t idx : it->second)
                            candidates.insert(idx);
                }
            }

            // ── Verify candidates; keep top-k in a max-heap ──────────────────
            using Elem = std::pair<int64_t, int64_t>; // (distance, orig_idx)
            std::priority_queue<Elem> heap;

            for (int32_t idx : candidates) {
                int64_t d = dist_hamming(queries[qi], _db[(size_t)idx]);
                if ((int64_t)heap.size() < (int64_t)k ||
                    d < heap.top().first) {
                    heap.push({d, (int64_t)idx});
                    if (heap.size() > k) heap.pop();
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

    int32_t m()     const { return _m; }
    size_t  n()     const { return _n; }
    size_t  nbytes() const { return _nbytes; }

private:
    int32_t _m;
    size_t  _nbytes = 0, _sub_nbytes = 0, _sub_nbits = 0, _n = 0;
    ndarrayli _db;
    std::vector<std::unordered_map<uint64_t, std::vector<int32_t>>> _tables;

    // Extract the t-th sub-string of the descriptor as a uint64_t key.
    inline uint64_t _extract_key(const arrayli& vec, int32_t t) const {
        uint64_t key = 0;
        std::memcpy(&key, vec.data() + (size_t)t * _sub_nbytes, _sub_nbytes);
        return key;
    }

    /*
     * Enumerate all uint64_t values within Hamming distance ≤ radius from key
     * (only the low sub_nbits bits are significant).
     *
     * Uses the recursive combination approach:
     *   base case:       push key (0 extra flips)
     *   recursive case:  for each bit b ≥ start_bit, flip b and recurse at
     *                    radius-1 starting from b+1.
     *
     * Result count: sum_{i=0}^{radius} C(sub_nbits, i).
     * For sub_nbits=64, radius=0: 1
     *                   radius=1: 65
     *                   radius=2: 2081
     */
    static void _enumerate_neighbors(
        uint64_t key, int sub_nbits, int radius,
        std::vector<uint64_t>& out, int start_bit)
    {
        out.push_back(key);
        if (radius == 0) return;
        for (int b = start_bit; b < sub_nbits; ++b)
            _enumerate_neighbors(key ^ (uint64_t(1) << b),
                                 sub_nbits, radius - 1, out, b + 1);
    }
};
