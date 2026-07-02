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
 * Query (Hamming radius r) — refined pigeonhole allocation
 * ────────────────────────────────────────────────────────
 *   Norouzi/Punjani/Fleet (PAMI 2014): instead of searching every sub-table
 *   at radius floor(r/m), let r' = r / m and a = r % m.  Search sub-tables
 *   0..a at radius r' and sub-tables a+1..m-1 at radius r'-1 (tables whose
 *   radius would be negative contribute nothing and are skipped).
 *
 *   Correctness proof: suppose a code within total distance ≤ r matched NO
 *   searched (table, radius) pair.  Then each of the first a+1 substrings
 *   has distance ≥ r'+1 and each of the remaining m-a-1 substrings has
 *   distance ≥ r', so the total distance would be at least
 *       (a+1)(r'+1) + (m-a-1)·r' = m·r' + a + 1 = r + 1 > r
 *   — a contradiction.  Hence every code within radius r matches at least
 *   one searched (table, radius) pair and the exactness guarantee holds.
 *
 *   For each searched sub-table t (radius r_t):
 *     1. Probe the precomputed XOR flip masks of popcount ≤ r_t
 *        (query-independent, built once per batch): key = qkey ^ mask.
 *     2. For every hit, dedup via a per-thread visited-stamp array and
 *        immediately verify the full Hamming distance against the flat
 *        code buffer, pushing into a bounded top-k max-heap.
 *
 * Complexity
 * ──────────
 *   Candidate collection: Σ_t C(sub_nbits, ≤ r_t)  hash lookups per query
 *   Verification:         O(|unique candidates| × d/64)  POPCNT evaluations
 *   vs. brute force:      O(N × d/64)
 *
 * For d=512, m=8, r=8 (r'=1, a=0): 17 + 7×1 = 24 bucket probes per query
 * (vs. 8×65 = 520 with the naive floor(r/m) allocation on 64-bit substrings).
 */

#include <DistanceFunctions.hpp>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(ENABLE_OMP_PARALLEL)
#include <omp.h>
#endif

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
        _flat.clear();
        _tables.clear();
        _n = 0;
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

        // Flat, contiguous code storage: one buffer, byte stride = _nbytes.
        // Avoids a pointer chase per verified candidate.
        _flat.resize(_n * _nbytes);
        for (size_t i = 0; i < _n; ++i)
            std::memcpy(_flat.data() + i * _nbytes, data[i].data(), _nbytes);

        // Dispatch the Hamming kernel on code width once (not per distance).
        _dist_fn = _select_kernel(_nbytes);

        _tables.assign((size_t)_m, {});
        for (size_t i = 0; i < _n; ++i) {
            const uint8_t* row = _flat.data() + i * _nbytes;
            for (int32_t t = 0; t < _m; ++t)
                _tables[(size_t)t][_extract_key(row, t)].push_back((int32_t)i);
        }

        _init_scratch_pool();
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
        if (nq == 0 || k == 0 || _n == 0 || _tables.empty() || radius < 0)
            return {std::move(all_idx), std::move(all_dist)};

        // Refined pigeonhole radius allocation (see header comment for proof):
        // tables 0..a search at radius r', tables a+1..m-1 at radius r'-1.
        const int32_t r_prime = radius / _m;
        const int32_t a = radius % _m;

        // Precompute the query-independent XOR flip masks once per batch,
        // grouped by ascending popcount so that the masks for radius w are
        // exactly the first mask_prefix[w] entries.  A table searched at
        // radius r_t probes table.find(qkey ^ masks[i]) for i < prefix[r_t].
        const int32_t w_max = std::min<int32_t>(r_prime, (int32_t)_sub_nbits);
        std::vector<uint64_t> masks;
        std::vector<size_t> mask_prefix((size_t)w_max + 1);
        for (int32_t w = 0; w <= w_max; ++w) {
            _gen_masks((int)_sub_nbits, w, 0, 0, masks);
            mask_prefix[(size_t)w] = masks.size();
        }

        // OpenMP 2.0 (the version MSVC ships) requires a SIGNED integral
        // loop variable. Use ptrdiff_t and cast nq once.
        const std::ptrdiff_t nq_signed = static_cast<std::ptrdiff_t>(nq);
#if defined(ENABLE_OMP_PARALLEL)
        #pragma omp parallel if (nq_signed > 1)
        {
            // One visited-stamp scratch per participating thread, leased from
            // a persistent pool.  Slot acquisition is an atomic test-and-set,
            // so this stays correct even when searchKNN is itself called from
            // an enclosing OMP parallel loop (nested single-thread teams all
            // report omp_get_thread_num() == 0 and must not share a slot).
            _ScratchSlot* slot = _try_acquire_slot((size_t)omp_get_thread_num());
            std::unique_ptr<_Scratch> local;
            _Scratch* scr = slot ? &slot->s : (local = std::unique_ptr<_Scratch>(new _Scratch())).get();
            _prepare_scratch(*scr);

            #pragma omp for schedule(dynamic)
            for (std::ptrdiff_t qi = 0; qi < nq_signed; ++qi)
                _search_one(queries[(size_t)qi], k, r_prime, a, w_max,
                            masks, mask_prefix, *scr,
                            all_idx[(size_t)qi], all_dist[(size_t)qi]);

            if (slot) slot->busy.clear(std::memory_order_release);
        }
#else
        _ScratchSlot* slot = _try_acquire_slot(0);
        std::unique_ptr<_Scratch> local;
        _Scratch* scr = slot ? &slot->s : (local = std::unique_ptr<_Scratch>(new _Scratch())).get();
        _prepare_scratch(*scr);

        for (std::ptrdiff_t qi = 0; qi < nq_signed; ++qi)
            _search_one(queries[(size_t)qi], k, r_prime, a, w_max,
                        masks, mask_prefix, *scr,
                        all_idx[(size_t)qi], all_dist[(size_t)qi]);

        if (slot) slot->busy.clear(std::memory_order_release);
#endif
        return {std::move(all_idx), std::move(all_dist)};
    }

    int32_t m()     const { return _m; }
    size_t  n()     const { return _n; }
    size_t  nbytes() const { return _nbytes; }

private:
    // ── Per-thread visited-stamp scratch ─────────────────────────────────────
    // stamp[i] == epoch  ⇔  candidate i already verified for the current query.
    // Bumping epoch resets the whole array in O(1); on uint32 wraparound the
    // array is zero-filled once and epoch restarts at 1.
    struct _Scratch {
        std::vector<uint32_t> stamp;
        uint32_t epoch = 0;
    };
    struct _ScratchSlot {
        std::atomic_flag busy = ATOMIC_FLAG_INIT;
        _Scratch s;
    };

    using DistFn = int64_t (*)(const uint8_t*, const uint8_t*, size_t);

    int32_t _m;
    size_t  _nbytes = 0, _sub_nbytes = 0, _sub_nbits = 0, _n = 0;
    std::vector<uint8_t> _flat; // contiguous codes, row stride = _nbytes
    DistFn  _dist_fn = &_dk_u8; // selected once in set()
    std::vector<std::unordered_map<uint64_t, std::vector<int32_t>>> _tables;
    mutable std::vector<std::unique_ptr<_ScratchSlot>> _scratch_pool;

    // ── Per-query search body (shared by the OMP and serial paths) ──────────
    void _search_one(const arrayli& query, size_t k,
                     int32_t r_prime, int32_t a, int32_t w_max,
                     const std::vector<uint64_t>& masks,
                     const std::vector<size_t>& mask_prefix,
                     _Scratch& scr,
                     std::vector<int64_t>& out_idx,
                     std::vector<int64_t>& out_dist) const {
        if (++scr.epoch == 0) { // uint32 wraparound: hard reset
            std::fill(scr.stamp.begin(), scr.stamp.end(), 0u);
            scr.epoch = 1;
        }
        const uint32_t ep = scr.epoch;
        const uint8_t* q = query.data();

        using Elem = std::pair<int64_t, int64_t>; // (distance, orig_idx)
        std::priority_queue<Elem> heap;

        for (int32_t t = 0; t < _m; ++t) {
            const int32_t r_t = (t <= a) ? r_prime : r_prime - 1;
            if (r_t < 0) break; // remaining tables all have radius r'-1 < 0

            const uint64_t qkey = _extract_key(q, t);
            const auto& table = _tables[(size_t)t];
            const size_t nmasks = mask_prefix[(size_t)std::min(r_t, w_max)];

            for (size_t mi = 0; mi < nmasks; ++mi) {
                auto it = table.find(qkey ^ masks[mi]);
                if (it == table.end()) continue;
                for (int32_t idx : it->second) {
                    if (scr.stamp[(size_t)idx] == ep) continue; // already verified
                    scr.stamp[(size_t)idx] = ep;
                    // Fused verification: compute the distance on first
                    // sighting instead of a second pass over a candidate set.
                    const int64_t d = _dist_fn(
                        q, _flat.data() + (size_t)idx * _nbytes, _nbytes);
                    if (heap.size() < k || d < heap.top().first) {
                        heap.push({d, (int64_t)idx});
                        if (heap.size() > k) heap.pop();
                    }
                }
            }
        }

        // ── Extract results in ascending distance order ──────────────────────
        out_idx.resize(heap.size());
        out_dist.resize(heap.size());
        for (size_t pos = heap.size(); pos-- > 0;) {
            out_idx[pos]  = heap.top().second;
            out_dist[pos] = heap.top().first;
            heap.pop();
        }
    }

    // ── Scratch pool management ──────────────────────────────────────────────
    void _init_scratch_pool() {
#if defined(ENABLE_OMP_PARALLEL)
        const size_t nslots = (size_t)std::max(1, omp_get_max_threads());
#else
        const size_t nslots = 1;
#endif
        while (_scratch_pool.size() < nslots)
            _scratch_pool.emplace_back(new _ScratchSlot());
    }

    _ScratchSlot* _try_acquire_slot(size_t hint) const {
        const size_t nslots = _scratch_pool.size();
        for (size_t i = 0; i < nslots; ++i) {
            _ScratchSlot* s = _scratch_pool[(hint + i) % nslots].get();
            if (!s->busy.test_and_set(std::memory_order_acquire)) return s;
        }
        return nullptr; // pool exhausted (rare) — caller allocates a local one
    }

    void _prepare_scratch(_Scratch& s) const {
        if (s.stamp.size() != _n) {
            s.stamp.assign(_n, 0u);
            s.epoch = 0;
        }
    }

    // ── Hamming kernels over the flat buffer (dispatched once in set()) ─────
    static int64_t _dk_512(const uint8_t* a, const uint8_t* b, size_t) {
        return hamming_u64<512>(reinterpret_cast<const uint64_t*>(a),
                                reinterpret_cast<const uint64_t*>(b));
    }
    static int64_t _dk_256(const uint8_t* a, const uint8_t* b, size_t) {
        return hamming_u64<256>(reinterpret_cast<const uint64_t*>(a),
                                reinterpret_cast<const uint64_t*>(b));
    }
    static int64_t _dk_128(const uint8_t* a, const uint8_t* b, size_t) {
        return hamming_u64<128>(reinterpret_cast<const uint64_t*>(a),
                                reinterpret_cast<const uint64_t*>(b));
    }
    static int64_t _dk_64(const uint8_t* a, const uint8_t* b, size_t) {
        return hamming_u64<64>(reinterpret_cast<const uint64_t*>(a),
                               reinterpret_cast<const uint64_t*>(b));
    }
    static int64_t _dk_32(const uint8_t* a, const uint8_t* b, size_t) {
        return (int64_t)hamming_u32<32>(reinterpret_cast<const uint32_t*>(a),
                                        reinterpret_cast<const uint32_t*>(b));
    }
    static int64_t _dk_u64(const uint8_t* a, const uint8_t* b, size_t nbytes) {
        const uint64_t* pa = reinterpret_cast<const uint64_t*>(a);
        const uint64_t* pb = reinterpret_cast<const uint64_t*>(b);
        const size_t nwords = nbytes / 8;
        int64_t h = 0;
        for (size_t i = 0; i < nwords; ++i)
            h += PYNEAR_POPCNT64(pa[i] ^ pb[i]);
        return h;
    }
    static int64_t _dk_u8(const uint8_t* a, const uint8_t* b, size_t nbytes) {
        int64_t h = 0;
        for (size_t i = 0; i < nbytes; ++i)
            h += PYNEAR_POPCNT32(static_cast<uint32_t>(a[i] ^ b[i]));
        return h;
    }

    static DistFn _select_kernel(size_t nbytes) {
        switch (nbytes) {
        case 64: return &_dk_512;
        case 32: return &_dk_256;
        case 16: return &_dk_128;
        case 8:  return &_dk_64;
        case 4:  return &_dk_32;
        default: break;
        }
        return (nbytes % 8 == 0) ? &_dk_u64 : &_dk_u8;
    }

    // Extract the t-th sub-string of a code (raw row pointer) as a uint64_t key.
    inline uint64_t _extract_key(const uint8_t* row, int32_t t) const {
        uint64_t key = 0;
        std::memcpy(&key, row + (size_t)t * _sub_nbytes, _sub_nbytes);
        return key;
    }

    /*
     * Generate all XOR masks of exactly `weight` bits set among the low
     * `nbits` bits, appended to `out` in deterministic lexicographic order.
     * Called for weight = 0..w_max so `out` is grouped by ascending popcount;
     * the masks for radius r are then exactly the first mask_prefix[r] entries.
     *
     * Count appended for weight w: C(nbits, w).
     */
    static void _gen_masks(int nbits, int weight, int start_bit, uint64_t cur,
                           std::vector<uint64_t>& out) {
        if (weight == 0) {
            out.push_back(cur);
            return;
        }
        for (int b = start_bit; b <= nbits - weight; ++b)
            _gen_masks(nbits, weight - 1, b + 1, cur | (uint64_t(1) << b), out);
    }
};
