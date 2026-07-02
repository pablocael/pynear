#pragma once
/*
 * IVFFlatBinaryIndex — Inverted File Index for binary (Hamming) descriptors.
 *
 * Build
 * ─────
 *   1. Binary k-means with K-Means++ initialisation.
 *      Centroids are majority-vote bit-strings.
 *   2. Assign every descriptor to its nearest centroid.
 *   3. Store codes in ONE flat contiguous buffer, reordered by cluster
 *      (Faiss-style). Inverted lists are (offset, count) ranges into that
 *      buffer plus a position → original-id remap, so probing a cluster is
 *      a sequential stream instead of a pointer chase.
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
#include <cstdint>
#include <cstring>
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
            _codes.clear();
            _ids.clear();
            _list_offsets.clear();
            _nbytes = 0;
            _nwords = 0;
            _ntail  = 0;
            return;
        }
        _nbytes = data[0].size();
        // Resolve the Hamming kernel shape once: whole 64-bit words + byte
        // tail. The scan then calls a raw-pointer popcount loop directly
        // instead of going through the per-call dist_hamming dispatch.
        _nwords = _nbytes / 8;
        _ntail  = _nbytes % 8;
        _db     = data;
        _build();
        // Codes now live (cluster-ordered) in _codes; drop the row-major copy.
        _db.clear();
        _db.shrink_to_fit();
    }

    /* Batch top-k search.  Returns (indices, distances). */
    std::tuple<std::vector<std::vector<int64_t>>,
               std::vector<std::vector<int64_t>>>
    searchKNN(const ndarrayli& queries, size_t k) const {
        size_t nq = queries.size();
        std::vector<std::vector<int64_t>> all_idx(nq), all_dist(nq);

        const int32_t nprobe = std::min(_nprobe, (int32_t)_centroids.size());
        const int32_t nc     = (int32_t)_centroids.size();
        const size_t  nbytes = _nbytes;
        const size_t  nwords = _nwords;
        const size_t  ntail  = _ntail;
        const uint8_t* codes = _codes.data();

        // Queries only touch read-only index state (_centroids, _codes, _ids,
        // _list_offsets) and write to their own all_idx/all_dist slots.
        //
        // OpenMP 2.0 (the version MSVC ships) requires a SIGNED integral
        // loop variable. Use ptrdiff_t and cast nq once.
        const std::ptrdiff_t nq_signed = static_cast<std::ptrdiff_t>(nq);
#if defined(ENABLE_OMP_PARALLEL)
        #pragma omp parallel if (nq_signed > 1)
#endif
        {
            // Per-thread scratch, reused across queries (was a per-query
            // allocation).
            std::vector<std::pair<int64_t, int32_t>> cdists(nc);

#if defined(ENABLE_OMP_PARALLEL)
            #pragma omp for schedule(dynamic)
#endif
            for (std::ptrdiff_t qi = 0; qi < nq_signed; ++qi) {
                const uint8_t* q = queries[qi].data();

                // ── Find nprobe nearest centroids ────────────────────────────
                for (int32_t c = 0; c < nc; ++c)
                    cdists[c] = {_hamming_raw(q, _centroids[c].data(),
                                              nwords, ntail),
                                 c};
                std::partial_sort(cdists.begin(), cdists.begin() + nprobe,
                                  cdists.end());

                // ── Scan chosen clusters, keep top-k in a max-heap ───────────
                using Elem = std::pair<int64_t, int64_t>; // (distance, original_idx)
                std::priority_queue<Elem> heap;

                for (int32_t p = 0; p < nprobe; ++p) {
                    int32_t c = cdists[p].second;
                    const size_t begin = _list_offsets[c];
                    const size_t end   = _list_offsets[c + 1];
                    const uint8_t* code = codes + begin * nbytes;
                    // Within a cluster, codes are stored in ascending
                    // original-id order, so the scan (and therefore heap
                    // tie-breaking) matches the original invlist order.
                    for (size_t j = begin; j < end; ++j, code += nbytes) {
                        int64_t d = _hamming_raw(q, code, nwords, ntail);
                        if ((int64_t)heap.size() < (int64_t)k ||
                            d < heap.top().first) {
                            heap.push({d, (int64_t)_ids[j]});
                            if (heap.size() > k) heap.pop();
                        }
                    }
                }

                // ── Extract results in ascending distance order ──────────────
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
    size_t   _nwords = 0; // whole 64-bit words per code
    size_t   _ntail  = 0; // trailing bytes (_nbytes % 8)

    ndarrayli _db;        // only populated during _build(); freed afterwards
    ndarrayli _centroids;

    // Flat cluster-ordered code storage (Faiss-style):
    //   _codes        – all codes back to back, grouped by cluster; within a
    //                   cluster codes are in ascending original-id order
    //   _ids          – flat position → original id
    //   _list_offsets – k+1 prefix offsets; cluster c occupies positions
    //                   [_list_offsets[c], _list_offsets[c+1])
    std::vector<uint8_t> _codes;
    std::vector<int32_t> _ids;
    std::vector<size_t>  _list_offsets;

    // Raw-pointer Hamming kernel: popcount over whole 64-bit words plus a
    // byte tail. Produces exactly the same integer as dist_hamming (a
    // popcount sum is independent of how the bytes are chunked). memcpy
    // loads compile to plain (unaligned-safe) 64-bit moves.
    static int64_t _hamming_raw(const uint8_t* a, const uint8_t* b,
                                size_t nwords, size_t ntail) {
        int64_t h = 0;
        for (size_t i = 0; i < nwords; ++i) {
            uint64_t wa, wb;
            std::memcpy(&wa, a + i * 8, sizeof(wa));
            std::memcpy(&wb, b + i * 8, sizeof(wb));
            h += PYNEAR_POPCNT64(wa ^ wb);
        }
        const uint8_t* pa = a + nwords * 8;
        const uint8_t* pb = b + nwords * 8;
        for (size_t i = 0; i < ntail; ++i)
            h += PYNEAR_POPCNT32((uint32_t)(pa[i] ^ pb[i]));
        return h;
    }

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
            // Each i is independent and min_d[i] is only touched by its own
            // iteration → parallelisation is order-independent.
#ifdef ENABLE_OMP_PARALLEL
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < (int64_t)n; ++i) {
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
        const size_t nbits = _nbytes * 8;

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

            // Update step: majority vote per bit. Branchless bit counting;
            // parallel threads accumulate private partial counts which are
            // then reduced — integer addition is order-independent, so the
            // result is identical to the serial scan.
            for (auto& bc : bit_counts)
                std::fill(bc.begin(), bc.end(), 0);
            std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

#ifdef ENABLE_OMP_PARALLEL
            #pragma omp parallel
            {
                std::vector<int32_t> local_bits((size_t)k * nbits, 0);
                std::vector<int32_t> local_counts((size_t)k, 0);

                #pragma omp for schedule(static) nowait
                for (int64_t i = 0; i < (int64_t)n; ++i) {
                    const int32_t c = labels[i];
                    ++local_counts[c];
                    int32_t* bp = local_bits.data() + (size_t)c * nbits;
                    const uint8_t* row = _db[i].data();
                    for (size_t b = 0; b < _nbytes; ++b) {
                        const uint8_t byte = row[b];
                        for (int bit = 0; bit < 8; ++bit)
                            bp[b * 8 + bit] += (byte >> bit) & 1;
                    }
                }

                #pragma omp critical
                {
                    for (int32_t c = 0; c < k; ++c) {
                        cluster_counts[c] += local_counts[c];
                        int32_t* dst = bit_counts[c].data();
                        const int32_t* src =
                            local_bits.data() + (size_t)c * nbits;
                        for (size_t j = 0; j < nbits; ++j) dst[j] += src[j];
                    }
                }
            }
#else
            for (size_t i = 0; i < n; ++i) {
                const int32_t c = labels[i];
                ++cluster_counts[c];
                int32_t* bp = bit_counts[c].data();
                const uint8_t* row = _db[i].data();
                for (size_t b = 0; b < _nbytes; ++b) {
                    const uint8_t byte = row[b];
                    for (int bit = 0; bit < 8; ++bit)
                        bp[b * 8 + bit] += (byte >> bit) & 1;
                }
            }
#endif

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

        // ── Build flat cluster-ordered storage ───────────────────────────────
        // Iterating i in ascending order keeps each cluster's codes in
        // ascending original-id order — exactly the order the per-cluster
        // invlists used to be appended in, preserving scan order and top-k
        // tie-breaking.
        _list_offsets.assign((size_t)k + 1, 0);
        for (size_t i = 0; i < n; ++i)
            ++_list_offsets[(size_t)labels[i] + 1];
        for (int32_t c = 0; c < k; ++c)
            _list_offsets[c + 1] += _list_offsets[c];

        _codes.resize(n * _nbytes);
        _ids.resize(n);
        std::vector<size_t> fill_pos(_list_offsets.begin(),
                                     _list_offsets.end() - 1);
        for (size_t i = 0; i < n; ++i) {
            const size_t pos = fill_pos[labels[i]]++;
            _ids[pos] = (int32_t)i;
            std::memcpy(_codes.data() + pos * _nbytes, _db[i].data(), _nbytes);
        }
    }
};
