#pragma once

/*
 * Lloyd's K-Means with K-Means++ initialisation.
 *
 * Assignment step is parallelised with OpenMP; update step is serial
 * (O(N·D) memory-bound, negligible vs. assignment at typical k values).
 *
 * Distance is Euclidean (L2), computed via SIMD dist_l2_f_avx2.
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <DistanceFunctions.hpp>

struct KMeansResult {
    std::vector<int32_t> labels;    // length N  — cluster index per point
    std::vector<float>   centroids; // length k*D, row-major float32
};

inline KMeansResult kmeans_l2(
    const float* data,
    size_t       n,
    size_t       d,
    size_t       k,
    size_t       max_iter,
    uint32_t     seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick_point(0, n - 1);
    std::uniform_real_distribution<float> unif01(0.f, 1.f);

    // ── K-Means++ initialisation ────────────────────────────────────────────
    std::vector<float> centroids(k * d);

    // First centroid: random point
    std::memcpy(centroids.data(), data + pick_point(rng) * d, d * sizeof(float));

    std::vector<float> min_sq(n, std::numeric_limits<float>::max());

    for (size_t ci = 1; ci < k; ++ci) {
        const float* prev = centroids.data() + (ci - 1) * d;

        // Update min squared distances w.r.t. the most-recently-added centroid
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            FlatSpan a{data + i * d, d};
            FlatSpan b{prev, d};
            float v = dist_l2_f_avx2(a, b);
            v = v * v;  // squared distance for weighted selection
            if (v < min_sq[i]) min_sq[i] = v;
        }

        // Weighted random selection (proportional to squared distance)
        double total = 0.0;
        for (float v : min_sq) total += (double)v;

        double r   = (double)unif01(rng) * total;
        double cum = 0.0;
        size_t chosen = n - 1;
        for (size_t i = 0; i < n; ++i) {
            cum += (double)min_sq[i];
            if (cum >= r) { chosen = i; break; }
        }
        std::memcpy(centroids.data() + ci * d, data + chosen * d, d * sizeof(float));
    }

    // ── Lloyd iterations ────────────────────────────────────────────────────
    std::vector<int32_t> labels(n, 0);
    std::vector<float>   new_centroids(k * d);
    std::vector<int64_t> counts(k);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        int64_t n_changed = 0;

        // ── Assignment (parallel) ────────────────────────────────────────────
        #pragma omp parallel for schedule(static) reduction(+:n_changed)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            FlatSpan xi{data + (size_t)i * d, d};
            float    best_dist = std::numeric_limits<float>::max();
            int32_t  best_c    = 0;
            for (size_t c = 0; c < k; ++c) {
                FlatSpan cj{centroids.data() + c * d, d};
                float dist = dist_l2_f_avx2(xi, cj);
                if (dist < best_dist) { best_dist = dist; best_c = (int32_t)c; }
            }
            if (labels[i] != best_c) {
                labels[i] = best_c;
                ++n_changed;
            }
        }

        if (n_changed == 0) break;

        // ── Update (serial) ──────────────────────────────────────────────────
        std::fill(new_centroids.begin(), new_centroids.end(), 0.f);
        std::fill(counts.begin(), counts.end(), 0LL);

        for (size_t i = 0; i < n; ++i) {
            int32_t      c  = labels[i];
            float*       cp = new_centroids.data() + c * d;
            const float* xi = data + i * d;
            ++counts[c];
            for (size_t j = 0; j < d; ++j) cp[j] += xi[j];
        }

        for (size_t c = 0; c < k; ++c) {
            float* cp = new_centroids.data() + c * d;
            if (counts[c] > 0) {
                float inv = 1.f / (float)counts[c];
                for (size_t j = 0; j < d; ++j) cp[j] *= inv;
            } else {
                // Reinitialise empty cluster to a random data point
                std::memcpy(cp, data + pick_point(rng) * d, d * sizeof(float));
            }
        }

        centroids.swap(new_centroids);
    }

    return KMeansResult{std::move(labels), std::move(centroids)};
}
