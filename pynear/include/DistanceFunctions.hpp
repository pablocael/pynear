#pragma once

#include <vector>

// x86-only intrinsic headers
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <nmmintrin.h>
#include <immintrin.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdint.h>
#include <stdio.h>

// Portable population count — maps to hardware popcount on every supported arch
#if defined(_MSC_VER)
#define PYNEAR_POPCNT64(x) static_cast<int64_t>(__popcnt64(x))
#define PYNEAR_POPCNT32(x) static_cast<int32_t>(__popcnt(x))
#else
#define PYNEAR_POPCNT64(x) static_cast<int64_t>(__builtin_popcountll(x))
#define PYNEAR_POPCNT32(x) static_cast<int32_t>(__builtin_popcount(x))
#endif

struct FlatSpan {
    const float* ptr;
    size_t sz;
    size_t size() const { return sz; }
    float operator[](size_t i) const { return ptr[i]; }
    const float* data() const { return ptr; }
};

// SQ8 — pointer to a vector of int8 values produced by scalar quantisation
// of the original floats. Carries no scale itself; the SQ8 owner (e.g.
// HNSWL2IndexSQ8) holds the global scale and scales final distances back
// to L2 if it wants them in float space. For ordering inside HNSW the
// scale is constant and can be omitted.
struct SQ8Span {
    const int8_t* ptr;
    size_t sz;
    size_t size() const { return sz; }
};

using arrayd = std::vector<double>;
using arrayf = FlatSpan;
using arrayli = std::vector<uint8_t>;
using ndarrayd = std::vector<arrayd>;
using ndarrayf = std::vector<arrayf>;
using ndarrayli = std::vector<arrayli>;

#if defined(_MSC_VER)
#define ALIGN_AS(bits) __declspec(align(bits))
#elif defined(__GNUC__)
#define ALIGN_AS(bits) __attribute__((__aligned__(bits)))
#endif

/* Hamming distances for multiples of 64 bits */
int64_t hamming_u64(const arrayli &p1, const arrayli &p2) {
    assert(p1.size() == p2.size());
    assert(p1.size() % 8 == 0);

    const uint64_t *bs1 = reinterpret_cast<const uint64_t *>(p1.data());
    const uint64_t *bs2 = reinterpret_cast<const uint64_t *>(p2.data());

    const size_t nwords = p1.size() / 8;
    size_t i;
    int64_t h = 0;
    for (i = 0; i < nwords; i++)
        h += PYNEAR_POPCNT64(bs1[i] ^ bs2[i]);
    return h;
}

template <size_t nbits> int64_t hamming_u64(const uint64_t *bs1, const uint64_t *bs2) {
    const size_t nwords = nbits / 64;
    size_t i;
    int64_t h = 0;
    for (i = 0; i < nwords; i++)
        h += PYNEAR_POPCNT64(bs1[i] ^ bs2[i]);
    return h;
}

/* Hamming distances for multiples of 32 bits */
int32_t hamming_u32(const arrayli &p1, const arrayli &p2) {
    assert(p1.size() == p2.size());
    assert(p1.size() % 4 == 0);

    const uint32_t *bs1 = reinterpret_cast<const uint32_t *>(p1.data());
    const uint32_t *bs2 = reinterpret_cast<const uint32_t *>(p2.data());

    const size_t nwords = p1.size() / 4;
    size_t i;
    int32_t h = 0;
    for (i = 0; i < nwords; i++)
        h += PYNEAR_POPCNT32(bs1[i] ^ bs2[i]);
    return h;
}

template <size_t nbits> int32_t hamming_u32(const uint32_t *bs1, const uint32_t *bs2) {
    const size_t nwords = nbits / 32;
    size_t i;
    int32_t h = 0;
    for (i = 0; i < nwords; i++)
        h += PYNEAR_POPCNT32(bs1[i] ^ bs2[i]);
    return h;
}

/* Hamming distances for multiples of 16 bits */
int16_t hamming_u16(const arrayli &p1, const arrayli &p2) {
    assert(p1.size() == p2.size());
    assert(p1.size() % 2 == 0);

    const uint16_t *bs1 = reinterpret_cast<const uint16_t *>(p1.data());
    const uint16_t *bs2 = reinterpret_cast<const uint16_t *>(p2.data());

    const size_t nwords = p1.size() / 2;
    size_t i;
    int16_t h = 0;
    for (i = 0; i < nwords; i++)
        h += PYNEAR_POPCNT32(bs1[i] ^ bs2[i]);
    return h;
}

template <size_t nbits> int16_t hamming_u16(const uint16_t *bs1, const uint16_t *bs2) {
    const size_t nwords = nbits / 16;
    size_t i;
    int16_t h = 0;
    for (i = 0; i < nwords; i++)
        h += PYNEAR_POPCNT32(bs1[i] ^ bs2[i]);
    return h;
}

/* Hamming distances for multiples of 8 bits */
int8_t hamming_u8(const arrayli &p1, const arrayli &p2) {
    assert(p1.size() == p2.size());

    const uint8_t *bs1 = p1.data();
    const uint8_t *bs2 = p2.data();

    const size_t nwords = p1.size();
    size_t i;
    int8_t h = 0;
    for (i = 0; i < nwords; i++)
        h += PYNEAR_POPCNT32(bs1[i] ^ bs2[i]);
    return h;
}

template <size_t nbits> int8_t hamming_u8(const uint8_t *bs1, const uint8_t *bs2) {
    const size_t nwords = nbits / 8;
    size_t i;
    int8_t h = 0;
    for (i = 0; i < nwords; i++)
        h += PYNEAR_POPCNT32(bs1[i] ^ bs2[i]);
    return h;
}

/* specialized (optimized) hamming functions — portable, no AVX needed */
template <> inline int8_t hamming_u8<8>(const uint8_t *pa, const uint8_t *pb) { return PYNEAR_POPCNT32(pa[0] ^ pb[0]); }

template <> inline int16_t hamming_u16<16>(const uint16_t *pa, const uint16_t *pb) { return PYNEAR_POPCNT32(pa[0] ^ pb[0]); }

template <> inline int32_t hamming_u32<32>(const uint32_t *pa, const uint32_t *pb) { return PYNEAR_POPCNT32(pa[0] ^ pb[0]); }

template <> inline int64_t hamming_u64<64>(const uint64_t *pa, const uint64_t *pb) { return PYNEAR_POPCNT64(pa[0] ^ pb[0]); }

template <> inline int64_t hamming_u64<128>(const uint64_t *pa, const uint64_t *pb) {
    return PYNEAR_POPCNT64(pa[0] ^ pb[0]) + PYNEAR_POPCNT64(pa[1] ^ pb[1]);
}

template <> inline int64_t hamming_u64<256>(const uint64_t *pa, const uint64_t *pb) {
    return PYNEAR_POPCNT64(pa[0] ^ pb[0]) + PYNEAR_POPCNT64(pa[1] ^ pb[1]) +
           PYNEAR_POPCNT64(pa[2] ^ pb[2]) + PYNEAR_POPCNT64(pa[3] ^ pb[3]);
}

template <> inline int64_t hamming_u64<512>(const uint64_t *pa, const uint64_t *pb) {
    return PYNEAR_POPCNT64(pa[0] ^ pb[0]) + PYNEAR_POPCNT64(pa[1] ^ pb[1]) + PYNEAR_POPCNT64(pa[2] ^ pb[2]) + PYNEAR_POPCNT64(pa[3] ^ pb[3]) +
           PYNEAR_POPCNT64(pa[4] ^ pb[4]) + PYNEAR_POPCNT64(pa[5] ^ pb[5]) + PYNEAR_POPCNT64(pa[6] ^ pb[6]) + PYNEAR_POPCNT64(pa[7] ^ pb[7]);
}

/* Scalar distance functions — always compiled, referenced by AVX #else fallbacks */

double dist_l2_d(const arrayd &p1, const arrayd &p2) {

    double result = 0;
    size_t i = p1.size();
    while (i--) {
        double d = (p1[i] - p2[i]);
        result += d * d;
    }

    return std::sqrt(result);
}

float dist_l2_f(const arrayf &p1, const arrayf &p2) {

    float result = 0.;
    size_t i = p1.size();
    while (i--) {
        float d = (p1[i] - p2[i]);
        result += d * d;
    }

    return std::sqrt(result);
}

float dist_l1_f(const arrayf &p1, const arrayf &p2) {
    /* L1 metric, also called Manhattan or taxicab metric */

    float result = 0.;
    size_t i = p1.size();
    while (i--) {
        result += std::fabs(p1[i] - p2[i]);
    }

    return result;
}

float dist_chebyshev_f(const arrayf &p1, const arrayf &p2) {
    /* Chebyshev distance metric, also called maximum metric or L_inf metric */

    float result = 0.;
    size_t i = p1.size();
    while (i--) {
        float distance = std::fabs(p1[i] - p2[i]);
        if (distance > result) {
            result = distance;
        }
    }

    return result;
}

/* AVX/AVX2 accelerated distance functions */
#if defined(__AVX__) || defined(__AVX2__)

inline double sum4(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_add_pd(vlow, vhigh);              // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); // reduce to scalar
}

inline float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

double dist_l2_d_avx2(const arrayd &p1, const arrayd &p2) {

    unsigned int i = p1.size() / 4;
    __m256d result = _mm256_set_pd(0, 0, 0, 0);

    while (i--) {
        __m256d x = _mm256_loadu_pd(&p1[4 * i]);
        __m256d y = _mm256_loadu_pd(&p2[4 * i]);

        /* Compute the difference between the two vectors */
        __m256d diff = _mm256_sub_pd(x, y);

        __m256d temp = _mm256_mul_pd(diff, diff);
        result = _mm256_add_pd(temp, result);
        /* Multipl_mm256_mul_psy and add to result */
        /* result = _mm256_fmadd_pd(diff, diff, result); */
    }

    double out = sum4(result);

    // Handle remainder elements (when size % 4 != 0)
    size_t processed = (p1.size() / 4) * 4;
    double remain = 0.0;
    for (size_t r = processed; r < p1.size(); r++) {
        double d = p1[r] - p2[r];
        remain += d * d;
    }
    return std::sqrt(out + remain);
}

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(int d, const float *x) {
    assert(0 <= d && d < 4);
    ALIGN_AS(16) float buf[4] = {0, 0, 0, 0};
    switch (d) {
    case 3:
        buf[2] = x[2];
    case 2:
        buf[1] = x[1];
    case 1:
        buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

// ─── int8 (SQ8) distance kernels ────────────────────────────────────────────
// Process 32 int8 lanes per AVX2 op (4 × wider than float). Combined with the
// 4 × memory-bandwidth reduction this is the biggest single perf lever we
// have on Skylake-era hardware. Result is int32 sum-of-squared-diffs in the
// quantised space; callers compose final L2 by multiplying by global scale².

inline int32_t dist_l2sq_sq8_avx2(const int8_t* x, const int8_t* y, size_t d) {
#if (defined(__AVX__) || defined(__AVX2__))
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= d; i += 32) {
        __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        __m256i vy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i));
        // Widen int8 → int16 in two halves so we can subtract without
        // saturation surprises (int8 - int8 can underflow int8 range).
        __m256i diff_lo = _mm256_sub_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 0)),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vy, 0)));
        __m256i diff_hi = _mm256_sub_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1)),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vy, 1)));
        // madd_epi16(a, a) computes pairwise a*a then adds adjacent pairs
        // → 8 int32 lanes, each holding the sum of two squared diffs.
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff_lo, diff_lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff_hi, diff_hi));
    }
    // Reduce 8 int32 lanes to a scalar.
    alignas(32) int32_t buf[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(buf), acc);
    int32_t result = 0;
    for (int j = 0; j < 8; j++) result += buf[j];
    // Scalar tail.
    for (; i < d; i++) {
        int32_t diff = static_cast<int32_t>(x[i]) - static_cast<int32_t>(y[i]);
        result += diff * diff;
    }
    return result;
#else
    int32_t result = 0;
    for (size_t i = 0; i < d; i++) {
        int32_t diff = static_cast<int32_t>(x[i]) - static_cast<int32_t>(y[i]);
        result += diff * diff;
    }
    return result;
#endif
}

// Wrapper matching the HNSWIndex distance-function-pointer signature.
inline int32_t dist_l2sq_sq8(const SQ8Span& p1, const SQ8Span& p2) {
    return dist_l2sq_sq8_avx2(p1.ptr, p2.ptr, p1.sz);
}

// 4-way batched SQ8 L2² distance — compute distance(query, v[i]) for i ∈ [0,n).
// 4 independent int32 accumulators expose ILP across the madd_epi16 chain.
inline void batch_l2sq_sq8_avx2(const int8_t* query, size_t d,
                                const int8_t* const* vectors,
                                size_t n,
                                int32_t* out) {
#if (defined(__AVX__) || defined(__AVX2__))
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i a0 = _mm256_setzero_si256();
        __m256i a1 = _mm256_setzero_si256();
        __m256i a2 = _mm256_setzero_si256();
        __m256i a3 = _mm256_setzero_si256();
        const int8_t* p0 = vectors[i + 0];
        const int8_t* p1 = vectors[i + 1];
        const int8_t* p2 = vectors[i + 2];
        const int8_t* p3 = vectors[i + 3];
        size_t j = 0;
        for (; j + 32 <= d; j += 32) {
            __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + j));
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + j));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + j));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2 + j));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p3 + j));

            __m256i q_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q, 0));
            __m256i q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q, 1));

            #define ACCUM(acc, v) do {                                                              \
                __m256i d_lo = _mm256_sub_epi16(q_lo,                                               \
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 0)));                          \
                __m256i d_hi = _mm256_sub_epi16(q_hi,                                               \
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)));                          \
                (acc) = _mm256_add_epi32((acc), _mm256_madd_epi16(d_lo, d_lo));                     \
                (acc) = _mm256_add_epi32((acc), _mm256_madd_epi16(d_hi, d_hi));                     \
            } while (0)

            ACCUM(a0, v0);
            ACCUM(a1, v1);
            ACCUM(a2, v2);
            ACCUM(a3, v3);

            #undef ACCUM
        }
        alignas(32) int32_t b0[8], b1[8], b2[8], b3[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(b0), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(b1), a1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(b2), a2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(b3), a3);
        int32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
        for (int k = 0; k < 8; k++) { r0 += b0[k]; r1 += b1[k]; r2 += b2[k]; r3 += b3[k]; }
        for (; j < d; j++) {
            int32_t qd = query[j];
            int32_t t;
            t = qd - p0[j]; r0 += t * t;
            t = qd - p1[j]; r1 += t * t;
            t = qd - p2[j]; r2 += t * t;
            t = qd - p3[j]; r3 += t * t;
        }
        out[i + 0] = r0; out[i + 1] = r1; out[i + 2] = r2; out[i + 3] = r3;
    }
    for (; i < n; i++) {
        out[i] = dist_l2sq_sq8_avx2(query, vectors[i], d);
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = dist_l2sq_sq8_avx2(query, vectors[i], d);
    }
#endif
}

// Squared sum of one vector — ||v||². Used to precompute per-DB norms.
inline float vec_l2sq_avx2(const float* v, size_t d) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    while (d >= 32) {
        __m256 v0 = _mm256_loadu_ps(v +  0);
        __m256 v1 = _mm256_loadu_ps(v +  8);
        __m256 v2 = _mm256_loadu_ps(v + 16);
        __m256 v3 = _mm256_loadu_ps(v + 24);
        s0 = _mm256_fmadd_ps(v0, v0, s0);
        s1 = _mm256_fmadd_ps(v1, v1, s1);
        s2 = _mm256_fmadd_ps(v2, v2, s2);
        s3 = _mm256_fmadd_ps(v3, v3, s3);
        v += 32; d -= 32;
    }
    while (d >= 8) {
        __m256 vv = _mm256_loadu_ps(v); v += 8;
        s0 = _mm256_fmadd_ps(vv, vv, s0);
        d -= 8;
    }
    __m256 sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float r = sum8(sum);
    for (size_t j = 0; j < d; j++) r += v[j] * v[j];
    return r;
}

// Dot product q·v with 4-way FMA ILP.
inline float dot_f_avx2(const float* q, const float* v, size_t d) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    while (d >= 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q +  0), _mm256_loadu_ps(v +  0), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q +  8), _mm256_loadu_ps(v +  8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 16), _mm256_loadu_ps(v + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 24), _mm256_loadu_ps(v + 24), s3);
        q += 32; v += 32; d -= 32;
    }
    while (d >= 8) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(v), s0);
        q += 8; v += 8; d -= 8;
    }
    __m256 sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float r = sum8(sum);
    for (size_t j = 0; j < d; j++) r += q[j] * v[j];
    return r;
}

// 8-way batched dot product. Eight accumulators fully saturate the FMA
// throughput on Intel Skylake+/AMD Zen+ (FMA latency 4 cycles × throughput
// 2/cycle ⇒ 8 in-flight to keep both FMA ports busy every cycle).
inline void batch_dot_f_avx2_8(const float* query, size_t d,
                                const float* const* vectors,
                                size_t n,
                                float* out_dots) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        const float* p0 = vectors[i + 0]; const float* p1 = vectors[i + 1];
        const float* p2 = vectors[i + 2]; const float* p3 = vectors[i + 3];
        const float* p4 = vectors[i + 4]; const float* p5 = vectors[i + 5];
        const float* p6 = vectors[i + 6]; const float* p7 = vectors[i + 7];
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 q = _mm256_loadu_ps(query + j);
            a0 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p0 + j), a0);
            a1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p1 + j), a1);
            a2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p2 + j), a2);
            a3 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p3 + j), a3);
            a4 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p4 + j), a4);
            a5 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p5 + j), a5);
            a6 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p6 + j), a6);
            a7 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p7 + j), a7);
        }
        float r[8] = {sum8(a0), sum8(a1), sum8(a2), sum8(a3),
                      sum8(a4), sum8(a5), sum8(a6), sum8(a7)};
        const float* ps[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
        for (; j < d; j++) {
            float qj = query[j];
            for (int k = 0; k < 8; k++) r[k] += qj * ps[k][j];
        }
        for (int k = 0; k < 8; k++) out_dots[i + k] = r[k];
    }
    // Tail: < 8 vectors — use the 4-way batch and single fallback.
    for (; i + 4 <= n; i += 4) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        const float* p0 = vectors[i + 0]; const float* p1 = vectors[i + 1];
        const float* p2 = vectors[i + 2]; const float* p3 = vectors[i + 3];
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 q = _mm256_loadu_ps(query + j);
            s0 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p0 + j), s0);
            s1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p1 + j), s1);
            s2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p2 + j), s2);
            s3 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p3 + j), s3);
        }
        float r0 = sum8(s0), r1 = sum8(s1), r2 = sum8(s2), r3 = sum8(s3);
        for (; j < d; j++) {
            r0 += query[j] * p0[j]; r1 += query[j] * p1[j];
            r2 += query[j] * p2[j]; r3 += query[j] * p3[j];
        }
        out_dots[i + 0] = r0; out_dots[i + 1] = r1;
        out_dots[i + 2] = r2; out_dots[i + 3] = r3;
    }
    for (; i < n; i++) {
        out_dots[i] = dot_f_avx2(query, vectors[i], d);
    }
}

// 4-way batched dot product. Same pattern as batch_l2sq but using FMA
// directly on q*v (one FMA per chunk instead of sub-then-square = two ops).
// Caller composes ||q-v||² = q_norm_sq + v_norm_sq - 2 q·v.
inline void batch_dot_f_avx2(const float* query, size_t d,
                              const float* const* vectors,
                              size_t n,
                              float* out_dots) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        const float* p0 = vectors[i + 0];
        const float* p1 = vectors[i + 1];
        const float* p2 = vectors[i + 2];
        const float* p3 = vectors[i + 3];
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 q = _mm256_loadu_ps(query + j);
            s0 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p0 + j), s0);
            s1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p1 + j), s1);
            s2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p2 + j), s2);
            s3 = _mm256_fmadd_ps(q, _mm256_loadu_ps(p3 + j), s3);
        }
        float r0 = sum8(s0), r1 = sum8(s1), r2 = sum8(s2), r3 = sum8(s3);
        for (; j < d; j++) {
            r0 += query[j] * p0[j];
            r1 += query[j] * p1[j];
            r2 += query[j] * p2[j];
            r3 += query[j] * p3[j];
        }
        out_dots[i + 0] = r0;
        out_dots[i + 1] = r1;
        out_dots[i + 2] = r2;
        out_dots[i + 3] = r3;
    }
    for (; i < n; i++) {
        out_dots[i] = dot_f_avx2(query, vectors[i], d);
    }
}

// Squared L2 — same memory and FMA pattern as L2, no final sqrt.
// Used internally by HNSW where ordering is preserved by squared distance
// and we save ~10 cycle sqrt latency per distance (~2.5 ns at 4 GHz).
// The HNSW adapter applies sqrt only to the final returned top-k.
inline float dist_l2sq_raw_avx2(const float* x, const float* y, size_t d) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    while (d >= 32) {
        __m256 x0 = _mm256_loadu_ps(x +  0);
        __m256 x1 = _mm256_loadu_ps(x +  8);
        __m256 x2 = _mm256_loadu_ps(x + 16);
        __m256 x3 = _mm256_loadu_ps(x + 24);
        __m256 d0 = _mm256_sub_ps(x0, _mm256_loadu_ps(y +  0));
        __m256 d1 = _mm256_sub_ps(x1, _mm256_loadu_ps(y +  8));
        __m256 d2 = _mm256_sub_ps(x2, _mm256_loadu_ps(y + 16));
        __m256 d3 = _mm256_sub_ps(x3, _mm256_loadu_ps(y + 24));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
        x += 32; y += 32; d -= 32;
    }
    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x); x += 8;
        __m256 my = _mm256_loadu_ps(y); y += 8;
        __m256 dd = _mm256_sub_ps(mx, my);
        s0 = _mm256_fmadd_ps(dd, dd, s0);
        d -= 8;
    }
    __m256 sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float r = sum8(sum);
    for (size_t j = 0; j < d; j++) {
        float t = x[j] - y[j];
        r += t * t;
    }
    return r;
}

// Batched squared L2 — 4-way to hide FMA latency.
inline void batch_l2sq_f_avx2(const float* query, size_t d,
                              const float* const* vectors,
                              size_t n,
                              float* out) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        const float* p0 = vectors[i + 0];
        const float* p1 = vectors[i + 1];
        const float* p2 = vectors[i + 2];
        const float* p3 = vectors[i + 3];
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 d0 = _mm256_sub_ps(q, _mm256_loadu_ps(p0 + j));
            __m256 d1 = _mm256_sub_ps(q, _mm256_loadu_ps(p1 + j));
            __m256 d2 = _mm256_sub_ps(q, _mm256_loadu_ps(p2 + j));
            __m256 d3 = _mm256_sub_ps(q, _mm256_loadu_ps(p3 + j));
            s0 = _mm256_fmadd_ps(d0, d0, s0);
            s1 = _mm256_fmadd_ps(d1, d1, s1);
            s2 = _mm256_fmadd_ps(d2, d2, s2);
            s3 = _mm256_fmadd_ps(d3, d3, s3);
        }
        float r0 = sum8(s0), r1 = sum8(s1), r2 = sum8(s2), r3 = sum8(s3);
        for (; j < d; j++) {
            float t;
            t = query[j] - p0[j]; r0 += t * t;
            t = query[j] - p1[j]; r1 += t * t;
            t = query[j] - p2[j]; r2 += t * t;
            t = query[j] - p3[j]; r3 += t * t;
        }
        out[i + 0] = r0;
        out[i + 1] = r1;
        out[i + 2] = r2;
        out[i + 3] = r3;
    }
    for (; i < n; i++) {
        out[i] = dist_l2sq_raw_avx2(query, vectors[i], d);
    }
}

// FlatSpan wrapper used as the function-pointer template parameter for
// HNSWIndex. Returns squared L2; the HNSW Python adapter applies sqrt
// when handing distances back to the caller so the public API still
// reports L2 (sqrt'd) distances.
inline float dist_l2sq_f_avx2(const arrayf &p1, const arrayf &p2) {
    return dist_l2sq_raw_avx2(p1.data(), p2.data(), p1.size());
}

// ─── AVX-512 paths (gated on __AVX512F__) ──────────────────────────────────
// Compile only when the target CPU supports AVX-512F. On Zen 4 / Sapphire
// Rapids / Xeon Gold these halve the distance-compute time versus AVX2 by
// processing 16 floats per op (vs 8). The AVX2 paths above remain the
// fallback for older hardware (and for Arrow Lake / Alder Lake clients
// which removed AVX-512 from the consumer line).

#if defined(__AVX512F__)

inline float sum16(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// 8-way batched dot product on AVX-512 — eight 512-bit accumulators saturate
// both FMA ports on Zen 4 (FMA latency 4 cycles × throughput 2/cycle ⇒ 8
// in-flight).
inline void batch_dot_f_avx512_8(const float* query, size_t d,
                                  const float* const* vectors,
                                  size_t n,
                                  float* out_dots) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps();
        __m512 a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
        const float* p[8];
        for (int k = 0; k < 8; k++) p[k] = vectors[i + k];
        size_t j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 q = _mm512_loadu_ps(query + j);
            a0 = _mm512_fmadd_ps(q, _mm512_loadu_ps(p[0] + j), a0);
            a1 = _mm512_fmadd_ps(q, _mm512_loadu_ps(p[1] + j), a1);
            a2 = _mm512_fmadd_ps(q, _mm512_loadu_ps(p[2] + j), a2);
            a3 = _mm512_fmadd_ps(q, _mm512_loadu_ps(p[3] + j), a3);
            a4 = _mm512_fmadd_ps(q, _mm512_loadu_ps(p[4] + j), a4);
            a5 = _mm512_fmadd_ps(q, _mm512_loadu_ps(p[5] + j), a5);
            a6 = _mm512_fmadd_ps(q, _mm512_loadu_ps(p[6] + j), a6);
            a7 = _mm512_fmadd_ps(q, _mm512_loadu_ps(p[7] + j), a7);
        }
        float r[8] = {sum16(a0), sum16(a1), sum16(a2), sum16(a3),
                      sum16(a4), sum16(a5), sum16(a6), sum16(a7)};
        for (; j < d; j++) {
            float qj = query[j];
            for (int k = 0; k < 8; k++) r[k] += qj * p[k][j];
        }
        for (int k = 0; k < 8; k++) out_dots[i + k] = r[k];
    }
    // Tail: fall through to the AVX2 path for any remaining < 8 vectors.
    if (i < n) {
        batch_dot_f_avx2(query, d, vectors + i, n - i, out_dots + i);
    }
}

// AVX-512 SQ8 distance — uses _mm512 int paths. 64 int8 lanes per op vs
// AVX2's 32. Same int16 widen → sub → madd chain.
inline int32_t dist_l2sq_sq8_avx512(const int8_t* x, const int8_t* y, size_t d) {
    __m512i acc = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= d; i += 64) {
        __m512i vx = _mm512_loadu_si512(reinterpret_cast<const void*>(x + i));
        __m512i vy = _mm512_loadu_si512(reinterpret_cast<const void*>(y + i));
        // Widen int8 → int16 in two halves.
        __m512i diff_lo = _mm512_sub_epi16(
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vx, 0)),
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vy, 0)));
        __m512i diff_hi = _mm512_sub_epi16(
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vx, 1)),
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vy, 1)));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(diff_lo, diff_lo));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(diff_hi, diff_hi));
    }
    int32_t result = _mm512_reduce_add_epi32(acc);
    for (; i < d; i++) {
        int32_t diff = static_cast<int32_t>(x[i]) - static_cast<int32_t>(y[i]);
        result += diff * diff;
    }
    return result;
}

#endif  // __AVX512F__

// 4-way batched L2 distance — compute distance(query, v[i]) for i in [0, n)
// processing 4 neighbours in parallel to expose instruction-level parallelism.
// Four independent FMA chains hide the FMA latency (~5 cycles on Skylake+)
// far better than computing distances sequentially.
//
// Single-vector fallback for the remainder (< 4 neighbours).
inline void batch_l2_f_avx2(const float* query, size_t d,
                            const float* const* vectors,
                            size_t n,
                            float* out) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        const float* p0 = vectors[i + 0];
        const float* p1 = vectors[i + 1];
        const float* p2 = vectors[i + 2];
        const float* p3 = vectors[i + 3];
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 d0 = _mm256_sub_ps(q, _mm256_loadu_ps(p0 + j));
            __m256 d1 = _mm256_sub_ps(q, _mm256_loadu_ps(p1 + j));
            __m256 d2 = _mm256_sub_ps(q, _mm256_loadu_ps(p2 + j));
            __m256 d3 = _mm256_sub_ps(q, _mm256_loadu_ps(p3 + j));
            s0 = _mm256_fmadd_ps(d0, d0, s0);
            s1 = _mm256_fmadd_ps(d1, d1, s1);
            s2 = _mm256_fmadd_ps(d2, d2, s2);
            s3 = _mm256_fmadd_ps(d3, d3, s3);
        }
        // Horizontal-add each accumulator + sqrt.
        float r0 = sum8(s0), r1 = sum8(s1), r2 = sum8(s2), r3 = sum8(s3);
        // Tail (dim % 8) — scalar.
        for (; j < d; j++) {
            float t;
            t = query[j] - p0[j]; r0 += t * t;
            t = query[j] - p1[j]; r1 += t * t;
            t = query[j] - p2[j]; r2 += t * t;
            t = query[j] - p3[j]; r3 += t * t;
        }
        out[i + 0] = std::sqrt(r0);
        out[i + 1] = std::sqrt(r1);
        out[i + 2] = std::sqrt(r2);
        out[i + 3] = std::sqrt(r3);
    }
    // Remainder — < 4 neighbours, inline the single-vector SIMD path
    // (avoids a forward-reference dependency on dist_l2_f_avx2 below).
    for (; i < n; i++) {
        const float* p = vectors[i];
        __m256 sum = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 dd = _mm256_sub_ps(q, _mm256_loadu_ps(p + j));
            sum = _mm256_fmadd_ps(dd, dd, sum);
        }
        float r = sum8(sum);
        for (; j < d; j++) {
            float t = query[j] - p[j];
            r += t * t;
        }
        out[i] = std::sqrt(r);
    }
}

float dist_l2_f_avx2(const arrayf &p1, const arrayf &p2) {
    unsigned int d = p1.size();
    // Four independent FMA accumulators hide FMA latency (~5 cycles on
    // Skylake+/Zen+). With a single accumulator this loop was latency-bound:
    // ~1 FMA every 5 cycles. With four accumulators we issue ~1 FMA per
    // cycle → ~4× throughput on the hot inner loop.
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();

    const float *x = p1.data();
    const float *y = p2.data();

    // 32 floats per iteration (4 lanes × 8 floats).
    while (d >= 32) {
        __m256 x0 = _mm256_loadu_ps(x +  0);
        __m256 x1 = _mm256_loadu_ps(x +  8);
        __m256 x2 = _mm256_loadu_ps(x + 16);
        __m256 x3 = _mm256_loadu_ps(x + 24);
        __m256 y0 = _mm256_loadu_ps(y +  0);
        __m256 y1 = _mm256_loadu_ps(y +  8);
        __m256 y2 = _mm256_loadu_ps(y + 16);
        __m256 y3 = _mm256_loadu_ps(y + 24);
        __m256 d0 = _mm256_sub_ps(x0, y0);
        __m256 d1 = _mm256_sub_ps(x1, y1);
        __m256 d2 = _mm256_sub_ps(x2, y2);
        __m256 d3 = _mm256_sub_ps(x3, y3);
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
        x += 32; y += 32;
        d -= 32;
    }
    // Drain remaining 8-float chunks into s0.
    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        s0 = _mm256_fmadd_ps(a_m_b1, a_m_b1, s0);
        d -= 8;
    }
    // Combine all four accumulators into one.
    __m256 msum1 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    float result = _mm_cvtss_f32(msum2);
    return std::sqrt(result);
}

float dist_l1_f_avx2(const arrayf &p1, const arrayf &p2) {
    /* SIMD L1 metric, also called Manhattan or taxicab metric */

    const float *vec1 = p1.data();
    const float *vec2 = p2.data();
    size_t size = p1.size();

    const size_t blocksize = 8;
    size_t i = 0;

    __m256 sum = _mm256_setzero_ps();
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (; i + blocksize <= size; i += blocksize) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);

        __m256 diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_and_ps(diff, abs_mask));
    }

    ALIGN_AS(32) float result[8];
    _mm256_store_ps(result, sum);

    float total_sum = 0.;
    for (int j = 0; j < 8; ++j) {
        total_sum += result[j];
    }

    // Calculate the remaining elements
    for (; i < size; ++i) {
        total_sum += std::fabs(vec1[i] - vec2[i]);
    }

    return total_sum;
}

float dist_chebyshev_f_avx2(const arrayf &p1, const arrayf &p2) {
    /* SIMD Chebyshev distance metric, also called maximum metric or L_inf metric */

    const float *vec1 = p1.data();
    const float *vec2 = p2.data();
    size_t size = p1.size();

    const size_t blocksize = 8;
    size_t i = 0;

    __m256 max_diff = _mm256_setzero_ps();
    const __m256 abs_mask_cheb = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (; i + blocksize <= size; i += blocksize) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);
        __m256 diff = _mm256_sub_ps(v1, v2);
        diff = _mm256_and_ps(diff, abs_mask_cheb); // Absolute value
        max_diff = _mm256_max_ps(max_diff, diff);
    }

    ALIGN_AS(32) float result[8];
    _mm256_store_ps(result, max_diff);

    float max_distance = result[0];
    for (int i = 1; i < 8; ++i) {
        max_distance = std::max(max_distance, result[i]);
    }

    // Calculate the remaining elements
    for (; i < size; ++i) {
        float diff = std::fabs(vec1[i] - vec2[i]);
        max_distance = std::max(max_distance, diff);
    }

    return max_distance;
}

#else // !(__AVX__ || __AVX2__) — scalar fallbacks for non-x86 platforms (e.g. arm64)

double dist_l2_d_avx2(const arrayd &p1, const arrayd &p2) { return dist_l2_d(p1, p2); }
float  dist_l2_f_avx2(const arrayf &p1, const arrayf &p2) { return dist_l2_f(p1, p2); }
float  dist_l1_f_avx2(const arrayf &p1, const arrayf &p2) { return dist_l1_f(p1, p2); }
float  dist_chebyshev_f_avx2(const arrayf &p1, const arrayf &p2) { return dist_chebyshev_f(p1, p2); }

#endif // __AVX__

int64_t dist_hamming(const arrayli &p1, const arrayli &p2) {
    size_t size = p1.size();

    if (size % 8 == 0) {
        return hamming_u64(p1, p2);
    } else if (size % 4 == 0) {
        return static_cast<int64_t>(hamming_u32(p1, p2));
    } else if (size % 2 == 0) {
        return static_cast<int64_t>(hamming_u16(p1, p2));
    } else {
        return static_cast<int64_t>(hamming_u8(p1, p2));
    }
}

inline int64_t dist_hamming_512(const arrayli &p1, const arrayli &p2) {

    return hamming_u64<512>(reinterpret_cast<const uint64_t *>(&p1[0]), reinterpret_cast<const uint64_t *>(&p2[0]));
}

inline int64_t dist_hamming_256(const arrayli &p1, const arrayli &p2) {

    return hamming_u64<256>(reinterpret_cast<const uint64_t *>(&p1[0]), reinterpret_cast<const uint64_t *>(&p2[0]));
}

inline int64_t dist_hamming_128(const arrayli &p1, const arrayli &p2) {

    return hamming_u64<128>(reinterpret_cast<const uint64_t *>(&p1[0]), reinterpret_cast<const uint64_t *>(&p2[0]));
}

inline int64_t dist_hamming_64(const arrayli &p1, const arrayli &p2) {
    return hamming_u64<64>(reinterpret_cast<const uint64_t *>(&p1[0]), reinterpret_cast<const uint64_t *>(&p2[0]));
}

inline int64_t dist_hamming_32(const arrayli &p1, const arrayli &p2) {

    return static_cast<int64_t>(hamming_u32<32>(reinterpret_cast<const uint32_t *>(&p1[0]), reinterpret_cast<const uint32_t *>(&p2[0])));
}

inline int64_t dist_hamming_16(const arrayli &p1, const arrayli &p2) {

    return static_cast<int64_t>(hamming_u16<16>(reinterpret_cast<const uint16_t *>(&p1[0]), reinterpret_cast<const uint16_t *>(&p2[0])));
}

inline int64_t dist_hamming_8(const arrayli &p1, const arrayli &p2) {

    return static_cast<int64_t>(hamming_u8<8>(reinterpret_cast<const uint8_t *>(&p1[0]), reinterpret_cast<const uint8_t *>(&p2[0])));
}
