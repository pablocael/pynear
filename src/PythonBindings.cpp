/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <VPTree.hpp>

#include <nmmintrin.h>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>

namespace py = pybind11;

using arrayd = std::vector<double>;
using arrayf = std::vector<float>;
using arrayli = std::vector<uint8_t>;
using ndarrayd = std::vector<arrayd>;
using ndarrayf = std::vector<arrayf>;
using ndarrayli = std::vector<arrayli>;

typedef float hamdis_t;

inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

/* Hamming distances for multiples of 64 bits */
template <size_t nbits>
hamdis_t hamming(const uint64_t* bs1, const uint64_t* bs2) {
    const size_t nwords = nbits / 64;
    size_t i;
    hamdis_t h = 0;
    for (i = 0; i < nwords; i++)
        h += popcount64(bs1[i] ^ bs2[i]);
    return h;
}

/* specialized (optimized) functions */
template <>
hamdis_t hamming<64>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]);
}

template <>
hamdis_t hamming<128>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]);
}

template <>
hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]) +
            popcount64(pa[2] ^ pb[2]) + popcount64(pa[3] ^ pb[3]);
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

inline double sum4(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

double dist_optimized_double(const arrayd& p1, const arrayd& p2) {

    unsigned int i = p1.size()/4;
    __m256d result = _mm256_set_pd(0, 0, 0, 0);

    while (i--) {
        __m256d x = _mm256_load_pd(&p1[4*i]);
        __m256d y = _mm256_load_pd(&p2[4*i]);

          /* Compute the difference between the two vectors */
        __m256d diff = _mm256_sub_pd(x, y);

        __m256d temp =_mm256_mul_pd(diff, diff);
        result = _mm256_add_pd(temp, result);
/* Multipl_mm256_mul_psy and add to result */
/* result = _mm256_fmadd_pd(diff, diff, result); */
    }

    // retrieve squared distance (avoid sqrt)
    return sum4(result);
}

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x)
{
    assert (0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
      case 3:
        buf[2] = x[2];
      case 2:
        buf[1] = x[1];
      case 1:
        buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}


float dist_optimized_float(const arrayf& p1, const arrayf& p2) {
    unsigned int d = p1.size();
    __m256 msum1 = _mm256_setzero_ps();

    const float* x = &(p1[0]);
    const float* y = &(p2[0]);

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
}

float distL2(const arrayd& p1, const arrayd& p2) {

    float result = 0;
    auto i = p1.size();
    while (i--) {
        float d = (p1[i] - p2[i]);
        result += d * d;
    }

    // retrieve squared distance (avoid sqrt)
    return result;
}

hamdis_t distHamming(const arrayli& p1, const arrayli& p2) {

    return hamming<256>(reinterpret_cast<const uint64_t*>(&p1[0]), reinterpret_cast<const uint64_t*>(&p2[0]));
}

inline int pop_count(uint64_t x, uint64_t y) {
    return __builtin_popcountll(x ^ y);
}

class VPTreeNumpyAdapter {
public:
    VPTreeNumpyAdapter() = default;

    void set(const ndarrayf& array) {
        _tree = vptree::VPTree<arrayf, dist_optimized_float>(array);
    }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<float>>> searchKNN(const ndarrayf& queries, unsigned int k) {


        std::vector<vptree::VPTree<arrayf,dist_optimized_float>::VPTreeSearchResultElement> results;
        _tree.searchKNN(queries, k, results);

        std::vector<std::vector<unsigned int>> indexes;
        std::vector<std::vector<float>> distances;
        indexes.resize(results.size());
        distances.resize(results.size());
        for(int i = 0; i < results.size(); ++i) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
        }

        // calculate sqrt of each distance since our distance L2 function returns squared distance
        for(auto& v: distances) {
            for(auto& d: v) {
                d = sqrt(d);
            }
        }
        return std::make_tuple(indexes, distances);
    }

    std::tuple<std::vector<unsigned int>, std::vector<float>> search1NN(const ndarrayf& queries) {


        std::vector<unsigned int> indices; std::vector<float> distances;
        _tree.search1NN(queries, indices, distances);

        for(auto& v: distances) {
            v = sqrt(v);
        }
        return std::make_tuple(std::move(indices), std::move(distances));
    }

private:
    vptree::VPTree<arrayf, dist_optimized_float> _tree;

};

class VPTreeBinaryNumpyAdapter {
public:
    VPTreeBinaryNumpyAdapter() = default;

    void set(const ndarrayli& array) {
        _tree = vptree::VPTree<arrayli, distHamming>(array);
    }

    std::tuple<std::vector<std::vector<unsigned int>>, std::vector<std::vector<float>>> searchKNN(const ndarrayli& queries, unsigned int k) {


        std::vector<vptree::VPTree<arrayli,distHamming>::VPTreeSearchResultElement> results;
        _tree.searchKNN(queries, k, results);

        std::vector<std::vector<unsigned int>> indexes;
        std::vector<std::vector<float>> distances;
        indexes.resize(results.size());
        distances.resize(results.size());
        for(int i = 0; i < results.size(); ++i) {
            indexes[i] = std::move(results[i].indexes);
            distances[i] = std::move(results[i].distances);
        }
        return std::make_tuple(indexes, distances);
    }

    std::tuple<std::vector<unsigned int>, std::vector<float>> search1NN(const ndarrayli& queries) {


        std::vector<unsigned int> indices; std::vector<float> distances;
        _tree.search1NN(queries, indices, distances);

        return std::make_tuple(std::move(indices), std::move(distances));
    }

private:
    vptree::VPTree<arrayli, distHamming> _tree;

};
PYBIND11_MODULE(pyvptree, m) {
    py::class_<VPTreeNumpyAdapter>(m, "VPTreeL2Index")
        .def(py::init<>())
        .def("set", &VPTreeNumpyAdapter::set)
        .def("searchKNN", &VPTreeNumpyAdapter::searchKNN)
        .def("search1NN", &VPTreeNumpyAdapter::search1NN);

    py::class_<VPTreeBinaryNumpyAdapter>(m, "VPTreeBinaryIndex")
        .def(py::init<>())
        .def("set", &VPTreeBinaryNumpyAdapter::set)
        .def("searchKNN", &VPTreeBinaryNumpyAdapter::searchKNN)
        .def("search1NN", &VPTreeBinaryNumpyAdapter::search1NN);
}
