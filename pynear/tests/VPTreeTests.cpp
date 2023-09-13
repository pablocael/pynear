#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <BuiltinSerializers.hpp>
#include <DistanceFunctions.hpp>
#include <MathUtils.hpp>
#include <SerializableVPTree.hpp>
#include <SerializedStateObject.hpp>
#include <VPTree.hpp>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <sstream>
#include <stdint.h>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <nmmintrin.h>
#endif

using namespace testing;

float distance(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) { return (v2 - v1).norm(); }
float distanceVector3(const std::vector<float> &v1, const std::vector<float> &v2) {
    double d = 0;
    for (int i = 0; i < 3; ++i) {
        d += (v2[i] - v1[i]) * (v2[i] - v1[i]);
    }
    return std::sqrt(d);
}

/* inline uint_fast8_t popcnt_u128(__uint128_t n) */
/* { */
/*     const uint64_t      n_hi    = n >> 64; */
/*     const uint64_t      n_lo    = n; */
/*     const uint_fast8_t  cnt_hi  = __builtin_popcountll(n_hi); */
/*     const uint_fast8_t  cnt_lo  = __builtin_popcountll(n_lo); */
/*     const uint_fast8_t  cnt     = cnt_hi + cnt_lo; */

/*     return  cnt; */
/* } */


namespace vptree::tests {
TEST(VPTests, TestHamming) {

    std::vector<unsigned char> b1 = {255, 255, 255, 255, 255, 255, 255, 255};
    std::vector<unsigned char> b2 = {255, 255, 255, 255, 255, 255, 255, 255};

    EXPECT_EQ(dist_hamming(b1, b2), 0);

    std::vector<unsigned char> b12 = {1,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    std::vector<unsigned char> b22 = {2,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    EXPECT_EQ(dist_hamming(b12, b22), 2);

    std::vector<unsigned char> b13 = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    std::vector<unsigned char> b23 = {0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    EXPECT_EQ(dist_hamming(b13, b23), 8);

    std::vector<unsigned char> b14 = {255, 255, 255, 255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    std::vector<unsigned char> b24 = {0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    EXPECT_EQ(dist_hamming(b14, b24), 9);

    std::vector<unsigned char> b15 = {255, 255, 255, 255, 255, 253, 255, 9,   255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    std::vector<unsigned char> b25 = {0,   255, 255, 255, 255, 255, 255, 0,   255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    EXPECT_EQ(dist_hamming(b15, b25), 11);

    std::vector<unsigned char> b16 = {255, 255, 255, 255, 255, 253, 255, 9,   255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    std::vector<unsigned char> b26 = {0,   255, 255, 255, 255, 255, 255, 0,   255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    EXPECT_EQ(dist_hamming(b16, b26), dist_hamming(b26, b16));

    std::vector<unsigned char> b17 = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    std::vector<unsigned char> b27 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(dist_hamming(b17, b27), 256);
}

TEST(VPTests, TestEmpty) {
    VPTree<Eigen::Vector3d, float, distance> tree;
    tree.set(std::vector<Eigen::Vector3d>());

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);

    std::vector<Eigen::Vector3d> queries;
    queries.resize(100);
    for (Eigen::Vector3d &point : queries) {
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    std::vector<int64_t> indices;
    std::vector<float> distances;
    EXPECT_THROW(tree.search1NN(queries, indices, distances), std::runtime_error);

    VPTree<Eigen::Vector3d, float, distance> treeEmpty2;
    EXPECT_THROW(treeEmpty2.search1NN(queries, indices, distances), std::runtime_error);

    VPTree<Eigen::Vector3d, float, distance> nonEmpty;
    nonEmpty.set(queries);
    EXPECT_NO_THROW(nonEmpty.search1NN(queries, indices, distances));
}

TEST(VPTests, TestToString) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);

    const unsigned int numPoints = 14001;
    std::vector<Eigen::Vector3d> points;
    points.reserve(numPoints);
    points.resize(numPoints);
    for (Eigen::Vector3d &point : points) {
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    VPTree<Eigen::Vector3d, float, distance> tree(points);

    std::stringstream ss;
    ss << tree;
}

TEST(VPTests, TestCopyVPTree) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);

    const unsigned int numPoints = 14001;
    std::vector<Eigen::Vector3d> points;
    points.reserve(numPoints);
    points.resize(numPoints);
    for (Eigen::Vector3d &point : points) {
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    VPTree<Eigen::Vector3d, float, distance> tree2;
    VPTree<Eigen::Vector3d, float, distance> tree(points);

    tree2 = tree;

    std::vector<Eigen::Vector3d> queries;
    queries.resize(100);
    for (Eigen::Vector3d &point : queries) {
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    std::vector<int64_t> indices;
    std::vector<float> distances;
    std::vector<int64_t> indices2;
    std::vector<float> distances2;
    tree.search1NN(queries, indices, distances);
    tree2.search1NN(queries, indices2, distances2);

    EXPECT_EQ(indices.size(), indices2.size());
    for (int i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], indices2[i]) << "Vectors x and y differ at index " << i;
        EXPECT_EQ(distances[i], distances2[i]) << "Vectors x and y differ at distance " << i;
    }
}

TEST(VPTests, TestCopySerializableVPTree) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);

    const unsigned int numPoints = 14001;
    std::vector<std::vector<float>> points;
    points.reserve(numPoints);
    points.resize(numPoints);
    for (auto &point : points) {
        point.resize(3);
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    SerializableVPTree<std::vector<float>, float, distanceVector3, vptree::ndarraySerializer<float>, vptree::ndarrayDeserializer<float>> tree2;
    SerializableVPTree<std::vector<float>, float, distanceVector3, vptree::ndarraySerializer<float>, vptree::ndarrayDeserializer<float>> tree(points);
    tree2 = tree;

    std::vector<std::vector<float>> queries;
    queries.resize(100);
    for (auto &point : queries) {
        point.resize(3);
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    std::vector<int64_t> indices;
    std::vector<float> distances;
    std::vector<int64_t> indices2;
    std::vector<float> distances2;
    tree.search1NN(queries, indices, distances);
    tree2.search1NN(queries, indices2, distances2);

    EXPECT_EQ(indices.size(), indices2.size());
    for (int i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], indices2[i]) << "Vectors x and y differ at index " << i;
        EXPECT_EQ(distances[i], distances2[i]) << "Vectors x and y differ at distance " << i;
    }
}

TEST(VPTests, TestSerialization) {

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);

    const unsigned int numPoints = 14001;
    std::vector<std::vector<float>> points;
    points.reserve(numPoints);
    points.resize(numPoints);
    for (auto &point : points) {
        point.resize(3);
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    SerializableVPTree<std::vector<float>, float, distanceVector3, vptree::ndarraySerializer<float>, vptree::ndarrayDeserializer<float>> tree2;
    SerializableVPTree<std::vector<float>, float, distanceVector3, vptree::ndarraySerializer<float>, vptree::ndarrayDeserializer<float>> tree(points);
    auto state = tree.serialize();
    tree2.deserialize(state);

    std::vector<std::vector<float>> queries;
    queries.resize(100);
    for (auto &point : queries) {
        point.resize(3);
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    std::vector<int64_t> indices;
    std::vector<float> distances;
    std::vector<int64_t> indices2;
    std::vector<float> distances2;
    tree.search1NN(queries, indices, distances);
    tree2.search1NN(queries, indices2, distances2);

    EXPECT_EQ(indices.size(), indices2.size());
    for (int i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], indices2[i]) << "Vectors x and y differ at index " << i;
        EXPECT_EQ(distances[i], distances2[i]) << "Vectors x and y differ at distance " << i;
    }
}

TEST(VPTests, TestSerializedStateObject) {
    SerializedStateObject state;

    struct TestStruct {
        int a;
        int b;
    };

    SerializedStateObjectWriter writer(state);
    writer.write(1);
    writer.write<std::string>(std::string("my string"));
    writer.write<TestStruct>({1, 2});

    std::vector<int64_t> testVector;
    testVector.resize(201);
    for (int i = 0; i < testVector.size(); ++i) {
        testVector[i] = i;
    }

    writer.writeUserVector<int64_t, vptree::vectorSerializer>(testVector);

    SerializedStateObjectReader reader(state);
    EXPECT_EQ(reader.read<int>(), 1);
    EXPECT_EQ(reader.read<std::string>(), "my string");

    auto stct = reader.read<TestStruct>();
    EXPECT_EQ(stct.a, 1);
    EXPECT_EQ(stct.b, 2);

    std::vector<int64_t> recoveredVector = reader.readUserVector<int64_t, vptree::vectorDeserializer>();
    EXPECT_EQ(recoveredVector.size(), 201);
}

TEST(VPTests, TestCreation) {

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);

    const unsigned int numPoints = 10000;
    std::vector<Eigen::Vector3d> points;
    points.reserve(numPoints);
    points.resize(numPoints);
    for (Eigen::Vector3d &point : points) {
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    auto start = std::chrono::steady_clock::now();

    VPTree<Eigen::Vector3d, float, distance> tree(points);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> diff = end - start;
}

TEST(VPTests, TestSearch) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);

    const unsigned int numPoints = 4e4;
    std::vector<Eigen::Vector3d> points;
    points.resize(numPoints);
    for (Eigen::Vector3d &point : points) {
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }

    auto start = std::chrono::steady_clock::now();
    VPTree<Eigen::Vector3d, float, distance> tree(points);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> diff = end - start;

    std::vector<Eigen::Vector3d> queries;
    queries.resize(5000);
    for (Eigen::Vector3d &point : queries) {
        point[0] = distribution(generator);
        point[1] = distribution(generator);
        point[2] = distribution(generator);
    }
    /* std::vector<VPTree<Eigen::Vector3d, float, distance>::VPTreeSearchResultElement> results; */
    std::vector<int64_t> indices;
    std::vector<float> distances;
    start = std::chrono::steady_clock::now();
    tree.search1NN(queries, indices, distances);
    end = std::chrono::steady_clock::now();
    diff = end - start;
}

} // namespace vptree::tests
