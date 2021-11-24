#include "gmock/gmock.h"

#include <VPTree.hpp>
#include <MathUtils.hpp>

#include <vector>
#include <random>
#include <chrono>
#include <random>
#include <iostream>
#include <Eigen/Core>

using namespace testing;

double distance( const Eigen::Vector3d& v1, const Eigen::Vector3d& v2 ) {

    return (v2-v1).norm();
}

namespace vptree::tests
{
    TEST(VPTests, TestCreation) {

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-10,10);

        const unsigned int numPoints = 10000;
        std::vector<Eigen::Vector3d> points;
        points.reserve(numPoints);
        points.resize(numPoints);
        for(Eigen::Vector3d& point: points) {
            point[0] = distribution(generator);
            point[1] = distribution(generator);
            point[2] = distribution(generator);
        }

        std::cout << "Building tree with " << numPoints << " points " << std::endl;
        auto start = std::chrono::steady_clock::now();

        VPTree<Eigen::Vector3d, distance> tree(points);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Process took" << diff.count() << " seconds " << std::endl;
    }

    TEST(VPTests, TestSearch) {

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-10,10);

        const unsigned int numPoints = 10000000;
        std::vector<Eigen::Vector3d> points;
        points.resize(numPoints);
        for(Eigen::Vector3d& point: points) {
            point[0] = distribution(generator);
            point[1] = distribution(generator);
            point[2] = distribution(generator);
        }

        std::cout << "Building tree with " << numPoints << " points " << std::endl;
        auto start = std::chrono::steady_clock::now();

        VPTree<Eigen::Vector3d, distance> tree(points);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Process took " << diff.count() << " seconds " << std::endl;

        std::cout << "Searching within the tree with " << numPoints << " points " << std::endl;

        std::vector<Eigen::Vector3d> queries;
        queries.resize(5000);
        for(Eigen::Vector3d& point: queries) {
            point[0] = distribution(generator);
            point[1] = distribution(generator);
            point[2] = distribution(generator);
        }
        /* std::vector<VPTree<Eigen::Vector3d,distance>::VPTreeSearchResultElement> results; */
        std::vector<unsigned int> indices; std::vector<double> distances;
        start = std::chrono::steady_clock::now();
        tree.search1NN(queries, indices, distances);
        end = std::chrono::steady_clock::now();
        diff = end - start;
        std::cout << "Process took " << diff.count() << " seconds " << std::endl;
    }
}
