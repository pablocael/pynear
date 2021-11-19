#include "gmock/gmock.h"

#include <VPTree.hpp>
#include <MathUtils.hpp>

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <Eigen/Core>

using namespace testing;

double distance( const Eigen::Vector3d& v1, const Eigen::Vector3d& v2 ) {

    return (v2-v1).norm();
}

namespace vptree::tests
{
    TEST(VPTests, TestCreation) {

        const unsigned int numPoints = 10000;
        std::vector<Eigen::Vector3d> points;
        points.reserve(numPoints);
        for(auto point: points) {
            point.setRandom();
        }

        std::cout << "Building tree with " << numPoints << " points " << std::endl;
        auto start = std::chrono::steady_clock::now();

        VPTree<Eigen::Vector3d, distance> tree(points);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Process took" << diff.count() << " seconds " << std::endl;
    }
}
