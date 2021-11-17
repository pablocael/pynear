#include "gmock/gmock.h"

#include <VPTree.hpp>
#include <MathUtils.hpp>

#include <vector>

using namespace testing;

namespace vptree::tests
{
    TEST(MathUtilsTest, Test) {

        std::vector<double> v = { 0, 0, 0, 0, 1, 1, 1, 1 };
        double s = vptree::math::distance2(v, 0, 1, 4);
        ASSERT_DOUBLE_EQ(s, 4);
    }
}
