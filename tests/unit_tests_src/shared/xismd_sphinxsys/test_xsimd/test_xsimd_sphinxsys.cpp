#include "large_data_containers.h"
#include "xsimd_eigen.h"

#include <gtest/gtest.h>
#include "xsimd_eigen.h"

#define EIGEN_DONT_VECTORIZE

using namespace SPH;
namespace xs = xsimd;

size_t vec_size = 110;

TEST(test_XsimdScalar, test_BasicOperations)
{
	StdLargeVec<Real> a, b;
	a.resize(vec_size, 1.0);
	b.resize(vec_size, 2.0);

	RealX x_sum(0.0);
	size_t floored_vec_size = vec_size - vec_size % XsimdSize;
	for (size_t i = 0; i < floored_vec_size; i += XsimdSize)
	{
		RealX ba = xs::load_aligned(&a[i]);
		RealX bb = xs::load_aligned(&b[i]);
		x_sum += (ba + bb) / 2.0;
	}
	Real sum = xs::reduce_add(x_sum);
	for (size_t i = floored_vec_size; i < vec_size; ++i)
	{
		sum += (a[i] + b[i]) / 2.0;
	}

	EXPECT_EQ(165.0, sum);
}

TEST(test_XsimdVecd, test_SimpleOperations)
{
	StdLargeVec<Vec2d> a, b;
	a.resize(vec_size, Vec2d(1.0, 1.0));
	b.resize(vec_size, Vec2d(2.0, 2.0));

	Vec2X x_sum = Vec2X::Zero();
	Vec2XHelper helper;
	Vec2X ba, bb;
	size_t floored_vec_size = vec_size - vec_size % XsimdSize;
	for (size_t i = 0; i < floored_vec_size; i += XsimdSize)
	{
		helper.assign(&a[i], ba);
		helper.assign(&b[i], bb);
		x_sum += (ba + bb) / 2.0;
	}

	Vec2d sum = Vec2d::Zero();
	helper.reduce(x_sum, sum);
	for (size_t i = floored_vec_size; i < vec_size; ++i)
	{
		sum += (a[i] + b[i]) / 2.0;
	}

	EXPECT_EQ(Vec2d(165.0, 165.0), sum);
}

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
