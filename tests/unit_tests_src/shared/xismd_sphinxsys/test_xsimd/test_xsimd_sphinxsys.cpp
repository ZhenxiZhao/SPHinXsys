#include "large_data_containers.h"
#include "xsimd_eigen.h"

#include <gtest/gtest.h>

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

TEST(test_XsimdVecd, test_VecdOperations)
{
	StdLargeVec<Vec3d> a, b;
	a.resize(vec_size, Vec3d(1.0, 1.0, 1.0));
	b.resize(vec_size, Vec3d(2.0, 2.0, 2.0));

	StdLargeVec<size_t> indexes;
	indexes.resize(vec_size, 0);
	for (size_t i = 0; i < indexes.size(); ++i)
	{
		indexes[i] = i;
	}

	Vec3dX x_sum = Vec3dX::Zero();
	Vec3dXHelper helper;
	Vec3dX ba, bb;
	size_t floored_vec_size = vec_size - vec_size % XsimdSize;
	for (size_t i = 0; i < floored_vec_size; i += XsimdSize)
	{
		helper.gather(a, &indexes[i], ba);
		helper.gather(b, &indexes[i], bb);
		x_sum += (ba + bb) / 2.0;
	}

	Vec3d sum = Vec3d::Zero();
	helper.reduce(x_sum, sum);
	for (size_t i = floored_vec_size; i < vec_size; ++i)
	{
		sum += (a[i] + b[i]) / 2.0;
	}

	EXPECT_EQ(Vec3d(165.0, 165.0, 165.0), sum);
}

TEST(test_XsimdMatd, test_MatdOperations)
{
	StdLargeVec<Mat2d> a, b;
	a.resize(vec_size, Mat2d{{1.0, 1.0}, {1.0, 1.0}});
	b.resize(vec_size, Mat2d{{2.0, 2.0}, {2.0, 2.0}});

	StdLargeVec<size_t> indexes;
	indexes.resize(vec_size, 0);
	for (size_t i = 0; i < indexes.size(); ++i)
	{
		indexes[i] = i;
	}

	Mat2dX x_sum = Mat2dX::Zero();
	Mat2dXHelper helper;
	Mat2dX ba, bb;
	size_t floored_vec_size = vec_size - vec_size % XsimdSize;
	for (size_t i = 0; i < floored_vec_size; i += XsimdSize)
	{
		helper.gather(a, &indexes[i], ba);
		helper.gather(b, &indexes[i], bb);
		x_sum += (ba + bb) / 2.0;
	}

	Mat2d sum = Mat2d::Zero();
	helper.reduce(x_sum, sum);
	for (size_t i = floored_vec_size; i < vec_size; ++i)
	{
		sum += (a[i] + b[i]) / 2.0;
	}
	Mat2d reference = Mat2d{{165.0, 165.0}, {165.0, 165.0}};
	EXPECT_EQ(reference, sum);
}

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
