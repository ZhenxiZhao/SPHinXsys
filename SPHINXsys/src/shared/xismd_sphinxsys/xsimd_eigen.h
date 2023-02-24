/* -------------------------------------------------------------------------*
 *								SPHinXsys									*
 * -------------------------------------------------------------------------*
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle*
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for	*
 * physical accurate simulation and aims to model coupled industrial dynamic*
 * systems including fluid, solid, multi-body dynamics and beyond with SPH	*
 * (smoothed particle hydrodynamics), a meshless computational method using	*
 * particle discretization.													*
 *																			*
 * SPHinXsys is partially funded by German Research Foundation				*
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1,			*
 *  HU1527/12-1 and HU1527/12-4												*
 *                                                                          *
 * Portions copyright (c) 2017-2022 Technical University of Munich and		*
 * the authors' affiliations.												*
 *                                                                          *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may  *
 * not use this file except in compliance with the License. You may obtain a*
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.       *
 *                                                                          *
 * ------------------------------------------------------------------------*/
/**
 * @file 	xsimd_eigen.h
 * @brief 	This the interface to use eigen and xsimd for vectorization.
 * @author	Xiangyu Hu
 */

#ifndef XSIMD_EIGEN_H
#define XSIMD_EIGEN_H

#include <base_data_package.h>

#include <include/xsimd/xsimd.hpp>

namespace SPH
{
    constexpr size_t XsimdSize = xsimd::simd_type<Real>::size;
    using RealX = xsimd::batch<Real, xsimd::default_arch>;
    class RealXHelper
    {
        StdLargeVec<Real> temp_;

    public:
        RealXHelper() : temp_(XsimdSize){};

        inline RealX load(Real *input)
        {
            return xsimd::load_aligned(input);
        }

        inline RealX gather(StdLargeVec<Real> &input, size_t *index)
        {
            for (size_t i = 0; i != XsimdSize; ++i)
            {
                temp_[i] = input[*(index + i)];
            }
            return xsimd::load_aligned(&temp_[0]);
        }

        inline Real reduce(const RealX &input)
        {
            return xsimd::reduce_add(input);
        }
    };
}

namespace Eigen
{
    template <>
    struct NumTraits<SPH::RealX>
        : GenericNumTraits<SPH::RealX>
    {
        typedef SPH::RealX Real;
        typedef SPH::RealX NonInteger;
        typedef SPH::RealX Nested;

        enum
        {
            IsComplex = 0,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 3,
            MulCost = 3
        };
    };
}

namespace SPH
{
    /** Vector with float point number.*/
    using Vec2dX = Eigen::Matrix<RealX, 2, 1>;
    using Vec3dX = Eigen::Matrix<RealX, 3, 1>;
    /** Small, 2*2 and 3*3, matrix with float point number in batches. */
    using Mat2dX = Eigen::Matrix<RealX, 2, 2>;
    using Mat3dX = Eigen::Matrix<RealX, 3, 3>;
    template <int NRow, int NCol>
    class MatXHelper
    {
        constexpr static int MatSize = NRow * NCol;
        Eigen::Matrix<Real, MatSize, XsimdSize> temp_;
        Eigen::Matrix<Real, XsimdSize, MatSize> temp_transpose_;

    public:
        MatXHelper(){};

        inline void load(Eigen::Matrix<Real, NRow, NCol> *input, Eigen::Matrix<RealX, NRow, NCol> &output)
        {
            for (size_t i = 0; i != XsimdSize; ++i)
            {
                Eigen::Matrix<Real, NRow, NCol> &eigen_vector = *(input + i);
                temp_.col(i) = Eigen::Map<Eigen::Matrix<Real, MatSize, 1>>(eigen_vector.data(), eigen_vector.size());
            }
            assign(output);
        };

        inline void gather(StdLargeVec<Eigen::Matrix<Real, NRow, NCol>> &input,
                           size_t *index, Eigen::Matrix<RealX, NRow, NCol> &output)
        {
            for (size_t i = 0; i != XsimdSize; ++i)
            {
                Eigen::Matrix<Real, NRow, NCol> &eigen_vector = input[*(index + i)];
                temp_.col(i) = Eigen::Map<Eigen::Matrix<Real, MatSize, 1>>(eigen_vector.data(), eigen_vector.size());
            }
            assign(output);
        };

        inline void reduce(const Eigen::Matrix<RealX, NRow, NCol> &input,
                           Eigen::Matrix<Real, NRow, NCol> &output)
        {
            for (size_t i = 0; i != NRow; ++i)
                for (size_t j = 0; j != NCol; ++j)
                {
                    output(i, j) = xsimd::reduce_add(input(i, j));
                }
        };

    private:
        inline void assign(Eigen::Matrix<RealX, NRow, NCol> &output)
        {
            temp_transpose_ = temp_.transpose();
            for (size_t i = 0; i != NRow; ++i)
                for (size_t j = 0; j != NCol; ++j)
                {
                    output(i, j) = xsimd::load_aligned(&temp_transpose_.col(j * NRow + i)[0]);
                }
        };
    };
    using Vec2dXHelper = MatXHelper<2, 1>;
    using Vec3dXHelper = MatXHelper<3, 1>;
    using Mat2dXHelper = MatXHelper<2, 2>;
    using Mat3dXHelper = MatXHelper<3, 3>;
}

#endif // XSIMD_EIGEN_H
