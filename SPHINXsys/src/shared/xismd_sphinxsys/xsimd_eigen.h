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

#include <base_data_type.h>

#include <include/xsimd/xsimd.hpp>

namespace SPH
{
    constexpr size_t XsimdSize = xsimd::simd_type<Real>::size;
    using RealX = xsimd::batch<Real, xsimd::default_arch>;

    inline RealX loadRealX(Real *input)
    {
        return xsimd::load_aligned(input);
    }

    template <int XSIMD_SIZE>
    RealX gatherRealX(StdLargeVec<Real> &input, size_t *index)
    {
        std::cout << "\n Error: getherRealX is not defined for the native architecture !" << std::endl;
        std::cout << __FILE__ << ':' << __LINE__ << std::endl;
        exit(1);
        return RealX();
    }

    template <>
    inline RealX gatherRealX<4>(StdLargeVec<Real> &input, size_t *index)
    {
        return RealX(input[*index], input[*(index + 1)], input[*(index + 2)], input[*(index + 3)]);
    }

    inline Real reduceRealX(const RealX &input)
    {
        return xsimd::reduce_add(input);
    }
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

    inline Vec2dX assignVecdX(const Vec2d &input)
    {
        return Vec2dX(RealX(input[0]), RealX(input[1]));
    }

    inline Vec3dX assignVecdX(const Vec3d &input)
    {
        return Vec3dX(RealX(input[0]), RealX(input[1]), RealX(input[2]));
    }

    template <int XSIMD_SIZE, int DIMENSION>
    Eigen::Matrix<RealX, DIMENSION, 1> loadVecdX(Eigen::Matrix<Real, DIMENSION, 1> *input)
    {
        std::cout << "\n Error: loadVecdX is not defined for the native architecture !" << std::endl;
        std::cout << __FILE__ << ':' << __LINE__ << std::endl;
        exit(1);
        return Eigen::Matrix<RealX, DIMENSION, 1>();
    }

    template <>
    inline Vec2dX loadVecdX<4>(Vec2d *input)
    {
        return Vec2dX(RealX((*input)[0], (*(input + 1))[0], (*(input + 2))[0], (*(input + 3))[0]),
                      RealX((*input)[1], (*(input + 1))[1], (*(input + 2))[1], (*(input + 3))[1]));
    }

    template <>
    inline Vec3dX loadVecdX<4>(Vec3d *input)
    {
        Real temp[3][4];
        for (size_t i = 0; i != 4; ++i)
            for (size_t j = 0; j != 3; ++j)
            {
                temp[j][i] = (*(input + i))[j];
            }
        return Vec3dX(loadRealX(&temp[0][0]), loadRealX(&temp[1][0]), loadRealX(&temp[2][0]));
    }

    template <int XSIMD_SIZE, int DIMENSION>
    Eigen::Matrix<RealX, DIMENSION, 1> gatherVecdX(StdLargeVec<Eigen::Matrix<Real, DIMENSION, 1>> &input, size_t *index)
    {
        std::cout << "\n Error: gatherVecdX is not defined for the native architecture !" << std::endl;
        std::cout << __FILE__ << ':' << __LINE__ << std::endl;
        exit(1);
        return Eigen::Matrix<RealX, DIMENSION, 1>();
    }

    template <>
    inline Vec2dX gatherVecdX<4>(StdLargeVec<Vec2d> &input, size_t *index)
    {
        return Vec2dX(RealX(input[*index][0], input[*(index + 1)][0], input[*(index + 2)][0], input[*(index + 3)][0]),
                      RealX(input[*index][1], input[*(index + 1)][1], input[*(index + 2)][1], input[*(index + 3)][1]));
    }

    template <>
    inline Vec3dX gatherVecdX<4>(StdLargeVec<Vec3d> &input, size_t *index)
    {
        Real temp[3][4];
        for (size_t i = 0; i != 4; ++i)
            for (size_t j = 0; j != 3; ++j)
            {
                temp[j][i] = input[*(index + i)][j];
            }

        return Vec3dX(loadRealX(&temp[0][0]), loadRealX(&temp[1][0]), loadRealX(&temp[2][0]));
    }

    inline Vec2d reduceVecdX(const Vec2dX &input)
    {
        return Vec2d(xsimd::reduce_add(input[0]), xsimd::reduce_add(input[1]));
    }

    inline Vec3d reduceVecdX(const Vec3dX &input)
    {
        return Vec3d(xsimd::reduce_add(input[0]), xsimd::reduce_add(input[1]), xsimd::reduce_add(input[2]));
    }

    template <int XSIMD_SIZE, int DIMENSION>
    Eigen::Matrix<RealX, DIMENSION, DIMENSION> loadMatdX(Eigen::Matrix<Real, DIMENSION, DIMENSION> *input)
    {
        std::cout << "\n Error: loadMatdX is not defined for the native architecture !" << std::endl;
        std::cout << __FILE__ << ':' << __LINE__ << std::endl;
        exit(1);
        return Eigen::Matrix<RealX, DIMENSION, DIMENSION>();
    }

    template <>
    inline Mat2dX loadMatdX<4>(Mat2d *input)
    {
        return Mat2dX{
            {RealX((*input)(0, 0), (*(input + 1))(0, 0), (*(input + 2))(0, 0), (*(input + 3))(0, 0)),
             RealX((*input)(0, 1), (*(input + 1))(0, 1), (*(input + 2))(0, 1), (*(input + 3))(0, 1))},
            {RealX((*input)(1, 0), (*(input + 1))(1, 0), (*(input + 2))(1, 0), (*(input + 3))(1, 0)),
             RealX((*input)(1, 1), (*(input + 1))(1, 1), (*(input + 2))(1, 1), (*(input + 3))(1, 1))}};
    }

    template <>
    inline Mat3dX loadMatdX<4>(Mat3d *input)
    {
        return Mat3dX{
            {RealX((*input)(0, 0), (*(input + 1))(0, 0), (*(input + 2))(0, 0), (*(input + 3))(0, 0)),
             RealX((*input)(0, 1), (*(input + 1))(0, 1), (*(input + 2))(0, 1), (*(input + 3))(0, 1)),
             RealX((*input)(0, 2), (*(input + 1))(0, 2), (*(input + 2))(0, 2), (*(input + 3))(0, 2))},
            {RealX((*input)(1, 0), (*(input + 1))(1, 0), (*(input + 2))(1, 0), (*(input + 3))(1, 0)),
             RealX((*input)(1, 1), (*(input + 1))(1, 1), (*(input + 2))(1, 1), (*(input + 3))(1, 1)),
             RealX((*input)(1, 2), (*(input + 1))(1, 2), (*(input + 2))(1, 2), (*(input + 3))(1, 2))},
            {RealX((*input)(2, 0), (*(input + 1))(2, 0), (*(input + 2))(2, 0), (*(input + 3))(2, 0)),
             RealX((*input)(2, 1), (*(input + 1))(2, 1), (*(input + 2))(2, 1), (*(input + 3))(2, 1)),
             RealX((*input)(2, 2), (*(input + 1))(2, 2), (*(input + 2))(2, 2), (*(input + 3))(2, 2))}};
    }

    template <int XSIMD_SIZE, int DIMENSION>
    Eigen::Matrix<RealX, DIMENSION, DIMENSION> gatherMatdX(StdLargeVec<Eigen::Matrix<Real, DIMENSION, DIMENSION>> &input, size_t *index)
    {
        std::cout << "\n Error: gatherMatdX is not defined for the native architecture !" << std::endl;
        std::cout << __FILE__ << ':' << __LINE__ << std::endl;
        exit(1);
        return Eigen::Matrix<RealX, DIMENSION, DIMENSION>();
    }

    template <>
    inline Mat2dX gatherMatdX<4>(StdLargeVec<Mat2d> &input, size_t *index)
    {
        return Mat2dX{
            {RealX(input[*index](0, 0), input[*(index + 1)](0, 0), input[*(index + 2)](0, 0), input[*(index + 3)](0, 0)),
             RealX(input[*index](0, 1), input[*(index + 1)](0, 1), input[*(index + 2)](0, 1), input[*(index + 3)](0, 1))},
            {RealX(input[*index](1, 0), input[*(index + 1)](1, 0), input[*(index + 2)](1, 0), input[*(index + 3)](1, 0)),
             RealX(input[*index](1, 1), input[*(index + 1)](1, 1), input[*(index + 2)](1, 1), input[*(index + 3)](1, 1))}};
    }

    template <>
    inline Mat3dX gatherMatdX<4>(StdLargeVec<Mat3d> &input, size_t *index)
    {
        return Mat3dX{
            {RealX(input[*index](0, 0), input[*(index + 1)](0, 0), input[*(index + 2)](0, 0), input[*(index + 3)](0, 0)),
             RealX(input[*index](0, 1), input[*(index + 1)](0, 1), input[*(index + 2)](0, 1), input[*(index + 3)](0, 1)),
             RealX(input[*index](0, 2), input[*(index + 1)](0, 2), input[*(index + 2)](0, 2), input[*(index + 3)](0, 2))},
            {RealX(input[*index](1, 0), input[*(index + 1)](1, 0), input[*(index + 2)](1, 0), input[*(index + 3)](1, 0)),
             RealX(input[*index](1, 1), input[*(index + 1)](1, 1), input[*(index + 2)](1, 1), input[*(index + 3)](1, 1)),
             RealX(input[*index](1, 2), input[*(index + 1)](1, 2), input[*(index + 2)](1, 2), input[*(index + 3)](1, 2))},
            {RealX(input[*index](2, 0), input[*(index + 1)](2, 0), input[*(index + 2)](2, 0), input[*(index + 3)](2, 0)),
             RealX(input[*index](2, 1), input[*(index + 1)](2, 1), input[*(index + 2)](2, 1), input[*(index + 3)](2, 1)),
             RealX(input[*index](2, 2), input[*(index + 1)](2, 2), input[*(index + 2)](2, 2), input[*(index + 3)](2, 2))}};
    }

    inline Mat2d reduceMatdX(const Mat2dX &input)
    {
        return Mat2d{{xsimd::reduce_add(input(0, 0)), xsimd::reduce_add(input(0, 1))},
                     {xsimd::reduce_add(input(1, 0)), xsimd::reduce_add(input(1, 1))}};
    }

    inline Mat3d reduceMatdX(const Mat3dX &input)
    {
        return Mat3d{{xsimd::reduce_add(input(0, 0)), xsimd::reduce_add(input(0, 1)), xsimd::reduce_add(input(0, 2))},
                     {xsimd::reduce_add(input(1, 0)), xsimd::reduce_add(input(1, 1)), xsimd::reduce_add(input(1, 2))},
                     {xsimd::reduce_add(input(2, 0)), xsimd::reduce_add(input(2, 1)), xsimd::reduce_add(input(2, 2))}};
    }
}

#endif // XSIMD_EIGEN_H
