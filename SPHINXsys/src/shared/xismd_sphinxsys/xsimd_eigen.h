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
#define EIGEN_DONT_VECTORIZE

namespace SPH
{
    using RealX = xsimd::batch<Real, xsimd::default_arch>;
    constexpr size_t XsimdSize = xsimd::simd_type<Real>::size;
}

namespace Eigen
{

    template <>
    struct NumTraits<SPH::RealX>
        : NumTraits<SPH::Real> // permits to get the epsilon, dummy_precision, lowest, highest functions
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
    /** Vector with float point number in batches.*/
    using Vec2X = Eigen::Matrix<RealX, 2, 1>;
    using Vec3X = Eigen::Matrix<RealX, 3, 1>;
    /** Small, 2*2 and 3*3, matrix with float point number in batches. */
    using Mat2X = Eigen::Matrix<RealX, 2, 2>;
    using Mat3X = Eigen::Matrix<RealX, 3, 3>;
}

#endif // XSIMD_EIGEN_H
