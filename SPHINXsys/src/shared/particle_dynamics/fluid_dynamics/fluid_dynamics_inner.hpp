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
 *  HU1527/12-1 and HU1527/12-4													*
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
 * @file 	fluid_dynamics_inner.hpp
 * @brief 	Here, we define the algorithm classes for fluid dynamics within the body.
 * @details 	We consider here weakly compressible fluids. The algorithms may be
 * 			different for free surface flow and the one without free surface.
 * @author	Chi ZHang and Xiangyu Hu
 */
#pragma once

#include "fluid_dynamics_inner.h"

namespace SPH
{
	//=====================================================================================================//
	namespace fluid_dynamics
	{
		//=================================================================================================//
		template <class RiemannSolverType>
		BaseIntegration1stHalf<RiemannSolverType>::BaseIntegration1stHalf(BaseInnerRelation &inner_relation)
			: BaseIntegration(inner_relation), riemann_solver_(fluid_, fluid_) {}
		//=================================================================================================//
		template <class RiemannSolverType>
		void BaseIntegration1stHalf<RiemannSolverType>::initialization(size_t index_i, Real dt)
		{
			rho_[index_i] += drho_dt_[index_i] * dt * 0.5;
			p_[index_i] = fluid_.getPressure(rho_[index_i]);
			pos_[index_i] += vel_[index_i] * dt * 0.5;
		}
		//=================================================================================================//
		template <class RiemannSolverType>
		void BaseIntegration1stHalf<RiemannSolverType>::update(size_t index_i, Real dt)
		{
			vel_[index_i] += (acc_prior_[index_i] + acc_[index_i]) * dt;
		}
		//=================================================================================================//
		template <class RiemannSolverType>
		Vecd BaseIntegration1stHalf<RiemannSolverType>::computeNonConservativeAcceleration(size_t index_i)
		{
			Vecd acceleration = acc_prior_[index_i] * rho_[index_i];
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Real dW_ijV_j = inner_neighborhood.dW_ijV_j_[n];
				const Vecd &e_ij = inner_neighborhood.e_ij_[n];

				acceleration += (p_[index_i] - p_[index_j]) * dW_ijV_j * e_ij;
			}
			return acceleration / rho_[index_i];
		}
		//=================================================================================================//
		template <class RiemannSolverType>
		void BaseIntegration1stHalf<RiemannSolverType>::interaction(size_t index_i, Real dt)
		{
			Neighborhood &ngh = inner_configuration_[index_i];
			size_t floor_size = ngh.current_size_ - ngh.current_size_ % XsimdSize;

			VecdX x_acceleration = VecdX::Zero();
			RealX x_rho_dissipation(0);
			RealX x_p_i = RealX(p_[index_i]);
			for (size_t n = 0; n < floor_size; n += XsimdSize)
			{
				RealX x_dW_ijV_j = loadRealX(&ngh.dW_ijV_j_[n]);
				RealX x_p_j = gatherRealX<XsimdSize>(p_, &ngh.j_[n]);

				x_acceleration -= (x_p_i + x_p_j) * x_dW_ijV_j * loadVecdX<XsimdSize>(&ngh.e_ij_[n]);
				x_rho_dissipation += riemann_solver_.DissipativeUJump(x_p_i - x_p_j) * x_dW_ijV_j;
			}

			Vecd acceleration = reduceVecdX(x_acceleration);
			Real rho_dissipation = reduceRealX(x_rho_dissipation);
			Real p_i = p_[index_i];
			for (size_t n = floor_size; n != ngh.current_size_; ++n)
			{
				Real dW_ijV_j = ngh.dW_ijV_j_[n];
				Real p_j = p_[ngh.j_[n]];

				acceleration -= (p_i + p_j) * dW_ijV_j * ngh.e_ij_[n];
				rho_dissipation += riemann_solver_.DissipativeUJump(p_i - p_j) * dW_ijV_j;
			}

			acc_[index_i] += acceleration / rho_[index_i];
			drho_dt_[index_i] = rho_dissipation * rho_[index_i];
		}
		//=================================================================================================//
		template <class RiemannSolverType>
		BaseIntegration2ndHalf<RiemannSolverType>::BaseIntegration2ndHalf(BaseInnerRelation &inner_relation)
			: BaseIntegration(inner_relation), riemann_solver_(fluid_, fluid_),
			  Vol_(particles_->Vol_), mass_(particles_->mass_) {}
		//=================================================================================================//
		template <class RiemannSolverType>
		void BaseIntegration2ndHalf<RiemannSolverType>::initialization(size_t index_i, Real dt)
		{
			pos_[index_i] += vel_[index_i] * dt * 0.5;
		}
		//=================================================================================================//
		template <class RiemannSolverType>
		void BaseIntegration2ndHalf<RiemannSolverType>::update(size_t index_i, Real dt)
		{
			rho_[index_i] += drho_dt_[index_i] * dt * 0.5;
			Vol_[index_i] = mass_[index_i] / rho_[index_i];
		}
		//=================================================================================================//
		template <class RiemannSolverType>
		void BaseIntegration2ndHalf<RiemannSolverType>::interaction(size_t index_i, Real dt)
		{
			Neighborhood &ngh = inner_configuration_[index_i];
			size_t floor_size = ngh.current_size_ - ngh.current_size_ % XsimdSize;

			RealX x_density_change_rate(0);
			VecdX x_p_dissipation = VecdX::Zero();
			VecdX x_vel_i = assignVecdX(vel_[index_i]);
			for (size_t n = 0; n < floor_size; n += XsimdSize)
			{
				RealX x_dW_ijV_j = loadRealX(&ngh.dW_ijV_j_[n]);
				VecdX x_e_ij = loadVecdX<XsimdSize>(&ngh.e_ij_[n]);

				RealX x_u_jump = (x_vel_i - gatherVecdX<XsimdSize>(vel_, &ngh.j_[n])).dot(x_e_ij);
				x_density_change_rate += x_u_jump * x_dW_ijV_j;
				x_p_dissipation += riemann_solver_.DissipativePJump(x_u_jump) * x_dW_ijV_j * x_e_ij;
			}

			Real density_change_rate = reduceRealX(x_density_change_rate);
			Vecd p_dissipation = reduceVecdX(x_p_dissipation);
			const Vecd &vel_i = vel_[index_i];
			for (size_t n = floor_size; n != ngh.current_size_; ++n)
			{
				Real dW_ijV_j = ngh.dW_ijV_j_[n];
				const Vecd &e_ij = ngh.e_ij_[n];

				Real u_jump = (vel_i - vel_[ngh.j_[n]]).dot(e_ij);
				density_change_rate += u_jump * dW_ijV_j;
				p_dissipation += riemann_solver_.DissipativePJump(u_jump) * dW_ijV_j * e_ij;
			}

			drho_dt_[index_i] += density_change_rate * rho_[index_i];
			acc_[index_i] = p_dissipation / rho_[index_i];
		};
		//=================================================================================================//
	}
	//=================================================================================================//
}
//=================================================================================================//