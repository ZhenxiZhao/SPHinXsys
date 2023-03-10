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
 * @file 	relax_dynamics.h
 * @brief 	This is the classes of particle relaxation in order to produce body fitted
 * 			initial particle distribution.
 * @author	Chi Zhang and Xiangyu Hu
 */

#ifndef RELAX_DYNAMICS_H
#define RELAX_DYNAMICS_H

#include "all_particle_dynamics.h"
#include "base_kernel.h"
#include "cell_linked_list.h"
#include "all_body_relations.h"
#include "general_dynamics.h"

namespace SPH
{
	class GeometryShape;
	class LevelSetShape;

	namespace relax_dynamics
	{
		typedef DataDelegateSimple<BaseParticles> RelaxDataDelegateSimple;

		typedef DataDelegateInner<BaseParticles> RelaxDataDelegateInner;

		typedef DataDelegateComplex<BaseParticles, BaseParticles> RelaxDataDelegateComplex;

		/**
		 * @class GetTimeStepSizeSquare
		 * @brief relaxation dynamics for particle initialization
		 * computing the square of time step size
		 */
		class GetTimeStepSizeSquare : public LocalDynamicsReduce<Real, ReduceMax>,
									  public RelaxDataDelegateSimple
		{
		protected:
			StdLargeVec<Vecd> &acc_;
			Real h_ref_;

		public:
			explicit GetTimeStepSizeSquare(SPHBody &sph_body);
			virtual ~GetTimeStepSizeSquare(){};

			Real reduce(size_t index_i, Real dt = 0.0);
			virtual Real outputResult(Real reduced_value);
		};

		/**
		 * @class RelaxationAccelerationInner
		 * @brief simple algorithm for physics relaxation
		 * without considering contact interaction.
		 * this is usually used for solid like bodies
		 */
		class RelaxationAccelerationInner : public LocalDynamics, public RelaxDataDelegateInner
		{
		public:
			explicit RelaxationAccelerationInner(BaseInnerRelation &inner_relation);
			virtual ~RelaxationAccelerationInner(){};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			StdLargeVec<Vecd> &acc_, &pos_;
		};

		/**
		 * @class RelaxationAccelerationInnerWithLevelSetCorrection
		 * @brief we constrain particles to a level function representing the interface.
		 */
		class RelaxationAccelerationInnerWithLevelSetCorrection : public RelaxationAccelerationInner
		{
		public:
			explicit RelaxationAccelerationInnerWithLevelSetCorrection(BaseInnerRelation &inner_relation);
			virtual ~RelaxationAccelerationInnerWithLevelSetCorrection() {};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			LevelSetShape *level_set_shape_;
			SPHAdaptation *sph_adaptation_;
		};

		/**
		 * @class RelaxationAccelerationFirstOrderInner
		 * @brief simple algorithm for physics relaxation
		 * without considering contact interaction by first order consistency.
		 */
		class RelaxationAccelerationFirstOrderInner : public LocalDynamics, public RelaxDataDelegateInner
		{
		public:
			explicit RelaxationAccelerationFirstOrderInner(BaseInnerRelation& inner_relation);
			virtual ~RelaxationAccelerationFirstOrderInner() {};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			Kernel* kernel_;
			StdLargeVec<Vecd>& acc_, & pos_;
			StdLargeVec<Matd>& stress_;
		};

		/**
		 * @class RelaxationAccelerationFirstOrderInnerWithLevelSetCorrection
		 * @brief we constrain particles to a level function representing the interface.
		 */
		class RelaxationAccelerationFirstOrderInnerWithLevelSetCorrection : public RelaxationAccelerationFirstOrderInner
		{
		public:
			explicit RelaxationAccelerationFirstOrderInnerWithLevelSetCorrection(
				BaseInnerRelation& inner_relation);
			virtual ~RelaxationAccelerationFirstOrderInnerWithLevelSetCorrection() {};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			LevelSetShape *level_set_shape_;
			SPHAdaptation *sph_adaptation_;
		};

		/**
		 * @class UpdateParticlePosition
		 * @brief update the particle position for a time step
		 */
		class UpdateParticlePosition : public LocalDynamics,
									   public RelaxDataDelegateSimple
		{
		protected:
			SPHAdaptation *sph_adaptation_;
			StdLargeVec<Vecd> &pos_, &acc_;

		public:
			explicit UpdateParticlePosition(SPHBody &sph_body);
			virtual ~UpdateParticlePosition(){};

			void update(size_t index_i, Real dt = 0.0);
		};

		/**
		 * @class UpdateSmoothingLengthRatioByShape
		 * @brief update the particle smoothing length ratio
		 */
		class UpdateSmoothingLengthRatioByShape : public LocalDynamics,
												  public RelaxDataDelegateSimple
		{
		protected:
			StdLargeVec<Real> &h_ratio_, &Vol_;
			StdLargeVec<Vecd> &pos_;
			Shape &target_shape_;
			ParticleRefinementByShape *particle_adaptation_;
			Real reference_spacing_;

		public:
			UpdateSmoothingLengthRatioByShape(SPHBody &sph_body, Shape &target_shape);
			explicit UpdateSmoothingLengthRatioByShape(SPHBody &sph_body);
			virtual ~UpdateSmoothingLengthRatioByShape(){};

			void update(size_t index_i, Real dt = 0.0);
		};

		/**
		 * @class RelaxationAccelerationComplex
		 * @brief compute relaxation acceleration while consider the present of contact bodies
		 * with considering contact interaction
		 * this is usually used for fluid like bodies
		 */
		class RelaxationAccelerationComplex : public LocalDynamics,
											  public RelaxDataDelegateComplex
		{
		public:
			explicit RelaxationAccelerationComplex(ComplexRelation &body_complex_relation);
			virtual ~RelaxationAccelerationComplex(){};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			StdLargeVec<Vecd> &acc_, &pos_;
		};

		/**
		 * @class ShapeSurfaceBounding
		 * @brief constrain surface particles by
		 * map constrained particles to geometry face and
		 * r = r + phi * norm (vector distance to face)
		 */
		class ShapeSurfaceBounding : public LocalDynamics,
									 public RelaxDataDelegateSimple
		{
		public:
			ShapeSurfaceBounding(NearShapeSurface &body_part);
			virtual ~ShapeSurfaceBounding(){};
			void update(size_t index_i, Real dt = 0.0);

		protected:
			StdLargeVec<Vecd> &pos_;
			LevelSetShape *level_set_shape_;
			Real constrained_distance_;
		};

		/**
		 * @class RelaxationStepInner
		 * @brief carry out particle relaxation step of particles within the body
		 */
		class RelaxationStepInner : public BaseDynamics<void>
		{
		public:
			explicit RelaxationStepInner(BaseInnerRelation &inner_relation,
										 bool level_set_correction = false);
			virtual ~RelaxationStepInner(){};
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface> &SurfaceBounding() { return surface_bounding_; };
			virtual void exec(Real dt = 0.0) override;
			virtual void parallel_exec(Real dt = 0.0) override;

		protected:
			RealBody *real_body_;
			BaseInnerRelation &inner_relation_;
			NearShapeSurface near_shape_surface_;
			UniquePtr<BaseDynamics<void>> relaxation_acceleration_inner_;
			ReduceDynamics<GetTimeStepSizeSquare> get_time_step_square_;
			SimpleDynamics<UpdateParticlePosition> update_particle_position_;
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface> surface_bounding_;
		};

		/**
		 * @class RelaxationStepFirstOrderInner
		 * @brief carry out particle relaxation step of particle within the body 
		 * with the first order consisitency.
		 */
		class RelaxationStepFirstOrderInner : public BaseDynamics<void>
		{
		public:
			explicit RelaxationStepFirstOrderInner(BaseInnerRelation& inner_relation,
				                                   bool level_set_correction = false);
			virtual ~RelaxationStepFirstOrderInner() {};
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface>& SurfaceBounding() { return surface_bounding_; };
			virtual void exec(Real dt = 0.0) override;
			virtual void parallel_exec(Real dt = 0.0) override;

		protected:
			RealBody* real_body_;
			BaseInnerRelation& inner_relation_;
			NearShapeSurface near_shape_surface_;
			UniquePtr<BaseDynamics<void>> relaxation_acceleration_inner_;
			ReduceDynamics<GetTimeStepSizeSquare> get_time_step_square_;
			SimpleDynamics<UpdateParticlePosition> update_particle_position_;
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface> surface_bounding_;
		};

		/**
		 * @class RelaxationAccelerationComplexWithLevelSetCorrection
		 * @brief compute relaxation acceleration while consider the present of contact bodies
		 * with considering contact interaction
		 * this is usually used for fluid like bodies
		 * we constrain particles with a level-set correction function when the fluid boundary is not contacted with solid.
		 */
		class RelaxationAccelerationComplexWithLevelSetCorrection : public RelaxationAccelerationComplex
		{
		public:
			RelaxationAccelerationComplexWithLevelSetCorrection(
				ComplexRelation &body_complex_relation, const std::string &shape_name);
			virtual ~RelaxationAccelerationComplexWithLevelSetCorrection(){};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			LevelSetShape *level_set_shape_;
			SPHAdaptation *sph_adaptation_;
		};

		/**
		 * @class RelaxationStepComplex
		 * @brief carry out particle relaxation step of particles within multi bodies
		 */
		class RelaxationStepComplex : public BaseDynamics<void>
		{
		public:
			explicit RelaxationStepComplex(ComplexRelation &body_complex_relation,
										   const std::string &shape_name, bool level_set_correction = false);
			virtual ~RelaxationStepComplex(){};
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface> &SurfaceBounding() { return surface_bounding_; };
			virtual void exec(Real dt = 0.0) override;
			virtual void parallel_exec(Real dt = 0.0) override;

		protected:
			RealBody *real_body_;
			ComplexRelation &complex_relation_;
			NearShapeSurface near_shape_surface_;
			UniquePtr<BaseDynamics<void>> relaxation_acceleration_complex_;
			ReduceDynamics<GetTimeStepSizeSquare> get_time_step_square_;
			SimpleDynamics<UpdateParticlePosition> update_particle_position_;
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface> surface_bounding_;
		};

		/** Here added by Bo for particle implicit relaxation and 0th and 1st consistency. */
		template <typename ErrorDataType, typename ParameterADataType, typename ParameterCDataType>
		struct ErrorAndParameters
		{
			ErrorDataType error_;
			ParameterADataType a_;
			ParameterCDataType c_;

			ErrorAndParameters<ErrorDataType, ParameterADataType, ParameterCDataType>() : error_(ZeroData<ErrorDataType>::value),
				                                                                          a_(ZeroData<ParameterADataType>::value),
				                                                                          c_(ZeroData<ParameterCDataType>::value) {};
		};

		/**
		 * @class RelaxationImplicitInner
		 * @brief carry out particle relaxation by position with implicit evolution.
		 */
		class RelaxationImplicitInner : public LocalDynamics, public RelaxDataDelegateInner
		{
		public:
			explicit RelaxationImplicitInner(BaseInnerRelation& inner_relation);
			virtual ~RelaxationImplicitInner() {};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			virtual ErrorAndParameters<Vecd, Matd, Matd> computeErrorAndParameters(size_t index_i, Real dt = 0.0);
			virtual void updateStates(size_t index_i, Real dt, const ErrorAndParameters<Vecd, Matd, Matd>& error_and_parameters);

			Kernel* kernel_;
			StdLargeVec<Real>& Vol_;
			StdLargeVec<Vecd>& pos_;
			StdLargeVec<Real> implicit_relaxation_residue_;
		};

		/**
		 * @class RelaxationImplicitInnerWithLevelSetCorrection
		 * @brief we constrain particles to a level function representing the interface.
		 */
		class RelaxationImplicitInnerWithLevelSetCorrection : public RelaxationImplicitInner
		{
		public:
			explicit RelaxationImplicitInnerWithLevelSetCorrection(BaseInnerRelation& inner_relation);
			virtual ~RelaxationImplicitInnerWithLevelSetCorrection() {};

		protected:
			virtual ErrorAndParameters<Vecd, Matd, Matd> computeErrorAndParameters(size_t index_i, Real dt = 0.0) override;

			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;
		};

		/**
		 * @class RelaxationEvolutionInner
		 * @brief carry out the particle relaxation evolution within the body
		 */
		class RelaxationEvolutionInner : public BaseDynamics<void>
		{
		public:
			explicit RelaxationEvolutionInner(BaseInnerRelation& inner_relation, bool level_set_correction = false);
			virtual ~RelaxationEvolutionInner() {};
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface>& SurfaceBounding() { return surface_bounding_; };
			virtual void exec(Real dt = 0.0) override;
			virtual void parallel_exec(Real dt = 0.0) override;

		protected:
			RealBody* real_body_;
			BaseInnerRelation& inner_relation_;
			NearShapeSurface near_shape_surface_;
			UniquePtr<BaseDynamics<void>> relaxation_evolution_inner_;
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface> surface_bounding_;
		};

		/** 
		 * @class RelaxationFirstOrderImplicitInner
		 * @brief carry out particle relaxation by first ordre consistency implicit evolution.
		 */
		class RelaxationFirstOrderImplicitInner : public LocalDynamics, public RelaxDataDelegateInner
		{
		public:
			explicit RelaxationFirstOrderImplicitInner(BaseInnerRelation& inner_relation);
			virtual ~RelaxationFirstOrderImplicitInner() {};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			virtual ErrorAndParameters<Vecd, Matd, Matd> computeErrorAndParameters(size_t index_i, Real dt = 0.0);
			virtual void updateStates(size_t index_i, Real dt, const ErrorAndParameters<Vecd, Matd, Matd>& error_and_parameters);

			Kernel* kernel_;
			StdLargeVec<Real>& Vol_;
			StdLargeVec<Vecd>& pos_;
			StdLargeVec<Matd>& stress_;
			StdLargeVec<Real> implicit_first_order_relaxation_residue_;
		};
		
		/**
		 * @class RelaxationFirstOrderEvolutionInnerWithLevelSetCorrection
		 * @brief we constrain particles to a level function representing the interface.
		 */
		class RelaxationFirstOrderImplicitInnerWithLevelSetCorrection : public RelaxationFirstOrderImplicitInner
		{
		public:
			explicit RelaxationFirstOrderImplicitInnerWithLevelSetCorrection(BaseInnerRelation& inner_relation);
			virtual ~RelaxationFirstOrderImplicitInnerWithLevelSetCorrection() {};

		protected:
			virtual ErrorAndParameters<Vecd, Matd, Matd> computeErrorAndParameters(size_t index_i, Real dt = 0.0) override;

			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;
		};

		/**
		 * @class RelaxationFirstOrderEvolutionInner
		 * @brief carry out the particle relaxation evolution from first order consistency within the body
		 */
		class RelaxationFirstOrderEvolutionInner : public BaseDynamics<void>
		{
		public:
			explicit RelaxationFirstOrderEvolutionInner(BaseInnerRelation& inner_relation, bool level_set_correction = false);
			virtual ~RelaxationFirstOrderEvolutionInner() {};
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface>& SurfaceBounding() { return surface_bounding_; };
			virtual void exec(Real dt = 0.0) override;
			virtual void parallel_exec(Real dt = 0.0) override;

		protected:
			RealBody* real_body_;
			BaseInnerRelation& inner_relation_;
			NearShapeSurface near_shape_surface_;
			UniquePtr<BaseDynamics<void>> relaxation_evolution_inner_;
			SimpleDynamics<ShapeSurfaceBounding, NearShapeSurface> surface_bounding_;
		};

		/**
		 * @class CalculateParticleVolume
		 * @brief calculate the particle volume by kernel integral
		 */
		class CalculateParticleVolume : public LocalDynamics, public RelaxDataDelegateInner
		{
		public:
			explicit CalculateParticleVolume(BaseInnerRelation& inner_relation);
			virtual ~CalculateParticleVolume() {};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			Real Vol_0_;
			Real W0_;
			StdLargeVec<Real> &Vol_;
			StdLargeVec<Vecd> &pos_;
			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;
		};

		/**
		 * @class CalcualteParticleStress
		 * @brief calculate the particle stress with first order consistency
		 */
		class CalculateParticleStress : public LocalDynamics, public RelaxDataDelegateInner
		{
		public:
			explicit CalculateParticleStress(BaseInnerRelation& inner_relation);
			virtual ~CalculateParticleStress() {};
			void interaction(size_t index_i, Real dt = 0.0);

		protected:
			StdLargeVec<Vecd> pos_;
			StdLargeVec<Matd> stress_;
			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;

		};

		/**
		 * @class ZeroOrderConsistencyEvolution
		 * @brief modify the particle volume to satisfy the zero order consistency. 
		 */
		class ZeroOrderConsistencyEvolution : public LocalDynamics, public RelaxDataDelegateInner
		{
		public:
			ZeroOrderConsistencyEvolution(BaseInnerRelation& inner_relation, const std::string correction_matrix);
			virtual ~ZeroOrderConsistencyEvolution() {};

		protected:
			virtual ErrorAndParameters<Vecd, Matd, Matd> computeErrorAndParameters(size_t index_i, Real dt = 0.0);
			virtual void updateStates(size_t index_i, Real dt, const ErrorAndParameters<Vecd, Matd, Matd>& error_and_parameters);
			void interaction(size_t index_i, Real dt = 0.0);

			Kernel* kernel_;
			StdLargeVec<Real>& Vol_;
			StdLargeVec<Vecd>& pos_;
			StdLargeVec<Matd>& B_;
			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;
			StdLargeVec<Real> residue_zero_order_consistency_;
		};

		class VolumetricMeasureConstrain : public LocalDynamics, public RelaxDataDelegateSimple
		{
		public:
			VolumetricMeasureConstrain(SPHBody& sph_body);
			virtual ~VolumetricMeasureConstrain() {};
			void UpdateInitialAveragedParameter(Real initial_averaged_volume_measure);
			void UpdateAveragedParameter(Real new_averaged_volume_measure);

		protected:
			Real initial_averaged_volume_measure_;
			Real new_averaged_volume_measure_;
			StdLargeVec<Real>& local_volume_measure_;
			void update(size_t index_i, Real dt = 0.0);
		};

		/**
		 * @class FirstOrderConsistencyEvolution
		 * @brief modify the B matrix to satisfy the first order consistency.
		 */
		class FirstOrderConsistencyEvolution : public LocalDynamics, public RelaxDataDelegateInner
		{
		public:
			FirstOrderConsistencyEvolution(BaseInnerRelation& inner_relation, const std::string correction_matrix);
			virtual ~FirstOrderConsistencyEvolution() {};

		protected:
			virtual ErrorAndParameters<Matd, Matd, Matd> computeErrorAndParameters(size_t index_i, Real dt = 0.0);
			virtual void updateStates(size_t index_i, Real dt, const ErrorAndParameters<Matd, Matd, Matd>& error_and_parameters);
			void interaction(size_t index_i, Real dt = 0.0);

			Kernel* kernel_;
			StdLargeVec<Real>& Vol_;
			StdLargeVec<Vecd>& pos_;
			StdLargeVec<Matd>& B_;
			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;
			StdLargeVec<Real> residue_first_order_consistency_;
		};

		/**
		 * @class ModificationStepForConsistency
		 * @brief modify the volume and B matrix for 0 order and 1 order consistency.
		 */
		class ModificationStepForConsistency : public BaseDynamics<void>, GeneralDataDelegateInner
		{
		public:
			ModificationStepForConsistency(BaseInnerRelation& inner_relation);
			virtual ~ModificationStepForConsistency() {};
			virtual void exec(Real dt = 0) override;
			virtual void parallel_exec(Real dt = 0) override;

		protected:
			StdLargeVec<Real> Vol_;
			StdLargeVec<Matd> B_;
			UniquePtr<BaseDynamics<void>> zero_order_consistency_evolution;
			ReduceAverage<QuantitySummation<Real>> total_averaged_volume_measure;
			SimpleDynamics<VolumetricMeasureConstrain> volume_measure_constrain;
			UniquePtr<BaseDynamics<void>> first_order_consistency_evolution;
		};

		/**
		 * @class UpdateParticleKineticEnergy
		 * @brief calculate the particle kinetic energy
		 */
		class UpdateParticleKineticEnergy : public LocalDynamics, public GeneralDataDelegateInner
		{
		public:
			UpdateParticleKineticEnergy(BaseInnerRelation& inner_relation);
			virtual ~UpdateParticleKineticEnergy() {};
			void interaction(size_t index_i, Real dt);

		protected:
			StdLargeVec<Vecd>& pos_;
			StdLargeVec<Real>& mass_;
			StdLargeVec<Matd>& stress_;
			StdLargeVec<Real> particle_kinetic_energy;
			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;
		};

		/**
		 * @class CheckZeroOrderConsistency
		 * @breif calculate the zero order consistency from kernel
		 */
		class CheckZeroOrderConsistency : public LocalDynamics, public GeneralDataDelegateInner
		{
		public:
			CheckZeroOrderConsistency(BaseInnerRelation& inner_relation);
			virtual ~CheckZeroOrderConsistency() {};
			void interaction(size_t index_i, Real dt);

		protected:
			Real W0_;
			StdLargeVec<Real>& Vol_;
			StdLargeVec<Vecd>& pos_;
			StdLargeVec<Matd>& stress_;
			StdLargeVec<Real> unit_zero_order_;
			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;
		};

		/**
		 * @class CheckFirstOrderConistency
		 * @brief calculate the first order consistency
		 */
		class CheckFirstOrderConsistency : public LocalDynamics, public GeneralDataDelegateInner
		{
		public:
			CheckFirstOrderConsistency(BaseInnerRelation& inner_relation);
			virtual ~CheckFirstOrderConsistency() {};
			void interaction(size_t index_i, Real dt);

		protected:
			StdLargeVec<Real>& Vol_;
			StdLargeVec<Vecd>& pos_;
			StdLargeVec<Matd>& stress_;
			StdLargeVec<Real> first_order_;
			LevelSetShape* level_set_shape_;
			SPHAdaptation* sph_adaptation_;
		};

		/**
		 * @class ShellMidSurfaceBounding
		 * @brief constrain particles by constraining particles to mid-surface.
		 * Note that level_set_refinement_ratio should be smaller than particle_spacing_ref_ / (0.05 * thickness_)
		 * because if level_set_refinement_ratio > particle_spacing_ref_ / (0.05 * thickness_),
		 * there will be no level set field.
		 */
		class ShellMidSurfaceBounding : public LocalDynamics,
										public RelaxDataDelegateInner
		{
		public:
			ShellMidSurfaceBounding(NearShapeSurface &body_part, BaseInnerRelation &inner_relation,
									Real thickness, Real level_set_refinement_ratio);
			virtual ~ShellMidSurfaceBounding(){};
			void update(size_t index_i, Real dt = 0.0);

		protected:
			StdLargeVec<Vecd> &pos_;
			Real constrained_distance_;
			Real particle_spacing_ref_, thickness_, level_set_refinement_ratio_;
			LevelSetShape *level_set_shape_;
		};

		/**
		 * @class ShellNormalDirectionPrediction
		 * @brief predict the normal direction of shell particles.
		 */
		class ShellNormalDirectionPrediction : public BaseDynamics<void>
		{
			const Real convergence_criterion_;
			const Real consistency_criterion_;

			void predictNormalDirection();
			void correctNormalDirection();

		public:
			explicit ShellNormalDirectionPrediction(BaseInnerRelation &inner_relation,
													Real thickness, Real consistency_criterion = cos(Pi / 20.0));
			virtual ~ShellNormalDirectionPrediction(){};

			virtual void exec(Real dt = 0.0) override;
			virtual void parallel_exec(Real dt = 0.0) override { exec(); };

		protected:
			class NormalPrediction : public RelaxDataDelegateSimple, public LocalDynamics
			{
				Real thickness_;
				LevelSetShape *level_set_shape_;
				StdLargeVec<Vecd> &pos_, &n_, n_temp_;

			public:
				NormalPrediction(SPHBody &sph_body, Real thickness);
				virtual ~NormalPrediction(){};
				void update(size_t index_i, Real dt = 0.0);
			};

			class PredictionConvergenceCheck : public LocalDynamicsReduce<bool, ReduceAND>,
											   public RelaxDataDelegateSimple
			{
			protected:
				const Real convergence_criterion_;
				StdLargeVec<Vecd> &n_, &n_temp_;

			public:
				PredictionConvergenceCheck(SPHBody &sph_body, Real convergence_criterion);
				virtual ~PredictionConvergenceCheck(){};

				bool reduce(size_t index_i, Real dt = 0.0);
			};

			class ConsistencyCorrection : public LocalDynamics, public RelaxDataDelegateInner
			{
			public:
				explicit ConsistencyCorrection(BaseInnerRelation &inner_relation, Real consistency_criterion);
				virtual ~ConsistencyCorrection(){};
				void interaction(size_t index_i, Real dt = 0.0);

			protected:
				std::mutex mutex_modify_neighbor_; /**< mutex exclusion for memory conflict */
				const Real consistency_criterion_;
				StdLargeVec<int> updated_indicator_; /**> 0 not updated, 1 updated with reliable prediction, 2 updated from a reliable neighbor */
				StdLargeVec<Vecd> &n_;
			};

			class ConsistencyUpdatedCheck : public LocalDynamicsReduce<bool, ReduceAND>,
											public RelaxDataDelegateSimple
			{
			protected:
				StdLargeVec<int> &updated_indicator_;

			public:
				explicit ConsistencyUpdatedCheck(SPHBody &sph_body);
				virtual ~ConsistencyUpdatedCheck(){};

				bool reduce(size_t index_i, Real dt = 0.0);
			};

			class SmoothingNormal : public ParticleSmoothing<Vecd>
			{
			public:
				explicit SmoothingNormal(BaseInnerRelation &inner_relation);
				virtual ~SmoothingNormal(){};
				void update(size_t index_i, Real dt = 0.0);

			protected:
			};

			SimpleDynamics<NormalPrediction> normal_prediction_;
			ReduceDynamics<PredictionConvergenceCheck> normal_prediction_convergence_check_;
			InteractionDynamics<ConsistencyCorrection> consistency_correction_;
			ReduceDynamics<ConsistencyUpdatedCheck> consistency_updated_check_;
			InteractionWithUpdate<SmoothingNormal> smoothing_normal_;
		};

		/**
		 * @class ShellRelaxationStepInner
		 * @brief carry out particle relaxation step of particles within the shell body
		 */
		class ShellRelaxationStepInner : public RelaxationStepInner
		{
		public:
			explicit ShellRelaxationStepInner(BaseInnerRelation &inner_relation, Real thickness,
											  Real level_set_refinement_ratio, bool level_set_correction = false);
			virtual ~ShellRelaxationStepInner(){};

			SimpleDynamics<UpdateParticlePosition> update_shell_particle_position_;
			SimpleDynamics<ShellMidSurfaceBounding, NearShapeSurface> mid_surface_bounding_;

			virtual void exec(Real dt = 0.0) override;
			virtual void parallel_exec(Real dt = 0.0) override;
		};
	}
}
#endif // RELAX_DYNAMICS_H