/**
 * @file 	heat_flux_steady.cpp
 * @brief 	This is the second test to demonstrate SPHInXsys as an optimization tool.
 * @details Consider a 2d block thermal domain with two constant temperature regions at the upper
 *          boundary and one heat flux region at the lower boundary. The optimization target is
 *			to achieve the lowest average temperature on the flux boundary by modifying the 
 *          distribution of the thermal diffusion rate in the domain with an extra conservation 
 *			constraint that the integral of the thermal diffusion rate in the entire domain is constant.
 * @author 	Bo Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" // using SPHinXsys library
using namespace SPH;   // namespace cite here
//----------------------------------------------------------------------
//	Global geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;					 // inner domain length
Real H = 1.0;					 // inner domain height
Real resolution_ref = H / 100.0; // reference resolution for discretization
Real BW = resolution_ref * 2.0;	 // boundary width
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Global parameters for physics state variables.
//----------------------------------------------------------------------
std::string variable_name = "Phi";
std::string flux_name = "HeatFlux";
std::string normal_vector = "UnitNormalVector";
std::string residue_name = "ThermalEquationResidue";
Real left_temperature = 300.0;
Real right_temperature = 350.0;
Real heat_flux = 2000;
Real heat_source = 0.0;
Real reference_temperature = right_temperature - left_temperature;
Real target_strength = 200.0;
Real learning_strength_ref = 1.0;
//----------------------------------------------------------------------
//	Global parameters for material properties or coefficient variables.
//----------------------------------------------------------------------
std::string coefficient_name = "ThermalDiffusivity";
std::string reference_coefficient = "ReferenceThermalDiffusivity";
Real diffusion_coff = 1.0;
//----------------------------------------------------------------------
//	Geometric regions used in the system.
//----------------------------------------------------------------------
Vec2d block_halfsize = Vec2d(0.5 * L, 0.5 * H);					 // local center at origin
Vec2d block_translation = block_halfsize;						 // translation to global coordinates
Vec2d constraint_halfsize = Vec2d(0.05 * L, 0.5 * BW);			 // constraint block half size
Vec2d left_constraint_translation = Vec2d(0.35 * L, L + 0.5 * BW); // left constraint
Vec2d right_constraint_translation = Vec2d(0.65 * L, L + 0.5 * BW); // right constraint
Vec2d lower_constraint_translation = Vec2d(0.5 * L, -0.5 * BW); // lower constraint
class Boundaries : public ComplexShape
{
public:
	explicit Boundaries(const std::string& shape_name)
		: ComplexShape(shape_name)
	{
		add<TransformShape<GeometricShapeBox>>(Transform2d(left_constraint_translation), constraint_halfsize);
		add<TransformShape<GeometricShapeBox>>(Transform2d(right_constraint_translation), constraint_halfsize);
		add<TransformShape<GeometricShapeBox>>(Transform2d(lower_constraint_translation), constraint_halfsize);
	}
};
std::vector<Vecd> flux_boundary_domain{
	Vecd(0.45*L,0), Vecd(0.45*L, resolution_ref), Vecd(0.55 * L, resolution_ref),
	Vecd(0.55 * L, 0), Vecd(0.45 * L, 0) };
MultiPolygon createFluxBoundaryDomain()
{
	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(flux_boundary_domain, ShapeBooleanOps::add);
	return multi_polygon;
}
//----------------------------------------------------------------------
//	Initial condition for temperature.
//----------------------------------------------------------------------
class DiffusionBodyInitialCondition : public ValueAssignment<Real>
{
public:
	explicit DiffusionBodyInitialCondition(SPHBody &diffusion_body)
		: ValueAssignment<Real>(diffusion_body, variable_name),
		  pos_(particles_->pos_){};
	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = 375.0 + 25.0 * (((double)rand() / (RAND_MAX)) - 0.5) * 2.0;
	};

protected:
	StdLargeVec<Vecd> &pos_;
};
//----------------------------------------------------------------------
//	Constraints for isothermal boundaries.
//----------------------------------------------------------------------
class IsothermalBoundariesConstraints : public ValueAssignment<Real>
{
public:
	explicit IsothermalBoundariesConstraints(SolidBody &isothermal_boundaries)
		: ValueAssignment<Real>(isothermal_boundaries, variable_name),
		  pos_(particles_->pos_){};

	void update(size_t index_i, Real dt)
	{
		if (pos_[index_i][1] >= H)
		{
			variable_[index_i] = pos_[index_i][0] > 0.5 ? right_temperature : left_temperature;
		}
	}

protected:
	StdLargeVec<Vecd> &pos_;
};
//----------------------------------------------------------------------
//	Constraints for heat_flux boundaries.
//----------------------------------------------------------------------
class FluxBoundaryConstraint : public ValueAssignment<Real>
{
public:
	explicit FluxBoundaryConstraint(SolidBody& heat_flux_boundary)
		: ValueAssignment<Real>(heat_flux_boundary, flux_name),
		pos_(particles_->pos_) {};

	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = pos_[index_i][1] < 0 ? heat_flux : 0.0;
	}

protected:
	StdLargeVec<Vecd> &pos_;
};
//----------------------------------------------------------------------
//	Initial coefficient distribution.
//----------------------------------------------------------------------
class DiffusivityDistribution : public ValueAssignment<Real>
{
public:
	explicit DiffusivityDistribution(SPHBody &diffusion_body)
		: ValueAssignment<Real>(diffusion_body, coefficient_name),
		  pos_(particles_->pos_){};
	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = diffusion_coff;
	};

protected:
	StdLargeVec<Vecd> &pos_;
};
//----------------------------------------------------------------------
//	Boundary unit normal vector.
//----------------------------------------------------------------------
class UpdateUnitNormalVector : public LocalDynamics,
							   public DissipationDataInner,
						  	   public DissipationDataContact
{
public:
	UpdateUnitNormalVector(ComplexRelation& body_complex_relation, 
		                   const std::string& variable_name) 
		: LocalDynamics(body_complex_relation.getInnerRelation().sph_body_),
		  DissipationDataInner(body_complex_relation.getInnerRelation()),
		  DissipationDataContact(body_complex_relation.getContactRelation()),
		  unit_normal_vector_(*particles_->getVariableByName<Vecd>(variable_name)) {};
	virtual ~UpdateUnitNormalVector() {};

	void interaction(size_t index_i, Real dt = 0.0)
	{
		for (size_t k = 0; k != this->contact_configuration_.size(); ++k)
		{
			Neighborhood& contact_neighborhood = (*this->contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				Real& dW_ijV_j_ = contact_neighborhood.dW_ijV_j_[n];
				Vecd& e_ij_ = contact_neighborhood.e_ij_[n];
				unit_normal_vector_[index_i] += dW_ijV_j_ * e_ij_;
			}
		}
		unit_normal_vector_[index_i] = unit_normal_vector_[index_i] / 
			(unit_normal_vector_[index_i].norm() + TinyReal);
	}

protected:
	StdLargeVec<Vecd>& unit_normal_vector_;
};
//----------------------------------------------------------------------
//	Coefficient reference for imposing coefficient evolution.
//----------------------------------------------------------------------
class DiffusivityReferenceAndIncrement : public ValueAssignment<Real>
{
public:
	DiffusivityReferenceAndIncrement(SPHBody &diffusion_body, const std::string &coefficient_name_ref)
		: ValueAssignment<Real>(diffusion_body, coefficient_name),
		  variable_ref_(*particles_->template getVariableByName<Real>(coefficient_name_ref)),
		  updated_increment_(*particles_->template getVariableByName<Real>("UpdatedIncrement")),
		  previous_increment_(*particles_->template getVariableByName<Real>("PreviousIncrement")){};
	void update(size_t index_i, Real dt)
	{
		variable_ref_[index_i] = variable_[index_i];
		previous_increment_[index_i] = updated_increment_[index_i];
	};

protected:
	StdLargeVec<Real> &variable_ref_;
	StdLargeVec<Real> updated_increment_, previous_increment_;
};
//----------------------------------------------------------------------
//	Equation residue to measure the solution convergence properties.
//----------------------------------------------------------------------
class ThermalEquationResidue
	: public DissipationDataWithWall,
	  public OperatorWithBoundary<LaplacianInner<Real, CoefficientByParticle<Real>>,
	                              LaplacianFromWall<Real, CoefficientByParticle<Real>>>
{
	Real source_;
	StdLargeVec<Real> &residue_;

public:
	ThermalEquationResidue(ComplexRelation &complex_relation,
						   const std::string &in_name, const std::string &out_name,
						   const std::string &eta_name, Real source)
		: DissipationDataWithWall(complex_relation.getContactRelation()),
		  OperatorWithBoundary<LaplacianInner<Real, CoefficientByParticle<Real>>,
		                       LaplacianFromWall<Real, CoefficientByParticle<Real>>>(
								   complex_relation, in_name, out_name, eta_name),
		  residue_(base_operator_.OutVariable()), source_(source)
	{
		for (size_t k = 0; k != contact_particles_.size(); ++k)
		{
			wall_flux_.push_back(contact_particles_[k]->template getVariableByName<Real>(flux_name));
			wall_normal_vector_.push_back(contact_particles_[k]->template getVariableByName<Vecd>(normal_vector));
		}   
	};
	void interaction(size_t index_i, Real dt)
	{
		OperatorWithBoundary<
			LaplacianInner<Real, CoefficientByParticle<Real>>,
			LaplacianFromWall<Real, CoefficientByParticle<Real>>>::interaction(index_i, dt);
		residue_[index_i] += source_;

		for (size_t k = 0; k < contact_configuration_.size(); ++k)
		{
			const StdLargeVec<Real>& heat_flux_k = *(wall_flux_[k]);
			const StdLargeVec<Vecd>& normal_vector_k = *(wall_normal_vector_[k]);
			const Neighborhood& contact_neighborhood = (*contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];
				residue_[index_i] += 2.0 * heat_flux_k[index_j] * contact_neighborhood.dW_ijV_j_[n] / 
					contact_neighborhood.e_ij_[n].dot(normal_vector_k[n]);
			}
		}
	};
protected:
	StdVec<StdLargeVec<Real>*> wall_flux_;
	StdVec<StdLargeVec<Vecd>*> wall_normal_vector_;
};
//----------------------------------------------------------------------
//	Source term for impose optimization target.
//----------------------------------------------------------------------
class ImposingTargetSource : public LocalDynamics, public GeneralDataDelegateSimple
{
public:
	ImposingTargetSource(SPHBody &sph_body, const std::string &variable_name, const Real &source_strength)
		: LocalDynamics(sph_body), GeneralDataDelegateSimple(sph_body),
		  variable_(*particles_->getVariableByName<Real>(variable_name)),
		  source_strength_(source_strength){};
	ImposingTargetSource(BodyPartByParticle& body_part, const std::string& variable_name, const Real& source_strength)
		: ImposingTargetSource(body_part.getSPHBody(), variable_name, source_strength) {};
	virtual ~ImposingTargetSource(){};
	void setSourceStrength(Real source_strength) { source_strength_ = source_strength; };
	void update(size_t index_i, Real dt)
	{
		Real increment = source_strength_ * dt;
		Real theta = increment < 0.0 ? SMIN((0.01 + Eps - variable_[index_i]) / increment, 1.0) : 1.0;
		variable_[index_i] += increment * theta;
	};

protected:
	StdLargeVec<Real> &variable_;
	Real source_strength_;
};
//----------------------------------------------------------------------
//	Evolution of the coefficient to achieve imposed target
//----------------------------------------------------------------------
class CoefficientEvolutionExplicit : public LocalDynamics, public DissipationDataInner
{
public:
	CoefficientEvolutionExplicit(BaseInnerRelation &inner_relation,
								 const std::string &variable_name, const std::string &eta)
		: LocalDynamics(inner_relation.sph_body_), DissipationDataInner(inner_relation),
		  rho_(particles_->rho_), source_(0.0),
		  variable_(*particles_->getVariableByName<Real>(variable_name)),
		  eta_(*particles_->template getVariableByName<Real>(eta)),
		  normal_vector_(*particles_->template getVariableByName<Vecd>(normal_vector))
	{
		particles_->registerVariable(change_rate_, "DiffusionCoefficientChangeRate");
		particles_->registerVariable(eta_ref_, reference_coefficient, [&](size_t i)
									 { return eta_[i]; });
		particles_->registerVariable(total_increment_, "TotalIncrement");
		particles_->registerVariable(updated_increment_, "UpdatedIncrement");
		particles_->registerVariable(previous_increment_, "PreviousIncrement");
	};
	virtual ~CoefficientEvolutionExplicit(){};

	void initialization(size_t index_i, Real dt)
	{
		updated_increment_[index_i] = eta_[index_i] - eta_ref_[index_i];
		total_increment_[index_i] = updated_increment_[index_i] + previous_increment_[index_i];
	};

	void interaction(size_t index_i, Real dt)
	{
		Real variable_i = variable_[index_i];
		Real eta_i = eta_[index_i];

		Real change_rate = source_;
		const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			Real b_ij = 2.0 * inner_neighborhood.dW_ijV_j_[n] / inner_neighborhood.r_ij_[n];
			size_t index_j = inner_neighborhood.j_[n];

			Real variable_diff = (variable_i - variable_[index_j]);
			Real variable_diff_abs = ABS(variable_diff);
			Real coefficient_ave = 0.5 * (total_increment_[index_i] + total_increment_[index_j]);
			Real coefficient_diff = 0.5 * (eta_i - eta_[index_j]);

			change_rate += b_ij * (coefficient_ave * variable_diff + coefficient_diff * variable_diff_abs);
		}
		change_rate_[index_i] = change_rate / rho_[index_i];
	};

	void update(size_t index_i, Real dt)
	{
		Real increment = change_rate_[index_i] * dt;
		Real theta = increment < 0.0 ? SMIN((0.01 + Eps - eta_[index_i]) / increment, 1.0) : 1.0;
		eta_[index_i] += increment * theta;
	};

	void setSource(Real source) { source_ = source; };

protected:
	StdLargeVec<Real> &rho_;
	StdLargeVec<Real> change_rate_;
	StdLargeVec<Real> &variable_;
	StdLargeVec<Vecd> &normal_vector_;
	StdLargeVec<Real> &eta_, eta_ref_; /**< variable damping coefficient */
	StdLargeVec<Real> total_increment_, updated_increment_, previous_increment_;
	Real source_;
};
//----------------------------------------------------------------------
//	Evolution of the coefficient to achieve imposed target from the wall
//----------------------------------------------------------------------
class CoefficientEvolutionWithWallExplicit : public CoefficientEvolutionExplicit,
	                                         public DissipationDataWithWall
{
public:
	CoefficientEvolutionWithWallExplicit(ComplexRelation& complex_relation,
		const std::string& variable_name, const std::string& eta)
		: CoefficientEvolutionExplicit(complex_relation.getInnerRelation(),
			variable_name, coefficient_name),
		DissipationDataWithWall(complex_relation.getContactRelation())
	{
		for (size_t k = 0; k != contact_particles_.size(); ++k)
		{
			wall_variable_.push_back(contact_particles_[k]->template getVariableByName<Real>(variable_name));
		}
	};
	virtual ~CoefficientEvolutionWithWallExplicit() {};

	void interaction(size_t index_i, Real dt)
	{
		CoefficientEvolutionExplicit::interaction(index_i, dt);

		Real change_rate = 0.0;
		for (size_t k = 0; k < contact_configuration_.size(); ++k)
		{
			const StdLargeVec<Real>& variable_k = *(wall_variable_[k]);
			const Neighborhood& contact_neighborhood = (*contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				Real b_ij = 2.0 * contact_neighborhood.dW_ijV_j_[n] / contact_neighborhood.r_ij_[n];
				size_t index_j = contact_neighborhood.j_[n];

				if (variable_k[index_j] != 0)
				{
					Real variable_diff = (variable_[index_i] - variable_k[index_j]);
					change_rate += b_ij * total_increment_[index_i] * variable_diff;
				}
			}
		}
		change_rate_[index_i] += change_rate / rho_[index_i];
	};

protected:
	StdVec<StdLargeVec<Real>*> wall_variable_;
};
//---------------------------------------------------------------------------
//	Evolution of the coefficient to achieve imposed target from the flux wall
//---------------------------------------------------------------------------
class CoefficientEvolutionWithFluxWallExplicit : public CoefficientEvolutionWithWallExplicit
{
public:
	CoefficientEvolutionWithFluxWallExplicit(ComplexRelation& complex_relation,
		                                     const std::string& variable_name, const std::string& eta,
		                                     const std::string& flux_name, const std::string normal_vector)
		: CoefficientEvolutionWithWallExplicit(complex_relation, variable_name, eta)
	{
		for (size_t k = 0; k != contact_particles_.size(); ++k)
		{
			wall_flux_.push_back(contact_particles_[k]->template getVariableByName<Real>(flux_name));
			wall_normal_vector_.push_back(contact_particles_[k]->template getVariableByName<Vecd>(normal_vector));
		}
	};
	virtual ~CoefficientEvolutionWithFluxWallExplicit(){};

	void interaction(size_t index_i, Real dt)
	{
		CoefficientEvolutionWithWallExplicit::interaction(index_i, dt);

		Real change_rate = 0.0;
		for (size_t k = 0; k < contact_configuration_.size(); ++k)
		{
			const StdLargeVec<Real>& heat_flux_k = *(wall_flux_[k]);
			const StdLargeVec<Vecd>& normal_vector_k = *(wall_normal_vector_[k]);
			const Neighborhood& contact_neighborhood = (*contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				Real b_ij = 2.0 * contact_neighborhood.dW_ijV_j_[n] / contact_neighborhood.r_ij_[n];
				size_t index_j = contact_neighborhood.j_[n];

				Real variable_diff_flux = -heat_flux_k[index_j] * contact_neighborhood.r_ij_[n] / eta_[index_i] /
					normal_vector_k[index_j].dot(contact_neighborhood.e_ij_[n]);
				change_rate += b_ij * total_increment_[index_i] * variable_diff_flux;
			}
		}
		change_rate_[index_i] += change_rate / rho_[index_i];
	};

protected:
	StdVec<StdLargeVec<Real>*> wall_flux_;
	StdVec<StdLargeVec<Vecd>*> wall_normal_vector_;
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main()
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	SolidBody diffusion_body(
		sph_system, makeShared<TransformShape<GeometricShapeBox>>(
						Transform2d(block_translation), block_halfsize, "DiffusionBody"));
	diffusion_body.defineParticlesAndMaterial<SolidParticles, Solid>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();
	//----------------------------------------------------------------------
	//	add extra discrete variables (not defined in the library)
	//----------------------------------------------------------------------
	StdLargeVec<Real> body_temperature;
	diffusion_body.addBodyState<Real>(body_temperature, variable_name);
	diffusion_body.addBodyStateForRecording<Real>(variable_name);
	diffusion_body.addBodyStateToRestart<Real>(variable_name);
	StdLargeVec<Real> diffusion_coefficient;
	diffusion_body.addBodyState<Real>(diffusion_coefficient, coefficient_name);
	diffusion_body.addBodyStateForRecording<Real>(coefficient_name);
	diffusion_body.addBodyStateToRestart<Real>(coefficient_name);
	StdLargeVec<Real> laplacian_residue;
	diffusion_body.addBodyState<Real>(laplacian_residue, residue_name);
	diffusion_body.addBodyStateForRecording<Real>(residue_name);
	StdLargeVec<Vecd> inner_normal_vector;
	diffusion_body.addBodyState<Vecd>(inner_normal_vector, normal_vector);
	diffusion_body.addBodyStateForRecording<Vecd>(normal_vector);

	SolidBody boundaries(sph_system, makeShared<Boundaries>("Boundaries"));
	boundaries.defineParticlesAndMaterial<SolidParticles, Solid>();
	boundaries.generateParticles<ParticleGeneratorLattice>();

	BodyRegionByParticle flux_boundary(diffusion_body, makeShared<MultiPolygonShape>(createFluxBoundaryDomain(), "FluxBoundary"));	
	//----------------------------------------------------------------------
	//	add extra discrete variables (not defined in the library)
	//----------------------------------------------------------------------
	StdLargeVec<Real> constrained_temperature;
	boundaries.addBodyState<Real>(constrained_temperature, variable_name);
	boundaries.addBodyStateForRecording<Real>(variable_name);
	StdLargeVec<Real> flux_strength;
	boundaries.addBodyState<Real>(flux_strength, flux_name);
	boundaries.addBodyStateForRecording<Real>(flux_name);
	StdLargeVec<Vecd> unit_normal_vector;
	boundaries.addBodyState<Vecd>(unit_normal_vector, normal_vector);
	boundaries.addBodyStateForRecording<Vecd>(normal_vector);
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	ComplexRelation diffusion_body_complex(diffusion_body, { &boundaries });
	ComplexRelation wall_boundary_complex(boundaries, { &diffusion_body });
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	SimpleDynamics<DiffusionBodyInitialCondition> diffusion_initial_condition(diffusion_body);
	SimpleDynamics<IsothermalBoundariesConstraints> isothermal_boundary_constraint(boundaries);
	SimpleDynamics<FluxBoundaryConstraint> flux_boundary_constraint(boundaries);

	SimpleDynamics<DiffusivityDistribution> coefficient_distribution(diffusion_body);
	SimpleDynamics<ConstraintTotalScalarAmount> constrain_total_coefficient(diffusion_body, coefficient_name);
	SimpleDynamics<ImposingTargetSource, BodyPartByParticle> target_source(flux_boundary, coefficient_name, target_strength);

	InteractionDynamics<UpdateUnitNormalVector> update_domain_vector(diffusion_body_complex, normal_vector);
	InteractionDynamics<UpdateUnitNormalVector> update_boundary_vector(wall_boundary_complex, normal_vector);

	InteractionDynamics<ThermalEquationResidue>
		thermal_equation_residue(diffusion_body_complex, variable_name, residue_name, coefficient_name, heat_source);
	ReduceDynamics<MaximumNorm<Real>> maximum_equation_residue(diffusion_body, residue_name);
	ReduceDynamics<QuantityMoment<Real>> total_coefficient(diffusion_body, coefficient_name);
	ReduceAverage<QuantitySummation<Real>, BodyPartByParticle> average_temperature(flux_boundary, variable_name);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	RestartIO restart_io(io_environment, sph_system.real_bodies_);
	//----------------------------------------------------------------------
	//	Thermal diffusivity optimization
	//----------------------------------------------------------------------
	InteractionSplit<DampingSplittingWithWallFluxCoefficientByParticle<Real>>
		implicit_heat_transfer_solver(diffusion_body_complex, variable_name, flux_name, 
			                          normal_vector, coefficient_name);
	Dynamics1Level<CoefficientEvolutionWithFluxWallExplicit>
		coefficient_evolution_with_wall(diffusion_body_complex, variable_name, 
			                            coefficient_name, flux_name, normal_vector);
	SimpleDynamics<DiffusivityReferenceAndIncrement>
		update_reference_and_increment(diffusion_body, reference_coefficient);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	diffusion_initial_condition.parallel_exec();
	isothermal_boundary_constraint.parallel_exec();
	flux_boundary_constraint.parallel_exec();
	update_domain_vector.parallel_exec();
	update_boundary_vector.parallel_exec();
	coefficient_distribution.parallel_exec();
	constrain_total_coefficient.setupInitialScalarAmount();
	thermal_equation_residue.parallel_exec();
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	int ite = 0;
	Real End_Time = 20.0;
	Real Observe_time = 0.01 * End_Time;
	Real dt = 1.0e-4;
	Real dt_coeff = SMIN(dt, 0.25 * resolution_ref * resolution_ref / reference_temperature);
	int target_steps = 20; // default number of iteration for imposing target
	bool imposing_target = true;
	Real allowed_equation_residue = 4.0e5;
	Real equation_residue_max = Infinity; // initial value
	//----------------------------------------------------------------------
	//	First output before the main loop.
	//----------------------------------------------------------------------
	write_states.writeToFile(ite);
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real relaxation_time = 0.0;
		while (relaxation_time < Observe_time)
		{
			// equation solving step
			implicit_heat_transfer_solver.parallel_exec(dt);
			relaxation_time += dt;
			GlobalStaticVariables::physical_time_ += dt;

			if (imposing_target)
			{
				// target imposing step
				update_reference_and_increment.parallel_exec();
				target_source.parallel_exec(dt_coeff);
				for (size_t k = 0; k != target_steps; ++k)
				{
					coefficient_evolution_with_wall.parallel_exec(dt_coeff);
					constrain_total_coefficient.parallel_exec();
				}
			}

			// residue evaluation step
			thermal_equation_residue.parallel_exec();
			Real residue_max_after_target = maximum_equation_residue.parallel_exec();
			if (residue_max_after_target > equation_residue_max && residue_max_after_target > allowed_equation_residue)
			{
				imposing_target = false; // imposing target skipped for next iteration
			}
			else
			{
				imposing_target = true;
				equation_residue_max = residue_max_after_target;
			}

			ite++;
			if (ite % 100 == 0)
			{
				std::cout << "N= " << ite << " Time: " << GlobalStaticVariables::physical_time_ << "	dt: " << dt << "\n";
				std::cout << "Total diffusivity is " << total_coefficient.parallel_exec() << "\n";
				std::cout << "Average temperature is " << average_temperature.parallel_exec() << "\n";
				std::cout << "Thermal equation maximum residue is " << equation_residue_max << "\n";
			}
		}

		write_states.writeToFile();
	}

	std::cout << "The final physical time has finished." << std::endl;
	return 0;
}
