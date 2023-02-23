/**
 * @file 	relaxation_evolution.cpp
 * @brief 	This is the first case by testing the relaxation with evolution method.
 * @author 	Bo Zhang
 */
#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec2d center(0.0, 0.0);	/**< Location of the cylinder center. */
Real radius = 1;		/**< Radius of the cylinder. */
Real resolution_ref = radius / 40.0;
Real BW = resolution_ref * 2.0;
BoundingBox system_domain_bounds(Vec2d(-BW - radius, -BW - radius), Vec2d(radius + BW, radius + BW));
//----------------------------------------------------------------------
//	Define geometries
//----------------------------------------------------------------------
class Insert : public MultiPolygonShape
{
public:
	explicit Insert(const std::string& shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addACircle(center, radius, 100, ShapeBooleanOps::add);
	}
};

int main(int ac, char* av[])
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem with global controls.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	sph_system.setRunParticleRelaxation(true);
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	SolidBody body(sph_system, makeShared<Insert>("InsertedBody"));
	body.defineBodyLevelSetShape()->writeLevelSet(io_environment);
	body.defineParticlesAndMaterial();
	body.addBodyStateForRecording<Vecd>("Position");
	(!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
		? body.generateParticles<ParticleGeneratorReload>(io_environment, body.getName())
		: body.generateParticles<ParticleGeneratorLattice>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	InnerRelation insert_body_inner(body);
	//----------------------------------------------------------------------
	//	Run particle relaxation for body-fitted distribution if chosen.
	//----------------------------------------------------------------------
	if (sph_system.RunParticleRelaxation())
	{
		//----------------------------------------------------------------------
		//	Methods used for particle relaxation.
		//----------------------------------------------------------------------
		/** Random reset the insert body particle posisiton. */
		SimpleDynamics<RandomizeParticlePosition> random_insert_body_particles(body);
		/** Write the body state to Vtp file. */
		BodyStatesRecordingToVtp write_insert_body_to_vtp(io_environment, { &body });
		/** Write the particle reload files. */
		ReloadParticleIO write_particle_reload_files(io_environment, { &body });

		/** An explicit physics relaxation step. */
		relax_dynamics::RelaxationStepInner relaxation_inner_explicit(insert_body_inner, true);
		/** An implicit evolution relaxation step. */
		relax_dynamics::RelaxationEvolutionInner relaxation_inner_implicit(insert_body_inner, true);
		/** An implicit modification for consistency. */
		relax_dynamics::ModificationStepForConsistency modification_for_consistency(insert_body_inner);
		
		/** Update relaxation residue. */
		InteractionDynamics<relax_dynamics::UpdateParticleKineticEnergy> 
			update_kinetic_energy(insert_body_inner);
		ReducedQuantityRecording<ReduceAverage<QuantitySummation<Real>>> 
			write_particle_averaged_kinetic_energy(io_environment, body, "particle_kinetic_energy");
		ReducedQuantityRecording<ReduceDynamics<QuantityMaximum<Real>>> 
			write_particle_maximum_kinetic_energy(io_environment, body, "particle_kinetic_energy");
		InteractionDynamics<relax_dynamics::CheckZeroOrderConsistency> 
			check_zero_order_consistency(insert_body_inner);
		ReducedQuantityRecording<ReduceAverage<QuantitySummation<Real>>> 
			write_particle_averaged_zero_order_consisitency(io_environment, body, "unit_zero_order");
		ReducedQuantityRecording<ReduceDynamics<QuantityMaximum<Real>>> 
			write_particle_maximum_zero_order_consisitency(io_environment, body, "unit_zero_order");
		ReducedQuantityRecording<ReduceAverage<QuantitySummation<Real>>> 
			write_particle_volume(io_environment, body, "VolumetricMeasure");
		ReducedQuantityRecording<ReduceAverage<QuantitySummation<Real>>> 
			write_particle_zero_order_error(io_environment, body, "residue_zero_order_consistency");
		ReducedQuantityRecording<ReduceAverage<QuantitySummation<Real>>> 
			write_particle_first_order_error(io_environment, body, "residue_first_order_consistency");
		//----------------------------------------------------------------------
		//	Particle relaxation starts here.
		//----------------------------------------------------------------------
		random_insert_body_particles.parallel_exec(0.25);
		//relaxation_inner_implicit.SurfaceBounding().parallel_exec();
		write_insert_body_to_vtp.writeToFile(0);
		//----------------------------------------------------------------------
		//	Relax particles of the insert body.
		//----------------------------------------------------------------------
		tick_count t1 = tick_count::now();
		int ite_p = 0;
		GlobalStaticVariables::physical_time_ = ite_p;
		Real dt = 1.0 / 10.0;
		while (ite_p < 2000)
		{
			relaxation_inner_implicit.parallel_exec(dt);

			if (ite_p % 20 == 0)
			{
				update_kinetic_energy.parallel_exec(dt);
				write_particle_averaged_kinetic_energy.writeToFile(ite_p);
				write_particle_maximum_kinetic_energy.writeToFile(ite_p);
				check_zero_order_consistency.exec(dt);
 				write_particle_averaged_zero_order_consisitency.writeToFile(ite_p);
				write_particle_maximum_zero_order_consisitency.writeToFile(ite_p);
			}

			ite_p += 1;
			GlobalStaticVariables::physical_time_ = ite_p;
			if (ite_p % 200 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
				write_particle_volume.writeToFile(ite_p);
				write_insert_body_to_vtp.writeToFile(ite_p);
			}
		}
		std::cout << "The physical relaxation process of body finish !" << std::endl;

		/*int ite_m = 2000;
		while (ite_m < 2400)
		{
			modification_for_consistency.exec(dt);
			ite_m += 1;
			GlobalStaticVariables::physical_time_ = ite_m;
			if (ite_m % 20 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Modification for the inserted body N = " << ite_m << "\n";
				write_insert_body_to_vtp.writeToFile(ite_m);
				write_particle_zero_order_error.writeToFile(ite_m);
				write_particle_volume.writeToFile(ite_m);
				write_particle_first_order_error.writeToFile(ite_m);
			}
		}
		std::cout << "The modification process is finished !" << std::endl;*/

		/** Output results. */
		write_particle_reload_files.writeToFile(0);
		tick_count t2 = tick_count::now();
		tick_count::interval_t tt;
		tt = t2 - t1;
		std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
		return 0;
	}
}
