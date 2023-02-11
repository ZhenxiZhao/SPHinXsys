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
std::string residue_name = "residue";
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

int main(int ac, char *av[])
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem with global controls.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	sph_system.setRunParticleRelaxation(true); // Tag for run particle relaxation for body-fitted distribution
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

	StdLargeVec<Real> residue_;
	body.addBodyState<Real>(residue_, residue_name);
	body.addBodyStateForRecording<Real>(residue_name);
	body.addBodyStateToRestart<Real>(residue_name);
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
		/** Random reset the insert body particle position. */
		SimpleDynamics<RandomizeParticlePosition> random_insert_body_particles(body);
		/** Write the body state to Vtp file. */
		BodyStatesRecordingToVtp write_insert_body_to_vtp(io_environment, {&body});
		/** Write the particle reload files. */
		ReloadParticleIO write_particle_reload_files(io_environment, {&body});
		/** A  Physics relaxation step. */
		relax_dynamics::RelaxationEvolutionInner relaxation_inner(insert_body_inner, true);
		/*InteractionDynamics<relax_dynamics::UpdateParticleKineticEnergy> 
			update_kinetic_energy(insert_body_inner, residue_name);
		ReduceAverage<QuantitySummation<Real>> average_residue(body, residue_name);*/
		//----------------------------------------------------------------------
		//	Particle relaxation starts here.
		//----------------------------------------------------------------------
		random_insert_body_particles.parallel_exec(0.25);
		relaxation_inner.SurfaceBounding().parallel_exec();
		write_insert_body_to_vtp.writeToFile(0);
		//----------------------------------------------------------------------
		//	Relax particles of the insert body.
		//----------------------------------------------------------------------
		std::string filefullpath_residue = io_environment.output_folder_ + "/" + "residue.dat";
		std::ofstream out_file_residue(filefullpath_residue.c_str(), std::ios::app);

		int ite_p = 0;
		Real dt = 1.0 / 200.0;
		while (ite_p < 2000)
		{
			relaxation_inner.parallel_exec(dt);
			ite_p += 1;
			if (ite_p % 200 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
				write_insert_body_to_vtp.writeToFile(ite_p);
			}
		}
		std::cout << "The physics relaxation process of inserted body finish !" << std::endl;
		/** Output results. */
		write_particle_reload_files.writeToFile(0);
		return 0;
	}
}
