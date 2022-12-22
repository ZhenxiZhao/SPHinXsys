/**
 * @file	standingwave.cpp
 * @brief	2D standingwave example.
 * @author	Yaru Ren, Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library.
using namespace SPH;   // Namespace cite here.
#define PI 3.1415926
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 2.0;					/**< Tank length. */
Real DH = 2.0;					/**< Tank height. */
Real LL = 2.0;						/**< Liquid column length. */
Real LH = 1.0;						/**< Liquid column height. */
Real particle_spacing_ref = 0.02;	/**< Initial reference particle spacing. */
Real BW = particle_spacing_ref * 4; /**< Extending width for boundary conditions. */
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(DL + BW, DH + BW));
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
Real rho0_f = 1000.0;						 /**< Reference density of fluid. */
Real gravity_g = 9.81;					 /**< Gravity force of fluid. */
Real U_max = 2.0 * sqrt(gravity_g * LH); /**< Characteristic velocity. */
Real c_f = 10.0 * U_max;				 /**< Reference sound speed. */
//----------------------------------------------------------------------
//	Geometric shapes used in this case.
//----------------------------------------------------------------------
Vec2d outer_wall_halfsize = Vec2d(0.5 * DL + BW, 0.5 * DH + BW);
Vec2d outer_wall_translation = Vec2d(-BW, -BW) + outer_wall_halfsize;
Vec2d inner_wall_halfsize = Vec2d(0.5 * DL, 0.5 * DH);
Vec2d inner_wall_translation = inner_wall_halfsize;
//----------------------------------------------------------------------
//	Complex shape for wall boundary, note that no partial overlap is allowed
//	for the shapes in a complex shape.
//----------------------------------------------------------------------
class WallBoundary : public ComplexShape
{
public:
	explicit WallBoundary(const std::string& shape_name) : ComplexShape(shape_name)
	{
		add<TransformShape<GeometricShapeBox>>(Transform2d(outer_wall_translation), outer_wall_halfsize);
		subtract<TransformShape<GeometricShapeBox>>(Transform2d(inner_wall_translation), inner_wall_halfsize);
	}
};
//----------------------------------------------------------------------
//	water block
//----------------------------------------------------------------------
std::vector<Vecd> createWaterBlockShape()
{
	int  Nh = 100;
	Real Lstep = DL / Nh;

	std::vector<Vecd> water_block_shape;
	water_block_shape.push_back(Vecd(0.0, 0.0));
	std::vector<Vecd> pnts;
	for (int n = 0; n <= Nh; n++)
	{
		Real lamda = 2.0;
		Real k = 2 * PI / lamda;
		Real x = n * Lstep;
		Real y = 0.1 * cos(PI * x);

		pnts.push_back(Vecd(x, y));
	}
	for (int n = 0; n <= Nh - 0; n++)
	{
		water_block_shape.push_back(Vecd(pnts[n][0], pnts[n][1] + 1.0));
	}
	water_block_shape.push_back(Vecd(LL, 0.0));
	water_block_shape.push_back(Vecd(0.0, 0.0));

	return water_block_shape;
}

class WaterBlock : public ComplexShape
{
public:
	explicit WaterBlock(const std::string& shape_name) : ComplexShape(shape_name)
	{
		MultiPolygon outer_boundary(createWaterBlockShape());
		add<MultiPolygonShape>(outer_boundary, "OuterBoundary");
	}
};
//----------------------------------------------------------------------
//	wave gauge
//----------------------------------------------------------------------
Real h = 1.3 * particle_spacing_ref;
MultiPolygon createWaveProbeShape()
{
	std::vector<Vecd> pnts;
	pnts.push_back(Vecd(1.0 - h, 0.0));
	pnts.push_back(Vecd(1.0 - h, DH));
	pnts.push_back(Vecd(1.0 + h, DH));
	pnts.push_back(Vecd(1.0 + h, 0.0));
	pnts.push_back(Vecd(1.0 - h, 0.0));

	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(pnts, ShapeBooleanOps::add);
	return multi_polygon;
}
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char* av[])
{
	//----------------------------------------------------------------------
	//	Build up an SPHSystem.
	//----------------------------------------------------------------------
	BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(DL + BW, DH + BW));
	SPHSystem sph_system(system_domain_bounds, particle_spacing_ref);
	/** Tag for run particle relaxation for the initial body fitted distribution. */
	sph_system.setRunParticleRelaxation(false);
	/** Tag for computation start with relaxed body fitted particles distribution. */
	sph_system.setReloadParticles(false);
	sph_system.generate_regression_data_ = true;
	sph_system.handleCommandlineOptions(ac, av);
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating bodies with corresponding materials and particles.
	//----------------------------------------------------------------------
	FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
	water_block.defineAdaptation<SPHAdaptation>(1.3, 1.0);
	water_block.defineComponentLevelSetShape("OuterBoundary");
	water_block.defineParticlesAndMaterial<FluidParticles, WeaklyCompressibleFluid>(rho0_f, c_f);
	// Using relaxed particle distribution if needed
	(!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
		? water_block.generateParticles<ParticleGeneratorReload>(io_environment, water_block.getName())
		: water_block.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
	wall_boundary.defineAdaptation<SPHAdaptation>(1.15, 1.0);
	wall_boundary.defineBodyLevelSetShape();
	wall_boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
	(!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
		? wall_boundary.generateParticles<ParticleGeneratorReload>(io_environment, wall_boundary.getName())
		: wall_boundary.generateParticles<ParticleGeneratorLattice>();
	wall_boundary.addBodyStateForRecording<Vecd>("NormalDirection");
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	ComplexRelation water_block_complex(water_block, { &wall_boundary });
	ComplexRelation wall_complex(wall_boundary, { &water_block });
	//----------------------------------------------------------------------
	/** check whether run particle relaxation for body fitted particle distribution. */
	if (sph_system.RunParticleRelaxation())
	{
		InnerRelation wall_inner(wall_boundary); // extra body topology only for particle relaxation
		/**
		 * @brief 	Methods used for particle relaxation.
		 */
		 /** Random reset the insert body particle position. */
		SimpleDynamics<RandomizeParticlePosition> random_water_body_particles(water_block);
		SimpleDynamics<RandomizeParticlePosition> random_wall_body_particles(wall_boundary);
		/** Write the body state to Vtp file. */
		BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
		/** Write the particle reload files. */
		ReloadParticleIO write_real_body_particle_reload_files(io_environment, sph_system.real_bodies_);

		/** A  Physics relaxation step. */
		relax_dynamics::RelaxationStepInner relaxation_step_inner(wall_inner, true);
		relax_dynamics::RelaxationStepComplex relaxation_step_complex(water_block_complex, "OuterBoundary", true);
		/**
		 * @brief 	Particle relaxation starts here.
		 */
		random_wall_body_particles.parallel_exec(0.25);
		random_water_body_particles.parallel_exec(0.25);
		relaxation_step_inner.SurfaceBounding().parallel_exec();
		relaxation_step_complex.SurfaceBounding().parallel_exec();
		write_real_body_states.writeToFile(0);

		/** relax particles of the insert body. */
		int ite_p = 0;
		while (ite_p < 1000)
		{
			relaxation_step_inner.parallel_exec();
			relaxation_step_complex.parallel_exec();
			ite_p += 1;
			if (ite_p % 200 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
				write_real_body_states.writeToFile(ite_p);
			}
		}
		std::cout << "The physics relaxation process of inserted body finish !" << std::endl;

		/** Output results. */
		write_real_body_particle_reload_files.writeToFile(0);
		return 0;
	}
	//----------------------------------------------------------------------
	//	Define the numerical methods used in the simulation.
	//	Note that there may be data dependence on the sequence of constructions.
	//----------------------------------------------------------------------
	InteractionDynamics<GlobalCorrectionMatrixComplex> corrected_configuration_fluid(water_block_complex);
	InteractionDynamics<GlobalCorrectionMatrixComplex> corrected_configuration_solid(wall_complex);
	Dynamics1Level<fluid_dynamics::Integration1stHalfRiemannCorrectWithWall> fluid_pressure_relaxation_correct(water_block_complex);
	Dynamics1Level<fluid_dynamics::Integration2ndHalfRiemannWithWall> fluid_density_relaxation(water_block_complex);
	InteractionWithUpdate<fluid_dynamics::DensitySummationFreeSurfaceComplex> fluid_density_by_summation(water_block_complex);
	SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
	SharedPtr<Gravity> gravity_ptr = makeShared<Gravity>(Vecd(0.0, -gravity_g));
	SimpleDynamics<TimeStepInitialization> fluid_step_initialization(water_block, gravity_ptr);
	ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> fluid_advection_time_step(water_block, U_max);
	ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> fluid_acoustic_time_step(water_block);
	/** We can output a method-specific particle data for debug */
	water_block.addBodyStateForRecording<Real>("Pressure");
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations, observations
	//	and regression tests of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp body_states_recording(io_environment, sph_system.real_bodies_);
	RestartIO restart_io(io_environment, sph_system.real_bodies_);
	RegressionTestDynamicTimeWarping<ReducedQuantityRecording<ReduceDynamics<TotalMechanicalEnergy>>>
		write_water_mechanical_energy(io_environment, water_block, gravity_ptr);
	/** WaveProbes. */
	BodyRegionByCell wave_probe_buffer_(water_block, makeShared<MultiPolygonShape>(createWaveProbeShape(), "WaveProbe"));
	RegressionTestDynamicTimeWarping<ReducedQuantityRecording<ReduceDynamics<fluid_dynamics::FreeSurfaceHeight, BodyRegionByCell>>>
		wave_probe(io_environment, wave_probe_buffer_);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	wall_boundary_normal_direction.parallel_exec();
	//----------------------------------------------------------------------
	//	Load restart file if necessary.
	//----------------------------------------------------------------------
	if (sph_system.RestartStep() != 0)
	{
		GlobalStaticVariables::physical_time_ = restart_io.readRestartFiles(sph_system.RestartStep());
		water_block.updateCellLinkedList();
		water_block_complex.updateConfiguration();
	}
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	size_t number_of_iterations = sph_system.RestartStep();
	int screen_output_interval = 100;
	int observation_sample_interval = screen_output_interval * 2;
	int restart_output_interval = screen_output_interval * 10;
	Real end_time = 3.0;
	Real output_interval = 0.1;
	//----------------------------------------------------------------------
	//	Statistics for CPU time
	//----------------------------------------------------------------------
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	tick_count::interval_t interval_computing_time_step;
	tick_count::interval_t interval_computing_fluid_pressure_relaxation;
	tick_count::interval_t interval_updating_configuration;
	tick_count time_instance;
	//----------------------------------------------------------------------
	//	First output before the main loop.
	//----------------------------------------------------------------------
	body_states_recording.writeToFile();
	write_water_mechanical_energy.writeToFile(number_of_iterations);
	wave_probe.writeToFile(number_of_iterations);
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < end_time)
	{
		Real integration_time = 0.0;
		/** Integrate time (loop) until the next output time. */
		while (integration_time < output_interval)
		{
			/** outer loop for dual-time criteria time-stepping. */
			time_instance = tick_count::now();
			fluid_step_initialization.parallel_exec();
			Real advection_dt = fluid_advection_time_step.parallel_exec();
			fluid_density_by_summation.parallel_exec();
			corrected_configuration_fluid.parallel_exec();
			corrected_configuration_solid.parallel_exec();
			interval_computing_time_step += tick_count::now() - time_instance;

			time_instance = tick_count::now();
			Real relaxation_time = 0.0;
			Real acoustic_dt = 0.0;
			while (relaxation_time < advection_dt)
			{
				/** inner loop for dual-time criteria time-stepping.  */
				acoustic_dt = fluid_acoustic_time_step.parallel_exec();
				fluid_pressure_relaxation_correct.parallel_exec(acoustic_dt);
				fluid_density_relaxation.parallel_exec(acoustic_dt);
				relaxation_time += acoustic_dt;
				integration_time += acoustic_dt;
				GlobalStaticVariables::physical_time_ += acoustic_dt;
			}
			interval_computing_fluid_pressure_relaxation += tick_count::now() - time_instance;

			/** screen output, write body reduced values and restart files  */
			if (number_of_iterations % screen_output_interval == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
					<< GlobalStaticVariables::physical_time_
					<< "	advection_dt = " << advection_dt << "	acoustic_dt = " << acoustic_dt << "\n";

				if (number_of_iterations % observation_sample_interval == 0 && number_of_iterations != sph_system.RestartStep())
				{
					write_water_mechanical_energy.writeToFile(number_of_iterations);
					wave_probe.writeToFile(number_of_iterations);
				}
				if (number_of_iterations % restart_output_interval == 0)
					restart_io.writeToFile(number_of_iterations);
			}
			number_of_iterations++;

			/** Update cell linked list and configuration. */
			time_instance = tick_count::now();
			water_block.updateCellLinkedListWithParticleSort(100);
			water_block_complex.updateConfiguration();
			wall_boundary.updateCellLinkedListWithParticleSort(100);
			wall_complex.updateConfiguration();
			interval_updating_configuration += tick_count::now() - time_instance;
		}

		body_states_recording.writeToFile();
		tick_count t2 = tick_count::now();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds()
		<< " seconds." << std::endl;
	std::cout << std::fixed << std::setprecision(9) << "interval_computing_time_step ="
		<< interval_computing_time_step.seconds() << "\n";
	std::cout << std::fixed << std::setprecision(9) << "interval_computing_fluid_pressure_relaxation = "
		<< interval_computing_fluid_pressure_relaxation.seconds() << "\n";
	std::cout << std::fixed << std::setprecision(9) << "interval_updating_configuration = "
		<< interval_updating_configuration.seconds() << "\n";

	if (sph_system.generate_regression_data_)
	{
		write_water_mechanical_energy.generateDataBase(1.0e-3);
		wave_probe.generateDataBase(1.0e-3);
	}
	else if (sph_system.RestartStep() == 0)
	{
		write_water_mechanical_energy.newResultTest();
		wave_probe.newResultTest();
	}

	return 0;
};
