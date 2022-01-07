/**
* @file 	twisting_column.h
* @brief 	This is the case setup for twisting_column.cpp.
* @author 	Chi Zhang and Xiangyu Hu
*/
#ifndef TEST_3D_TWISTING_COLUMN_CASE_H
#define TEST_3D_TWISTING_COLUMN_CASE_H

#include "sphinxsys.h"
using namespace SPH;

//----------------------------------------------------------------------
//	Global geometry parameters.
//----------------------------------------------------------------------
Real PL = 6.0; /**< X-direction domain. */
Real PH = 1.0; /**< Y-direction domain. */
Real PW = 1.0; /**< Z-direction domain. */
Real particle_spacing_ref = PH / 10.0;
/** YOU can try PW = 0.2 and particle_spacing_ref = PH / 10.0 to see an interesting test. */
Real BW = particle_spacing_ref * 0.0; /**< no wall boundary in this case. */
Real SL = particle_spacing_ref * 1.0; /**< Length of the holder is one layer particle. */
Vecd halfsize_column(0.5 * (PL + SL), 0.5 * PH, 0.5 * PW);
Vecd translation_column(0.5 * (PL - SL), 0.0, 0.0);
Vecd halfsize_holder(0.5 * (SL + BW), 0.5 * (PH + BW), 0.5 * (PW + BW));
Vecd translation_holder(-0.5 * (SL + BW), 0.0, 0.0);

Vec3d domain_lower_bound(-SL - BW, -0.5 * (PH + BW), -0.5 * (PW + BW));
Vec3d domain_upper_bound(PL, 0.5 * (PH + BW), 0.5 * (PW + BW));
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
int resolution(20);
//----------------------------------------------------------------------
//	Material properties and global parameters
//----------------------------------------------------------------------
Real rho0_s = 1100.0; /**< Reference density. */
Real poisson = 0.45;  /**< Poisson ratio. */
Real Youngs_modulus = 1.7e7;
Real angular_0 = -400.0;

/** Define the body. */
class Column : public SolidBody
{
public:
	Column(SPHSystem &system, const std::string &body_name)
		: SolidBody(system, body_name, makeShared<SPHAdaptation>(1.15, 1.0))
	{
		body_shape_.add<TriangleMeshShapeBrick>(halfsize_column, resolution, translation_column);
		body_shape_.add<TriangleMeshShapeBrick>(halfsize_holder, resolution, translation_holder);
	}
};
/**
 * application dependent initial condition
 */
class InitialCondition
	: public solid_dynamics::ElasticDynamicsInitialCondition
{
public:
	explicit InitialCondition(SolidBody &body)
		: solid_dynamics::ElasticDynamicsInitialCondition(body){};

protected:
	void Update(size_t index_i, Real dt) override
	{
		Real x = pos_n_[index_i][0];
		Real y = pos_n_[index_i][1];
		Real z = pos_n_[index_i][2];
		Real angular_velocity = angular_0 * sin((M_PI * x) / (2.0 * PL));
		Real local_radius = sqrt(pow(y, 2.0) + pow(z, 2.0));
		Real angular = atan2(y, z);

		if (x > 0.0)
		{
			vel_n_[index_i][1] = angular_velocity * local_radius * cos(angular);
			vel_n_[index_i][2] = -angular_velocity * local_radius * sin(angular);
		}
	};
};

//define an observer particle generator
class ObserverParticleGenerator : public ParticleGeneratorDirect
{
public:
	ObserverParticleGenerator() : ParticleGeneratorDirect()
	{
		positions_volumes_.push_back(std::make_pair(Vecd(PL, 0.0, 0.0), 0.0));
	}
};
#endif //TEST_3D_TWISTING_COLUMN_CASE_H