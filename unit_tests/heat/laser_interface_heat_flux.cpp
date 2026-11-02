#include <gtest/gtest.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_interface_heat_flux.hpp>

#include <cmath>

using namespace MeltPoolDG::Heat;

namespace
{
  using VectorizedArrayType = dealii::VectorizedArray<double>;

  double
  evaluate_flux(const LaserInterfaceHeatFlux<2, double> &flux, const dealii::Tensor<1, 2> &normal)
  {
    dealii::Tensor<1, 2, VectorizedArrayType> normal_vectorized;
    for (unsigned int d = 0; d < 2; ++d)
      normal_vectorized[d] = normal[d];

    return flux.value(dealii::Point<2, VectorizedArrayType>(), normal_vectorized)[0];
  }

  LaserData<double>
  create_laser_data()
  {
    LaserData<double> laser;
    laser.model               = LaserModelType::interface_projection_sharp;
    laser.intensity_profile   = LaserIntensityProfileType::uniform;
    laser.power               = 100.0;
    laser.power_end_time      = 1.0;
    laser.absorptivity_gas    = 0.4;
    laser.absorptivity_liquid = 0.4;
    laser.post(/*dim*/ 2); // default beam direction: (0, -1)
    return laser;
  }
} // namespace

TEST(LaserInterfaceHeatFlux, ProjectsIntensityInBeamDirection)
{
  const auto laser = create_laser_data();

  LaserBeam<2, double> beam(laser);
  beam.reset(0.0);

  const LaserInterfaceHeatFlux<2, double> flux(laser, beam);

  // the normal vector points into the liquid, i.e. in beam direction for a flat surface
  EXPECT_DOUBLE_EQ(evaluate_flux(flux, dealii::Tensor<1, 2>({0.0, -1.0})), 0.4 * 100.0);
  EXPECT_DOUBLE_EQ(evaluate_flux(flux, dealii::Tensor<1, 2>({std::sqrt(0.5), -std::sqrt(0.5)})),
                   0.4 * 100.0 * std::sqrt(0.5));
  // surfaces parallel to or facing away from the beam are not heated
  EXPECT_DOUBLE_EQ(evaluate_flux(flux, dealii::Tensor<1, 2>({1.0, 0.0})), 0.0);
  EXPECT_DOUBLE_EQ(evaluate_flux(flux, dealii::Tensor<1, 2>({0.0, 1.0})), 0.0);
}

TEST(LaserInterfaceHeatFlux, FollowsBeamPower)
{
  const auto laser = create_laser_data();

  LaserBeam<2, double> beam(laser);
  beam.reset(0.0);

  const LaserInterfaceHeatFlux<2, double> flux(laser, beam);

  // the laser is switched off at the power end time
  beam.advance(1.0);
  EXPECT_DOUBLE_EQ(evaluate_flux(flux, dealii::Tensor<1, 2>({0.0, -1.0})), 0.0);
}
