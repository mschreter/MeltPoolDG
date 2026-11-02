#include <gtest/gtest.h>

#include <deal.II/base/point.h>

#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_intensity_profiles.hpp>

using namespace MeltPoolDG::Heat;

namespace
{
  LaserData<double>
  create_laser_data()
  {
    LaserData<double> laser;
    laser.model             = LaserModelType::volumetric;
    laser.intensity_profile = LaserIntensityProfileType::uniform;
    laser.power             = 100.0;
    laser.post(/*dim*/ 2);
    return laser;
  }
} // namespace

TEST(LaserBeam, ConstantPowerIsSwitchedOffAtEndTime)
{
  auto laser           = create_laser_data();
  laser.power_end_time = 1.0;

  LaserBeam<2, double> beam(laser);
  beam.reset(0.0);
  EXPECT_DOUBLE_EQ(beam.get_power(), 100.0);

  beam.advance(0.5);
  EXPECT_DOUBLE_EQ(beam.get_power(), 100.0);

  beam.advance(0.5);
  EXPECT_DOUBLE_EQ(beam.get_power(), 0.0);
}

TEST(LaserBeam, RampPowerIncreasesLinearly)
{
  auto laser             = create_laser_data();
  laser.power_over_time  = "ramp";
  laser.power_start_time = 1.0;
  laser.power_end_time   = 3.0;

  LaserBeam<2, double> beam(laser);
  beam.reset(0.0);
  EXPECT_DOUBLE_EQ(beam.get_power(), 0.0);

  beam.advance(2.0);
  EXPECT_DOUBLE_EQ(beam.get_power(), 50.0);

  beam.advance(2.0);
  EXPECT_DOUBLE_EQ(beam.get_power(), 100.0);
}

TEST(LaserBeam, MovesInScanDirection)
{
  auto laser       = create_laser_data();
  laser.scan_speed = 2.0;

  LaserBeam<2, double> beam(laser);
  beam.reset(0.0);

  beam.advance(0.25);
  EXPECT_DOUBLE_EQ(beam.get_position()[0], 0.5);
  EXPECT_DOUBLE_EQ(beam.get_position()[1], 0.0);

  // a reset moves the beam back to its starting position
  beam.reset(0.0);
  EXPECT_DOUBLE_EQ(beam.get_position()[0], 0.0);
}

TEST(LaserBeam, IntensityProfileFollowsBeam)
{
  auto laser           = create_laser_data();
  laser.power_end_time = 1.0;

  LaserBeam<2, double> beam(laser);
  beam.reset(0.0);

  const auto profile = create_intensity_profile(laser, beam, true /*is_volumetric*/);
  EXPECT_DOUBLE_EQ(profile->value(dealii::Point<2>(0.3, 0.1)), 100.0);

  // the profile follows the beam without further updates
  beam.advance(1.0);
  EXPECT_DOUBLE_EQ(profile->value(dealii::Point<2>(0.3, 0.1)), 0.0);
}
