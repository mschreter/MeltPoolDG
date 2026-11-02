#include <deal.II/base/exceptions.h>

#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <sstream>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  LaserBeam<dim, number>::LaserBeam(const LaserData<number> &laser_data_in)
    : laser_data(laser_data_in)
    , direction(laser_data.template get_beam_direction<dim>())
    , position(laser_data.template get_starting_position<dim>())
  {
    AssertThrow(
      laser_data.power_end_time > laser_data.power_start_time,
      ExcMessage(
        "For the temporal ramp distribution of the laser power,"
        " the parameter laser power end time must be larger than laser power start time."));
  }

  template <int dim, typename number>
  void
  LaserBeam<dim, number>::reset(const number start_time)
  {
    time = start_time;
    update_power();
    position = laser_data.template get_starting_position<dim>();
  }

  template <int dim, typename number>
  void
  LaserBeam<dim, number>::advance(const number dt)
  {
    time += dt;
    position += laser_data.template get_scan_direction<dim>() * laser_data.scan_speed * dt;
    update_power();
  }

  template <int dim, typename number>
  number
  LaserBeam<dim, number>::get_time() const
  {
    return time;
  }

  template <int dim, typename number>
  number
  LaserBeam<dim, number>::get_power() const
  {
    return power;
  }

  template <int dim, typename number>
  const Point<dim> &
  LaserBeam<dim, number>::get_position() const
  {
    return position;
  }

  template <int dim, typename number>
  const Tensor<1, dim, number> &
  LaserBeam<dim, number>::get_direction() const
  {
    return direction;
  }

  template <int dim, typename number>
  void
  LaserBeam<dim, number>::print(const ConditionalOStream &pcout) const
  {
    std::ostringstream str;
    str << "current laser position: " << position;
    Journal::print_line(pcout, str.str(), "laser");
    str.str("");
    str << "current laser intensity: " << relative_power;
    Journal::print_line(pcout, str.str(), "laser");
  }

  template <int dim, typename number>
  void
  LaserBeam<dim, number>::update_power()
  {
    if (laser_data.power_over_time == "ramp")
      {
        relative_power = (time - laser_data.power_start_time) /
                         (laser_data.power_end_time - laser_data.power_start_time);
        relative_power = std::clamp(relative_power, number(0.0), number(1.0));
      }
    else if (laser_data.power_over_time == "constant")
      relative_power = (time >= laser_data.power_end_time) ? 0.0 : 1.0;
    else
      AssertThrow(false, ExcNotImplemented());

    power = laser_data.power * relative_power;
  }

  template class LaserBeam<1, double>;
  template class LaserBeam<2, double>;
  template class LaserBeam<3, double>;
} // namespace MeltPoolDG::Heat
