#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>

namespace MeltPoolDG::Heat
{
  /**
   * State of the laser beam over time.
   *
   * This class keeps track of the current laser power according to the temporal power
   * distribution, the current position of the beam spot center according to the scan movement,
   * and the beam direction. It is independent of any finite element discretization and of the
   * spatial intensity distribution, see create_intensity_profile(), which evaluates the beam
   * state on demand.
   */
  template <int dim, typename number>
  class LaserBeam final
  {
  public:
    explicit LaserBeam(const LaserData<number> &laser_data_in);

    /**
     * Reset the beam to its starting position at time @p start_time.
     */
    void
    reset(const number start_time);

    /**
     * Advance the beam by the time increment @p dt according to the scan speed and the
     * temporal power distribution.
     */
    void
    advance(const number dt);

    number
    get_time() const;

    /**
     * Current laser power.
     */
    number
    get_power() const;

    /**
     * Current position of the beam spot center.
     */
    const dealii::Point<dim> &
    get_position() const;

    /**
     * Unit vector of the beam direction.
     */
    const dealii::Tensor<1, dim, number> &
    get_direction() const;

    /**
     * Print the current position and relative power of the beam.
     */
    void
    print(const ConditionalOStream &pcout) const;

  private:
    /**
     * Compute the power from the temporal power distribution at the current time.
     */
    void
    update_power();

    const LaserData<number> &laser_data;

    const dealii::Tensor<1, dim, number> direction;

    number time = 0.0;

    // current power relative to the nominal power (between 0 and 1)
    number relative_power = 0.0;

    number power = 0.0;

    dealii::Point<dim> position;
  };
} // namespace MeltPoolDG::Heat
