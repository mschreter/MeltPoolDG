#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_data.hpp>

#include <cmath>
#include <functional>
#include <memory>

namespace MeltPoolDG::Heat
{
  /**
   * Computes the projection factor for angled laser interface interactions.
   *
   * @param normal_vector the unit normal vector pointing inside the liquid domain
   */
  template <int dim, typename number>
  inline number
  compute_projection_factor(const dealii::Tensor<1, dim, number> &laser_direction,
                            const dealii::Tensor<1, dim, number> &normal_vector)
  {
    Assert(std::abs(laser_direction.norm() - 1.0) < 1e-8,
           dealii::ExcMessage("The laser direction must be a unit vector"));
    Assert(std::abs(normal_vector.norm() - 1.0) < 1e-8,
           dealii::ExcMessage("The laser direction must be a unit vector"));

    const number fac = normal_vector * laser_direction;
    if (fac < 0.0)
      return 0.0;
    return fac;
  }

  /**
   * Same as above for vectorized normal vectors.
   */
  template <int dim, typename number>
  inline dealii::VectorizedArray<number>
  compute_projection_factor(
    const dealii::Tensor<1, dim, number>                          &laser_direction,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal_vector)
  {
    Assert(std::abs(laser_direction.norm() - 1.0) < 1e-8,
           dealii::ExcMessage("The laser direction must be a unit vector"));

    const auto fac = normal_vector * laser_direction;
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(fac, 0.0, 0.0, fac);
  }

  /**
   * Computes the minimum distance between the point @param p and the infinite line that is defined
   * by the @param line_base and the @param line_direction, which must be a unit vector.
   */
  template <int dim, typename number>
  inline number
  compute_distance_to_line(const dealii::Point<dim, number>     &p,
                           const dealii::Point<dim, number>     &line_base,
                           const dealii::Tensor<1, dim, number> &line_direction)
  {
    Assert(std::abs(line_direction.norm() - 1.0) < 1e-8,
           dealii::ExcMessage("The line direction must be a unit vector"));

    if (dim == 1)
      return 0.0;
    else if (dim == 2)
      return std::abs(dealii::cross_product_2d(p - line_base) * line_direction);
    else if (dim == 3)
      return dealii::cross_product_3d(p - line_base, line_direction).norm();
    else
      Assert(false, dealii::ExcNotImplemented());
    return 0.0;
  }

  /**
   * Uniform intensity profile, which equals the power everywhere.
   */
  template <int dim, typename number>
  class UniformIntensityProfile : public dealii::Function<dim, number>
  {
  public:
    explicit UniformIntensityProfile(const std::function<number(void)> &get_power_in)
      : dealii::Function<dim, number>(1)
      , get_power(get_power_in)
    {}

    inline number
    value(const dealii::Point<dim> &, const unsigned int /*component*/) const final
    {
      return get_power();
    }

  private:
    const std::function<number(void)> get_power;
  };

  /**
   * Gauss laser profile for projection-based laser models
   *
   * intensity = power / ( radius^2 * pi/2 ) * exp( -2 * ( distance / radius )^2 )
   *
   * where the distance is between the point @param p and the laser beam center line defined by the
   * @param laser_position and the @param laser_direction.
   */
  template <int dim, typename number>
  class GaussProjectionIntensityProfile : public dealii::Function<dim, number>
  {
  public:
    struct State
    {
      const number                     power;
      const dealii::Point<dim, number> position;
    };

    /**
     * Constructor for a laser beam with constant @p power and @p laser_position.
     */
    GaussProjectionIntensityProfile(const number                          power,
                                    const number                          radius_in,
                                    const dealii::Point<dim, number>     &laser_position,
                                    const dealii::Tensor<1, dim, number> &laser_direction_in)
      : GaussProjectionIntensityProfile(radius_in,
                                        laser_direction_in,
                                        [state = State{power, laser_position}]() { return state; })
    {}

    /**
     * Constructor for a laser beam, whose current power and position are returned by
     * @p get_state_in.
     */
    GaussProjectionIntensityProfile(const number                            radius_in,
                                    const dealii::Tensor<1, dim, number>   &laser_direction_in,
                                    const std::function<const State(void)> &get_state_in)
      : dealii::Function<dim, number>(1)
      , radius(radius_in)
      , laser_direction(laser_direction_in)
      , get_state(get_state_in)
    {
      AssertThrow(radius > 0.0,
                  dealii::ExcMessage("The laser beam radius must be greater than zero! Abort.."));
    }

    inline number
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const final
    {
      const State  state              = get_state();
      const number peak_power_density = state.power / (radius * radius * dealii::numbers::PI_2);
      const number s = compute_distance_to_line(p, state.position, laser_direction) / radius;
      return peak_power_density * std::exp(-2. * s * s);
    }

  private:
    const number                         radius;
    const dealii::Tensor<1, dim, number> laser_direction;

    const std::function<const State(void)> get_state;
  };

  /**
   * Volumetric spherical Gauss intensity profile
   *
   * intensity = power / ( radius * sqrt(pi/2) )^3 * exp( -2 * ( distance / radius )^2 )
   *
   * where the distance is between the point @param p and the laser beam center defined by the @param laser_position
   */
  template <int dim, typename number>
  class GaussVolumetricIntensityProfile : public dealii::Function<dim, number>
  {
  public:
    struct State
    {
      const number                     power;
      const dealii::Point<dim, number> position;
    };

    GaussVolumetricIntensityProfile(const number                            radius_in,
                                    const std::function<const State(void)> &get_state_in)
      : dealii::Function<dim, number>(1)
      , radius(radius_in)
      , get_state(get_state_in)
    {
      AssertThrow(radius > 0.0,
                  dealii::ExcMessage("The laser beam radius must be greater than zero! Abort.."));
    }

    inline number
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const final
    {
      const State  state = get_state();
      const number peak_power_density =
        state.power / dealii::Utilities::fixed_power<3>(radius * std::sqrt(dealii::numbers::PI_2));
      const number s = p.distance(state.position) / radius;
      return peak_power_density * std::exp(-2. * s * s);
    }

  private:
    const number radius;

    const std::function<const State(void)> get_state;
  };

  /**
   * This class implements the laser heat source model of Gusarov et al. (2009).
   * DOI: 10.1115/1.3109245
   *
   *   ^ dim-1             2*R
   *   |                  <--->
   *   ----> x            | | |
   *                      | | |  laser beam
   *                      | | |     power
   *                      | | |
   *                      | | |
   *
   *    ---------------------------------------^
   *             ++                  ++        |
   *               ++      q       ++          |  layer thickness
   *                 +++        +++            |
   *                     ++++++                v
   *
   * The laser beam is direction is assumed to be in the negative dim-1 direction.
   */
  template <int dim, typename number>
  class GusarovIntensityProfile : public dealii::Function<dim, number>
  {
  public:
    struct State
    {
      const number                     power;
      const dealii::Point<dim, number> position;
    };

    GusarovIntensityProfile(const LaserData<number>::GusarovData   &data_in,
                            const number                            radius_in,
                            const std::function<const State(void)> &get_state_in)
      : dealii::Function<dim, number>(1)
      , data(data_in)
      , radius(radius_in)
      , lambda(data.extinction_coefficient * data.layer_thickness)
      , a(std::sqrt(1. - data.reflectivity))
      , D((1. - a) * (1. - a - data.reflectivity * (1. + a)) * std::exp(-2. * a * lambda) -
          (1. + a) * (1. + a - data.reflectivity * (1. - a)) * std::exp(2. * a * lambda))
      , get_state(get_state_in)
    {
      AssertThrow(radius > 0.0,
                  dealii::ExcMessage("The laser beam radius must be greater than zero! Abort.."));
    }

    inline number
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const final
    {
      const State  state    = get_state();
      const number distance = p.distance(state.position);
      const number z        = -(p[dim - 1] - state.position[dim - 1]);
      const number xi       = z * data.extinction_coefficient;

      return z >= data.layer_thickness or z < state.position[dim - 1] ?
               0. :
               -data.extinction_coefficient * power_density(state.power, distance) * dq_dxi(xi);
    }

  private:
    inline number
    power_density(const number power, const number distance) const
    {
      return distance <= radius ? 3. * power / (dealii::numbers::PI * radius * radius) *
                                    dealii::Utilities::fixed_power<2>(1. - distance / radius) *
                                    dealii::Utilities::fixed_power<2>(1 + distance / radius) :
                                  0.0;
    }

    inline number
    dq_dxi(const number xi) const
    {
      const number &rho = data.reflectivity;

      // clang-format off
      return xi < lambda ?
             ((3 - 3 * rho) *(std::exp(-xi) + rho * std::exp(xi - 2 * lambda)))
             / // ------------------------------------------------------
             (4 * rho - 3)
             +
             2 * a * a * rho
             / // ------------------
             (D * (4 * rho - 3))
             *
             (
                     std::exp(-lambda) * (1 - rho * rho) *
                     (std::exp(-2 * a * xi) * (a - 1) + std::exp(2 * a * xi) * (a + 1))
                     -
                     (rho * std::exp(-2 * lambda) + 3) *
                     (std::exp(2 * a * (xi - lambda)) * (1 - a - rho * (a + 1)) -
                      std::exp(2 * a * (lambda - xi)) * (a + rho * (a - 1) + 1))
             ) :
             0.0;
      // clang-format on
    }

    const LaserData<number>::GusarovData &data;
    const number                          radius;
    const number                          lambda;
    const number                          a;
    const number                          D;

    const std::function<const State(void)> get_state;
  };

  /**
   * Create the intensity profile requested by @p laser_data, which evaluates the current power
   * and position of @p beam, i.e. it follows the beam without further updates. For a volumetric
   * heat source
   * (@p is_volumetric), the Gauss profile depends on the distance to the beam center, otherwise
   * on the distance to the beam axis.
   */
  template <int dim, typename number>
  std::shared_ptr<dealii::Function<dim, number>>
  create_intensity_profile(const LaserData<number>      &laser_data,
                           const LaserBeam<dim, number> &beam,
                           const bool                    is_volumetric)
  {
    switch (laser_data.intensity_profile)
      {
        case LaserIntensityProfileType::uniform:
          return std::make_shared<UniformIntensityProfile<dim, number>>(
            [&beam]() { return beam.get_power(); });
        case LaserIntensityProfileType::Gauss:
          if (is_volumetric)
            return std::make_shared<GaussVolumetricIntensityProfile<dim, number>>(
              laser_data.radius,
              [&beam]() -> const GaussVolumetricIntensityProfile<dim, number>::State {
                return {.power = beam.get_power(), .position = beam.get_position()};
              });
          else
            return std::make_shared<GaussProjectionIntensityProfile<dim, number>>(
              laser_data.radius,
              beam.get_direction(),
              [&beam]() -> const GaussProjectionIntensityProfile<dim, number>::State {
                return {.power = beam.get_power(), .position = beam.get_position()};
              });
        case LaserIntensityProfileType::Gusarov:
          return std::make_shared<GusarovIntensityProfile<dim, number>>(
            laser_data.gusarov,
            laser_data.radius,
            [&beam]() -> const GusarovIntensityProfile<dim, number>::State {
              return {.power = beam.get_power(), .position = beam.get_position()};
            });
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
          return nullptr;
      }
  }
} // namespace MeltPoolDG::Heat
