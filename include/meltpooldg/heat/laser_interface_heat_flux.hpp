#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/heat/heat_source_receiver.hpp>
#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_data.hpp>

#include <memory>

namespace MeltPoolDG::Heat
{
  /**
   * Laser heat flux across the phase interface for heat operations, which evaluate the
   * interfacial heat flux themselves along the sharp interface (CutFEM).
   *
   * The intensity profile I of the laser is projected in beam direction d onto the interface with
   * the unit normal vector n, pointing into the liquid:
   *
   * q = absorptivity * I(x) * max(0, n · d).
   *
   * The intensity profile follows the current state of the laser beam.
   */
  template <int dim, typename number>
  class LaserInterfaceHeatFlux final : public InterfaceHeatFlux<dim, number>
  {
  public:
    using VectorizedArrayType = typename InterfaceHeatFlux<dim, number>::VectorizedArrayType;

    LaserInterfaceHeatFlux(const LaserData<number> &laser_data, const LaserBeam<dim, number> &beam);

    VectorizedArrayType
    value(const dealii::Point<dim, VectorizedArrayType>     &p,
          const dealii::Tensor<1, dim, VectorizedArrayType> &normal) const override;

  private:
    const std::shared_ptr<const dealii::Function<dim, number>> intensity_profile;
    const dealii::Tensor<1, dim, number>                       laser_direction;
    const number                                               absorptivity;
  };
} // namespace MeltPoolDG::Heat
