#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <memory>

namespace MeltPoolDG::Heat
{
  /**
   * Interface of a heat operation that accepts heat sources given as DoF vectors.
   */
  template <int dim, typename number>
  class HeatSourceReceiver
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    virtual ~HeatSourceReceiver() = default;

    /**
     * Volumetric heat source, given as DoF vector.
     */
    virtual VectorType &
    get_heat_source() = 0;

    /**
     * Heat source in weak form, which is directly added to the right-hand side, e.g. an
     * interfacial heat flux integrated along a sharp interface.
     */
    virtual VectorType &
    get_user_rhs() = 0;
  };

  /**
   * Heat flux across the phase interface into the liquid.
   */
  template <int dim, typename number>
  class InterfaceHeatFlux
  {
  public:
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    virtual ~InterfaceHeatFlux() = default;

    /**
     * Evaluate the heat flux at the points @p p with the unit normal vectors @p normal,
     * pointing into the liquid.
     */
    virtual VectorizedArrayType
    value(const dealii::Point<dim, VectorizedArrayType>     &p,
          const dealii::Tensor<1, dim, VectorizedArrayType> &normal) const = 0;
  };

  /**
   * Interface of a heat operation that evaluates an interfacial heat flux itself along the
   * sharp interface.
   */
  template <int dim, typename number>
  class InterfaceHeatFluxReceiver
  {
  public:
    virtual ~InterfaceHeatFluxReceiver() = default;

    virtual void
    register_interface_heat_flux(
      std::shared_ptr<const InterfaceHeatFlux<dim, number>> interface_heat_flux) = 0;
  };
} // namespace MeltPoolDG::Heat
