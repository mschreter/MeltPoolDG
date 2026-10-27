#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <functional>
#include <memory>
#include <variant>

namespace MeltPoolDG::Heat
{
  /**
   * Volumetric heat source given as a DoF vector @p field with the DoF layout of the heat
   * operator (without Dirichlet constraints).
   */
  template <int dim, typename number>
  struct VolumetricHeatSource
  {
    const dealii::LinearAlgebra::distributed::Vector<number> &field;
  };

  /**
   * Heat flux @p flux across the interface, caused by radiation with the direction
   * @p direction. A heat operator that evaluates the heat flux itself at the (sharp) interface
   * uses @p flux and @p direction. Otherwise, the heat operator calls @p add_weak_form, which
   * adds the weak form of the heat flux, i.e. the surface integral over the interface tested with
   * the shape functions of the heat operator, to the given right-hand-side vector.
   */
  template <int dim, typename number>
  struct InterfaceHeatFlux
  {
    std::shared_ptr<const dealii::Function<dim, number>>                      flux;
    dealii::Tensor<1, dim, number>                                            direction;
    std::function<void(dealii::LinearAlgebra::distributed::Vector<number> &)> add_weak_form;
  };

  /**
   * Heat source contribution that is passed to HeatOperationBase::add_heat_source(). The
   * contribution references data of its provider and is only valid until the provider computes
   * the next contribution.
   */
  template <int dim, typename number>
  using HeatSourceContribution =
    std::variant<VolumetricHeatSource<dim, number>, InterfaceHeatFlux<dim, number>>;
} // namespace MeltPoolDG::Heat
