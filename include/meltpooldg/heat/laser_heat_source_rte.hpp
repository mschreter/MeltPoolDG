#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_source_receiver.hpp>
#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_operation.hpp>

#include <map>
#include <memory>

namespace MeltPoolDG::Heat
{
  /**
   * Laser heat source model based on the radiative transport equation (RTE), see
   * RadiativeTransport::RadiativeTransportOperation. The intensity profile of the laser is
   * prescribed as inflow boundary condition of the RTE, which allows to account for shadowing
   * effects.
   */
  template <int dim, typename number>
  class LaserHeatSourceRTE final : public LaserHeatSourceBase<dim, number>
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    LaserHeatSourceRTE(ScratchData<dim, dim, number>         &scratch_data_in,
                       const PeriodicBoundaryConditions<dim> &periodic_bc_in,
                       const LaserData<number>               &laser_data,
                       const LaserBeam<dim, number>          &beam,
                       const LaserCouplingData<dim, number>  &coupling_data,
                       HeatSourceReceiver<dim, number>       &heat_operation_in);

    void
    distribute_dofs(const FiniteElementData &fe_data) override;

    void
    setup_constraints() override;

    void
    distribute_constraints() override;

    void
    reinit() override;

    void
    attach_vectors(DoFHandlerAndVectorDataType<dim, VectorType> &data) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    void
    apply() override;

  private:
    ScratchData<dim, dim, number>         &scratch_data;
    const PeriodicBoundaryConditions<dim> &periodic_bc;
    const unsigned int                     heat_no_bc_dof_idx;
    HeatSourceReceiver<dim, number>       &heat_operation;

    dealii::DoFHandler<dim>           dof_handler;
    dealii::AffineConstraints<number> constraints_dirichlet;
    dealii::AffineConstraints<number> hanging_node_constraints;
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      dirichlet_boundary_condition;

    std::unique_ptr<RadiativeTransport::RadiativeTransportOperation<dim, number>> rte_operation;

    VectorType heat_source;
  };
} // namespace MeltPoolDG::Heat
