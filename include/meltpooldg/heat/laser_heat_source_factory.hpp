#pragma once

#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_source_receiver.hpp>
#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>

#include <memory>

namespace MeltPoolDG::Heat
{
  /**
   * Create the laser heat source model requested by @p laser_data for a heat operation, which
   * accepts heat sources as DoF vectors (diffuse heat operator). The model follows the state of
   * @p beam and adds its heat source to @p heat_operation.
   *
   * For heat operations that evaluate the interfacial heat flux themselves (CutFEM), use
   * LaserInterfaceHeatFlux instead.
   */
  template <int dim, typename number>
  std::unique_ptr<LaserHeatSourceBase<dim, number>>
  create_laser_heat_source(ScratchData<dim, dim, number>         &scratch_data,
                           const PeriodicBoundaryConditions<dim> &periodic_bc,
                           const LaserData<number>               &laser_data,
                           const LaserBeam<dim, number>          &beam,
                           const LaserCouplingData<dim, number>  &coupling_data,
                           HeatSourceReceiver<dim, number>       &heat_operation);
} // namespace MeltPoolDG::Heat
