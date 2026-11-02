#include <deal.II/base/exceptions.h>

#include <meltpooldg/heat/laser_heat_source_factory.hpp>
#include <meltpooldg/heat/laser_heat_source_projection_based.hpp>
#include <meltpooldg/heat/laser_heat_source_rte.hpp>
#include <meltpooldg/heat/laser_heat_source_volumetric.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  std::unique_ptr<LaserHeatSourceBase<dim, number>>
  create_laser_heat_source(ScratchData<dim, dim, number>         &scratch_data,
                           const PeriodicBoundaryConditions<dim> &periodic_bc,
                           const LaserData<number>               &laser_data,
                           const LaserBeam<dim, number>          &beam,
                           const LaserCouplingData<dim, number>  &coupling_data,
                           HeatSourceReceiver<dim, number>       &heat_operation)
  {
    switch (laser_data.model)
      {
        case LaserModelType::volumetric:
          return std::make_unique<LaserHeatSourceVolumetric<dim, number>>(
            scratch_data, laser_data, beam, coupling_data.heat_no_bc_dof_idx, heat_operation);
        case LaserModelType::interface_projection_regularized:
        case LaserModelType::interface_projection_sharp:
        case LaserModelType::interface_projection_sharp_conforming:
          return std::make_unique<LaserHeatSourceProjectionBased<dim, number>>(
            scratch_data, laser_data, beam, coupling_data, heat_operation);
        case LaserModelType::RTE:
          return std::make_unique<LaserHeatSourceRTE<dim, number>>(
            scratch_data, periodic_bc, laser_data, beam, coupling_data, heat_operation);
        default:
          AssertThrow(false,
                      ExcMessage(
                        "No requested laser model found. Please specify the "
                        "heat source model in the laser section of the input parameters."));
          return nullptr;
      }
  }

  template std::unique_ptr<LaserHeatSourceBase<1, double>>
  create_laser_heat_source(ScratchData<1, 1, double> &,
                           const PeriodicBoundaryConditions<1> &,
                           const LaserData<double> &,
                           const LaserBeam<1, double> &,
                           const LaserCouplingData<1, double> &,
                           HeatSourceReceiver<1, double> &);
  template std::unique_ptr<LaserHeatSourceBase<2, double>>
  create_laser_heat_source(ScratchData<2, 2, double> &,
                           const PeriodicBoundaryConditions<2> &,
                           const LaserData<double> &,
                           const LaserBeam<2, double> &,
                           const LaserCouplingData<2, double> &,
                           HeatSourceReceiver<2, double> &);
  template std::unique_ptr<LaserHeatSourceBase<3, double>>
  create_laser_heat_source(ScratchData<3, 3, double> &,
                           const PeriodicBoundaryConditions<3> &,
                           const LaserData<double> &,
                           const LaserBeam<3, double> &,
                           const LaserCouplingData<3, double> &,
                           HeatSourceReceiver<3, double> &);
} // namespace MeltPoolDG::Heat
