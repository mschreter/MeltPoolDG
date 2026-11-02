#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_values.h>

#include <meltpooldg/heat/laser_heat_source_volumetric.hpp>
#include <meltpooldg/heat/laser_intensity_profiles.hpp>

#include <vector>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  LaserHeatSourceVolumetric<dim, number>::LaserHeatSourceVolumetric(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const LaserData<number>             &laser_data,
    const LaserBeam<dim, number>        &beam,
    const unsigned int                   heat_dof_idx_in,
    HeatSourceReceiver<dim, number>     &heat_operation_in)
    : scratch_data(scratch_data_in)
    , intensity_profile(create_intensity_profile(laser_data, beam, true /*is_volumetric*/))
    , heat_dof_idx(heat_dof_idx_in)
    , heat_operation(heat_operation_in)
  {}

  template <int dim, typename number>
  void
  LaserHeatSourceVolumetric<dim, number>::apply()
  {
    scratch_data.initialize_dof_vector(heat_source, heat_dof_idx);
    compute_volumetric_heat_source(heat_source);
    heat_operation.get_heat_source() += heat_source;
  }

  template <int dim, typename number>
  void
  LaserHeatSourceVolumetric<dim, number>::compute_volumetric_heat_source(
    VectorType &heat_source_vector) const
  {
    FEValues<dim> heat_source_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(heat_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(heat_dof_idx).get_fe().get_unit_support_points()),
      update_quadrature_points);

    const unsigned int dofs_per_cell =
      scratch_data.get_dof_handler(heat_dof_idx).get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : scratch_data.get_dof_handler(heat_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            heat_source_eval.reinit(cell);

            for (const auto q : heat_source_eval.quadrature_point_indices())
              heat_source_vector[local_dof_indices[q]] =
                intensity_profile->value(heat_source_eval.quadrature_point(q));
          }
      }

    scratch_data.get_constraint(heat_dof_idx).distribute(heat_source_vector);
  }

  template class LaserHeatSourceVolumetric<1, double>;
  template class LaserHeatSourceVolumetric<2, double>;
  template class LaserHeatSourceVolumetric<3, double>;
} // namespace MeltPoolDG::Heat
