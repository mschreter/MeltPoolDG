#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>

#include <meltpooldg/heat/laser_heat_source_rte.hpp>
#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <string>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  LaserHeatSourceRTE<dim, number>::LaserHeatSourceRTE(
    ScratchData<dim, dim, number>         &scratch_data_in,
    const PeriodicBoundaryConditions<dim> &periodic_bc_in,
    const LaserData<number>               &laser_data,
    const LaserBeam<dim, number>          &beam,
    const LaserCouplingData<dim, number>  &coupling_data,
    HeatSourceReceiver<dim, number>       &heat_operation_in)
    : scratch_data(scratch_data_in)
    , periodic_bc(periodic_bc_in)
    , heat_no_bc_dof_idx(coupling_data.heat_no_bc_dof_idx)
    , heat_operation(heat_operation_in)
  {
    AssertThrow(coupling_data.rte_data != nullptr, ExcInternalError());
    AssertThrow(coupling_data.level_set_as_heaviside,
                ExcMessage("The RTE laser model requires a heaviside!"));
    AssertThrow(not coupling_data.rte_inflow_boundary_ids.empty(),
                ExcMessage("The RTE laser model requires the simulation case to attach the "
                           "boundary condition \"laser_inflow\" for \"intensity\"!"));

    const auto &triangulation =
      scratch_data.get_dof_handler(coupling_data.level_set_dof_idx).get_triangulation();
    dof_handler.reinit(triangulation);

    // one DoFHandler slot per constraint
    scratch_data.attach_dof_handler(dof_handler);
    scratch_data.attach_dof_handler(dof_handler);
    const unsigned int rte_dof_idx = scratch_data.attach_constraint_matrix(constraints_dirichlet);
    const unsigned int rte_hanging_nodes_dof_idx =
      scratch_data.attach_constraint_matrix(hanging_node_constraints);

    // the intensity profile follows the beam and is applied as inflow boundary condition
    const auto intensity_profile =
      create_intensity_profile(laser_data, beam, false /*is_volumetric*/);
    for (const auto boundary_id : coupling_data.rte_inflow_boundary_ids)
      {
        if (coupling_data.print_boundary_ids)
          Journal::print_line(scratch_data.get_pcout(1),
                              "RTE boundary id = " + std::to_string(boundary_id),
                              "laser");
        dirichlet_boundary_condition[boundary_id] = intensity_profile;
      }

    const auto        &rte_data = *coupling_data.rte_data;
    const unsigned int rte_quad_idx =
      rte_data.fe.type == FiniteElementType::FE_SimplexP ?
        scratch_data.attach_quadrature(QGaussSimplex<dim>(rte_data.fe.get_n_q_points())) :
        scratch_data.attach_quadrature(QGauss<dim>(rte_data.fe.get_n_q_points()));

    rte_operation = std::make_unique<RadiativeTransport::RadiativeTransportOperation<dim, number>>(
      scratch_data,
      rte_data,
      laser_data.template get_beam_direction<dim>(),
      *coupling_data.level_set_as_heaviside,
      rte_dof_idx,
      rte_hanging_nodes_dof_idx,
      rte_quad_idx,
      coupling_data.level_set_dof_idx);
  }

  template <int dim, typename number>
  void
  LaserHeatSourceRTE<dim, number>::distribute_dofs(const FiniteElementData &fe_data)
  {
    FiniteElementUtils::distribute_dofs<dim, 1>(fe_data, dof_handler);
  }

  template <int dim, typename number>
  void
  LaserHeatSourceRTE<dim, number>::setup_constraints()
  {
    rte_operation->setup_constraints(scratch_data, dirichlet_boundary_condition, periodic_bc);
  }

  template <int dim, typename number>
  void
  LaserHeatSourceRTE<dim, number>::distribute_constraints()
  {
    rte_operation->distribute_constraints();
  }

  template <int dim, typename number>
  void
  LaserHeatSourceRTE<dim, number>::reinit()
  {
    rte_operation->reinit();
  }

  template <int dim, typename number>
  void
  LaserHeatSourceRTE<dim, number>::attach_vectors(
    DoFHandlerAndVectorDataType<dim, VectorType> &data)
  {
    data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
      rte_operation->attach_vectors(vectors);
    });
  }

  template <int dim, typename number>
  void
  LaserHeatSourceRTE<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    rte_operation->attach_output_vectors(data_out);
  }

  template <int dim, typename number>
  void
  LaserHeatSourceRTE<dim, number>::apply()
  {
    // update the inflow boundary condition according to the current laser beam
    setup_constraints();
    rte_operation->solve();

    scratch_data.initialize_dof_vector(heat_source, heat_no_bc_dof_idx);
    rte_operation->compute_heat_source(heat_source, heat_no_bc_dof_idx, true /*zero_out*/);
    heat_operation.get_heat_source() += heat_source;
  }

  template class LaserHeatSourceRTE<1, double>;
  template class LaserHeatSourceRTE<2, double>;
  template class LaserHeatSourceRTE<3, double>;
} // namespace MeltPoolDG::Heat
