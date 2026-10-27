#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>

#include <meltpooldg/core/boundary_conditions.hpp>
#include <meltpooldg/core/material.hpp>
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/heat_diffuse_operator.hpp>
#include <meltpooldg/heat/heat_operation_base.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <map>
#include <memory>
#include <tuple>
#include <vector>


namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  class HeatDiffuseOperation : public HeatOperationBase<dim, number>
  {
  private:
    using VectorType      = typename HeatOperationBase<dim, number>::VectorType;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    const ScratchData<dim, dim, number>                                               &scratch_data;
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;
    const PeriodicBoundaryConditions<dim>                                             &periodic_bc;
    /**
     * parameters
     */
    const HeatData<number>                      &heat_data;
    const TimeIntegration::TimeIterator<number> &time_iterator;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int heat_dof_idx;
    const unsigned int heat_no_bc_dof_idx;
    const unsigned int heat_quad_idx;

    // These are the primary solution variables of this module, which will be also publicly
    // accessible for output_results.
    TimeIntegration::SolutionHistory<VectorType> solution_history;
    VectorType                                   predictor_buffer;
    VectorType                                   heat_source;
    VectorType                                   user_rhs;
    VectorType                                   interface_temperature;

    // for output only
    mutable VectorType user_rhs_projected;

    std::unique_ptr<Predictor<VectorType, number>> predictor;

    // optional flow velocity for internal convection
    const unsigned int vel_dof_idx;
    const VectorType  *velocity;

    // optional level-set heaviside field for two phase flow
    const unsigned int ls_dof_idx;
    const VectorType  *level_set_as_heaviside;

    NewtonRaphsonSolver<number, VectorType> newton;

    std::unique_ptr<HeatDiffuseMultiPhaseOperator<dim, number>> heat_operator;

    Preconditioner<dim, VectorType, number> preconditioner;

    // determine whether solution vectors are prepared for time advance
    bool ready_for_time_advance = false;

    std::unique_ptr<LevelSet::Tools::NearestPoint<dim, double>> nearest_point_search;

  public:
    HeatDiffuseOperation(
      const ScratchData<dim, dim, number>                               &scratch_data_in,
      const std::shared_ptr<const BoundaryConditionManager<dim, number>> heat_bc_manager,
      const PeriodicBoundaryConditions<dim>                             &periodic_bc_in,
      const HeatData<number>                                            &heat_data_in,
      const Material<number>                                            &material,
      const TimeIntegration::TimeIterator<number>                       &time_iterator,
      unsigned int                                                       heat_dof_idx_in,
      unsigned int                                                       heat_no_bc_dof_idx_in,
      unsigned int                                                       heat_quad_idx_in,
      unsigned int                                                       vel_dof_idx_in = 0,
      const VectorType                                                  *velocity_in    = nullptr,
      unsigned int                                                       ls_dof_idx_in  = 0,
      const VectorType *level_set_as_heaviside_in                                       = nullptr,
      const bool        do_solidifiaction                                               = false);

    void
    register_evaporative_mass_flux(VectorType        *evaporative_mass_flux_in,
                                   const unsigned int evapor_mass_flux_dof_idx_in,
                                   const Evaporation::EvaporationData<number> &evapor_data);

    void
    register_surface_mesh(
      const std::vector<
        std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                   std::vector<dealii::Point<dim>> /*quad_points*/,
                   std::vector<number> /*weights*/
                   >> &surface_mesh_info_in);

    void
    distribute_dofs(ScratchData<dim, dim, number> &mutable_scratch_data) const override;

    void
    setup_constraints(ScratchData<dim, dim, number> &mutable_scratch_data) const override;

    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function_temperature) override;

    void
    reinit() override;

    void
    init_time_advance() override;

    void
    solve() override;

    void
    solve(const bool do_finish_time_step);

    void
    finish_time_advance();

    void
    register_interface_projection_data(
      const VectorType                         &distance,
      const BlockVectorType                    &normal_vector,
      const LevelSet::NearestPointData<number> &nearest_point_data) override;

    void
    compute_interface_temperature() override;

    void
    reset_heat_sources() override;

    void
    add_heat_source(const HeatSourceContribution<dim, number> &heat_source_in) override;

    void
    attach_vectors(std::vector<VectorType *> &vectors) override;

    void
    distribute_constraints() override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    void
    attach_output_vectors_failed_step(GenericDataOut<dim, number> &data_out) const override;

    const VectorType &
    get_temperature() const override;

    VectorType &
    get_temperature() override;

    const VectorType &
    get_interface_temperature() const override;

    VectorType &
    get_interface_temperature() override;

    const VectorType &
    get_heat_source() const override;

    VectorType &
    get_heat_source() override;

    const VectorType &
    get_user_rhs() const;

    VectorType &
    get_user_rhs();

  private:
    void
    setup_newton();
  };
} // namespace MeltPoolDG::Heat
