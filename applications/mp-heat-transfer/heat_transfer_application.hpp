#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/material.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_operation_base.hpp>
#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>

#include <functional>
#include <memory>

#include "heat_transfer_case.hpp"


namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  class HeatTransferApplication
  {
  private:
    using CaseType        = HeatTransferCase<dim, number>;
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    const std::unique_ptr<CaseType> simulation_case;

    TimeIntegration::SolutionHistory<VectorType> velocity_hist;
    VectorType                                   level_set_as_heaviside;
    VectorType                                   level_set;

    std::shared_ptr<TimeIntegration::TimeIterator<number>> time_iterator;
    dealii::DoFHandler<dim>                                dof_handler;
    dealii::DoFHandler<dim>                                dof_handler_velocity;
    dealii::DoFHandler<dim>                                dof_handler_level_set;

    dealii::AffineConstraints<number> heat_dirichlet_constraints;
    dealii::AffineConstraints<number> heat_hanging_nodes_constraints;
    dealii::AffineConstraints<number> velocity_hanging_nodes_constraints;
    dealii::AffineConstraints<number> level_set_hanging_nodes_constraints;

    unsigned int heat_dof_idx;
    unsigned int heat_no_bc_dof_idx;
    unsigned int heat_quad_idx;
    unsigned int velocity_dof_idx;
    unsigned int level_set_dof_idx;

    // optional DoFHandler index for the HeatCutOperation's continuous DoFs
    // default value is invalid so we don't accidentally use a different DoFHandler
    unsigned int heat_continuous_no_bc_dof_idx = -1;

    std::shared_ptr<ScratchData<dim, dim, number>>  scratch_data;
    std::shared_ptr<HeatOperationBase<dim, number>> heat_operation;
    std::shared_ptr<Material<number>>               material;
    std::shared_ptr<Postprocessor<dim, number>>     post_processor;

    std::shared_ptr<dealii::Function<dim>> velocity_field_function;
    std::shared_ptr<dealii::Function<dim>> heaviside_field_function;
    std::shared_ptr<dealii::Function<dim>> level_set_field_function;

    std::shared_ptr<LaserBeam<dim, number>>           laser_beam;
    std::shared_ptr<LaserHeatSourceBase<dim, number>> laser_heat_source;

    // AMR
    AMR::MarkCellsForRefinementType<dim>            mark_cells_for_refinement = {};
    AttachDoFHandlerAndVectorsType<dim, VectorType> attach_vectors_for_amr    = {};
    std::function<void()>                           amr_post                  = {};

  public:
    HeatTransferApplication(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();

  private:
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize();

    void
    setup_dof_system();

    void
    compute_field_vector(VectorType            &vector,
                         unsigned               dof_idx,
                         dealii::Function<dim> &field_function);

    /*
     *  perform output of results
     */
    void
    output_results(const bool output_not_converged = false);
    /*
     * collect all relevant output data
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;
    /*
     *  perform adaptive mesh refinement
     */
    void
    refine_mesh(const bool is_initial_solution = false);
  };
} // namespace MeltPoolDG::Heat
