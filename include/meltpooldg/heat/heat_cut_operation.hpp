#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/core/boundary_conditions.hpp>
#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/solution_transfer.hpp>
#include <meltpooldg/heat/heat_cut_operator.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/heat_operation_base.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>

#include <functional>
#include <map>
#include <memory>
#include <vector>


namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  class HeatCutOperation : public HeatOperationBase<dim, number>
  {
  private:
    using VectorType      = typename HeatOperationBase<dim, number>::VectorType;
    using BlockVectorType = typename HeatOperationBase<dim, number>::BlockVectorType;

    const ScratchData<dim, dim, number>                                               &scratch_data;
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;
    const PeriodicBoundaryConditions<dim>                                             &periodic_bc;
    const HeatData<number>                                                            &heat_data;

    const TimeIntegration::TimeIterator<number> &time_iterator;

    // ScratchData's DoFHandler indices for ..
    const unsigned int heat_cut_dof_idx;        // .. CutFEM DoFs with Dirichlet BCs
    const unsigned int heat_cut_no_bc_dof_idx;  // .. CutFEM DoFs without Dirichlet BCs
    const unsigned int heat_cont_no_bc_dof_idx; // .. continuous DoFs without Dirichlet BCs
    // ScratchData's Quadrature index
    const unsigned int heat_quad_idx;

    TimeIntegration::SolutionHistory<VectorType> solution_history;
    VectorType                                   interface_temperature;
    // TODO: note that this is not yet implemented in the operator
    VectorType volumetric_heat_source;

    // level set which defines the interface with its zero contour
    const unsigned int ls_dof_idx;
    const VectorType  &level_set;
    // The mesh classifier loops over all non-artificial cells and retrieves all level-set DoF
    // values to classify the cell. We use the @param mc_level_set vector that uses the matrix-based DoF layout.
    mutable VectorType                                        mc_level_set;
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier_old;

    CutUtil::SolutionTransferOperator<dim, number>  cut_solution_transfer;
    std::function<void(VectorType &)>               reinit_vector;
    std::function<void()>                           setup_dof_system;
    AttachDoFHandlerAndVectorsType<dim, VectorType> attach_all_vectors;

    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
      mapping_info_surface;
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
      mapping_info_cells;

    NewtonRaphsonSolver<number, VectorType> newton;

    std::unique_ptr<HeatCutOperator<dim, number>> heat_operator;

    Preconditioner<dim, VectorType, number> preconditioner;

    std::unique_ptr<LevelSet::Tools::NearestPoint<dim, double>> nearest_point_search;

    // bool indicating whether solution vectors are prepared for time advance
    bool ready_for_time_advance = false;

    /// the following two parameters are used to manipulate the behaviour of setup_dof_system during
    /// cut solution transfer:
    // - bool indicating whether the level set vector is ready to generate intersected quadrature
    bool ready_to_generate_intersected_quadrature = true;
    // - bool indicating whether to skip classifying cells according to the new interface position
    bool skip_classify_cells = false;

  public:
    HeatCutOperation(
      const ScratchData<dim, dim, number>                               &scratch_data_in,
      const std::shared_ptr<const BoundaryConditionManager<dim, number>> heat_bc_manager,
      const PeriodicBoundaryConditions<dim>                             &periodic_bc_in,
      const HeatData<number>                                            &heat_data_in,
      const MaterialData<number>                                        &material_data_in,
      const Evaporation::EvaporationData<number>                        &evapor_data_in,
      const TimeIntegration::TimeIterator<number>                       &time_iterator_in,
      const unsigned int                                                 heat_cut_dof_idx_in,
      const unsigned int                                                 heat_cut_no_bc_dof_idx_in,
      const unsigned int heat_continuous_no_bc_dof_idx_in,
      const unsigned int heat_quad_idx_in,
      const bool         do_solidification_in,
      const unsigned int ls_dof_idx_in,
      const VectorType  &level_set_in,
      const unsigned int vel_dof_idx_in  = -1,
      const VectorType  *velocity_in     = nullptr,
      const VectorType  *velocity_old_in = nullptr);

    /**
     * Before this operator can adapt_to_new_interface_position(), attach the following lambdas:
     * @param setup_dof_system_in same as in AMR::refine_grid()
     * @param attach_vectors_in same as in AMR::refine_grid(), only necessary if this operation is used
     *                          with non-cut operations in a partitioned application
     */
    void
    register_lambdas_for_solution_transfer(
      const std::function<void()>                          &setup_dof_system_in,
      const AttachDoFHandlerAndVectorsType<dim, VectorType> attach_vectors_in = {});

  private:
    void
    classify_cells() const;

    void
    compute_intersected_quadrature();

    void
    adapt_to_new_interface_position();

  public:
    void
    distribute_dofs(ScratchData<dim, dim, number> &mutable_scratch_data) const override;

    void
    setup_constraints(ScratchData<dim, dim, number> &mutable_scratch_data) const override;

    void
    reinit() override;

    void
    set_initial_condition(const dealii::Function<dim> &initial_temperature) override;

    void
    distribute_constraints() override;

    void
    init_time_advance() override;

    void
    solve() override;

    number
    compute_L2_norm() const;

    void
    register_interface_projection_data(
      const VectorType                         &distance,
      const BlockVectorType                    &normal_vector,
      const LevelSet::NearestPointData<double> &nearest_point_data) override;

    void
    compute_interface_temperature() override;

    void
    reset_heat_sources() override;

    void
    add_heat_source(const HeatSourceContribution<dim, number> &heat_source) override;

    /**
     * register vectors for adaptive mesh refinement solution transfer
     */
    void
    attach_vectors(std::vector<VectorType *> &vectors) override;

    /**
     * attach vectors for output
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    void
    attach_output_vectors_failed_step(GenericDataOut<dim, number> &data_out) const override;

    /*
     * getters
     */
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

  private:
    void
    setup_newton();

    void
    update_ghost_values() const;
  };
} // namespace MeltPoolDG::Heat
