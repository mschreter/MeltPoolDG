#include <meltpooldg/heat/heat_cut_operation.hpp>
//

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/cut/cut_norm.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/functions.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <algorithm>
#include <string>
#include <type_traits>
#include <variant>


namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
  construct_dirichlet_bc_map(const bool                                              two_phase,
                             const std::map<dealii::types::boundary_id,
                                            std::shared_ptr<dealii::Function<dim>>> &dirichlet_bc)
  {
    if (two_phase)
      // For the two-phase case, the initial temperature function must be set up with 2 components
      // because the FESystem for cut is set up with 2 to handle both phases.
      {
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> heat_bc;
        for (const auto &bc : dirichlet_bc)
          heat_bc[bc.first] =
            std::make_shared<Functions::NComponentFunction<dim, number>>(*bc.second);
        return heat_bc;
      }
    else
      return dirichlet_bc;
  }

  template <int dim, typename number>
  HeatCutOperation<dim, number>::HeatCutOperation(
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
    const unsigned int vel_dof_idx_in,
    const VectorType  *velocity_in,
    const VectorType  *velocity_old_in)
    : scratch_data(scratch_data_in)
    , dirichlet_bc(
        construct_dirichlet_bc_map<dim, number>(heat_data_in.cut.two_phase,
                                                heat_bc_manager->get_bc_of_type("dirichlet")))
    , periodic_bc(periodic_bc_in)
    , heat_data(heat_data_in)
    , time_iterator(time_iterator_in)
    , heat_cut_dof_idx(heat_cut_dof_idx_in)
    , heat_cut_no_bc_dof_idx(heat_cut_no_bc_dof_idx_in)
    , heat_cont_no_bc_dof_idx(heat_continuous_no_bc_dof_idx_in)
    , heat_quad_idx(heat_quad_idx_in)
    , solution_history(std::max(2U, heat_data.predictor.n_old_solution_vectors))
    , ls_dof_idx(ls_dof_idx_in)
    , level_set(level_set_in)
    , cut_solution_transfer(heat_data.cut.stabilization.ghost_penalty,
                            heat_data.cut.two_phase,
                            heat_data.verbosity_level)
    , mapping_info_surface(scratch_data.get_mapping(),
                           dealii::update_values | dealii::update_gradients |
                             dealii::update_JxW_values | dealii::update_normal_vectors)
    , newton(heat_data.nlsolve)
  {
    AssertThrow(heat_data.linear_solver.do_matrix_free, dealii::ExcNotImplemented());

    // liquid domain
    mapping_info_cells.push_back(
      std::make_shared<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>(
        scratch_data.get_mapping(),
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
          dealii::update_normal_vectors));
    // gas domain
    if (heat_data.cut.two_phase)
      mapping_info_cells.push_back(
        std::make_shared<
          dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>(
          scratch_data.get_mapping(),
          dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_normal_vectors));

    heat_operator =
      std::make_unique<HeatCutOperator<dim, number>>(scratch_data,
                                                     heat_data,
                                                     material_data_in,
                                                     evapor_data_in,
                                                     heat_cut_dof_idx,
                                                     heat_cut_no_bc_dof_idx,
                                                     heat_cont_no_bc_dof_idx,
                                                     heat_quad_idx,
                                                     solution_history.get_current_solution(),
                                                     mapping_info_surface,
                                                     mapping_info_cells,
                                                     do_solidification_in,
                                                     vel_dof_idx_in,
                                                     velocity_in,
                                                     velocity_old_in);

    preconditioner = make_preconditioner<dim, number, HeatCutOperator<dim, number>, VectorType>(
      heat_data.linear_solver.preconditioner_type,
      heat_operator.get(),
      scratch_data,
      heat_cut_dof_idx);

    mesh_classifier = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      scratch_data.get_dof_handler(ls_dof_idx), mc_level_set);
    mesh_classifier_old = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      scratch_data.get_dof_handler(ls_dof_idx), mc_level_set);

    setup_newton();

    reinit_vector = [this](VectorType &vec) {
      const auto &dh = scratch_data.get_dof_handler(heat_cut_dof_idx);
      vec.reinit(dh.locally_owned_dofs(),
                 scratch_data.get_locally_relevant_dofs(heat_cut_dof_idx),
                 dh.get_mpi_communicator());
    };
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::register_lambdas_for_solution_transfer(
    const std::function<void()>                          &setup_dof_system_in,
    const AttachDoFHandlerAndVectorsType<dim, VectorType> attach_vectors_in)
  {
    setup_dof_system   = setup_dof_system_in;
    attach_all_vectors = attach_vectors_in;
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::classify_cells() const
  {
    if (skip_classify_cells)
      return;

    const auto &ls_dof_handler = scratch_data.get_dof_handler(ls_dof_idx);
    mc_level_set.reinit(ls_dof_handler.locally_owned_dofs(),
                        dealii::DoFTools::extract_locally_relevant_dofs(ls_dof_handler),
                        scratch_data.get_mpi_comm(ls_dof_idx));
    mc_level_set.copy_locally_owned_data_from(level_set);
    mc_level_set.update_ghost_values();
    mesh_classifier->reclassify();
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::adapt_to_new_interface_position()
  {
    // We must make sure that compute_intersected_quadrature() is not run during the solution
    // transfer, since it can be part of setup_dof_system. The reason is, that during solution
    // transfer the level set solution vector is not available.
    ready_to_generate_intersected_quadrature = false;

    std::swap(mesh_classifier_old, mesh_classifier);
    classify_cells();

    // During cut solution transfer, the new dof system is set up, which usually includes
    // classify_cells(). Since we just classified the cells above, we can skip that operation.
    skip_classify_cells = true;

    {
      const ScopedName                 scope_n("cut_solution_transfer");
      const dealii::TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

      Assert(setup_dof_system != nullptr,
             dealii::ExcMessage("You must register the setup_dof_system lambda function first!"));

      std::vector<const VectorType *> cut_solution_vectors;
      cut_solution_vectors.reserve(solution_history.size());
      solution_history.apply(
        [&cut_solution_vectors](VectorType &vec) { cut_solution_vectors.push_back(&vec); });

      // transfer old solution according to the new interface position,
      // the matrix-free object is reinitialized within the reinit function
      cut_solution_transfer.reinit(
        const_cast<dealii::DoFHandler<dim> &>(scratch_data.get_dof_handler(heat_cut_dof_idx)),
        const_cast<dealii::Triangulation<dim> &>(scratch_data.get_triangulation()),
        cut_solution_vectors,
        *mesh_classifier_old,
        *mesh_classifier,
        reinit_vector,
        setup_dof_system,
        attach_all_vectors);
    }

    {
      const auto  &updated_solutions = cut_solution_transfer.get_updated_solutions();
      unsigned int i                 = 0;
      solution_history.apply([this, &updated_solutions, &i](VectorType &vec) {
        scratch_data.initialize_dof_vector(vec, heat_cut_dof_idx);
        vec.copy_locally_owned_data_from(updated_solutions[i]);
        vec.update_ghost_values();
        ++i;
      });
    }

    // generate intersected quadrature for new interface position.
    ready_to_generate_intersected_quadrature = true;
    compute_intersected_quadrature();

    // re-enable classifying cells for future operations
    skip_classify_cells = false;
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::compute_intersected_quadrature()
  {
    if (not ready_to_generate_intersected_quadrature)
      return;

    const ScopedName                 scope_n("intersected_quadrature");
    const dealii::TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    level_set.update_ghost_values();

    CutUtil::compute_intersected_quadrature(mapping_info_cells,
                                            mapping_info_surface,
                                            scratch_data.get_dof_handler(ls_dof_idx),
                                            level_set,
                                            scratch_data.get_matrix_free(),
                                            heat_data.fe.degree,
                                            heat_data.cut.two_phase);
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::distribute_dofs(
    ScratchData<dim, dim, number> &mutable_scratch_data) const
  {
    AssertThrow(heat_data.fe.type == FiniteElementType::FE_Q,
                dealii::ExcMessage("For now, only standard FE_Q elements are supported."));
    Assert(&mutable_scratch_data.get_dof_handler(heat_cut_dof_idx) ==
             &mutable_scratch_data.get_dof_handler(heat_cut_no_bc_dof_idx),
           dealii::ExcMessage(
             "Please make sure to use the same DoFHandler for the two constraint indices!"));
    Assert(&mutable_scratch_data.get_dof_handler(heat_cut_dof_idx) !=
             &mutable_scratch_data.get_dof_handler(heat_cont_no_bc_dof_idx),
           dealii::ExcMessage(
             "Please make sure to use different DoFHandlers for the cut and continuous indices!"));

    classify_cells();

    dealii::FE_Q<dim>             fe_q(heat_data.fe.degree);
    dealii::FE_Nothing<dim>       fe_n;
    dealii::hp::FECollection<dim> fe_collection;

    if (heat_data.cut.two_phase)
      {
        fe_collection.push_back(dealii::FESystem<dim, dim>(fe_q, 1, fe_n, 1)); // liquid
        fe_collection.push_back(
          dealii::FESystem<dim, dim>(fe_q, 1, fe_q, 1)); // intersected (liquid and gas)
        fe_collection.push_back(dealii::FESystem<dim, dim>(fe_n, 1, fe_q, 1)); // gas
      }
    else // single phase
      {
        fe_collection.push_back(fe_q); // liquid
        fe_collection.push_back(fe_q); // intersected
        fe_collection.push_back(fe_n); // gas
      }

    auto &cut_dof_handler = mutable_scratch_data.get_dof_handler(heat_cut_dof_idx);
    CutUtil::set_fe_index<dim>(cut_dof_handler, *mesh_classifier, false /* set_future */);
    cut_dof_handler.distribute_dofs(fe_collection);

    FiniteElementUtils::distribute_dofs<dim, 1>(
      heat_data.fe, mutable_scratch_data.get_dof_handler(heat_cont_no_bc_dof_idx));
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::setup_constraints(
    ScratchData<dim, dim, number> &mutable_scratch_data) const
  {
    Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim, number>(
      mutable_scratch_data, dirichlet_bc, periodic_bc, heat_cut_dof_idx, heat_cut_no_bc_dof_idx);
    Constraints::make_HNC_plus_PBC<dim>(mutable_scratch_data, periodic_bc, heat_cont_no_bc_dof_idx);
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::reinit()
  {
    DoFMonitor<number>::add_n_dofs("heat::n_dofs",
                                   scratch_data.get_dof_handler(heat_cut_dof_idx).n_dofs());

    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, heat_cut_no_bc_dof_idx); });

    if (nearest_point_search)
      scratch_data.initialize_dof_vector(interface_temperature, heat_cont_no_bc_dof_idx);

    // TODO can we avoid initializing this vector if its not used?
    scratch_data.initialize_dof_vector(volumetric_heat_source, heat_cont_no_bc_dof_idx);

    heat_operator->reinit();

    preconditioner.reinit();

    compute_intersected_quadrature();
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::set_initial_condition(
    const dealii::Function<dim> &initial_temperature)
  {
    if (heat_data.cut.two_phase)
      // For the two-phase case, the initial temperature function must be set up with 2 components
      // because the FESystem for cut is set up with 2 to handle both phases.
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       scratch_data.get_dof_handler(heat_cut_dof_idx),
                                       Functions::NComponentFunction<dim, number>(
                                         initial_temperature),
                                       solution_history.get_current_solution());
    else
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       scratch_data.get_dof_handler(heat_cut_dof_idx),
                                       initial_temperature,
                                       solution_history.get_current_solution());

    scratch_data.get_constraint(heat_cut_dof_idx)
      .distribute(solution_history.get_current_solution());
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::distribute_constraints()
  {}

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::setup_newton()
  {
    newton.residual = [&](const VectorType & /*solution_update*/, VectorType &rhs) {
      update_ghost_values();
      heat_operator->create_rhs(rhs, solution_history.get_recent_old_solution());
    };

    newton.solve_with_jacobian = [this](const VectorType &rhs, VectorType &solution_update) -> int {
      return LinearSolver::solve<VectorType>(*heat_operator,
                                             solution_update,
                                             rhs,
                                             heat_data.linear_solver,
                                             preconditioner,
                                             "heat_operation");
    };

    newton.reinit_vector = [&](VectorType &v) {
      scratch_data.initialize_dof_vector(v, heat_cut_dof_idx);
    };

    newton.distribute_constraints = [&](VectorType &v) {
      scratch_data.get_constraint(heat_cut_dof_idx).distribute(v);
    };

    newton.norm_of_solution_vector = [this]() -> number { return this->compute_L2_norm(); };
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::init_time_advance()
  {
    // TODO detect whether the level set has changed
    adapt_to_new_interface_position();

    // Using solution_history.commit_old_solutions();  messes up the partitioner.
    // This is a workaround.
    auto &solutions = const_cast<std::vector<VectorType> &>(solution_history.get_all_solutions());
    for (int i = solutions.size() - 1; i >= 1; --i)
      solutions[i].swap(solutions[i - 1]);

    // predictor: copy old solution
    // Copying the vector messes up the partitioner. As a workaround we use the copy constructor to
    // construct a temporary vector that is then swap in place.
    VectorType temp = solution_history.get_recent_old_solution();
    solution_history.get_current_solution().swap(temp);

    heat_operator->init_time_advance(time_iterator.get_current_time_increment());

    ready_for_time_advance = true;
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::solve()
  {
    if (not ready_for_time_advance)
      init_time_advance();

    const ScopedName                 scope_n("solve");
    const dealii::TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    update_ghost_values();
    preconditioner.update();

    try
      {
        newton.solve(solution_history.get_current_solution());
      }
    catch (const ExcNewtonDidNotConverge &)
      {
        AssertThrow(false, ExcHeatTransferNoConvergence());
      }

    ready_for_time_advance = false;
  }


  template <int dim, typename number>
  number
  HeatCutOperation<dim, number>::compute_L2_norm() const
  {
    return CutUtil::compute_cut_norm(solution_history.get_current_solution(),
                                     scratch_data.get_matrix_free(),
                                     mapping_info_cells,
                                     heat_data.cut.two_phase,
                                     dealii::FE_Q<dim>(heat_data.fe.degree),
                                     heat_cut_dof_idx,
                                     heat_quad_idx,
                                     CutUtil::NormType::L2_norm);
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::register_interface_projection_data(
    const VectorType                         &distance,
    const BlockVectorType                    &normal_vector,
    const LevelSet::NearestPointData<double> &nearest_point_data)
  {
    nearest_point_search = std::make_unique<LevelSet::Tools::NearestPoint<dim, double>>(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_dof_idx),
      distance,
      normal_vector,
      scratch_data.get_remote_point_evaluation(heat_cut_no_bc_dof_idx),
      nearest_point_data
      /*, TODO timer output */);
    scratch_data.initialize_dof_vector(interface_temperature, heat_cont_no_bc_dof_idx);
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::compute_interface_temperature()
  {
    const ScopedName                 scope_n("project_interface_temperature");
    const dealii::TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    interface_temperature = 0;

    AssertThrow(
      nearest_point_search,
      dealii::ExcMessage(
        "Before computing the interface temperature, you must register the necessary data "
        "for interface projection using register_interface_projection_data()!"));

    nearest_point_search->reinit(&scratch_data.get_dof_handler(heat_cut_dof_idx),
                                 &scratch_data.get_dof_handler(heat_cont_no_bc_dof_idx));

    nearest_point_search->extend_interface_values(interface_temperature, get_temperature());

    scratch_data.get_constraint(heat_cont_no_bc_dof_idx).distribute(interface_temperature);
    interface_temperature.update_ghost_values();
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::reset_heat_sources()
  {
    volumetric_heat_source = 0;
    heat_operator->register_laser_intensity_function_and_direction(nullptr, {});
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::add_heat_source(
    const HeatSourceContribution<dim, number> &heat_source)
  {
    std::visit(
      [&](const auto &contribution) {
        using T = std::decay_t<decltype(contribution)>;
        if constexpr (std::is_same_v<T, VolumetricHeatSource<dim, number>>)
          {
            (void)contribution;
            AssertThrow(false,
                        dealii::ExcMessage(
                          "The CutFEM heat operator does not support volumetric heat sources, "
                          "e.g. the laser models volumetric, interface_projection_regularized "
                          "and RTE. Abort..."));
          }
        else if constexpr (std::is_same_v<T, InterfaceHeatFlux<dim, number>>)
          {
            AssertThrow(contribution.flux,
                        dealii::ExcMessage("The CutFEM heat operator requires the interfacial "
                                           "heat flux as a function. Abort..."));
            heat_operator->register_laser_intensity_function_and_direction(contribution.flux,
                                                                           contribution.direction);
          }
        else
          static_assert(sizeof(T) == 0, "Heat source contribution not handled.");
      },
      heat_source);
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::update_ghost_values() const
  {
    if (not solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().update_ghost_values();
    if (not solution_history.get_recent_old_solution().has_ghost_elements())
      solution_history.get_recent_old_solution().update_ghost_values();
    heat_operator->update_ghost_values();
  }

  /**
   * register vectors for adaptive mesh refinement solution transfer
   */
  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::attach_vectors(std::vector<VectorType *> &vectors)
  {
    vectors.reserve(solution_history.size());
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::attach_output_vectors(GenericDataOut<dim, number> &data_out) const
  {
    if (heat_data.cut.two_phase)
      {
        const std::vector component_interpretation(
          2, dealii::DataComponentInterpretation::component_is_scalar);
        data_out.add_data_vector(scratch_data.get_dof_handler(heat_cut_dof_idx),
                                 solution_history.get_current_solution(),
                                 std::vector<std::string>{"temperature", "temperature_gas"},
                                 component_interpretation);
        data_out.add_data_vector(scratch_data.get_dof_handler(heat_cut_dof_idx),
                                 solution_history.get_recent_old_solution(),
                                 std::vector<std::string>{"temperature_old", "temperature_old_gas"},
                                 component_interpretation);
      }
    else // one-phase
      {
        data_out.add_data_vector(scratch_data.get_dof_handler(heat_cut_dof_idx),
                                 solution_history.get_current_solution(),
                                 "temperature");
        data_out.add_data_vector(scratch_data.get_dof_handler(heat_cut_dof_idx),
                                 solution_history.get_recent_old_solution(),
                                 "temperature_old");
      }

    if (nearest_point_search)
      data_out.add_data_vector(scratch_data.get_dof_handler(heat_cont_no_bc_dof_idx),
                               interface_temperature,
                               "interface_temperature");

    data_out.add_data_vector(scratch_data.get_dof_handler(heat_cont_no_bc_dof_idx),
                             volumetric_heat_source,
                             "heat_source");
  }

  template <int dim, typename number>
  void
  HeatCutOperation<dim, number>::attach_output_vectors_failed_step(
    GenericDataOut<dim, number> &data_out) const
  {
    if (heat_data.cut.two_phase)
      {
        const std::vector component_interpretation(
          2, dealii::DataComponentInterpretation::component_is_scalar);
        data_out.add_data_vector(
          scratch_data.get_dof_handler(heat_cut_no_bc_dof_idx),
          newton.get_solution_update(),
          std::vector<std::string>{"temperature_newton_last_solution_update",
                                   "temperature_newton_last_solution_update_gas"},
          component_interpretation,
          true /* force output */);
        data_out.add_data_vector(scratch_data.get_dof_handler(heat_cut_no_bc_dof_idx),
                                 newton.get_residual(),
                                 std::vector<std::string>{"temperature_newton_failed_residual",
                                                          "temperature_newton_failed_residual_gas"},
                                 component_interpretation,
                                 true /* force output */);
      }
    else
      {
        data_out.add_data_vector(scratch_data.get_dof_handler(heat_cut_no_bc_dof_idx),
                                 newton.get_solution_update(),
                                 "temperature_newton_last_solution_update",
                                 true /* force output */);
        data_out.add_data_vector(scratch_data.get_dof_handler(heat_cut_no_bc_dof_idx),
                                 newton.get_residual(),
                                 "temperature_newton_failed_residual",
                                 true /* force output */);
      }
  }

  template <int dim, typename number>
  const typename HeatCutOperation<dim, number>::VectorType &
  HeatCutOperation<dim, number>::get_temperature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  typename HeatCutOperation<dim, number>::VectorType &
  HeatCutOperation<dim, number>::get_temperature()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  const typename HeatCutOperation<dim, number>::VectorType &
  HeatCutOperation<dim, number>::get_interface_temperature() const
  {
    return interface_temperature;
  }

  template <int dim, typename number>
  typename HeatCutOperation<dim, number>::VectorType &
  HeatCutOperation<dim, number>::get_interface_temperature()
  {
    return interface_temperature;
  }

  template <int dim, typename number>
  const typename HeatCutOperation<dim, number>::VectorType &
  HeatCutOperation<dim, number>::get_heat_source() const
  {
    return volumetric_heat_source;
  }

  template <int dim, typename number>
  typename HeatCutOperation<dim, number>::VectorType &
  HeatCutOperation<dim, number>::get_heat_source()
  {
    return volumetric_heat_source;
  }



  template class HeatCutOperation<1, double>;
  template class HeatCutOperation<2, double>;
  template class HeatCutOperation<3, double>;
} // namespace MeltPoolDG::Heat
