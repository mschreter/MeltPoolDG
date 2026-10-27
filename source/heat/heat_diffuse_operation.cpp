#include <meltpooldg/heat/heat_diffuse_operation.hpp>
//

#include <deal.II/base/exceptions.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <type_traits>
#include <variant>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  HeatDiffuseOperation<dim, number>::HeatDiffuseOperation(
    const ScratchData<dim, dim, number>                               &scratch_data_in,
    const std::shared_ptr<const BoundaryConditionManager<dim, number>> heat_bc_manager,
    const PeriodicBoundaryConditions<dim>                             &periodic_bc_in,
    const HeatData<number>                                            &heat_data_in,
    const Material<number>                                            &material,
    const TimeIntegration::TimeIterator<number>                       &time_iterator,
    const unsigned int                                                 heat_dof_idx_in,
    const unsigned int                                                 heat_no_bc_dof_idx_in,
    const unsigned int                                                 heat_quad_idx_in,
    const unsigned int                                                 vel_dof_idx_in,
    const VectorType                                                  *velocity_in,
    const unsigned int                                                 ls_dof_idx_in,
    const VectorType                                                  *level_set_as_heaviside_in,
    const bool                                                         do_solidification)
    : scratch_data(scratch_data_in)
    , dirichlet_bc(heat_bc_manager->get_bc_of_type("dirichlet"))
    , periodic_bc(periodic_bc_in)
    , heat_data(heat_data_in)
    , time_iterator(time_iterator)
    , heat_dof_idx(heat_dof_idx_in)
    , heat_no_bc_dof_idx(heat_no_bc_dof_idx_in)
    , heat_quad_idx(heat_quad_idx_in)
    , solution_history(std::max(heat_data.predictor.n_old_solution_vectors,
                                2U /*TODO: include time integration scheme*/))
    , vel_dof_idx(vel_dof_idx_in)
    , velocity(velocity_in)
    , ls_dof_idx(ls_dof_idx_in)
    , level_set_as_heaviside(level_set_as_heaviside_in)
    , newton(heat_data.nlsolve)
  {
    heat_operator = std::make_unique<HeatDiffuseMultiPhaseOperator<dim, number>>(
      scratch_data,
      heat_bc_manager,
      heat_data,
      material,
      heat_dof_idx,
      heat_quad_idx,
      heat_no_bc_dof_idx,
      solution_history.get_current_solution(),
      solution_history.get_recent_old_solution(),
      heat_source,
      vel_dof_idx,
      velocity,
      ls_dof_idx,
      level_set_as_heaviside,
      do_solidification);

    preconditioner =
      make_preconditioner<dim, number, HeatDiffuseMultiPhaseOperator<dim, number>, VectorType>(
        heat_data.linear_solver.preconditioner_type,
        heat_operator.get(),
        scratch_data,
        heat_dof_idx);

    setup_newton();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::setup_newton()
  {
    newton.residual = [&](const VectorType & /*evaluation_point*/, VectorType &rhs) {
      // solely homogeneous dirichlet bc are distributed for the
      // corrected temperature field in the newton solver
      heat_operator->pre();
      rhs.copy_locally_owned_data_from(user_rhs);
      heat_operator->create_rhs(rhs, solution_history.get_recent_old_solution());
    };

    newton.solve_with_jacobian = [&](const VectorType &rhs, VectorType &solution_update) -> int {
      return LinearSolver::solve<VectorType, OperatorMatrixFree<dim, number>>(
        *heat_operator,
        solution_update,
        rhs,
        heat_data.linear_solver,
        preconditioner,
        "heat_operation");
    };

    newton.reinit_vector = [&](VectorType &v) {
      scratch_data.initialize_dof_vector(v, heat_dof_idx);
    };

    newton.distribute_constraints = [&](VectorType &v) {
      scratch_data.get_constraint(heat_dof_idx).distribute(v);
    };

    newton.norm_of_solution_vector = [this]() -> number {
      return MeltPoolDG::VectorTools::compute_norm<dim, number>(
        solution_history.get_current_solution(), scratch_data, heat_dof_idx, heat_quad_idx);
    };
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::register_evaporative_mass_flux(
    VectorType                                 *evaporative_mass_flux_in,
    const unsigned int                          evapor_mass_flux_dof_idx_in,
    const Evaporation::EvaporationData<number> &evapor_data)
  {
    if (evaporative_mass_flux_in)
      heat_operator->register_evaporative_mass_flux(evaporative_mass_flux_in,
                                                    evapor_mass_flux_dof_idx_in,
                                                    evapor_data);
    else
      heat_operator->register_evaporative_mass_flux(nullptr, 0, evapor_data);
  }


  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::register_surface_mesh(
    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<number> /*weights*/
                                 >> &surface_mesh_info_in)
  {
    heat_operator->register_surface_mesh(surface_mesh_info_in);
  }


  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::distribute_dofs(
    ScratchData<dim, dim, number> &mutable_scratch_data) const
  {
    Assert(&mutable_scratch_data.get_dof_handler(heat_dof_idx) ==
             &mutable_scratch_data.get_dof_handler(heat_no_bc_dof_idx),
           dealii::ExcMessage(
             "Please make sure to use the same DoFHandler for the two constraint indices!"));
    FiniteElementUtils::distribute_dofs<dim, 1>(heat_data.fe,
                                                mutable_scratch_data.get_dof_handler(heat_dof_idx));
  }


  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::setup_constraints(
    ScratchData<dim, dim, number> &mutable_scratch_data) const
  {
    Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim, number>(
      mutable_scratch_data, dirichlet_bc, periodic_bc, heat_dof_idx, heat_no_bc_dof_idx);
  }


  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function_temperature)
  {
    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(heat_dof_idx),
                                     initial_field_function_temperature,
                                     solution_history.get_current_solution());

    if (heat_data.enable_time_dependent_bc)
      MeltPoolDG::Constraints::make_DBC_and_HNC_and_merge_HNC_into_DBC<dim, number>(
        const_cast<ScratchData<dim, dim, number> &>(scratch_data),
        dirichlet_bc,
        heat_dof_idx,
        heat_no_bc_dof_idx);

    scratch_data.get_constraint(heat_dof_idx).distribute(solution_history.get_current_solution());
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::reinit()
  {
    DoFMonitor<number>::add_n_dofs("heat::n_dofs",
                                   scratch_data.get_dof_handler(heat_dof_idx).n_dofs());

    scratch_data.initialize_dof_vector(solution_history.get_current_solution(), heat_dof_idx);
    solution_history.apply_old(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, heat_no_bc_dof_idx); });

    scratch_data.initialize_dof_vector(predictor_buffer, heat_dof_idx);
    scratch_data.initialize_dof_vector(user_rhs, heat_no_bc_dof_idx);
    scratch_data.initialize_dof_vector(heat_source, heat_no_bc_dof_idx);
    if (nearest_point_search)
      scratch_data.initialize_dof_vector(interface_temperature, heat_no_bc_dof_idx);

    heat_operator->reinit();

    preconditioner.reinit();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::init_time_advance()
  {
    heat_operator->reset_time_increment(time_iterator.get_current_time_increment());

    if (heat_data.enable_time_dependent_bc)
      {
        Constraints::make_DBC_and_HNC_and_merge_HNC_into_DBC<dim, number>(
          const_cast<ScratchData<dim, dim, number> &>(scratch_data),
          dirichlet_bc,
          heat_dof_idx,
          heat_no_bc_dof_idx);
      }

    if (not predictor)
      predictor = std::make_unique<Predictor<VectorType, number>>(heat_data.predictor,
                                                                  solution_history,
                                                                  &time_iterator);
    VectorType predictor_rhs;
    if (this->heat_data.predictor.type == PredictorType::least_squares_projection)
      {
        // solely homogeneous dirichlet bc are distributed for the
        // corrected temperature field in the newton solver
        heat_operator->pre();
        predictor_rhs = user_rhs;
        heat_operator->create_rhs(
          predictor_rhs,
          solution_history.get_current_solution() //= old_solution for current time step
        );
        heat_operator->post();
      }
    scratch_data.initialize_dof_vector(predictor_buffer, heat_dof_idx);
    predictor->vmult(*heat_operator, predictor_buffer, predictor_rhs);

    // apply constraints to predictor
    scratch_data.get_constraint(heat_dof_idx).distribute(solution_history.get_current_solution());

    ready_for_time_advance = true;
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::solve()
  {
    solve(true);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::solve(const bool do_finish_time_step)
  {
    const ScopedName         scope_n("solve");
    const TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    if (not ready_for_time_advance)
      init_time_advance();

    if (not heat_data.linear_solver.do_matrix_free)
      AssertThrow(false, ExcNotImplemented());

    // setup preconditioner
    heat_operator->pre();
    preconditioner.update();

    try
      {
        newton.solve(solution_history.get_current_solution());
      }
    catch (const ExcNewtonDidNotConverge &)
      {
        AssertThrow(false, ExcHeatTransferNoConvergence());
      }

    if (do_finish_time_step)
      finish_time_advance();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::finish_time_advance()
  {
    heat_operator->post();
    solution_history.update_ghost_values();
    ready_for_time_advance = false;
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::register_interface_projection_data(
    const VectorType                         &distance,
    const BlockVectorType                    &normal_vector,
    const LevelSet::NearestPointData<number> &nearest_point_data)
  {
    nearest_point_search = std::make_unique<LevelSet::Tools::NearestPoint<dim, double>>(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_dof_idx),
      distance,
      normal_vector,
      scratch_data.get_remote_point_evaluation(heat_no_bc_dof_idx),
      nearest_point_data
      /*, TODO timer output */);
    scratch_data.initialize_dof_vector(interface_temperature, heat_no_bc_dof_idx);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::compute_interface_temperature()
  {
    const ScopedName         scope_n("project_interface_temperature");
    const TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    AssertThrow(
      nearest_point_search,
      dealii::ExcMessage(
        "Before computing the interface temperature, you must register the necessary data "
        "for interface projection using register_interface_projection_data()!"));

    nearest_point_search->reinit(&scratch_data.get_dof_handler(heat_dof_idx));

    nearest_point_search->template extend_interface_values(interface_temperature,
                                                           get_temperature());

    scratch_data.get_constraint(heat_no_bc_dof_idx).distribute(interface_temperature);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::reset_heat_sources()
  {
    heat_source = 0;
    user_rhs    = 0;
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::add_heat_source(
    const HeatSourceContribution<dim, number> &heat_source_in)
  {
    std::visit(
      [&](const auto &contribution) {
        using T = std::decay_t<decltype(contribution)>;
        if constexpr (std::is_same_v<T, VolumetricHeatSource<dim, number>>)
          {
            heat_source += contribution.field;
            if (heat_source.has_ghost_elements())
              heat_source.update_ghost_values();
          }
        else if constexpr (std::is_same_v<T, InterfaceHeatFlux<dim, number>>)
          {
            AssertThrow(contribution.add_weak_form,
                        ExcMessage("The diffuse heat operator requires the weak form of the "
                                   "interfacial heat flux, which is not provided. Abort..."));
            contribution.add_weak_form(user_rhs);
          }
        else
          static_assert(sizeof(T) == 0, "Heat source contribution not handled.");
      },
      heat_source_in);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::attach_vectors(std::vector<VectorType *> &vectors)
  {
    vectors.reserve(solution_history.size() + (nearest_point_search ? 1 : 0));

    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });

    if (nearest_point_search)
      vectors.push_back(&interface_temperature);

    heat_operator->attach_vectors(vectors);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::distribute_constraints()
  {
    scratch_data.get_constraint(heat_dof_idx).distribute(solution_history.get_current_solution());
    solution_history.apply_old(
      [this](VectorType &v) { scratch_data.get_constraint(heat_no_bc_dof_idx).distribute(v); });
    heat_operator->distribute_constraints();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(heat_dof_idx),
                             solution_history.get_current_solution(),
                             "temperature");

    data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                             solution_history.get_recent_old_solution(),
                             "temperature_old");

    data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                             heat_source,
                             "heat_source");

    // evaporative heat source/sink
    heat_operator->attach_output_vectors(data_out);

    if (nearest_point_search)
      data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                               interface_temperature,
                               "interface_temperature");

    data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                             user_rhs,
                             "heat_user_rhs");

    if (data_out.is_requested("heat_user_rhs_projected"))
      {
        scratch_data.initialize_dof_vector(user_rhs_projected, heat_no_bc_dof_idx);
        VectorTools::project_vector<1>(scratch_data.get_mapping(),
                                       scratch_data.get_dof_handler(heat_dof_idx),
                                       scratch_data.get_constraint(heat_dof_idx),
                                       scratch_data.get_quadrature(heat_quad_idx),
                                       user_rhs,
                                       user_rhs_projected);
        data_out.add_data_vector(scratch_data.get_dof_handler(heat_dof_idx),
                                 user_rhs_projected,
                                 "heat_user_rhs_projected");
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::attach_output_vectors_failed_step(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                             newton.get_solution_update(),
                             "temperature_newton_last_solution_update",
                             true /* force output */);
    data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                             newton.get_residual(),
                             "temperature_newton_failed_residual",
                             true /* force output */);
  }

  template <int dim, typename number>
  const typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_temperature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_temperature()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  const typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_interface_temperature() const
  {
    return interface_temperature;
  }

  template <int dim, typename number>
  typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_interface_temperature()
  {
    return interface_temperature;
  }

  template <int dim, typename number>
  const typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_heat_source() const
  {
    return heat_source;
  }

  template <int dim, typename number>
  typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_heat_source()
  {
    return heat_source;
  }

  template <int dim, typename number>
  const typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_user_rhs() const
  {
    return user_rhs;
  }

  template <int dim, typename number>
  typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_user_rhs()
  {
    return user_rhs;
  }

  template class HeatDiffuseOperation<1, double>;
  template class HeatDiffuseOperation<2, double>;
  template class HeatDiffuseOperation<3, double>;
} // namespace MeltPoolDG::Heat
