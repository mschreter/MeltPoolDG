#include <meltpooldg/heat/heat_cut_operator.hpp>
//

#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation_data.h>
#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/material.templates.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/phase_change/evaporative_cooling.templates.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <algorithm>
#include <functional>


namespace MeltPoolDG::Heat
{
  // for dim = 2,3 the velocity type is dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
  // but for dim = 1 its simply dealii::VectorizedArray<number>
  // this alias helps derive the correct type from the Evaluator, which needs to inherit from
  // dealii::FEEvaluationBase
  template <typename Evaluation>
  using VelocityType = typename FECellIntegrator<Evaluation::dimension,
                                                 Evaluation::dimension,
                                                 typename Evaluation::ScalarNumber>::value_type;

  static constexpr dealii::EvaluationFlags::EvaluationFlags evaluate_values =
    dealii::EvaluationFlags::values;
  static constexpr dealii::EvaluationFlags::EvaluationFlags evaluate_gradients =
    dealii::EvaluationFlags::gradients;


  namespace internal
  {
    template <int dim, typename number>
    inline dealii::VectorizedArray<number>
    compute_interface_heat_flux(
      const InterfaceHeatFlux<dim, number> *interface_heat_flux,
      const dealii::FEPointEvaluation<1, dim, dim, dealii::VectorizedArray<number>>
                        &temperature_eval,
      const unsigned int q)
    {
      if (interface_heat_flux == nullptr)
        return dealii::VectorizedArray<number>(0.0);

      return interface_heat_flux->value(temperature_eval.quadrature_point(q),
                                        temperature_eval.normal_vector(q));
    }

    template <typename Evaluation>
    inline std::tuple<typename Evaluation::value_type, typename Evaluation::value_type>
    get_liquid_material_parameters(const Evaluation *temperature_eval,
                                   const Material<typename Evaluation::ScalarNumber> &material,
                                   const unsigned int                                 q)
    {
      if (temperature_eval == nullptr)
        return {/*conductivity*/ material.get_data().liquid.thermal_conductivity,
                /*cv*/ material.get_data().liquid.density *
                  material.get_data().liquid.specific_heat_capacity};

      const auto material_values =
        material.template compute_parameters<typename Evaluation::value_type>(
          temperature_eval->get_value(q),
          MaterialUpdateFlags::density | MaterialUpdateFlags::specific_heat_capacity |
            MaterialUpdateFlags::thermal_conductivity);
      return {/*conductivity*/ material_values.thermal_conductivity,
              /*cv*/ material_values.density * material_values.specific_heat_capacity};
    }

    template <typename Evaluation>
    inline std::tuple<typename Evaluation::value_type,
                      typename Evaluation::value_type,
                      typename Evaluation::value_type,
                      typename Evaluation::value_type>
    get_liquid_material_parameters_and_derivatives(
      const Evaluation                                  *temperature_eval,
      const Material<typename Evaluation::ScalarNumber> &material,
      const unsigned int                                 q)
    {
      if (temperature_eval == nullptr)
        return {/*conductivity*/ material.get_data().liquid.thermal_conductivity,
                /*cv*/ material.get_data().liquid.density *
                  material.get_data().liquid.specific_heat_capacity,
                /*d_conductivity_d_T*/ 0.0,
                /*d_cv_d_T*/ 0.0};

      const auto material_values =
        material.template compute_parameters<typename Evaluation::value_type>(
          temperature_eval->get_value(q),
          MaterialUpdateFlags::density | MaterialUpdateFlags::specific_heat_capacity |
            MaterialUpdateFlags::thermal_conductivity |
            MaterialUpdateFlags::d_thermal_conductivity_d_T |
            MaterialUpdateFlags::d_specific_heat_capacity_d_T | MaterialUpdateFlags::d_density_d_T);
      return {/*conductivity*/ material_values.thermal_conductivity,
              /*cv*/ material_values.density * material_values.specific_heat_capacity,
              /*d_conductivity_d_T*/ material_values.d_thermal_conductivity_d_T,
              /*d_cv_d_T*/ material_values.d_specific_heat_capacity_d_T * material_values.density +
                material_values.d_density_d_T * material_values.specific_heat_capacity};
    }

    template <typename Evaluation, typename VectorType>
    inline void
    reinit_and_read_plain(Evaluation        &evaluator,
                          const VectorType  &dof_vector,
                          const unsigned int cell_index)
    {
      evaluator.reinit(cell_index);
      evaluator.read_dof_values_plain(dof_vector);
    };


    // tangent operator domain integral
    template <typename Evaluation>
    inline void
    tangent_domain_integral(
      Evaluation                            &evaluator,
      const Evaluation                      *T_eval,
      const typename Evaluation::value_type &conductivity,
      const typename Evaluation::value_type &cv, // density * specific_heat_capacity
      [[maybe_unused]] const typename Evaluation::value_type &d_conductivity_d_T,
      [[maybe_unused]] const typename Evaluation::value_type &d_cv_d_T,
      const VelocityType<Evaluation>                         &velocity,
      const typename Evaluation::ScalarNumber                 ost_factor, // delta_t * theta
      const unsigned int                                      q)
    {
      const auto flux_1 = ost_factor * evaluator.get_gradient(q);

      // heat storage
      auto val = cv * evaluator.get_value(q);

      // convective heat flux
      val += cv * scalar_product(velocity, flux_1);

      // conductive heat flux
      auto grad = conductivity * flux_1;

      if (T_eval)
        {
          const auto flux_2 = ost_factor * T_eval->get_gradient(q) * evaluator.get_value(q);

          // temperature-dependent material parameters terms
          val += d_cv_d_T * T_eval->get_value(q) * evaluator.get_value(q);
          val += d_cv_d_T * scalar_product(velocity, flux_2);
          grad += d_conductivity_d_T * flux_2;
        }

      evaluator.submit_value(val, q);
      evaluator.submit_gradient(grad, q);
    }


    // residual domain integral
    template <typename Evaluation>
    inline void
    residual_domain_integral(
      Evaluation                             &evaluator, // T^n+1
      const Evaluation                       &Told_eval,
      const typename Evaluation::value_type  &conductivity_new,
      const typename Evaluation::value_type  &cv_new, // density * specific_heat_capacity
      const typename Evaluation::value_type  &conductivity_old,
      const typename Evaluation::value_type  &cv_old, // density * specific_heat_capacity
      const VelocityType<Evaluation>         &velocity,
      const VelocityType<Evaluation>         &velocity_old,
      const typename Evaluation::ScalarNumber ost_factor_implicit, // delta_t * theta
      const typename Evaluation::ScalarNumber ost_factor_explicit, // delta_t * (1. - theta)
      const unsigned int                      q)
    {
      const auto flux_1 = ost_factor_implicit * evaluator.get_gradient(q);
      const auto flux_2 = ost_factor_explicit * Told_eval.get_gradient(q);

      // heat storage
      auto val = cv_new * evaluator.get_value(q) - cv_old * Told_eval.get_value(q);

      // conductive heat flux
      evaluator.submit_gradient((conductivity_new * flux_1 + conductivity_old * flux_2) * -1., q);

      // convective heat flux
      val +=
        cv_new * scalar_product(velocity, flux_1) + cv_old * scalar_product(velocity_old, flux_2);

      evaluator.submit_value(val * -1., q);
    }


    // tangent operator immersed boundary integral (one-phase case)
    // only if evaporative cooling is enabled
    template <typename Evaluation>
    inline void
    tangent_immersed_boundary_integral(
      Evaluation       &evaluator,
      const Evaluation &T_eval,
      const std::function<typename Evaluation::value_type(const typename Evaluation::value_type &)>
                                             &compute_qVapor_derivative,
      const typename Evaluation::ScalarNumber ost_factor, // delta_t * theta
      const unsigned int                      q)
    {
      // temperature-dependent evaporation-induced heat flux
      evaluator.submit_value(-ost_factor * compute_qVapor_derivative(T_eval.get_value(q)) *
                               evaluator.get_value(q),
                             q);
    }


    // residual immersed boundary integral (one-domain case)
    template <typename Evaluation>
    inline void
    residual_immersed_boundary_integral(
      Evaluation       &evaluator, // T^n+1
      const Evaluation *Told_eval,
      [[maybe_unused]] const std::function<
        typename Evaluation::value_type(const typename Evaluation::value_type &)> &compute_qVapor,
      [[maybe_unused]] const typename Evaluation::ScalarNumber
        ost_factor_implicit, // delta_t * theta
      [[maybe_unused]] const typename Evaluation::ScalarNumber
                                             ost_factor_explicit,    // delta_t * (1. - theta)
      const typename Evaluation::value_type &laser_heat_flux_factor, // laser_heat_flux * delta_t
      const unsigned int                     q)
    {
      // laser heat flux
      auto val = -laser_heat_flux_factor;

      if (Told_eval)
        {
          Assert(compute_qVapor != nullptr, dealii::ExcInternalError());

          // temperature-dependent evaporation-induced heat flux
          val += -ost_factor_implicit * compute_qVapor(evaluator.get_value(q));

          // temperature-dependent evaporation-induced heat flux
          val += -ost_factor_explicit * compute_qVapor(Told_eval->get_value(q));
        }
      evaluator.submit_value(val * -1., q);
    }


    // tangent operator interface integral (two-domain case)
    template <typename Evaluation>
    inline void
    tangent_interface_integral(
      Evaluation                                  &evaluator_l,
      Evaluation                                  &evaluator_g,
      [[maybe_unused]] const Evaluation           *T_eval_l,
      [[maybe_unused]] const Evaluation           *T_eval_g,
      [[maybe_unused]] const std::function<typename Evaluation::value_type(
        const typename Evaluation::value_type &)> &compute_qVapor_derivative,
      const typename Evaluation::ScalarNumber      conductivity_l,
      const typename Evaluation::ScalarNumber      conductivity_g,
      const typename Evaluation::ScalarNumber      ost_factor,     // delta_t * theta
      const typename Evaluation::ScalarNumber      nitsche_factor, // delta_t * gamma_Gamma / h
      const typename Evaluation::ScalarNumber      kappa_l,
      const typename Evaluation::ScalarNumber      kappa_g,
      const unsigned int                           q)
    {
      const auto &eval_l   = evaluator_l.get_value(q);
      const auto &eval_g   = evaluator_g.get_value(q);
      const auto  val_jump = eval_l - eval_g;
      const auto &normal_l = evaluator_l.normal_vector(q);
      const auto  normal_g = -normal_l;
      const auto &grad_l   = evaluator_l.get_gradient(q);
      const auto &grad_g   = evaluator_g.get_gradient(q);

      // heat flux over interface - interface integral from divergence theorem
      auto flux_1 = -ost_factor * (kappa_l * conductivity_l * grad_l * normal_l +
                                   kappa_g * conductivity_g * grad_g * normal_g);
      // Nitsche term
      flux_1 += nitsche_factor * val_jump;
      // submit to [v]
      auto val_l = flux_1;
      auto val_g = -flux_1;

      if (T_eval_l)
        {
          Assert(T_eval_g != nullptr, dealii::ExcInternalError());
          Assert(compute_qVapor_derivative != nullptr, dealii::ExcInternalError());
          // temperature-dependent evaporation-induced heat flux
          val_l +=
            -ost_factor * kappa_g * compute_qVapor_derivative(T_eval_l->get_value(q)) * eval_l;
          val_g +=
            -ost_factor * kappa_l * compute_qVapor_derivative(T_eval_g->get_value(q)) * eval_g;
        }

      // symmetry term
      const auto flux_2 = -ost_factor * val_jump;
      // submit to {k ∂_n v}
      evaluator_l.submit_gradient(flux_2 * conductivity_l * kappa_l * normal_l, q);
      evaluator_g.submit_gradient(flux_2 * conductivity_g * kappa_g * normal_g, q);

      evaluator_l.submit_value(val_l, q);
      evaluator_g.submit_value(val_g, q);
    }


    // residual interface integral (two-domain case)
    template <typename Evaluation>
    inline void
    residual_interface_integral(
      Evaluation       &evaluator_l, // T^n+1_l
      Evaluation       &evaluator_g, // T^n+1_g
      const Evaluation &Told_eval_l,
      const Evaluation &Told_eval_g,
      [[maybe_unused]] const std::function<
        typename Evaluation::value_type(const typename Evaluation::value_type &)> &compute_qVapor,
      const typename Evaluation::ScalarNumber                                      conductivity_l,
      const typename Evaluation::ScalarNumber                                      conductivity_g,
      const typename Evaluation::ScalarNumber ost_factor_implicit,    // delta_t * theta
      const typename Evaluation::ScalarNumber ost_factor_explicit,    // delta_t * (1. - theta)
      const typename Evaluation::ScalarNumber nitsche_factor,         // delta_t * gamma_Gamma / h
      const typename Evaluation::value_type  &laser_heat_flux_factor, // laser_heat_flux * delta_t
      const typename Evaluation::ScalarNumber kappa_l,
      const typename Evaluation::ScalarNumber kappa_g,
      const bool                              enable_evapor_cooling,
      const bool                              do_rhs_symm_term,
      const unsigned int                      q)
    {
      const auto &normal_l     = evaluator_l.normal_vector(q);
      const auto  normal_g     = -normal_l;
      const auto &T_new_val_l  = evaluator_l.get_value(q);
      const auto &T_new_val_g  = evaluator_g.get_value(q);
      const auto  T_new_jump   = T_new_val_l - T_new_val_g;
      const auto &T_new_grad_l = evaluator_l.get_gradient(q);
      const auto &T_new_grad_g = evaluator_g.get_gradient(q);

      // heat flux over interface - interface integral from divergence theorem
      auto flux_1 = -ost_factor_implicit * (kappa_l * conductivity_l * T_new_grad_l * normal_l +
                                            kappa_g * conductivity_g * T_new_grad_g * normal_g);
      // Nitsche term
      flux_1 += nitsche_factor * T_new_jump;

      // symmetry term
      auto flux_2 = -ost_factor_implicit * T_new_jump;

      // laser heat flux
      auto val_l = -kappa_g * laser_heat_flux_factor;
      auto val_g = -kappa_l * laser_heat_flux_factor;

      if (enable_evapor_cooling)
        {
          // temperature-dependent evaporation-induced heat flux
          val_l += -ost_factor_implicit * kappa_g * compute_qVapor(T_new_val_l);
          val_g += -ost_factor_implicit * kappa_l * compute_qVapor(T_new_val_g);
        }

      const auto T_old_val_l = Told_eval_l.get_value(q);
      const auto T_old_val_g = Told_eval_g.get_value(q);
      const auto T_old_jump  = T_old_val_l - T_old_val_g;

      // heat flux over interface - interface integral from divergence theorem
      flux_1 +=
        -ost_factor_explicit * (kappa_l * conductivity_l * Told_eval_l.get_gradient(q) * normal_l +
                                kappa_g * conductivity_g * Told_eval_g.get_gradient(q) * normal_g);

      if (do_rhs_symm_term)
        {
          // symmetry term
          flux_2 += -ost_factor_explicit * T_old_jump;
        }

      if (enable_evapor_cooling)
        {
          // temperature-dependent evaporation-induced heat flux
          val_l += -ost_factor_explicit * kappa_g * compute_qVapor(T_old_val_l);
          val_g += -ost_factor_explicit * kappa_l * compute_qVapor(T_old_val_g);
        }

      // submit to [v]
      val_l += flux_1;
      val_g += -flux_1;

      // submit to {k ∂_n v}
      evaluator_l.submit_gradient(flux_2 * conductivity_l * kappa_l * normal_l * -1., q);
      evaluator_g.submit_gradient(flux_2 * conductivity_g * kappa_g * normal_g * -1., q);

      evaluator_l.submit_value(val_l * -1., q);
      evaluator_g.submit_value(val_g * -1., q);
    }


    // ghost-penalty terms for FE_Q elements (p=1)
    template <typename Evaluation>
    inline void
    ghost_penalty_face_integral(
      Evaluation                             &evaluator_minus,
      Evaluation                             &evaluator_plus,
      const typename Evaluation::ScalarNumber conductivity,
      const typename Evaluation::ScalarNumber ost_factor, // delta_t * theta
      const typename Evaluation::ScalarNumber h,
      const typename Evaluation::ScalarNumber gamma_M_degree_1,
      const typename Evaluation::ScalarNumber gamma_A_degree_1,
      const unsigned int                      q,
      const typename Evaluation::ScalarNumber factor = 1.0)
    {
      const auto normal_grad_jump =
        evaluator_minus.get_normal_derivative(q) - evaluator_plus.get_normal_derivative(q);

      // stiffness stabilization
      auto gp = ost_factor * gamma_A_degree_1 * conductivity * h / 3. * normal_grad_jump;

      // mass stabilization
      gp += gamma_M_degree_1 * dealii::Utilities::fixed_power<3>(h) / 3. * normal_grad_jump;

      // submit to [∂_n v]
      evaluator_minus.submit_normal_derivative(gp * factor, q);
      evaluator_plus.submit_normal_derivative(-gp * factor, q);
    }
  } // namespace internal


  template <int dim, typename number>
  HeatCutOperator<dim, number>::HeatCutOperator(
    const ScratchData<dim, dim, number>        &scratch_data_in,
    const HeatData<number>                     &heat_data_in,
    const MaterialData<number>                 &material_data_in,
    const Evaporation::EvaporationData<number> &evapor_data_in,
    const unsigned int                          heat_cut_dof_idx_in,
    const unsigned int                          heat_cut_no_bc_dof_idx_in,
    const unsigned int                          heat_continuous_no_bc_dof_idx_in,
    const unsigned int                          heat_quad_idx_in,
    const VectorType                           &temperature_in,
    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
      &mapping_info_interface_in,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
                      &mapping_info_cells_in,
    const bool         do_solidification_in,
    const unsigned int vel_dof_idx_in,
    const VectorType  *velocity_in,
    const VectorType  *velocity_old_in)
    : scratch_data(scratch_data_in)
    , heat_data(heat_data_in)
    , material(material_data_in,
               do_solidification_in ? MaterialTypes::liquid_solid : MaterialTypes::liquid)
    , heat_cut_dof_idx(heat_cut_dof_idx_in)
    , heat_cut_no_bc_dof_idx(heat_cut_no_bc_dof_idx_in)
    , heat_continuous_no_bc_dof_idx(heat_continuous_no_bc_dof_idx_in)
    , heat_quad_idx(heat_quad_idx_in)
    , temperature(temperature_in)
    , mapping_info_surface(mapping_info_interface_in)
    , mapping_info_cells(mapping_info_cells_in)
    , reference_finite_element_heat(heat_data.fe.degree)
    , n_dofs_per_cell_heat(reference_finite_element_heat.dofs_per_cell)
    , reference_finite_element_vel(dealii::FE_Q<dim>(heat_data.fe.degree), dim)
    , n_dofs_per_cell_vel(reference_finite_element_vel.dofs_per_cell)
    , kappa_l(material.get_data().gas.thermal_conductivity /
              (material.get_data().gas.thermal_conductivity +
               material.get_data().liquid.thermal_conductivity))
    , kappa_g(material.get_data().liquid.thermal_conductivity /
              (material.get_data().gas.thermal_conductivity +
               material.get_data().liquid.thermal_conductivity))
    , vel_dof_idx(vel_dof_idx_in)
    , velocity(velocity_in)
    , velocity_old(velocity_old_in)
    , do_solidification(do_solidification_in)
  {
    // TODO external heat source

    AssertThrow(heat_data.linear_solver.do_matrix_free, dealii::ExcNotImplemented());
    AssertThrow(heat_data.fe.type == FiniteElementType::FE_Q,
                dealii::ExcMessage("only standard FE_Q elements are supported for now"));
    AssertThrow(heat_data.fe.degree == 1, dealii::ExcMessage("only degree 1 is supported for now"));

    material.get_data().check_parameters_heat_transfer(heat_data.cut.two_phase, do_solidification);

    if (evapor_data_in.evaporative_cooling.enable)
      {
        evapor_cooling = std::make_unique<Evaporation::EvaporativeCooling<number>>(
          evapor_data_in, material.get_data(), true /*setup_internal_mass_flux_operator*/);

        if (evapor_data_in.evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux ==
            "true")
          AssertThrow(not do_solidification or material.get_data().solid.specific_heat_capacity ==
                                                 material.get_data().liquid.specific_heat_capacity,
                      dealii::ExcMessage(
                        "The equation for specific enthalpy for evaporative cooling "
                        "assumes equality between the solid and liquid "
                        "phase heat capacity! Abort..."));
      }

    if (velocity != nullptr)
      AssertThrow(velocity_old != nullptr, dealii::ExcInternalError());
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::register_interface_heat_flux(
    std::shared_ptr<const InterfaceHeatFlux<dim, number>> interface_heat_flux_in)
  {
    interface_heat_flux = std::move(interface_heat_flux_in);
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::reinit()
  {
    // precompute cell size (constant cell size for homogeneous Cartesian mesh)
    cell_side_length = scratch_data.get_min_cell_size();

    // precompute Nitsche term factor for the two-phase case
    if (heat_data.cut.two_phase)
      weighted_nitsche_factor = heat_data.cut.stabilization.nitsche_parameter *
                                (material.get_data().liquid.thermal_conductivity *
                                 material.get_data().gas.thermal_conductivity) /
                                (material.get_data().liquid.thermal_conductivity +
                                 material.get_data().gas.thermal_conductivity) /
                                cell_side_length;
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::init_time_advance(const double dt)
  {
    AssertThrowZeroTimeIncrement(dt);
    this->reset_time_increment(dt);
    ost_factor_implicit = this->time_increment * heat_data.cut.theta;
    ost_factor_explicit = this->time_increment * (1. - heat_data.cut.theta);
    if (heat_data.cut.two_phase)
      nitsche_factor = this->time_increment * weighted_nitsche_factor;
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::update_ghost_values() const
  {
    if (velocity and not velocity->has_ghost_elements())
      velocity->update_ghost_values();
    if (velocity_old and not velocity_old->has_ghost_elements())
      velocity_old->update_ghost_values();
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    // TODO dst.zero_out_ghost_values();

    scratch_data.get_matrix_free().loop(
      &HeatCutOperator::tangent_cell_loop,
      &HeatCutOperator::tangent_inner_face_loop,
      &HeatCutOperator::tangent_boundary_face_loop,
      this,
      dst,
      src,
      true,
      dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::DataAccessOnFaces::
        gradients,
      dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::DataAccessOnFaces::
        gradients);
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::create_rhs(VectorType       &residual,
                                           const VectorType &temperature_old) const
  {
    // TODO residual.zero_out_ghost_values();

    scratch_data.get_matrix_free().loop(
      &HeatCutOperator::residual_cell_loop,
      &HeatCutOperator::residual_inner_face_loop,
      &HeatCutOperator::residual_boundary_face_loop,
      this,
      residual,
      temperature_old,
      true,
      dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::DataAccessOnFaces::
        gradients,
      dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::DataAccessOnFaces::
        gradients);
  }

  template <int dim, typename number>
  dealii::VectorizedArray<number>
  HeatCutOperator<dim, number>::compute_qVapor(const dealii::VectorizedArray<number> &T) const
  {
    return evapor_cooling->compute_evaporative_cooling(T);
  }

  template <int dim, typename number>
  dealii::VectorizedArray<number>
  HeatCutOperator<dim, number>::compute_qVapor_derivative(
    const dealii::VectorizedArray<number> &T) const
  {
    return evapor_cooling
      ->compute_evaporative_cooling_derivative_with_temperature_dependent_mass_flux(T);
  }


  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::tangent_cell_operation_liquid(const unsigned int cell_batch,
                                                              DomainEval<>      &eval_l,
                                                              DomainEval<>      *T_eval_l,
                                                              DomainEval<dim>   *vel_eval,
                                                              const bool do_reinit_cell) const
  {
    Assert(eval_l.get_active_fe_index() == CutUtil::CellCategory::liquid,
           dealii::ExcInternalError());

    if (T_eval_l and do_reinit_cell)
      {
        T_eval_l->reinit(cell_batch);
        T_eval_l->gather_evaluate(temperature, evaluate_values | evaluate_gradients);
      }
    if (vel_eval and do_reinit_cell)
      {
        vel_eval->reinit(cell_batch);
        vel_eval->gather_evaluate(*velocity, evaluate_values);
      }

    for (const unsigned int q : eval_l.quadrature_point_indices())
      {
        const auto [conductivity_l, cv_l, d_conductivity_d_T, d_cv_d_T] =
          internal::get_liquid_material_parameters_and_derivatives(T_eval_l, material, q);
        const auto vel = vel_eval ? vel_eval->get_value(q) : typename DomainEval<dim>::value_type();
        internal::tangent_domain_integral(eval_l,
                                          T_eval_l,
                                          conductivity_l,
                                          cv_l,
                                          d_conductivity_d_T,
                                          d_cv_d_T,
                                          vel,
                                          ost_factor_implicit,
                                          q);
      }
  }

  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::tangent_cell_operation_gas(const unsigned int cell_batch,
                                                           DomainEval<>      &eval_g,
                                                           DomainEval<dim>   *vel_eval,
                                                           const bool         do_reinit_cell) const
  {
    Assert(heat_data.cut.two_phase, dealii::ExcInternalError());
    Assert(eval_g.get_active_fe_index() == CutUtil::CellCategory::gas, dealii::ExcInternalError());

    if (vel_eval and do_reinit_cell)
      {
        vel_eval->reinit(cell_batch);
        vel_eval->gather_evaluate(*velocity, evaluate_values);
      }

    for (const unsigned int q : eval_g.quadrature_point_indices())
      internal::tangent_domain_integral<DomainEval<>>(
        eval_g,
        nullptr,
        material.get_data().gas.thermal_conductivity,
        material.get_data().gas.density * material.get_data().gas.specific_heat_capacity,
        0.0,
        0.0,
        vel_eval ? vel_eval->get_value(q) : typename DomainEval<dim>::value_type(),
        ost_factor_implicit,
        q);
  }


  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::tangent_cell_operation_intersected_one_phase(
    const unsigned int cell_batch,
    DomainEval<>      &eval_cell_l,
    PointEval<>       &eval_subdomain_l,
    PointEval<>       *eval_interface_l,
    DomainEval<>      *T_eval_cell_l,
    PointEval<>       *T_eval_subdomain_l,
    PointEval<>       *T_eval_interface_l,
    DomainEval<dim>   *vel_eval,
    PointEval<dim>    *vel_eval_subdomain_l,
    const bool         do_reinit_cell) const
  {
    Assert(not heat_data.cut.two_phase, dealii::ExcInternalError());
    Assert(eval_cell_l.get_active_fe_index() == CutUtil::CellCategory::intersected,
           dealii::ExcInternalError());

    if (T_eval_cell_l and do_reinit_cell)
      {
        T_eval_cell_l->reinit(cell_batch);
        T_eval_cell_l->read_dof_values(temperature);
      }
    if (vel_eval and do_reinit_cell)
      {
        vel_eval->reinit(cell_batch);
        vel_eval->read_dof_values(*velocity);
      }

    for (unsigned int cell_lane = 0;
         cell_lane < eval_cell_l.get_matrix_free().n_active_entries_per_cell_batch(cell_batch);
         ++cell_lane)
      {
        // evaluate for liquid domain integral
        CutUtil::evaluate_intersected_domain<dim, number>(eval_subdomain_l,
                                                          eval_cell_l,
                                                          evaluate_values | evaluate_gradients,
                                                          cell_batch,
                                                          cell_lane,
                                                          n_dofs_per_cell_heat);
        if (T_eval_subdomain_l)
          // evaluate T^n+1 for liquid domain integral
          CutUtil::evaluate_intersected_domain<dim, number>(*T_eval_subdomain_l,
                                                            *T_eval_cell_l,
                                                            evaluate_values | evaluate_gradients,
                                                            cell_batch,
                                                            cell_lane,
                                                            n_dofs_per_cell_heat);
        if (eval_interface_l)
          // evaluate for immersed boundary integral
          CutUtil::evaluate_intersected_domain<dim, number>(*eval_interface_l,
                                                            eval_cell_l,
                                                            evaluate_values,
                                                            cell_batch,
                                                            cell_lane,
                                                            n_dofs_per_cell_heat);
        if (T_eval_interface_l)
          // evaluate T^n+1 for immersed boundary integral
          CutUtil::evaluate_intersected_domain<dim, number>(*T_eval_interface_l,
                                                            *T_eval_cell_l,
                                                            evaluate_values,
                                                            cell_batch,
                                                            cell_lane,
                                                            n_dofs_per_cell_heat);
        // evaluate velocity for liquid domain integral
        if (vel_eval)
          CutUtil::evaluate_intersected_domain<dim, number, dim>(*vel_eval_subdomain_l,
                                                                 *vel_eval,
                                                                 evaluate_values,
                                                                 cell_batch,
                                                                 cell_lane,
                                                                 n_dofs_per_cell_vel);

        // liquid domain integral
        for (const unsigned int q_batch : eval_subdomain_l.quadrature_point_indices())
          {
            const auto [conductivity_l, cv_l, d_conductivity_d_T, d_cv_d_T] =
              internal::get_liquid_material_parameters_and_derivatives(T_eval_subdomain_l,
                                                                       material,
                                                                       q_batch);
            const auto vel = vel_eval_subdomain_l ? vel_eval_subdomain_l->get_value(q_batch) :
                                                    typename DomainEval<dim>::value_type();
            internal::tangent_domain_integral(eval_subdomain_l,
                                              T_eval_subdomain_l,
                                              conductivity_l,
                                              cv_l,
                                              d_conductivity_d_T,
                                              d_cv_d_T,
                                              vel,
                                              ost_factor_implicit,
                                              q_batch);
          }
        eval_subdomain_l.integrate(
          dealii::StridedArrayView<number, n_lanes>(&eval_cell_l.begin_dof_values()[0][cell_lane],
                                                    n_dofs_per_cell_heat),
          evaluate_values | evaluate_gradients);

        if (eval_interface_l)
          {
            // immersed boundary integral
            for (const unsigned int q_batch : eval_interface_l->quadrature_point_indices())
              internal::tangent_immersed_boundary_integral(
                *eval_interface_l,
                *T_eval_interface_l,
                [&](const dealii::VectorizedArray<number> &T) {
                  return compute_qVapor_derivative(T);
                },
                ost_factor_implicit,
                q_batch);

            eval_interface_l->integrate(
              dealii::StridedArrayView<number, n_lanes>(
                &eval_cell_l.begin_dof_values()[0][cell_lane], n_dofs_per_cell_heat),
              evaluate_values,
              true
              /*specify flag 'true' for summing the integrated values into the solution values*/);
          }
      }
  }


  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::tangent_cell_operation_intersected_two_phase(
    const unsigned int cell_batch,
    DomainEval<>      &eval_cell_l,
    DomainEval<>      &eval_cell_g,
    PointEval<>       &eval_subdomain_l,
    PointEval<>       &eval_subdomain_g,
    PointEval<>       &eval_interface_l,
    PointEval<>       &eval_interface_g,
    DomainEval<>      *T_eval_cell_l,
    DomainEval<>      *T_eval_cell_g,
    PointEval<>       *T_eval_subdomain_l,
    PointEval<>       *T_eval_interface_l,
    PointEval<>       *T_eval_interface_g,
    DomainEval<dim>   *vel_eval,
    PointEval<dim>    *vel_eval_subdomain_l,
    PointEval<dim>    *vel_eval_subdomain_g,
    const bool         do_reinit_cell) const
  {
    Assert(heat_data.cut.two_phase, dealii::ExcInternalError());
    Assert(eval_cell_l.get_active_fe_index() == CutUtil::CellCategory::intersected,
           dealii::ExcInternalError());

    if (T_eval_cell_l and do_reinit_cell)
      {
        T_eval_cell_l->reinit(cell_batch);
        T_eval_cell_l->read_dof_values(temperature);
      }
    if (T_eval_cell_g and do_reinit_cell)
      {
        T_eval_cell_g->reinit(cell_batch);
        T_eval_cell_g->read_dof_values(temperature);
      }
    if (vel_eval and do_reinit_cell)
      {
        vel_eval->reinit(cell_batch);
        vel_eval->read_dof_values(*velocity);
      }

    for (unsigned int cell_lane = 0;
         cell_lane < eval_cell_l.get_matrix_free().n_active_entries_per_cell_batch(cell_batch);
         ++cell_lane)
      {
        // evaluate for liquid domain integral
        CutUtil::evaluate_intersected_domain<dim, number>(eval_subdomain_l,
                                                          eval_cell_l,
                                                          evaluate_values | evaluate_gradients,
                                                          cell_batch,
                                                          cell_lane,
                                                          n_dofs_per_cell_heat);
        // evaluate liquid side for interface integral
        CutUtil::evaluate_intersected_domain<dim, number>(eval_interface_l,
                                                          eval_cell_l,
                                                          evaluate_values | evaluate_gradients,
                                                          cell_batch,
                                                          cell_lane,
                                                          n_dofs_per_cell_heat);
        // evaluate for gas domain integral
        CutUtil::evaluate_intersected_domain<dim, number>(eval_subdomain_g,
                                                          eval_cell_g,
                                                          evaluate_values | evaluate_gradients,
                                                          cell_batch,
                                                          cell_lane,
                                                          n_dofs_per_cell_heat);
        // evaluate gas side for interface integral
        CutUtil::evaluate_intersected_domain<dim, number>(eval_interface_g,
                                                          eval_cell_g,
                                                          evaluate_values | evaluate_gradients,
                                                          cell_batch,
                                                          cell_lane,
                                                          n_dofs_per_cell_heat);
        if (T_eval_subdomain_l)
          CutUtil::evaluate_intersected_domain<dim, number>(*T_eval_subdomain_l,
                                                            *T_eval_cell_l,
                                                            evaluate_values | evaluate_gradients,
                                                            cell_batch,
                                                            cell_lane,
                                                            n_dofs_per_cell_heat);
        if (T_eval_interface_l)
          // evaluate T^n+1_l for interface integral
          CutUtil::evaluate_intersected_domain<dim, number>(*T_eval_interface_l,
                                                            *T_eval_cell_l,
                                                            evaluate_values,
                                                            cell_batch,
                                                            cell_lane,
                                                            n_dofs_per_cell_heat);
        if (T_eval_interface_g)
          // evaluate T^n+1_g for interface integral
          CutUtil::evaluate_intersected_domain<dim, number>(*T_eval_interface_g,
                                                            *T_eval_cell_g,
                                                            evaluate_values,
                                                            cell_batch,
                                                            cell_lane,
                                                            n_dofs_per_cell_heat);
        if (vel_eval)
          {
            // evaluate velocity for liquid domain integral
            CutUtil::evaluate_intersected_domain<dim, number, dim>(*vel_eval_subdomain_l,
                                                                   *vel_eval,
                                                                   evaluate_values,
                                                                   cell_batch,
                                                                   cell_lane,
                                                                   n_dofs_per_cell_vel);
            // evaluate velocity for gas domain integral
            CutUtil::evaluate_intersected_domain<dim, number, dim>(*vel_eval_subdomain_g,
                                                                   *vel_eval,
                                                                   evaluate_values,
                                                                   cell_batch,
                                                                   cell_lane,
                                                                   n_dofs_per_cell_vel);
          }

        // liquid domain integral
        for (const unsigned int q_batch : eval_subdomain_l.quadrature_point_indices())
          {
            const auto [conductivity_l, cv_l, d_conductivity_d_T_l, d_cv_d_T_l] =
              internal::get_liquid_material_parameters_and_derivatives(T_eval_subdomain_l,
                                                                       material,
                                                                       q_batch);
            const auto vel = vel_eval_subdomain_l ? vel_eval_subdomain_l->get_value(q_batch) :
                                                    typename DomainEval<dim>::value_type();
            internal::tangent_domain_integral(eval_subdomain_l,
                                              T_eval_subdomain_l,
                                              conductivity_l,
                                              cv_l,
                                              d_conductivity_d_T_l,
                                              d_cv_d_T_l,
                                              vel,
                                              ost_factor_implicit,
                                              q_batch);
          }
        eval_subdomain_l.integrate(
          dealii::StridedArrayView<number, n_lanes>(&eval_cell_l.begin_dof_values()[0][cell_lane],
                                                    n_dofs_per_cell_heat),
          evaluate_values | evaluate_gradients);

        // gas domain integral
        for (const unsigned int q_batch : eval_subdomain_g.quadrature_point_indices())
          {
            internal::tangent_domain_integral<PointEval<>>(
              eval_subdomain_g,
              nullptr,
              material.get_data().gas.thermal_conductivity,
              material.get_data().gas.density * material.get_data().gas.specific_heat_capacity,
              0.0,
              0.0,
              vel_eval_subdomain_g ? vel_eval_subdomain_g->get_value(q_batch) :
                                     typename DomainEval<dim>::value_type(),
              ost_factor_implicit,
              q_batch);
          }
        eval_subdomain_g.integrate(
          dealii::StridedArrayView<number, n_lanes>(&eval_cell_g.begin_dof_values()[0][cell_lane],
                                                    n_dofs_per_cell_heat),
          evaluate_values | evaluate_gradients);

        // immersed interface integral
        for (const unsigned int q_batch : eval_interface_l.quadrature_point_indices())
          internal::tangent_interface_integral(
            eval_interface_l,
            eval_interface_g,
            T_eval_interface_l,
            T_eval_interface_g,
            [&](const dealii::VectorizedArray<number> &T) { return compute_qVapor_derivative(T); },
            material.get_data()
              .liquid.thermal_conductivity, // TODO temperature-dependent at interface?
            material.get_data().gas.thermal_conductivity,
            ost_factor_implicit,
            nitsche_factor,
            kappa_l,
            kappa_g,
            q_batch);

        eval_interface_l.integrate(
          dealii::StridedArrayView<number, n_lanes>(&eval_cell_l.begin_dof_values()[0][cell_lane],
                                                    n_dofs_per_cell_heat),
          evaluate_values | evaluate_gradients,
          true
          /*specify flag 'true' for summing the integrated values into the solution values*/);

        eval_interface_g.integrate(
          dealii::StridedArrayView<number, n_lanes>(&eval_cell_g.begin_dof_values()[0][cell_lane],
                                                    n_dofs_per_cell_heat),
          evaluate_values | evaluate_gradients,
          true
          /*specify flag 'true' for summing the integrated values into the solution values*/);
      }
  }


  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::tangent_cell_loop(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &dst,
    const VectorType                                                       &src,
    const std::pair<unsigned int, unsigned int>                            &cell_batch_range) const
  {
    std::unique_ptr<DomainEval<dim>> vel_eval;
    if (velocity)
      vel_eval = std::make_unique<DomainEval<dim>>(matrix_free, vel_dof_idx, heat_quad_idx);

    const auto cell_category = matrix_free.get_cell_range_category(cell_batch_range);
    if (cell_category == CutUtil::CellCategory::liquid)
      {
        DomainEval                    eval_l(matrix_free,
                          heat_cut_dof_idx /*dof_no*/,
                          heat_quad_idx /*quad_no*/,
                          0 /*selected component*/,
                          cell_category /*active_fe_index*/);
        std::unique_ptr<DomainEval<>> T_eval_l;
        if (do_solidification)
          T_eval_l = std::make_unique<DomainEval<>>(eval_l);

        for (unsigned int cell_batch = cell_batch_range.first; cell_batch < cell_batch_range.second;
             ++cell_batch)
          {
            eval_l.reinit(cell_batch);
            eval_l.gather_evaluate(src, evaluate_values | evaluate_gradients);

            tangent_cell_operation_liquid(cell_batch, eval_l, T_eval_l.get(), vel_eval.get());

            eval_l.integrate_scatter(evaluate_values | evaluate_gradients, dst);
          }
      }
    else if (cell_category == CutUtil::CellCategory::gas and heat_data.cut.two_phase)
      {
        DomainEval eval_g(matrix_free,
                          heat_cut_dof_idx /*dof_no*/,
                          heat_quad_idx /*quad_no*/,
                          1 /*selected component*/,
                          cell_category /*active_fe_index*/);

        for (unsigned int cell_batch = cell_batch_range.first; cell_batch < cell_batch_range.second;
             ++cell_batch)
          {
            eval_g.reinit(cell_batch);
            eval_g.gather_evaluate(src, evaluate_values | evaluate_gradients);

            tangent_cell_operation_gas(cell_batch, eval_g, vel_eval.get());

            eval_g.integrate_scatter(evaluate_values | evaluate_gradients, dst);
          }
      }
    else if (cell_category == CutUtil::CellCategory::intersected and not heat_data.cut.two_phase)
      {
        // use FEEvaluation and FEPointEvaluation in combination for intersected cells
        DomainEval                    eval_cell_l(matrix_free,
                               heat_cut_dof_idx /*dof_no*/,
                               heat_quad_idx /*quad_no*/,
                               0 /*selected component*/,
                               cell_category /*active_fe_index*/);
        std::unique_ptr<DomainEval<>> T_eval_cell_l;
        if (evapor_cooling or do_solidification)
          T_eval_cell_l = std::make_unique<DomainEval<>>(eval_cell_l);

        PointEval                    eval_subdomain_l(*mapping_info_cells[0],
                                   reference_finite_element_heat,
                                   0 /*selected component*/,
                                   true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<>> T_eval_subdomain_l;
        if (do_solidification)
          T_eval_subdomain_l =
            std::make_unique<PointEval<>>(*mapping_info_cells[0],
                                          reference_finite_element_heat,
                                          0 /*selected component*/,
                                          true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<dim>> vel_eval_subdomain_l;
        if (vel_eval)
          vel_eval_subdomain_l =
            std::make_unique<PointEval<dim>>(*mapping_info_cells[0],
                                             reference_finite_element_vel,
                                             0 /*selected component*/,
                                             true /*force_lexicographic_numbering*/);

        std::unique_ptr<PointEval<>> eval_interface_l;
        std::unique_ptr<PointEval<>> T_eval_interface_l;
        if (evapor_cooling)
          {
            eval_interface_l =
              std::make_unique<PointEval<>>(mapping_info_surface,
                                            reference_finite_element_heat,
                                            0 /*selected component*/,
                                            true /*force_lexicographic_numbering*/);
            T_eval_interface_l =
              std::make_unique<PointEval<>>(mapping_info_surface,
                                            reference_finite_element_heat,
                                            0 /*selected component*/,
                                            true /*force_lexicographic_numbering*/);
          }

        for (unsigned int cell_batch = cell_batch_range.first; cell_batch < cell_batch_range.second;
             ++cell_batch)
          {
            eval_cell_l.reinit(cell_batch);
            eval_cell_l.read_dof_values(src);

            tangent_cell_operation_intersected_one_phase(cell_batch,
                                                         eval_cell_l,
                                                         eval_subdomain_l,
                                                         eval_interface_l.get(),
                                                         T_eval_cell_l.get(),
                                                         T_eval_subdomain_l.get(),
                                                         T_eval_interface_l.get(),
                                                         vel_eval.get(),
                                                         vel_eval_subdomain_l.get());

            eval_cell_l.distribute_local_to_global(dst);
          }
      }
    else if (cell_category == CutUtil::CellCategory::intersected and heat_data.cut.two_phase)
      {
        // use FEEvaluation and FEPointEvaluation in combination for intersected cells
        DomainEval                    eval_cell_l(matrix_free,
                               heat_cut_dof_idx /*dof_no*/,
                               heat_quad_idx /*quad_no*/,
                               0 /*selected component*/,
                               cell_category /*active_fe_index*/);
        std::unique_ptr<DomainEval<>> T_eval_cell_l;
        if (evapor_cooling or do_solidification)
          T_eval_cell_l = std::make_unique<DomainEval<>>(eval_cell_l);

        PointEval                    eval_subdomain_l(*mapping_info_cells[0],
                                   reference_finite_element_heat,
                                   0 /*selected component*/,
                                   true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<>> T_eval_subdomain_l;
        if (do_solidification)
          T_eval_subdomain_l =
            std::make_unique<PointEval<>>(*mapping_info_cells[0],
                                          reference_finite_element_heat,
                                          0 /*selected component*/,
                                          true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<dim>> vel_eval_subdomain_l;
        if (vel_eval)
          vel_eval_subdomain_l =
            std::make_unique<PointEval<dim>>(*mapping_info_cells[0],
                                             reference_finite_element_vel,
                                             0 /*selected component*/,
                                             true /*force_lexicographic_numbering*/);

        PointEval                    eval_interface_l(mapping_info_surface,
                                   reference_finite_element_heat,
                                   0 /*selected component*/,
                                   true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<>> T_eval_interface_l;
        if (evapor_cooling)
          T_eval_interface_l =
            std::make_unique<PointEval<>>(mapping_info_surface,
                                          reference_finite_element_heat,
                                          0 /*selected component*/,
                                          true /*force_lexicographic_numbering*/);

        DomainEval                    eval_cell_g(matrix_free,
                               heat_cut_dof_idx /*dof_no*/,
                               heat_quad_idx /*quad_no*/,
                               1 /*selected component*/,
                               cell_category /*active_fe_index*/);
        std::unique_ptr<DomainEval<>> T_eval_cell_g;
        if (evapor_cooling)
          T_eval_cell_g = std::make_unique<DomainEval<>>(eval_cell_g);

        PointEval                       eval_subdomain_g(*mapping_info_cells[1],
                                   reference_finite_element_heat,
                                   0 /*selected component*/,
                                   true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<dim>> vel_eval_subdomain_g;
        if (vel_eval)
          vel_eval_subdomain_g =
            std::make_unique<PointEval<dim>>(*mapping_info_cells[1],
                                             reference_finite_element_vel,
                                             0 /*selected component*/,
                                             true /*force_lexicographic_numbering*/);

        PointEval                    eval_interface_g(mapping_info_surface,
                                   reference_finite_element_heat,
                                   0 /*selected component*/,
                                   true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<>> T_eval_interface_g;
        if (evapor_cooling)
          T_eval_interface_g =
            std::make_unique<PointEval<>>(mapping_info_surface,
                                          reference_finite_element_heat,
                                          0 /*selected component*/,
                                          true /*force_lexicographic_numbering*/);

        for (unsigned int cell_batch = cell_batch_range.first; cell_batch < cell_batch_range.second;
             ++cell_batch)
          {
            eval_cell_l.reinit(cell_batch);
            eval_cell_g.reinit(cell_batch);
            eval_cell_l.read_dof_values(src);
            eval_cell_g.read_dof_values(src);

            tangent_cell_operation_intersected_two_phase(cell_batch,
                                                         eval_cell_l,
                                                         eval_cell_g,
                                                         eval_subdomain_l,
                                                         eval_subdomain_g,
                                                         eval_interface_l,
                                                         eval_interface_g,
                                                         T_eval_cell_l.get(),
                                                         T_eval_cell_g.get(),
                                                         T_eval_subdomain_l.get(),
                                                         T_eval_interface_l.get(),
                                                         T_eval_interface_g.get(),
                                                         vel_eval.get(),
                                                         vel_eval_subdomain_l.get(),
                                                         vel_eval_subdomain_g.get());

            eval_cell_l.distribute_local_to_global(dst);
            eval_cell_g.distribute_local_to_global(dst);
          }
      }
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::tangent_inner_face_loop(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &dst,
    const VectorType                                                       &src,
    const std::pair<unsigned int, unsigned int>                            &face_batch_range) const
  {
    const auto              face_category = matrix_free.get_face_range_category(face_batch_range);
    const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);

    if (face_type == CutUtil::FaceType::intersected_face or
        face_type == CutUtil::FaceType::mixed_face_liquid_intersected or
        face_type == CutUtil::FaceType::mixed_face_intersected_liquid)
      {
        auto liquid_minus_category = CutUtil::CellCategory::intersected;
        auto liquid_plus_category  = CutUtil::CellCategory::intersected;

        if (face_type == CutUtil::FaceType::mixed_face_liquid_intersected)
          {
            liquid_minus_category = CutUtil::CellCategory::liquid;
          }
        else if (face_type == CutUtil::FaceType::mixed_face_intersected_liquid)
          {
            liquid_plus_category = CutUtil::CellCategory::liquid;
          }
        else
          {
            AssertThrow(face_type == CutUtil::FaceType::intersected_face,
                        dealii::ExcInternalError());
          }

        FaceEval eval_minus_l(matrix_free,
                              true /*is_interior_face*/,
                              heat_cut_dof_idx /*dof_no*/,
                              heat_quad_idx /*quad_no*/,
                              0 /*selected component*/,
                              liquid_minus_category /*active_fe_index*/);
        FaceEval eval_plus_l(matrix_free,
                             false /*is_interior_face*/,
                             heat_cut_dof_idx /*dof_no*/,
                             heat_quad_idx /*quad_no*/,
                             0 /*selected component*/,
                             liquid_plus_category /*active_fe_index*/);

        for (unsigned int face_batch = face_batch_range.first; face_batch < face_batch_range.second;
             face_batch++)
          {
            eval_minus_l.reinit(face_batch);
            eval_plus_l.reinit(face_batch);
            eval_minus_l.gather_evaluate(src, evaluate_gradients);
            eval_plus_l.gather_evaluate(src, evaluate_gradients);

            for (const unsigned int q : eval_minus_l.quadrature_point_indices())
              internal::ghost_penalty_face_integral(
                eval_minus_l,
                eval_plus_l,
                material.get_data().liquid.thermal_conductivity,
                ost_factor_implicit,
                cell_side_length,
                heat_data.cut.stabilization.ghost_penalty.gamma_M_degree_1,
                heat_data.cut.stabilization.ghost_penalty.gamma_A_degree_1,
                q);

            eval_minus_l.integrate_scatter(evaluate_gradients, dst);
            eval_plus_l.integrate_scatter(evaluate_gradients, dst);
          }
      }
    if ((face_type == CutUtil::FaceType::intersected_face or
         face_type == CutUtil::FaceType::mixed_face_gas_intersected or
         face_type == CutUtil::FaceType::mixed_face_intersected_gas) and
        heat_data.cut.two_phase)
      {
        auto gas_minus_category = CutUtil::CellCategory::intersected;
        auto gas_plus_category  = CutUtil::CellCategory::intersected;

        if (face_type == CutUtil::FaceType::mixed_face_gas_intersected)
          {
            gas_minus_category = CutUtil::CellCategory::gas;
          }
        else if (face_type == CutUtil::FaceType::mixed_face_intersected_gas)
          {
            gas_plus_category = CutUtil::CellCategory::gas;
          }
        else
          {
            AssertThrow(face_type == CutUtil::FaceType::intersected_face,
                        dealii::ExcInternalError());
          }


        FaceEval eval_minus_g(matrix_free,
                              true /*is_interior_face*/,
                              heat_cut_dof_idx /*dof_no*/,
                              heat_quad_idx /*quad_no*/,
                              1 /*selected component*/,
                              gas_minus_category /*active_fe_index*/);

        FaceEval eval_plus_g(matrix_free,
                             false /*is_interior_face*/,
                             heat_cut_dof_idx /*dof_no*/,
                             heat_quad_idx /*quad_no*/,
                             1 /*selected component*/,
                             gas_plus_category /*active_fe_index*/);

        for (unsigned int face_batch = face_batch_range.first; face_batch < face_batch_range.second;
             face_batch++)
          {
            eval_minus_g.reinit(face_batch);
            eval_plus_g.reinit(face_batch);
            eval_minus_g.gather_evaluate(src, evaluate_gradients);
            eval_plus_g.gather_evaluate(src, evaluate_gradients);

            for (const unsigned int q : eval_minus_g.quadrature_point_indices())
              internal::ghost_penalty_face_integral(
                eval_minus_g,
                eval_plus_g,
                material.get_data().gas.thermal_conductivity,
                ost_factor_implicit,
                cell_side_length,
                heat_data.cut.stabilization.ghost_penalty.gamma_M_degree_1,
                heat_data.cut.stabilization.ghost_penalty.gamma_A_degree_1,
                q);

            eval_minus_g.integrate_scatter(evaluate_gradients, dst);
            eval_plus_g.integrate_scatter(evaluate_gradients, dst);
          }
      }
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::tangent_boundary_face_loop(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &dst,
    const VectorType                                                       &src,
    const std::pair<unsigned int, unsigned int>                            &face_batch_range) const
  {
    (void)matrix_free;
    (void)dst;
    (void)src;
    (void)face_batch_range;
    // TODO temperature-dependent Neumann BC
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::residual_cell_loop(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &residual,
    const VectorType                                                       &temperature_old,
    const std::pair<unsigned int, unsigned int>                            &cell_batch_range) const
  {
    std::unique_ptr<DomainEval<dim>> vel_eval;
    std::unique_ptr<DomainEval<dim>> vel_old_eval;
    if (velocity)
      {
        vel_eval     = std::make_unique<DomainEval<dim>>(matrix_free, vel_dof_idx, heat_quad_idx);
        vel_old_eval = std::make_unique<DomainEval<dim>>(*vel_eval);
      }

    const auto cell_category = matrix_free.get_cell_range_category(cell_batch_range);
    if (cell_category == CutUtil::CellCategory::liquid)
      {
        DomainEval   T_new_eval_l(matrix_free,
                                heat_cut_dof_idx /*dof_no*/,
                                heat_quad_idx /*quad_no*/,
                                0 /*selected component*/,
                                cell_category /*active_fe_index*/);
        DomainEval<> T_old_eval_l(T_new_eval_l);

        for (unsigned int cell_batch = cell_batch_range.first; cell_batch < cell_batch_range.second;
             ++cell_batch)
          {
            internal::reinit_and_read_plain(T_new_eval_l, temperature, cell_batch);
            T_new_eval_l.evaluate(evaluate_values | evaluate_gradients);
            internal::reinit_and_read_plain(T_old_eval_l, temperature_old, cell_batch);
            T_old_eval_l.evaluate(evaluate_values | evaluate_gradients);
            if (vel_eval)
              {
                internal::reinit_and_read_plain(*vel_eval, *velocity, cell_batch);
                vel_eval->evaluate(evaluate_values);
                internal::reinit_and_read_plain(*vel_old_eval, *velocity_old, cell_batch);
                vel_old_eval->evaluate(evaluate_values);
              }

            for (const unsigned int q : T_new_eval_l.quadrature_point_indices())
              {
                const auto [conductivity_new_l, cv_new_l] =
                  internal::get_liquid_material_parameters(
                    do_solidification ? &T_new_eval_l : nullptr, material, q);
                const auto [conductivity_old_l, cv_old_l] =
                  internal::get_liquid_material_parameters(
                    do_solidification ? &T_old_eval_l : nullptr, material, q);
                const auto vel =
                  vel_eval ? vel_eval->get_value(q) : typename DomainEval<dim>::value_type();
                const auto vel_old = vel_old_eval ? vel_old_eval->get_value(q) :
                                                    typename DomainEval<dim>::value_type();
                internal::residual_domain_integral(T_new_eval_l,
                                                   T_old_eval_l,
                                                   conductivity_new_l,
                                                   cv_new_l,
                                                   conductivity_old_l,
                                                   cv_old_l,
                                                   vel,
                                                   vel_old,
                                                   ost_factor_implicit,
                                                   ost_factor_explicit,
                                                   q);
              }
            T_new_eval_l.integrate_scatter(evaluate_values | evaluate_gradients, residual);
          }
      }
    else if (cell_category == CutUtil::CellCategory::gas and heat_data.cut.two_phase)
      {
        DomainEval    T_new_eval_g(matrix_free,
                                heat_cut_dof_idx /*dof_no*/,
                                heat_quad_idx /*quad_no*/,
                                1 /*selected component*/,
                                cell_category /*active_fe_index*/);
        DomainEval<1> T_old_eval_g(T_new_eval_g);

        for (unsigned int cell_batch = cell_batch_range.first; cell_batch < cell_batch_range.second;
             ++cell_batch)
          {
            internal::reinit_and_read_plain(T_new_eval_g, temperature, cell_batch);
            T_new_eval_g.evaluate(evaluate_values | evaluate_gradients);
            internal::reinit_and_read_plain(T_old_eval_g, temperature_old, cell_batch);
            T_old_eval_g.evaluate(evaluate_values | evaluate_gradients);
            if (vel_eval)
              {
                internal::reinit_and_read_plain(*vel_eval, *velocity, cell_batch);
                vel_eval->evaluate(evaluate_values);
                internal::reinit_and_read_plain(*vel_old_eval, *velocity_old, cell_batch);
                vel_old_eval->evaluate(evaluate_values);
              }

            for (const unsigned int q : T_new_eval_g.quadrature_point_indices())
              {
                const auto &conductivity_g = material.get_data().gas.thermal_conductivity;
                const auto  cv_g =
                  material.get_data().gas.density * material.get_data().gas.specific_heat_capacity;
                const auto vel =
                  vel_eval ? vel_eval->get_value(q) : typename DomainEval<dim>::value_type();
                const auto vel_old = vel_old_eval ? vel_old_eval->get_value(q) :
                                                    typename DomainEval<dim>::value_type();
                internal::residual_domain_integral(T_new_eval_g,
                                                   T_old_eval_g,
                                                   conductivity_g,
                                                   cv_g,
                                                   conductivity_g,
                                                   cv_g,
                                                   vel,
                                                   vel_old,
                                                   ost_factor_implicit,
                                                   ost_factor_explicit,
                                                   q);
              }
            T_new_eval_g.integrate_scatter(evaluate_values | evaluate_gradients, residual);
          }
      }
    else if (cell_category == CutUtil::CellCategory::intersected and not heat_data.cut.two_phase)
      {
        // use FEEvaluation and FEPointEvaluation in combination for intersected cells
        DomainEval                      T_new_eval_cell_l(matrix_free,
                                     heat_cut_dof_idx /*dof_no*/,
                                     heat_quad_idx /*quad_no*/,
                                     0 /*selected component*/,
                                     cell_category /*active_fe_index*/);
        DomainEval<1>                   T_old_eval_cell_l(T_new_eval_cell_l);
        PointEval                       T_new_eval_subdomain_l(*mapping_info_cells[0],
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        PointEval                       T_old_eval_subdomain_l(*mapping_info_cells[0],
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<dim>> vel_eval_subdomain_l;
        std::unique_ptr<PointEval<dim>> vel_old_eval_subdomain_l;
        if (vel_eval)
          {
            vel_eval_subdomain_l =
              std::make_unique<PointEval<dim>>(*mapping_info_cells[0],
                                               reference_finite_element_vel,
                                               0 /*selected component*/,
                                               true /*force_lexicographic_numbering*/);
            vel_old_eval_subdomain_l =
              std::make_unique<PointEval<dim>>(*mapping_info_cells[0],
                                               reference_finite_element_vel,
                                               0 /*selected component*/,
                                               true /*force_lexicographic_numbering*/);
          }
        PointEval                    T_new_eval_interface_l(mapping_info_surface,
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<>> T_old_eval_interface_l;
        if (evapor_cooling)
          T_old_eval_interface_l =
            std::make_unique<PointEval<>>(mapping_info_surface,
                                          reference_finite_element_heat,
                                          0 /*selected component*/,
                                          true /*force_lexicographic_numbering*/);

        for (unsigned int cell_batch = cell_batch_range.first; cell_batch < cell_batch_range.second;
             ++cell_batch)
          {
            internal::reinit_and_read_plain(T_new_eval_cell_l, temperature, cell_batch);
            internal::reinit_and_read_plain(T_old_eval_cell_l, temperature_old, cell_batch);
            if (vel_eval)
              {
                internal::reinit_and_read_plain(*vel_eval, *velocity, cell_batch);
                internal::reinit_and_read_plain(*vel_old_eval, *velocity_old, cell_batch);
              }

            for (unsigned int cell_lane = 0;
                 cell_lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
                 ++cell_lane)
              {
                // evaluate T^n+1 for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_new_eval_subdomain_l,
                                                                  T_new_eval_cell_l,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                // evaluate T^n+1 for immersed boundary integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  T_new_eval_interface_l,
                  T_new_eval_cell_l,
                  T_old_eval_interface_l ? evaluate_values : dealii::EvaluationFlags::nothing,
                  cell_batch,
                  cell_lane,
                  n_dofs_per_cell_heat);
                // evaluate T^n for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_old_eval_subdomain_l,
                                                                  T_old_eval_cell_l,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                if (T_old_eval_interface_l)
                  // evaluate T^n for immersed boundary integral
                  CutUtil::evaluate_intersected_domain<dim, number>(*T_old_eval_interface_l,
                                                                    T_old_eval_cell_l,
                                                                    evaluate_values,
                                                                    cell_batch,
                                                                    cell_lane,
                                                                    n_dofs_per_cell_heat);
                if (vel_eval_subdomain_l)
                  {
                    // evaluate velocity for liquid domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(*vel_eval_subdomain_l,
                                                                           *vel_eval,
                                                                           evaluate_values,
                                                                           cell_batch,
                                                                           cell_lane,
                                                                           n_dofs_per_cell_vel);
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_old_eval_subdomain_l,
                      *vel_old_eval,
                      evaluate_values,
                      cell_batch,
                      cell_lane,
                      n_dofs_per_cell_vel);
                  }


                // liquid domain integral
                for (const unsigned int q_batch : T_new_eval_subdomain_l.quadrature_point_indices())
                  {
                    const auto [conductivity_new_l, cv_new_l] =
                      internal::get_liquid_material_parameters(
                        do_solidification ? &T_new_eval_subdomain_l : nullptr, material, q_batch);
                    const auto [conductivity_old_l, cv_old_l] =
                      internal::get_liquid_material_parameters(
                        do_solidification ? &T_old_eval_subdomain_l : nullptr, material, q_batch);
                    const auto vel     = vel_eval_subdomain_l ?
                                           vel_eval_subdomain_l->get_value(q_batch) :
                                           typename DomainEval<dim>::value_type();
                    const auto vel_old = vel_old_eval_subdomain_l ?
                                           vel_old_eval_subdomain_l->get_value(q_batch) :
                                           typename DomainEval<dim>::value_type();
                    internal::residual_domain_integral(T_new_eval_subdomain_l,
                                                       T_old_eval_subdomain_l,
                                                       conductivity_new_l,
                                                       cv_new_l,
                                                       conductivity_old_l,
                                                       cv_old_l,
                                                       vel,
                                                       vel_old,
                                                       ost_factor_implicit,
                                                       ost_factor_explicit,
                                                       q_batch);
                  }
                T_new_eval_subdomain_l.integrate(
                  dealii::StridedArrayView<number, n_lanes>(
                    &T_new_eval_cell_l.begin_dof_values()[0][cell_lane], n_dofs_per_cell_heat),
                  evaluate_values | evaluate_gradients);

                // immersed boundary integral
                for (const unsigned int q_batch : T_new_eval_interface_l.quadrature_point_indices())
                  {
                    internal::residual_immersed_boundary_integral(
                      T_new_eval_interface_l,
                      T_old_eval_interface_l.get(),
                      [&](const dealii::VectorizedArray<number> &T) { return compute_qVapor(T); },
                      ost_factor_implicit,
                      ost_factor_explicit,
                      internal::compute_interface_heat_flux(interface_heat_flux.get(),
                                                            T_new_eval_interface_l,
                                                            q_batch) *
                        this->time_increment,
                      q_batch);
                  }
                T_new_eval_interface_l.integrate(
                            dealii::StridedArrayView<number, n_lanes>(
                                    &T_new_eval_cell_l.begin_dof_values()[0][cell_lane], n_dofs_per_cell_heat),
                            evaluate_values,
                            true
                            /*specify flag 'true' for summing the integrated values into the solution values*/);
              }
            T_new_eval_cell_l.distribute_local_to_global(residual);
          }
      }
    else if (cell_category == CutUtil::CellCategory::intersected and heat_data.cut.two_phase)
      {
        // use FEEvaluation and FEPointEvaluation in combination for intersected cells
        DomainEval                      T_new_eval_cell_l(matrix_free,
                                     heat_cut_dof_idx /*dof_no*/,
                                     heat_quad_idx /*quad_no*/,
                                     0 /*selected component*/,
                                     cell_category /*active_fe_index*/);
        DomainEval<1>                   T_old_eval_cell_l(T_new_eval_cell_l);
        PointEval                       T_new_eval_subdomain_l(*mapping_info_cells[0],
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        PointEval                       T_old_eval_subdomain_l(*mapping_info_cells[0],
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<dim>> vel_eval_subdomain_l;
        std::unique_ptr<PointEval<dim>> vel_old_eval_subdomain_l;
        if (vel_eval)
          {
            vel_eval_subdomain_l =
              std::make_unique<PointEval<dim>>(*mapping_info_cells[0],
                                               reference_finite_element_vel,
                                               0 /*selected component*/,
                                               true /*force_lexicographic_numbering*/);
            vel_old_eval_subdomain_l =
              std::make_unique<PointEval<dim>>(*mapping_info_cells[0],
                                               reference_finite_element_vel,
                                               0 /*selected component*/,
                                               true /*force_lexicographic_numbering*/);
          }
        PointEval                       T_new_eval_interface_l(mapping_info_surface,
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        PointEval                       T_old_eval_interface_l(mapping_info_surface,
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        DomainEval                      T_new_eval_cell_g(matrix_free,
                                     heat_cut_dof_idx /*dof_no*/,
                                     heat_quad_idx /*quad_no*/,
                                     1 /*selected component*/,
                                     cell_category /*active_fe_index*/);
        DomainEval<1>                   T_old_eval_cell_g(T_new_eval_cell_g);
        PointEval                       T_new_eval_subdomain_g(*mapping_info_cells[1],
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        PointEval                       T_old_eval_subdomain_g(*mapping_info_cells[1],
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        std::unique_ptr<PointEval<dim>> vel_eval_subdomain_g;
        std::unique_ptr<PointEval<dim>> vel_old_eval_subdomain_g;
        if (vel_eval)
          {
            vel_eval_subdomain_g =
              std::make_unique<PointEval<dim>>(*mapping_info_cells[1],
                                               reference_finite_element_vel,
                                               0 /*selected component*/,
                                               true /*force_lexicographic_numbering*/);
            vel_old_eval_subdomain_g =
              std::make_unique<PointEval<dim>>(*mapping_info_cells[1],
                                               reference_finite_element_vel,
                                               0 /*selected component*/,
                                               true /*force_lexicographic_numbering*/);
          }
        PointEval T_new_eval_interface_g(mapping_info_surface,
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);
        PointEval T_old_eval_interface_g(mapping_info_surface,
                                         reference_finite_element_heat,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);

        for (unsigned int cell_batch = cell_batch_range.first; cell_batch < cell_batch_range.second;
             ++cell_batch)
          {
            internal::reinit_and_read_plain(T_new_eval_cell_l, temperature, cell_batch);
            internal::reinit_and_read_plain(T_new_eval_cell_g, temperature, cell_batch);
            internal::reinit_and_read_plain(T_old_eval_cell_l, temperature_old, cell_batch);
            internal::reinit_and_read_plain(T_old_eval_cell_g, temperature_old, cell_batch);
            if (vel_eval)
              {
                internal::reinit_and_read_plain(*vel_eval, *velocity, cell_batch);
                internal::reinit_and_read_plain(*vel_old_eval, *velocity_old, cell_batch);
              }

            for (unsigned int cell_lane = 0;
                 cell_lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
                 ++cell_lane)
              {
                // evaluate T^n+1_l for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_new_eval_subdomain_l,
                                                                  T_new_eval_cell_l,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                // evaluate T^n+1_l for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_new_eval_interface_l,
                                                                  T_new_eval_cell_l,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                // evaluate T^n+1_g for gas domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_new_eval_subdomain_g,
                                                                  T_new_eval_cell_g,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                // evaluate T^n+1_g for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_new_eval_interface_g,
                                                                  T_new_eval_cell_g,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                // evaluate T^n_l for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_old_eval_subdomain_l,
                                                                  T_old_eval_cell_l,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                // evaluate T^n_l for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_old_eval_interface_l,
                                                                  T_old_eval_cell_l,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                // evaluate T^n_g for gas domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_old_eval_subdomain_g,
                                                                  T_old_eval_cell_g,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                // evaluate T^n_g for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(T_old_eval_interface_g,
                                                                  T_old_eval_cell_g,
                                                                  evaluate_values |
                                                                    evaluate_gradients,
                                                                  cell_batch,
                                                                  cell_lane,
                                                                  n_dofs_per_cell_heat);
                if (vel_eval)
                  {
                    // evaluate velocity for liquid domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(*vel_eval_subdomain_l,
                                                                           *vel_eval,
                                                                           evaluate_values,
                                                                           cell_batch,
                                                                           cell_lane,
                                                                           n_dofs_per_cell_vel);
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_old_eval_subdomain_l,
                      *vel_old_eval,
                      evaluate_values,
                      cell_batch,
                      cell_lane,
                      n_dofs_per_cell_vel);
                    // evaluate velocity for gas domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(*vel_eval_subdomain_g,
                                                                           *vel_eval,
                                                                           evaluate_values,
                                                                           cell_batch,
                                                                           cell_lane,
                                                                           n_dofs_per_cell_vel);
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_old_eval_subdomain_g,
                      *vel_eval,
                      evaluate_values,
                      cell_batch,
                      cell_lane,
                      n_dofs_per_cell_vel);
                  }

                // liquid domain integral
                for (const unsigned int q_batch : T_new_eval_subdomain_l.quadrature_point_indices())
                  {
                    const auto [conductivity_new_l, cv_new_l] =
                      internal::get_liquid_material_parameters(
                        do_solidification ? &T_new_eval_subdomain_l : nullptr, material, q_batch);
                    const auto [conductivity_old_l, cv_old_l] =
                      internal::get_liquid_material_parameters(
                        do_solidification ? &T_old_eval_subdomain_l : nullptr, material, q_batch);
                    const auto vel     = vel_eval_subdomain_l ?
                                           vel_eval_subdomain_l->get_value(q_batch) :
                                           typename DomainEval<dim>::value_type();
                    const auto vel_old = vel_old_eval_subdomain_l ?
                                           vel_old_eval_subdomain_l->get_value(q_batch) :
                                           typename DomainEval<dim>::value_type();
                    internal::residual_domain_integral(T_new_eval_subdomain_l,
                                                       T_old_eval_subdomain_l,
                                                       conductivity_new_l,
                                                       cv_new_l,
                                                       conductivity_old_l,
                                                       cv_old_l,
                                                       vel,
                                                       vel_old,
                                                       ost_factor_implicit,
                                                       ost_factor_explicit,
                                                       q_batch);
                  }
                T_new_eval_subdomain_l.integrate(
                  dealii::StridedArrayView<number, n_lanes>(
                    &T_new_eval_cell_l.begin_dof_values()[0][cell_lane], n_dofs_per_cell_heat),
                  evaluate_values | evaluate_gradients);

                // gas domain integral
                for (const unsigned int q_batch : T_new_eval_subdomain_g.quadrature_point_indices())
                  {
                    const auto &conductivity_g = material.get_data().gas.thermal_conductivity;
                    const auto  cv_g           = material.get_data().gas.density *
                                      material.get_data().gas.specific_heat_capacity;
                    const auto vel     = vel_eval_subdomain_g ?
                                           vel_eval_subdomain_g->get_value(q_batch) :
                                           typename DomainEval<dim>::value_type();
                    const auto vel_old = vel_old_eval_subdomain_g ?
                                           vel_old_eval_subdomain_g->get_value(q_batch) :
                                           typename DomainEval<dim>::value_type();
                    internal::residual_domain_integral(T_new_eval_subdomain_g,
                                                       T_old_eval_subdomain_g,
                                                       conductivity_g,
                                                       cv_g,
                                                       conductivity_g,
                                                       cv_g,
                                                       vel,
                                                       vel_old,
                                                       ost_factor_implicit,
                                                       ost_factor_explicit,
                                                       q_batch);
                  }
                T_new_eval_subdomain_g.integrate(
                  dealii::StridedArrayView<number, n_lanes>(
                    &T_new_eval_cell_g.begin_dof_values()[0][cell_lane], n_dofs_per_cell_heat),
                  evaluate_values | evaluate_gradients);

                // interface integral
                for (const unsigned int q_batch : T_new_eval_interface_l.quadrature_point_indices())
                  {
                    internal::residual_interface_integral(
                      T_new_eval_interface_l,
                      T_new_eval_interface_g,
                      T_old_eval_interface_l,
                      T_old_eval_interface_g,
                      [&](const dealii::VectorizedArray<number> &T) { return compute_qVapor(T); },
                      material.get_data()
                        .liquid.thermal_conductivity, // TODO temperature-dependent at interface?
                      material.get_data().gas.thermal_conductivity,
                      ost_factor_implicit,
                      ost_factor_explicit,
                      nitsche_factor,
                      internal::compute_interface_heat_flux(interface_heat_flux.get(),
                                                            T_new_eval_interface_l,
                                                            q_batch) *
                        this->time_increment,
                      kappa_l,
                      kappa_g,
                      evapor_cooling != nullptr,
                      heat_data.cut.do_explicit_symmetry_term,
                      q_batch);
                  }
                T_new_eval_interface_l.integrate(
                            dealii::StridedArrayView<number, n_lanes>(
                                    &T_new_eval_cell_l.begin_dof_values()[0][cell_lane], n_dofs_per_cell_heat),
                            evaluate_values | evaluate_gradients,
                            true
                            /*specify flag 'true' for summing the integrated values into the solution values*/);
                T_new_eval_interface_g.integrate(
                            dealii::StridedArrayView<number, n_lanes>(
                                    &T_new_eval_cell_g.begin_dof_values()[0][cell_lane], n_dofs_per_cell_heat),
                            evaluate_values | evaluate_gradients,
                            true
                            /*specify flag 'true' for summing the integrated values into the solution values*/);
              }
            T_new_eval_cell_l.distribute_local_to_global(residual);
            T_new_eval_cell_g.distribute_local_to_global(residual);
          }
      }
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::residual_inner_face_loop(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &residual,
    [[maybe_unused]] const VectorType                                      &temperature_old,
    const std::pair<unsigned int, unsigned int>                            &face_batch_range) const
  {
    const auto              face_category = matrix_free.get_face_range_category(face_batch_range);
    const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);
    if (face_type == CutUtil::FaceType::intersected_face or
        face_type == CutUtil::FaceType::mixed_face_liquid_intersected or
        face_type == CutUtil::FaceType::mixed_face_intersected_liquid)
      {
        auto liquid_minus_category = CutUtil::CellCategory::intersected;
        auto liquid_plus_category  = CutUtil::CellCategory::intersected;

        if (face_type == CutUtil::FaceType::mixed_face_liquid_intersected)
          {
            liquid_minus_category = CutUtil::CellCategory::liquid;
          }
        else if (face_type == CutUtil::FaceType::mixed_face_intersected_liquid)
          {
            liquid_plus_category = CutUtil::CellCategory::liquid;
          }
        else
          {
            AssertThrow(face_type == CutUtil::FaceType::intersected_face,
                        dealii::ExcInternalError());
          }

        FaceEval T_new_eval_minus_l(matrix_free,
                                    true /*is_interior_face*/,
                                    heat_cut_dof_idx /*dof_no*/,
                                    heat_quad_idx /*quad_no*/,
                                    0 /*selected component*/,
                                    liquid_minus_category /*active_fe_index*/);
        FaceEval T_new_eval_plus_l(matrix_free,
                                   false /*is_interior_face*/,
                                   heat_cut_dof_idx /*dof_no*/,
                                   heat_quad_idx /*quad_no*/,
                                   0 /*selected component*/,
                                   liquid_plus_category /*active_fe_index*/);

        for (unsigned int face_batch = face_batch_range.first; face_batch < face_batch_range.second;
             face_batch++)
          {
            internal::reinit_and_read_plain(T_new_eval_minus_l, temperature, face_batch);
            internal::reinit_and_read_plain(T_new_eval_plus_l, temperature, face_batch);
            T_new_eval_minus_l.evaluate(evaluate_gradients);
            T_new_eval_plus_l.evaluate(evaluate_gradients);

            for (const unsigned int q : T_new_eval_minus_l.quadrature_point_indices())
              {
                internal::ghost_penalty_face_integral(
                  T_new_eval_minus_l,
                  T_new_eval_plus_l,
                  material.get_data().liquid.thermal_conductivity,
                  ost_factor_implicit,
                  cell_side_length,
                  heat_data.cut.stabilization.ghost_penalty.gamma_M_degree_1,
                  heat_data.cut.stabilization.ghost_penalty.gamma_A_degree_1,
                  q,
                  -1.0);
              }
            T_new_eval_minus_l.integrate_scatter(evaluate_gradients, residual);
            T_new_eval_plus_l.integrate_scatter(evaluate_gradients, residual);
          }
      }
    if ((face_type == CutUtil::FaceType::intersected_face or
         face_type == CutUtil::FaceType::mixed_face_gas_intersected or
         face_type == CutUtil::FaceType::mixed_face_intersected_gas) and
        heat_data.cut.two_phase)
      {
        auto gas_minus_category = CutUtil::CellCategory::intersected;
        auto gas_plus_category  = CutUtil::CellCategory::intersected;

        if (face_type == CutUtil::FaceType::mixed_face_gas_intersected)
          {
            gas_minus_category = CutUtil::CellCategory::gas;
          }
        else if (face_type == CutUtil::FaceType::mixed_face_intersected_gas)
          {
            gas_plus_category = CutUtil::CellCategory::gas;
          }
        else
          {
            AssertThrow(face_type == CutUtil::FaceType::intersected_face,
                        dealii::ExcInternalError());
          }

        FaceEval T_new_eval_minus_g(matrix_free,
                                    true /*is_interior_face*/,
                                    heat_cut_dof_idx /*dof_no*/,
                                    heat_quad_idx /*quad_no*/,
                                    1 /*selected component*/,
                                    gas_minus_category /*active_fe_index*/);

        FaceEval T_new_eval_plus_g(matrix_free,
                                   false /*is_interior_face*/,
                                   heat_cut_dof_idx /*dof_no*/,
                                   heat_quad_idx /*quad_no*/,
                                   1 /*selected component*/,
                                   gas_plus_category /*active_fe_index*/);

        for (unsigned int face_batch = face_batch_range.first; face_batch < face_batch_range.second;
             face_batch++)
          {
            internal::reinit_and_read_plain(T_new_eval_minus_g, temperature, face_batch);
            internal::reinit_and_read_plain(T_new_eval_plus_g, temperature, face_batch);
            T_new_eval_minus_g.evaluate(evaluate_gradients);
            T_new_eval_plus_g.evaluate(evaluate_gradients);

            for (const unsigned int q : T_new_eval_minus_g.quadrature_point_indices())
              {
                internal::ghost_penalty_face_integral(
                  T_new_eval_minus_g,
                  T_new_eval_plus_g,
                  material.get_data().gas.thermal_conductivity,
                  ost_factor_implicit,
                  cell_side_length,
                  heat_data.cut.stabilization.ghost_penalty.gamma_M_degree_1,
                  heat_data.cut.stabilization.ghost_penalty.gamma_A_degree_1,
                  q,
                  -1.0);
              }
            T_new_eval_minus_g.integrate_scatter(evaluate_gradients, residual);
            T_new_eval_plus_g.integrate_scatter(evaluate_gradients, residual);
          }
      }
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::residual_boundary_face_loop(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &residual,
    const VectorType                                                       &temperature_old,
    const std::pair<unsigned int, unsigned int>                            &face_batch_range) const
  {
    (void)matrix_free;
    (void)residual;
    (void)temperature_old;
    (void)face_batch_range;
    // TODO non-zero Neumann BC
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, heat_cut_dof_idx);

    dealii::TrilinosWrappers::SparseMatrix dummy;
    internal_compute_diagonal_or_system_matrix(diagonal, dummy, true);

    // invert
    const double linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = std::abs(i) > 1.0e-14 * linfty_norm ? 1.0 / i : 1.0;
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::compute_system_matrix_from_matrixfree(
    dealii::TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    VectorType dummy;
    internal_compute_diagonal_or_system_matrix(dummy, system_matrix, false);
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::internal_compute_diagonal_or_system_matrix(
    [[maybe_unused]] VectorType                             &diagonal,
    [[maybe_unused]] dealii::TrilinosWrappers::SparseMatrix &system_matrix,
    const bool                                               do_diagonal) const
  {
    const auto &matrix_free = scratch_data.get_matrix_free();

    // use FEEvaluation and FEPointEvaluation in combination for intersected cells
    std::unique_ptr<DomainEval<>> T_eval_l;
    std::unique_ptr<DomainEval<>> T_eval_intersected_l;
    if (evapor_cooling or do_solidification)
      {
        T_eval_l =
          std::make_unique<DomainEval<>>(matrix_free,
                                         heat_cut_dof_idx /*dof_no*/,
                                         heat_quad_idx /*quad_no*/,
                                         0 /*selected component*/,
                                         CutUtil::CellCategory::liquid /*active_fe_index*/);
        T_eval_intersected_l =
          std::make_unique<DomainEval<>>(matrix_free,
                                         heat_cut_dof_idx /*dof_no*/,
                                         heat_quad_idx /*quad_no*/,
                                         0 /*selected component*/,
                                         CutUtil::CellCategory::intersected /*active_fe_index*/);
      }
    std::unique_ptr<DomainEval<dim>> vel_eval;
    if (velocity)
      vel_eval = std::make_unique<DomainEval<dim>>(matrix_free, vel_dof_idx, heat_quad_idx);

    PointEval                    eval_subdomain_l(*mapping_info_cells[0],
                               reference_finite_element_heat,
                               0 /*selected component*/,
                               true /*force_lexicographic_numbering*/);
    std::unique_ptr<PointEval<>> T_eval_subdomain_l;
    if (do_solidification)
      T_eval_subdomain_l = std::make_unique<PointEval<>>(*mapping_info_cells[0],
                                                         reference_finite_element_heat,
                                                         0 /*selected component*/,
                                                         true /*force_lexicographic_numbering*/);
    std::unique_ptr<PointEval<dim>> vel_eval_subdomain_l;
    if (vel_eval)
      vel_eval_subdomain_l =
        std::make_unique<PointEval<dim>>(*mapping_info_cells[0],
                                         reference_finite_element_vel,
                                         0 /*selected component*/,
                                         true /*force_lexicographic_numbering*/);

    PointEval                    eval_interface_l(mapping_info_surface,
                               reference_finite_element_heat,
                               0 /*selected component*/,
                               true /*force_lexicographic_numbering*/);
    std::unique_ptr<PointEval<>> T_eval_interface_l;
    if (evapor_cooling)
      T_eval_interface_l = std::make_unique<PointEval<>>(mapping_info_surface,
                                                         reference_finite_element_heat,
                                                         0 /*selected component*/,
                                                         true /*force_lexicographic_numbering*/);

    std::unique_ptr<DomainEval<>>   T_eval_intersected_g;
    std::unique_ptr<PointEval<>>    eval_subdomain_g;
    std::unique_ptr<PointEval<dim>> vel_eval_subdomain_g;
    std::unique_ptr<PointEval<>>    eval_interface_g;
    std::unique_ptr<PointEval<>>    T_eval_interface_g;
    if (heat_data.cut.two_phase)
      {
        if (evapor_cooling)
          {
            T_eval_intersected_g = std::make_unique<DomainEval<>>(
              matrix_free,
              heat_cut_dof_idx /*dof_no*/,
              heat_quad_idx /*quad_no*/,
              1 /*selected component*/,
              CutUtil::CellCategory::intersected /*active_fe_index*/);
            T_eval_interface_g =
              std::make_unique<PointEval<>>(mapping_info_surface,
                                            reference_finite_element_heat,
                                            0 /*selected component*/,
                                            true /*force_lexicographic_numbering*/);
          }

        eval_subdomain_g = std::make_unique<PointEval<>>(*mapping_info_cells[1],
                                                         reference_finite_element_heat,
                                                         0 /*selected component*/,
                                                         true /*force_lexicographic_numbering*/);

        if (vel_eval)
          vel_eval_subdomain_g =
            std::make_unique<PointEval<dim>>(*mapping_info_cells[1],
                                             reference_finite_element_vel,
                                             0 /*selected component*/,
                                             true /*force_lexicographic_numbering*/);

        eval_interface_g = std::make_unique<PointEval<>>(mapping_info_surface,
                                                         reference_finite_element_heat,
                                                         0 /*selected component*/,
                                                         true /*force_lexicographic_numbering*/);
      }

    unsigned int old_cell_batch = dealii::numbers::invalid_unsigned_int;

    dealii::MatrixFreeTools::internal::
      ComputeMatrixScratchData<dim, dealii::VectorizedArray<number>, false /*is_face_*/>
        data_cell;
    dealii::MatrixFreeTools::internal::
      ComputeMatrixScratchData<dim, dealii::VectorizedArray<number>, true /*is_face_*/>
        data_face;

    if (heat_data.cut.two_phase)
      {
        data_cell.dof_numbers               = {heat_cut_dof_idx, heat_cut_dof_idx};
        data_cell.quad_numbers              = {heat_quad_idx, heat_quad_idx};
        data_cell.n_components              = {1, 1};
        data_cell.first_selected_components = {0, 1};
        data_cell.batch_type                = {0, 0}; // 0 for cell

        data_face.dof_numbers  = {heat_cut_dof_idx,
                                  heat_cut_dof_idx,
                                  heat_cut_dof_idx,
                                  heat_cut_dof_idx};
        data_face.quad_numbers = {heat_quad_idx, heat_quad_idx, heat_quad_idx, heat_quad_idx};
        data_face.n_components = {1, 1, 1, 1};
        data_face.first_selected_components = {0, 0, 1, 1};
        data_face.batch_type = {1, 2, 1, 2}; // 1 for interior face, 2 for exterior face
      }
    else
      {
        data_cell.dof_numbers               = {heat_cut_dof_idx};
        data_cell.quad_numbers              = {heat_quad_idx};
        data_cell.n_components              = {1};
        data_cell.first_selected_components = {0};
        data_cell.batch_type                = {0}; // 0 for cell

        data_face.dof_numbers               = {heat_cut_dof_idx, heat_cut_dof_idx};
        data_face.quad_numbers              = {heat_quad_idx, heat_quad_idx};
        data_face.n_components              = {1, 1};
        data_face.first_selected_components = {0, 0};
        data_face.batch_type                = {1, 2}; // 1 for interior face, 2 for exterior face
      }



    data_cell.op_create = [&](const std::pair<unsigned int, unsigned int> &cell_range) {
      std::vector<
        std::unique_ptr<dealii::FEEvaluationData<dim, dealii::VectorizedArray<number>, false>>>
        eval_data;

      const auto emplace_eval = [&](const unsigned int index) {
        eval_data.emplace_back(
          std::make_unique<DomainEval<>>(matrix_free,
                                         cell_range,
                                         data_cell.dof_numbers[index],
                                         data_cell.quad_numbers[index],
                                         data_cell.first_selected_components[index]));
      };

      const auto cell_category = matrix_free.get_cell_range_category(cell_range);

      if (cell_category == CutUtil::CellCategory::liquid or
          cell_category == CutUtil::CellCategory::intersected)
        {
          emplace_eval(0);
        }

      if ((cell_category == CutUtil::CellCategory::gas or
           cell_category == CutUtil::CellCategory::intersected) and
          heat_data.cut.two_phase)
        {
          emplace_eval(1);
        }

      return eval_data;
    };



    data_cell.op_reinit = [&](auto &evaluators, const unsigned cell_index) {
      for (unsigned int i = 0; i < evaluators.size(); ++i)
        static_cast<DomainEval<> &>(*evaluators[i]).reinit(cell_index);
    };

    data_cell.op_compute = [&](auto &evaluators) {
      auto &eval_1 = static_cast<DomainEval<> &>(*evaluators[0]);

      const unsigned int cell_batch    = eval_1.get_current_cell_index();
      const unsigned int cell_category = eval_1.get_active_fe_index();

      if (cell_category == CutUtil::CellCategory::liquid)
        {
          eval_1.evaluate(evaluate_values | evaluate_gradients);

          tangent_cell_operation_liquid(
            cell_batch, eval_1, T_eval_l.get(), vel_eval.get(), cell_batch != old_cell_batch);

          eval_1.integrate(evaluate_values | evaluate_gradients);
        }
      else if (cell_category == CutUtil::CellCategory::gas and heat_data.cut.two_phase)
        {
          eval_1.evaluate(evaluate_values | evaluate_gradients);

          tangent_cell_operation_gas(cell_batch,
                                     eval_1,
                                     vel_eval.get(),
                                     cell_batch != old_cell_batch);

          eval_1.integrate(evaluate_values | evaluate_gradients);
        }
      else if (cell_category == CutUtil::CellCategory::intersected and not heat_data.cut.two_phase)
        {
          tangent_cell_operation_intersected_one_phase(cell_batch,
                                                       eval_1,
                                                       eval_subdomain_l,
                                                       evapor_cooling ? &eval_interface_l : nullptr,
                                                       T_eval_intersected_l.get(),
                                                       T_eval_subdomain_l.get(),
                                                       T_eval_interface_l.get(),
                                                       vel_eval.get(),
                                                       vel_eval_subdomain_l.get(),
                                                       cell_batch != old_cell_batch);
        }
      else if (cell_category == CutUtil::CellCategory::intersected and heat_data.cut.two_phase)
        {
          auto &eval_2 = static_cast<DomainEval<> &>(*evaluators[1]);

          tangent_cell_operation_intersected_two_phase(cell_batch,
                                                       eval_1,
                                                       eval_2,
                                                       eval_subdomain_l,
                                                       *eval_subdomain_g,
                                                       eval_interface_l,
                                                       *eval_interface_g,
                                                       T_eval_intersected_l.get(),
                                                       T_eval_intersected_g.get(),
                                                       T_eval_subdomain_l.get(),
                                                       T_eval_interface_l.get(),
                                                       T_eval_interface_g.get(),
                                                       vel_eval.get(),
                                                       vel_eval_subdomain_l.get(),
                                                       vel_eval_subdomain_g.get(),
                                                       cell_batch != old_cell_batch);
        }
      old_cell_batch = cell_batch;
    };



    data_face.op_create = [&](const std::pair<unsigned int, unsigned int> &face_range) {
      std::vector<
        std::unique_ptr<dealii::FEEvaluationData<dim, dealii::VectorizedArray<number>, true>>>
        eval_data;

      const auto emplace_face_eval = [&](const unsigned int index) {
        bool       is_interior_face;
        const auto batch_type = data_face.batch_type[index];
        if (batch_type == 1)
          is_interior_face = true;
        else if (batch_type == 2)
          is_interior_face = false;
        else
          AssertThrow(
            false,
            dealii::ExcMessage(
              "The face batch type must either be 1 (interior face) or 2 (exterior face)!"));
        eval_data.emplace_back(
          std::make_unique<FaceEval>(matrix_free,
                                     face_range,
                                     is_interior_face,
                                     data_face.dof_numbers[index],
                                     data_face.quad_numbers[index],
                                     data_face.first_selected_components[index]));
      };


      const auto              face_category = matrix_free.get_face_range_category(face_range);
      const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);

      if (face_type == CutUtil::FaceType ::intersected_face or
          face_type == CutUtil::FaceType ::mixed_face_liquid_intersected or
          face_type == CutUtil::FaceType ::mixed_face_intersected_liquid)
        {
          emplace_face_eval(0);
          emplace_face_eval(1);
        }
      if ((face_type == CutUtil::FaceType ::intersected_face or
           face_type == CutUtil::FaceType ::mixed_face_gas_intersected or
           face_type == CutUtil::FaceType ::mixed_face_intersected_gas) and
          heat_data.cut.two_phase)
        {
          emplace_face_eval(2);
          emplace_face_eval(3);
        }

      return eval_data;
    };

    data_face.op_reinit = [](auto &evaluators, const unsigned face_index) {
      for (unsigned int i = 0; i < evaluators.size(); ++i)
        static_cast<FaceEval &>(*evaluators[i]).reinit(face_index);
    };

    data_face.op_compute = [&](auto &evaluators) {
      auto &eval_minus_l = static_cast<FaceEval &>(*evaluators[0]);
      auto &eval_plus_l  = static_cast<FaceEval &>(*evaluators[1]);

      const unsigned int      face_batch    = eval_minus_l.get_cell_or_face_batch_id();
      const auto              face_category = matrix_free.get_face_category(face_batch);
      const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);

      if (face_type == CutUtil::FaceType ::intersected_face or
          face_type == CutUtil::FaceType ::mixed_face_liquid_intersected or
          face_type == CutUtil::FaceType ::mixed_face_intersected_liquid)
        {
          eval_minus_l.evaluate(evaluate_gradients);
          eval_plus_l.evaluate(evaluate_gradients);

          for (const unsigned int q : eval_minus_l.quadrature_point_indices())
            internal::ghost_penalty_face_integral(
              eval_minus_l,
              eval_plus_l,
              material.get_data().liquid.thermal_conductivity,
              ost_factor_implicit,
              cell_side_length,
              heat_data.cut.stabilization.ghost_penalty.gamma_M_degree_1,
              heat_data.cut.stabilization.ghost_penalty.gamma_A_degree_1,
              q);

          eval_minus_l.integrate(evaluate_gradients);
          eval_plus_l.integrate(evaluate_gradients);
        }
      if ((face_type == CutUtil::FaceType ::intersected_face or
           face_type == CutUtil::FaceType ::mixed_face_gas_intersected or
           face_type == CutUtil::FaceType ::mixed_face_intersected_gas) and
          heat_data.cut.two_phase)
        {
          const int eval_idx = evaluators.size() == 4 ? 2 : 0;

          auto &eval_minus_g = static_cast<FaceEval &>(*evaluators[eval_idx]);
          auto &eval_plus_g  = static_cast<FaceEval &>(*evaluators[eval_idx + 1]);

          eval_minus_g.evaluate(evaluate_gradients);
          eval_plus_g.evaluate(evaluate_gradients);

          for (const unsigned int q : eval_minus_g.quadrature_point_indices())
            internal::ghost_penalty_face_integral(
              eval_minus_g,
              eval_plus_g,
              material.get_data().gas.thermal_conductivity,
              ost_factor_implicit,
              cell_side_length,
              heat_data.cut.stabilization.ghost_penalty.gamma_M_degree_1,
              heat_data.cut.stabilization.ghost_penalty.gamma_A_degree_1,
              q);

          eval_minus_g.integrate(evaluate_gradients);
          eval_plus_g.integrate(evaluate_gradients);
        }
    };

    if (do_diagonal)
      {
        std::vector<VectorType *> dummy(1);
        dummy[0] = &diagonal;
        dealii::MatrixFreeTools::internal::
          compute_diagonal<dim, number, dealii::VectorizedArray<number>>(
            matrix_free, data_cell, data_face, {} /*data_boundary*/, diagonal, dummy);
      }
    else // compute matrix
      {
        dealii::MatrixFreeTools::internal::
          compute_matrix<dim, number, dealii::VectorizedArray<number>>(matrix_free,
                                                                       scratch_data.get_constraint(
                                                                         heat_cut_dof_idx),
                                                                       data_cell,
                                                                       data_face,
                                                                       {} /*data_boundary*/,
                                                                       system_matrix);
      }
  }



  template class HeatCutOperator<1, double>;
  template class HeatCutOperator<2, double>;
  template class HeatCutOperator<3, double>;
} // namespace MeltPoolDG::Heat
