#include <deal.II/base/array_view.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/heat/laser_heat_source_projection_based.hpp>
#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/level_set/normal_vector_operator.hpp>
#include <meltpooldg/utilities/dof_tools.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  LaserHeatSourceProjectionBased<dim, number>::LaserHeatSourceProjectionBased(
    const ScratchData<dim, dim, number>  &scratch_data_in,
    const LaserData<number>              &laser_data_in,
    const LaserBeam<dim, number>         &beam,
    const LaserCouplingData<dim, number> &coupling_data_in,
    HeatSourceReceiver<dim, number>      &heat_operation_in)
    : scratch_data(scratch_data_in)
    , laser_data(laser_data_in)
    , coupling_data(coupling_data_in)
    , heat_operation(heat_operation_in)
    , intensity_profile(create_intensity_profile(laser_data, beam, false /*is_volumetric*/))
    , laser_direction(laser_data.template get_beam_direction<dim>())
    , variable_properties_over_interface(coupling_data.variable_properties_over_interface)
  {
    delta_phase_weighted =
      create_phase_weighted_delta_approximation(laser_data.delta_approximation_phase_weighted);

    AssertThrow(coupling_data.level_set_as_heaviside != nullptr,
                ExcMessage("The projection-based laser models require a heaviside field."));
  }

  template <int dim, typename number>
  bool
  LaserHeatSourceProjectionBased<dim, number>::requires_inner_face_loops() const
  {
    return laser_data.model == LaserModelType::interface_projection_sharp_conforming;
  }

  template <int dim, typename number>
  void
  LaserHeatSourceProjectionBased<dim, number>::apply()
  {
    const BlockVectorType *normal_vector =
      coupling_data.get_normal_vector ? &coupling_data.get_normal_vector() : nullptr;

    switch (laser_data.model)
      {
          case LaserModelType::interface_projection_regularized: {
            scratch_data.initialize_dof_vector(heat_source, coupling_data.heat_no_bc_dof_idx);
            compute_regularized_heat_source(heat_source, normal_vector);
            heat_operation.get_heat_source() += heat_source;
            break;
          }
          case LaserModelType::interface_projection_sharp: {
            add_sharp_heat_source(heat_operation.get_user_rhs(), normal_vector);
            break;
          }
          case LaserModelType::interface_projection_sharp_conforming: {
            add_sharp_conforming_heat_source(heat_operation.get_user_rhs(), normal_vector);
            break;
          }
        default:
          AssertThrow(false, ExcNotImplemented());
      }
  }


  template <int dim, typename number>
  number
  LaserHeatSourceProjectionBased<dim, number>::local_compute_interfacial_heat_source(
    const Point<dim>             &p,
    const Tensor<1, dim, number> &normal_vector,
    const number                  delta_value,
    const number                  heaviside) const
  {
    const auto projection_factor = compute_projection_factor(laser_direction, normal_vector);

    const auto weight =
      variable_properties_over_interface ? heaviside : ((heaviside > 0.5) ? 1.0 : 0.0);
    const auto absorptivity = LevelSet::Tools::interpolate(weight,
                                                           laser_data.absorptivity_gas,
                                                           laser_data.absorptivity_liquid);

    return intensity_profile->value(p) * projection_factor * absorptivity * delta_value;
  }

  template <int dim, typename number>
  void
  LaserHeatSourceProjectionBased<dim, number>::compute_regularized_heat_source(
    VectorType            &heat_source_vector,
    const BlockVectorType *normal_vector) const
  {
    const unsigned int heat_dof_idx        = coupling_data.heat_no_bc_dof_idx;
    const VectorType  &level_set_heaviside = *coupling_data.level_set_as_heaviside;
    const unsigned int ls_dof_idx          = coupling_data.level_set_dof_idx;
    const unsigned int normal_dof_idx      = coupling_data.normal_dof_idx;

    heat_source_vector = 0.0;

    const bool update_ghosts = !level_set_heaviside.has_ghost_elements();

    if (update_ghosts)
      level_set_heaviside.update_ghost_values();

    bool normal_update_ghosts = true;
    if (normal_vector)
      {
        normal_update_ghosts = !normal_vector->has_ghost_elements();
        if (normal_update_ghosts)
          normal_vector->update_ghost_values();
      }

    const number tolerance_normal_vector =
      UtilityFunctions::compute_numerical_zero_of_norm<dim, number>(
        scratch_data.get_triangulation(), scratch_data.get_mapping());

    FEValues<dim> heat_source_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(heat_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(heat_dof_idx).get_fe().get_unit_support_points()),
      update_quadrature_points);

    const VectorType              *used_level_set  = &level_set_heaviside;
    unsigned int                   used_ls_dof_idx = ls_dof_idx;
    std::unique_ptr<FEValues<dim>> ls_heaviside_eval;

    ls_heaviside_eval = std::make_unique<FEValues<dim>>(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(heat_dof_idx).get_fe().get_unit_support_points()),
      update_values | update_gradients);

    FEValues<dim> normal_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(normal_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(heat_dof_idx).get_fe().get_unit_support_points()),
      update_values);

    const unsigned int dofs_per_cell =
      scratch_data.get_dof_handler(heat_dof_idx).get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<number> ls_heaviside_at_q(ls_heaviside_eval->n_quadrature_points);
    std::vector<dealii::Tensor<1, dim, number>> grad_ls_heaviside_at_q(
      ls_heaviside_eval->n_quadrature_points);
    std::vector<Tensor<1, dim>> normal_at_q(dofs_per_cell, Tensor<1, dim>());

    /*
     * Interpolate the level set DoFs to the temperature DoFs
     */
    VectorType interpolated_vec;

    if (scratch_data.is_FE_Q_iso_Q_1(ls_dof_idx))
      {
        const auto ls_to_temperature_grad_interpolation_matrix =
          DoFTools::create_dof_interpolation_matrix<dim, number>(
            scratch_data.get_dof_handler(heat_dof_idx),
            scratch_data.get_dof_handler(ls_dof_idx),
            false);

        ls_heaviside_eval = std::make_unique<FEValues<dim>>(
          scratch_data.get_mapping(),
          scratch_data.get_dof_handler(heat_dof_idx).get_fe(),
          Quadrature<dim>(
            scratch_data.get_dof_handler(heat_dof_idx).get_fe().get_unit_support_points()),
          update_values | update_gradients);

        std::vector<types::global_dof_index> ls_local_dof_indices(
          scratch_data.get_dof_handler(ls_dof_idx).get_fe().n_dofs_per_cell());

        // create vector of interpolated values of level set at DoF points of the temperature field
        used_level_set  = &interpolated_vec;
        used_ls_dof_idx = heat_dof_idx;
        scratch_data.initialize_dof_vector(interpolated_vec, heat_dof_idx);

        for (const auto &cell : scratch_data.get_triangulation().active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                TriaIterator<DoFCellAccessor<dim, dim, false>> ls_dof_cell(
                  &scratch_data.get_triangulation(),
                  cell->level(),
                  cell->index(),
                  &scratch_data.get_dof_handler(ls_dof_idx));
                ls_dof_cell->get_dof_indices(ls_local_dof_indices);

                TriaIterator<DoFCellAccessor<dim, dim, false>> T_dof_cell(
                  &scratch_data.get_triangulation(),
                  cell->level(),
                  cell->index(),
                  &scratch_data.get_dof_handler(heat_dof_idx));

                T_dof_cell->get_dof_indices(local_dof_indices);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    number interpolated_value = 0;

                    /* Interpolate the level set Φ from the support points of the level set space j
                     * to the one of the temperature space i, using the interpolation matrix P
                     * _
                     * Φ   = P   · Φ
                     *  i     ij    j
                     */
                    for (unsigned int j = 0;
                         j < scratch_data.get_dof_handler(ls_dof_idx).get_fe().n_dofs_per_cell();
                         ++j)
                      interpolated_value += ls_to_temperature_grad_interpolation_matrix(i, j) *
                                            level_set_heaviside[ls_local_dof_indices[j]];

                    // Store the interpolated values at the support points of the pressure space
                    interpolated_vec[local_dof_indices[i]] = interpolated_value;
                  }
              }
          }

        interpolated_vec.compress(VectorOperation::insert);
        interpolated_vec.update_ghost_values();
      }

    // count the number of nodal assembly entries
    VectorType heat_source_vector_multiplicity;
    heat_source_vector_multiplicity.reinit(heat_source_vector);

    for (const auto &cell : scratch_data.get_triangulation().active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            TriaIterator<DoFCellAccessor<dim, dim, false>> ls_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(used_ls_dof_idx));

            TriaIterator<DoFCellAccessor<dim, dim, false>> T_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(heat_dof_idx));

            TriaIterator<DoFCellAccessor<dim, dim, false>> normal_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(normal_dof_idx));

            T_dof_cell->get_dof_indices(local_dof_indices);

            // fill multiplicity
            Vector<number> heat_source_vector_multiplicity_local(dofs_per_cell);
            for (auto &val : heat_source_vector_multiplicity_local)
              val = 1.0;
            scratch_data.get_constraint(heat_dof_idx)
              .distribute_local_to_global(heat_source_vector_multiplicity_local,
                                          local_dof_indices,
                                          heat_source_vector_multiplicity);


            heat_source_eval.reinit(T_dof_cell);
            normal_eval.reinit(normal_dof_cell);
            ls_heaviside_eval->reinit(ls_dof_cell);

            ls_heaviside_eval->get_function_gradients(*used_level_set, grad_ls_heaviside_at_q);
            ls_heaviside_eval->get_function_values(*used_level_set, ls_heaviside_at_q);

            // use filtered normal vector computation ..
            if (normal_vector)
              LevelSet::NormalVectorOperator<dim, number>::get_unit_normals_at_quadrature(
                normal_eval, *normal_vector, normal_at_q, tolerance_normal_vector);

            Vector<number> heat_source_vector_local(dofs_per_cell);

            for (const auto q : heat_source_eval.quadrature_point_indices())
              {
                const number grad_ls_norm = grad_ls_heaviside_at_q[q].norm();
                const number delta_value =
                  delta_phase_weighted == nullptr ?
                    grad_ls_norm :
                    grad_ls_norm * delta_phase_weighted->compute_weight(ls_heaviside_at_q[q]);

                if (delta_value == 0.0)
                  {
                    heat_source_vector_local[q] = 0;
                    continue;
                  }

                // ... or use (unfiltered) gradient of the level set function
                if (normal_vector == nullptr)
                  normal_at_q[q] = grad_ls_heaviside_at_q[q] / grad_ls_norm;

                heat_source_vector_local[q] =
                  local_compute_interfacial_heat_source(heat_source_eval.quadrature_point(q),
                                                        normal_at_q[q],
                                                        delta_value,
                                                        ls_heaviside_at_q[q]);
              }

            scratch_data.get_constraint(heat_dof_idx)
              .distribute_local_to_global(heat_source_vector_local,
                                          local_dof_indices,
                                          heat_source_vector);
          }
      }

    heat_source_vector.compress(VectorOperation::add);
    heat_source_vector_multiplicity.compress(VectorOperation::add);

    /*
     * average the nodally assembled values to smoothen discontinuous gradients of
     * the level set field
     */
    for (unsigned int i = 0; i < heat_source_vector_multiplicity.locally_owned_size(); ++i)
      if (heat_source_vector_multiplicity.local_element(i) > 1.0)
        heat_source_vector.local_element(i) /= heat_source_vector_multiplicity.local_element(i);

    scratch_data.get_constraint(heat_dof_idx).distribute(heat_source_vector);

    heat_source_vector.zero_out_ghost_values();

    if (update_ghosts)
      level_set_heaviside.zero_out_ghost_values();
    if (normal_vector && normal_update_ghosts)
      normal_vector->zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  LaserHeatSourceProjectionBased<dim, number>::add_sharp_conforming_heat_source(
    VectorType            &heat_rhs,
    const BlockVectorType *normal_vector) const
  {
    const unsigned int heat_dof_idx        = coupling_data.heat_no_bc_dof_idx;
    const VectorType  &level_set_heaviside = *coupling_data.level_set_as_heaviside;
    const unsigned int ls_dof_idx          = coupling_data.level_set_dof_idx;
    const unsigned int normal_dof_idx      = coupling_data.normal_dof_idx;
    const unsigned int heat_quad_idx       = coupling_data.heat_quad_idx;

    const bool update_ghosts = !level_set_heaviside.has_ghost_elements();
    if (update_ghosts)
      level_set_heaviside.update_ghost_values();

    bool normal_update_ghosts = true;
    if (normal_vector)
      {
        normal_update_ghosts = !normal_vector->has_ghost_elements();

        if (normal_update_ghosts)
          normal_vector->update_ghost_values();
      }

    // step 1: set material ID of cells
    LevelSet::Tools::set_material_id_from_level_set(scratch_data, ls_dof_idx, level_set_heaviside);

    // step 2: evaluate and fill rhs
    FEFaceIntegrator<dim, 1, number> ls_eval(scratch_data.get_matrix_free(),
                                             true /*is_interior_face*/,
                                             ls_dof_idx,
                                             heat_quad_idx);

    FEFaceIntegrator<dim, 1, number> rhs_eval(scratch_data.get_matrix_free(),
                                              true /*is_interior_face*/,
                                              heat_dof_idx,
                                              heat_quad_idx);

    std::unique_ptr<FEFaceIntegrator<dim, dim, number>> normal_eval;

    VectorizedArray<number> local_result_face = 0.0;
    Point<dim>              position_v;
    Point<dim>              normal_v;

    if (normal_vector)
      normal_eval = std::make_unique<FEFaceIntegrator<dim, dim, number>>(
        scratch_data.get_matrix_free(), true, normal_dof_idx, heat_quad_idx);

    std::pair<unsigned int, unsigned int> face_range = {
      0, scratch_data.get_matrix_free().n_inner_face_batches()};

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        ls_eval.reinit(face);
        ls_eval.read_dof_values(level_set_heaviside);
        ls_eval.evaluate(normal_vector ? EvaluationFlags::values :
                                         EvaluationFlags::values | EvaluationFlags::gradients);

        if (normal_vector)
          {
            normal_eval->reinit(face);
            normal_eval->read_dof_values(*normal_vector);
            normal_eval->evaluate(EvaluationFlags::values);
          }

        rhs_eval.reinit(face);

        // collect lanes that need to be processed
        std::vector<unsigned int> process_lanes;
        for (unsigned int v = 0;
             v < scratch_data.get_matrix_free().n_active_entries_per_face_batch(face);
             ++v)
          {
            const auto face_iter_inner =
              scratch_data.get_matrix_free().get_face_iterator(face, v, true);
            const auto face_iter_outer =
              scratch_data.get_matrix_free().get_face_iterator(face, v, false);

            // check if surrounding cells have different materials
            if (face_iter_inner.first->material_id() != face_iter_outer.first->material_id())
              process_lanes.emplace_back(v);
          }

        // loop over quadrature points
        for (unsigned int q = 0; q < ls_eval.n_q_points; ++q)
          {
            auto unit_normal =
              normal_vector ? MeltPoolDG::VectorTools::to_vector<dim>(normal_eval->get_value(q)) :
                              ls_eval.get_gradient(q);
            unit_normal = unit_normal / std::max(unit_normal.norm(), VectorizedArray<number>(1e-6));

            // loop over relavant lanes
            local_result_face = 0.0;
            for (const auto &v : process_lanes)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  position_v[d] = ls_eval.quadrature_point(q)[d][v];
                for (unsigned int d = 0; d < dim; ++d)
                  normal_v[d] = unit_normal[d][v];

                local_result_face[v] = local_compute_interfacial_heat_source(
                  position_v, normal_v, 1.0 /*delta value*/, ls_eval.get_value(q)[v]);
              }
            rhs_eval.submit_value(local_result_face, q);
          }
        rhs_eval.integrate_scatter(EvaluationFlags::values, heat_rhs);
      }

    heat_rhs.compress(VectorOperation::add);

    if (update_ghosts)
      level_set_heaviside.zero_out_ghost_values();
    if (normal_vector && normal_update_ghosts)
      normal_vector->zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  LaserHeatSourceProjectionBased<dim, number>::add_sharp_heat_source(
    VectorType            &heat_rhs,
    const BlockVectorType *normal_vector) const
  {
    const unsigned int heat_dof_idx        = coupling_data.heat_no_bc_dof_idx;
    const VectorType  &level_set_heaviside = *coupling_data.level_set_as_heaviside;
    const unsigned int ls_dof_idx          = coupling_data.level_set_dof_idx;
    const unsigned int normal_dof_idx      = coupling_data.normal_dof_idx;

    const bool update_ghosts = !level_set_heaviside.has_ghost_elements();
    if (update_ghosts)
      level_set_heaviside.update_ghost_values();

    bool normal_update_ghosts = true;
    if (normal_vector)
      {
        normal_update_ghosts = !normal_vector->has_ghost_elements();
        if (normal_update_ghosts)
          normal_vector->update_ghost_values();
      }

    FEPointEvaluation<1, dim> ls(scratch_data.get_mapping(),
                                 scratch_data.get_fe(ls_dof_idx),
                                 normal_vector ? update_values : update_values | update_gradients);

    std::unique_ptr<FEPointEvaluation<dim, dim>> normal_vals;
    std::unique_ptr<FESystem<dim>>               fe_normal;

    if (normal_vector)
      {
        fe_normal   = std::make_unique<FESystem<dim>>(scratch_data.get_fe(normal_dof_idx), dim);
        normal_vals = std::make_unique<FEPointEvaluation<dim, dim>>(scratch_data.get_mapping(),
                                                                    *fe_normal,
                                                                    update_values);
      }

    FEPointEvaluation<1, dim> heat_source_vals(scratch_data.get_mapping(),
                                               scratch_data.get_fe(heat_dof_idx),
                                               update_values);

    std::vector<number>                  buffer;
    std::vector<number>                  buffer_dim;
    std::vector<types::global_dof_index> local_dof_indices;

    LevelSet::Tools::evaluate_at_interface<dim, number>(
      scratch_data.get_dof_handler(ls_dof_idx),
      scratch_data.get_mapping(),
      level_set_heaviside,
      [&](const auto &cell, const auto &points_real, const auto &points, const auto &weights) {
        // dof indices level set
        local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
        buffer.resize(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);

        const unsigned int n_points = points.size();

        const ArrayView<const Point<dim>> unit_points(points.data(), n_points);
        const ArrayView<const number>     JxW(weights.data(), n_points);

        // gather evaluate level set for the points at the interface
        ls.reinit(cell, unit_points);
        scratch_data.get_constraint(ls_dof_idx)
          .get_dof_values(level_set_heaviside,
                          local_dof_indices.begin(),
                          buffer.begin(),
                          buffer.end());
        ls.evaluate(buffer,
                    normal_vector ? EvaluationFlags::values :
                                    EvaluationFlags::values | EvaluationFlags::gradients);

        // gather_evaluate unit normal vector for the points at the interface
        if (normal_vals && normal_vector)
          {
            normal_vals->reinit(cell, unit_points);
            buffer_dim.resize(fe_normal->n_dofs_per_cell()); // @todo: times dim

            for (int d = 0; d < dim; ++d)
              {
                cell->get_dof_values(normal_vector->block(d), buffer.begin(), buffer.end());

                for (unsigned int c = 0; c < scratch_data.get_fe(normal_dof_idx).n_dofs_per_cell();
                     ++c)
                  buffer_dim[fe_normal->component_to_system_index(d, c)] = buffer[c];
              }

            // normalize
            for (unsigned int c = 0; c < cell->get_fe().n_dofs_per_cell(); ++c)
              {
                number norm = 0.0;
                for (int d = 0; d < dim; ++d)
                  norm += Utilities::fixed_power<2>(
                    buffer_dim[fe_normal->component_to_system_index(d, c)]);

                norm = std::max(1e-6, std::sqrt(norm));

                for (int d = 0; d < dim; ++d)
                  buffer_dim[fe_normal->component_to_system_index(d, c)] /= norm;
              }
            normal_vals->evaluate(make_array_view(buffer_dim), EvaluationFlags::values);
          }

        TriaIterator<DoFCellAccessor<dim, dim, false>> T_dof_cell(&scratch_data.get_triangulation(),
                                                                  cell->level(),
                                                                  cell->index(),
                                                                  &scratch_data.get_dof_handler(
                                                                    heat_dof_idx));

        local_dof_indices.resize(T_dof_cell->get_fe().n_dofs_per_cell());
        T_dof_cell->get_dof_indices(local_dof_indices);
        buffer.resize(T_dof_cell->get_fe().n_dofs_per_cell());

        heat_source_vals.reinit(T_dof_cell, unit_points);

        for (unsigned int q = 0; q < n_points; ++q)
          {
            // If a normal vector field is given, use it to compute the unit normal to the
            // interface. Otherwise use the gradient of the level set field.
            const auto unit_normal =
              normal_vector ? MeltPoolDG::VectorTools::to_vector<dim>(normal_vals->get_value(q)) :
                              ls.get_gradient(q) / std::max(ls.get_gradient(q).norm(), 1e-6);

            const auto ls_at_q = ls.get_value(q);
            const auto result =
              local_compute_interfacial_heat_source(points_real[q], unit_normal, 1.0, ls_at_q) *
              JxW[q];

            heat_source_vals.submit_value(result, q);
          }

        // integrate laser heat source
        heat_source_vals.test_and_sum(buffer, EvaluationFlags::values);

        scratch_data.get_constraint(heat_dof_idx)
          .distribute_local_to_global(buffer, local_dof_indices, heat_rhs);
      },
      0.5, /*contour value*/
      3 /*n_subdivisions*/);

    heat_rhs.compress(VectorOperation::add);

    heat_rhs.zero_out_ghost_values();

    if (update_ghosts)
      level_set_heaviside.zero_out_ghost_values();
    if (normal_vector && normal_update_ghosts)
      normal_vector->zero_out_ghost_values();

    if (false)
      {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;

        DataOut<dim> data_out;
        data_out.set_flags(flags);

        data_out.add_data_vector(scratch_data.get_dof_handler(heat_dof_idx),
                                 heat_rhs,
                                 "heat_source");
        data_out.build_patches(scratch_data.get_mapping());
        std::string output = "heat_source.vtu";
        data_out.write_vtu_in_parallel(output, scratch_data.get_mpi_comm());
      }
  }

  template class LaserHeatSourceProjectionBased<1, double>;
  template class LaserHeatSourceProjectionBased<2, double>;
  template class LaserHeatSourceProjectionBased<3, double>;
} // namespace MeltPoolDG::Heat
