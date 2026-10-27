#include <meltpooldg/level_set/nearest_point.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.templates.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#ifdef DEAL_II_WITH_ARBORX
#  include <deal.II/arborx/distributed_tree.h>
#endif

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/rtree.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>


namespace MeltPoolDG::LevelSet::Tools
{
  namespace internal
  {
    /**
     * Compute unit tangent vectors orthogonal to the unit normal vector @p unit_normal. If the
     * normal vector is zero, i.e., it could not be determined, the tangents are set to zero.
     */
    template <int dim>
    std::array<dealii::Point<dim>, dim - 1>
    compute_unit_tangents(const dealii::Point<dim> &unit_normal)
    {
      std::array<dealii::Point<dim>, dim - 1> unit_tangents;

      if (unit_normal.norm_square() == 0)
        return unit_tangents;

      if constexpr (dim == 2)
        {
          unit_tangents[0][0] = unit_normal[1];
          unit_tangents[0][1] = -unit_normal[0];
        }
      else if constexpr (dim == 3)
        {
          // choose an auxiliary unit vector that is not (nearly) parallel to the normal vector
          const dealii::Point<dim> temp_vec = std::abs(unit_normal[0]) < 0.9 ?
                                                dealii::Point<dim>::unit_vector(0) :
                                                dealii::Point<dim>::unit_vector(1);

          unit_tangents[0] = temp_vec - (temp_vec * unit_normal) * unit_normal;
          unit_tangents[0] /= unit_tangents[0].norm();
          unit_tangents[1] = cross_product_3d(unit_normal, unit_tangents[0]);
        }

      return unit_tangents;
    }

    /**
     * Compute the offsets of the point @p y relative to the point @p x in the directions of
     * @p unit_tangents.
     */
    template <int dim>
    std::array<double, dim - 1>
    compute_tangential_offsets(const dealii::Point<dim>                      &y,
                               const dealii::Point<dim>                      &x,
                               const std::array<dealii::Point<dim>, dim - 1> &unit_tangents)
    {
      const auto distance_vec = y - x;

      std::array<double, dim - 1> offsets;
      for (unsigned int d = 0; d < dim - 1; ++d)
        offsets[d] = distance_vec * unit_tangents[d];

      return offsets;
    }

    /**
     * Compute the maximum absolute value of the entries of @p values. Returns zero if
     * @p values is empty.
     */
    template <typename Container>
    typename Container::value_type
    max_abs(const Container &values)
    {
      typename Container::value_type max_value = 0;
      for (const auto &v : values)
        max_value = std::max(max_value, std::abs(v));
      return max_value;
    }
  } // namespace internal

  template <int dim, typename number>
  NearestPoint<dim, number>::NearestPoint(
    const dealii::Mapping<dim>                                &mapping,
    const dealii::DoFHandler<dim>                             &dof_handler_signed_distance,
    const VectorType                                          &signed_distance,
    const BlockVectorType                                     &normal_vector,
    dealii::Utilities::MPI::RemotePointEvaluation<dim, dim>   &remote_point_evaluation,
    const NearestPointData<number>                            &nearest_point_data,
    std::optional<std::reference_wrapper<dealii::TimerOutput>> timer_output)
    : NearestPoint(mapping,
                   dof_handler_signed_distance,
                   signed_distance,
                   remote_point_evaluation,
                   nearest_point_data,
                   timer_output)
  {
    this->normal_vector = &normal_vector;
  }

  template <int dim, typename number>
  NearestPoint<dim, number>::NearestPoint(
    const dealii::Mapping<dim>                                &mapping,
    const dealii::DoFHandler<dim>                             &dof_handler_signed_distance,
    const VectorType                                          &signed_distance,
    dealii::Utilities::MPI::RemotePointEvaluation<dim, dim>   &remote_point_evaluation,
    const NearestPointData<number>                            &nearest_point_data,
    std::optional<std::reference_wrapper<dealii::TimerOutput>> timer_output)
    : mapping(mapping)
    , dof_handler_ls(dof_handler_signed_distance)
    , signed_distance(signed_distance)
    , normal_vector(nullptr)
    , nearest_point_data(nearest_point_data)
    , remote_point_evaluation(remote_point_evaluation)
    , tol_distance(nearest_point_data.rel_tol *
                   dealii::GridTools::minimal_cell_diameter(dof_handler_ls.get_triangulation()) /
                   std::sqrt(dim))
    , narrow_band_threshold(nearest_point_data.narrow_band_threshold > 0 ?
                              nearest_point_data.narrow_band_threshold :
                              signed_distance.linfty_norm() * 0.9999)
    , tolerance_normal_vector(UtilityFunctions::compute_numerical_zero_of_norm<dim, number>(
        dof_handler_ls.get_triangulation(),
        mapping))
    , mpi_comm(dof_handler_ls.get_mpi_communicator())
    , pcout(std::cout,
            dealii::Utilities::MPI::this_mpi_process(
              dof_handler_signed_distance.get_mpi_communicator()) == 0)
    , timer_output(timer_output)
  {
    if (nearest_point_data.verbosity_level > 0)
      {
        Journal::print_line(pcout,
                            "narrow band threshold " + std::to_string(narrow_band_threshold) +
                              "    isocontour = " + std::to_string(nearest_point_data.isocontour),
                            "nearest_point");
      }
    AssertThrow(dim <= 2 or
                  nearest_point_data.type != NearestPointType::closest_point_normal_collinear,
                dealii::ExcNotImplemented());

    AssertThrow(narrow_band_threshold > 0,
                dealii::ExcMessage("Narrow band threshold must be larger than zero."));
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::reinit(const dealii::DoFHandler<dim> *dof_handler_src_in,
                                    const dealii::DoFHandler<dim> *dof_handler_dst_in)
  {
    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope =
        std::make_unique<dealii::TimerOutput::Scope>(timer_output.value(),
                                                     ScopedName("nearest_point::reinit"));

    const bool signed_distance_update_ghosts = not signed_distance.has_ghost_elements();
    if (signed_distance_update_ghosts)
      signed_distance.update_ghost_values();

    clear_cached_data();

    register_dof_handlers(dof_handler_src_in, dof_handler_dst_in);

    collect_narrow_band_support_points();

    run_projection_algorithm();

    if (signed_distance_update_ghosts)
      signed_distance.zero_out_ghost_values();

    is_reinit_called = true;
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::reinit(const dealii::DoFHandler<dim>         &dof_handler,
                                    const std::vector<dealii::Point<dim>> &interface_points)
  {
    AssertThrow(nearest_point_data.type == NearestPointType::nearest_point_fast,
                dealii::ExcMessage("Providing interface points is only supported for the "
                                   "\"nearest_point_fast\" algorithm."));
    AssertThrow(CutUtil::get_cut_type(dof_handler) == CutUtil::CutPhaseType::not_cut,
                dealii::ExcMessage("The DoFHandler must not be set up for CutFEM!"));
    AssertThrow(dealii::Utilities::MPI::sum(interface_points.size(), mpi_comm) > 0,
                dealii::ExcMessage("No interface points provided."));

    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope =
        std::make_unique<dealii::TimerOutput::Scope>(timer_output.value(),
                                                     ScopedName("nearest_point::reinit"));

    const bool signed_distance_update_ghosts = not signed_distance.has_ghost_elements();
    if (signed_distance_update_ghosts)
      signed_distance.update_ghost_values();

    clear_cached_data();

    register_dof_handlers(&dof_handler, nullptr);

    collect_narrow_band_support_points();

    if (signed_distance_update_ghosts)
      signed_distance.zero_out_ghost_values();

    // set initial guess, which is overwritten by the nearest interface point
    projected_points_at_interface = stencil;

    setup_nearest_point_fast(interface_points);

    interface_points_from_user = true;
    is_reinit_called           = true;
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::clear_cached_data()
  {
    is_reinit_called           = false;
    input_vector_is_cut        = false;
    interface_points_from_user = false;
    stencil.clear();
    dof_indices.clear();
    projected_points_at_interface.clear();
    total_points_rpe = 0;
    points_not_found.clear();
    locally_owned_surface_indices.clear();
    ghost_surface_indices.clear();
    surface_cells_and_unit_points.clear();
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::register_dof_handlers(const dealii::DoFHandler<dim> *src,
                                                   const dealii::DoFHandler<dim> *dst)
  {
    AssertThrow(src != nullptr, dealii::ExcInternalError());
    dof_handler_src = src;

    const CutUtil::CutPhaseType cut_type = CutUtil::get_cut_type(*src);
    if (cut_type == CutUtil::CutPhaseType::not_cut)
      {
        Assert(dst == nullptr, dealii::ExcNotImplemented());
        dof_handler_dst     = nullptr;
        input_vector_is_cut = false;
      }
    else
      {
        AssertThrow(
          dst != nullptr,
          dealii::ExcMessage(
            "If the dof_handler_src is set up for CutFEM you must provide a dof_handler_dst "
            "that is set up for continuous FEM with the identical number of components as "
            "dof_handler_src!"));
        dof_handler_dst = dst;
        AssertThrow(CutUtil::get_cut_type(*dof_handler_dst) == CutUtil::CutPhaseType::not_cut,
                    dealii::ExcMessage("The destination DoFHandler must not be setup for CutFEM!"));
        input_vector_is_cut = true;
      }
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::collect_narrow_band_support_points()
  {
    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
        timer_output.value(),
        ScopedName("nearest_point::reinit::collect_narrow_band_support_points"));

    const auto &used_dst_dof_handler = input_vector_is_cut ? *dof_handler_dst : *dof_handler_src;
    const auto  n_components         = used_dst_dof_handler.get_fe().n_components();

    if (input_vector_is_cut)
      {
        unsigned int cut_n_comp = dof_handler_src->get_fe().n_components();
        if (CutUtil::get_cut_type(*dof_handler_src) == CutUtil::CutPhaseType::two_phase_cut)
          cut_n_comp /= 2;

        AssertThrow(n_components == cut_n_comp,
                    dealii::ExcMessage(
                      "Mismatch in number of components between src/dst DoFHandlers"));
      }

    const auto &unit_support_point =
      used_dst_dof_handler.get_fe().base_element(0).get_unit_support_points();

    dealii::FEValues<dim> signed_distance_eval(mapping,
                                               dof_handler_ls.get_fe(),
                                               dealii::Quadrature<dim>(unit_support_point),
                                               dealii::update_values);
    dealii::FEValues<dim> dst_eval(mapping,
                                   used_dst_dof_handler.get_fe(),
                                   dealii::Quadrature<dim>(unit_support_point),
                                   dealii::update_quadrature_points);

    // fill initial evaluation points with node coordinates (stencil)
    const unsigned int                           n_points = dst_eval.get_quadrature().size();
    const unsigned int                           n_dofs   = n_points * n_components;
    std::vector<number>                          signed_distance_values(n_points);
    std::vector<dealii::types::global_dof_index> local_dof_indices(n_dofs);

    typename dealii::DoFHandler<dim>::active_cell_iterator dst_cell =
      used_dst_dof_handler.begin_active();

    std::unordered_set<dealii::types::global_dof_index> point_processed{};

    const dealii::Utilities::MPI::Partitioner partitioner_support_points(
      used_dst_dof_handler.locally_owned_dofs(), used_dst_dof_handler.get_mpi_communicator());

    for (const auto &cell : dof_handler_ls.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            signed_distance_eval.reinit(cell);
            signed_distance_eval.get_function_values(signed_distance, signed_distance_values);

            dst_eval.reinit(dst_cell);
            dst_cell->get_dof_indices(local_dof_indices);

            for (const auto q : dst_eval.quadrature_point_indices())
              {
                // early return if point is outside of narrow band
                if (std::abs(signed_distance_values[q]) >= narrow_band_threshold)
                  continue;

                const unsigned int first_dof_index =
                  local_dof_indices[used_dst_dof_handler.get_fe().component_to_system_index(0, q)];
                // check only for first component
                if (not partitioner_support_points.in_local_range(first_dof_index))
                  continue;

                // early return if point was already collected in stencil
                if (point_processed.count(first_dof_index) > 0)
                  continue;

                point_processed.insert(first_dof_index);

                const auto &current_point = dst_eval.quadrature_point(q);
                stencil.emplace_back(current_point);

                // create a list of component-wise dof indices
                std::vector<dealii::types::global_dof_index> temp_local_dof_indices;
                for (unsigned int c = 0; c < n_components; ++c)
                  temp_local_dof_indices.emplace_back(
                    local_dof_indices[used_dst_dof_handler.get_fe().component_to_system_index(c,
                                                                                              q)]);
                dof_indices.emplace_back(temp_local_dof_indices);
              }
          }
        ++dst_cell;
      }

    if (nearest_point_data.verbosity_level > 0)
      Journal::print_line(pcout,
                          "no. stencil points: " +
                            std::to_string(dealii::Utilities::MPI::sum(stencil.size(), mpi_comm)),
                          "nearest_point");

    Assert(dealii::Utilities::MPI::sum(stencil.size(), mpi_comm) > 0,
           dealii::ExcMessage("number of points in narrow band equal to zero."));
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::run_projection_algorithm()
  {
    // set initial guess for closest point projection
    projected_points_at_interface = stencil;

    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope =
        std::make_unique<dealii::TimerOutput::Scope>(timer_output.value(),
                                                     ScopedName("nearest_point::reinit::project"));
    switch (nearest_point_data.type)
      {
        case NearestPointType::nearest_point:
          local_compute_nearest_point();
          break;
        case NearestPointType::nearest_point_fast:
          local_compute_nearest_point_fast();
          break;
        case NearestPointType::closest_point_normal:
          local_compute_normal_correction(projected_points_at_interface);
          break;
        case NearestPointType::closest_point_normal_collinear:
          local_compute_normal_and_tangential_correction(projected_points_at_interface);
          break;
        case NearestPointType::closest_point_normal_collinear_coquerelle:
          local_compute_normal_and_tangential_correction_coquerelle(projected_points_at_interface);
          break;
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

    if (nearest_point_data.verbosity_level > 0 &&
        nearest_point_data.type != NearestPointType::nearest_point &&
        nearest_point_data.type != NearestPointType::nearest_point_fast)
      {
        Journal::print_line(pcout,
                            "total number of RPE points: " +
                              std::to_string(
                                dealii::Utilities::MPI::sum(total_points_rpe, mpi_comm)),
                            "nearest_point");
      }

    if (nearest_point_data.type != NearestPointType::nearest_point_fast)
      {
        AssertThrow(dof_indices.size() == projected_points_at_interface.size(),
                    dealii::ExcMessage(
                      "Mismatched sizes between DOF indices and projected points."));
        std::unique_ptr<dealii::TimerOutput::Scope> timer_scope_local;
        if (timer_output)
          timer_scope_local = std::make_unique<dealii::TimerOutput::Scope>(
            timer_output.value(), ScopedName("nearest_point::reinit::project::rpe"));

        remote_point_evaluation.reinit(projected_points_at_interface,
                                       dof_handler_ls.get_triangulation(),
                                       mapping);
      }
  }

  template <int dim, typename number>
  const std::vector<dealii::Point<dim>> &
  NearestPoint<dim, number>::get_projected_points() const
  {
    AssertThrow(is_reinit_called, dealii::ExcMessage("You need to call reinit() first."));

    return projected_points_at_interface;
  }

  template <int dim, typename number>
  const std::vector<std::vector<dealii::types::global_dof_index>> &
  NearestPoint<dim, number>::get_projected_points_dof_indices() const
  {
    AssertThrow(is_reinit_called, dealii::ExcMessage("You need to call reinit() first."));

    return dof_indices;
  }

  template <int dim, typename number>
  template <int n_components>
  void
  NearestPoint<dim, number>::extend_interface_values(
    VectorType                                &solution_out,
    const VectorType                          &solution_in,
    const bool                                 zero_out,
    const std::function<number(const number)> &operation) const
  {
    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
        timer_output.value(), ScopedName("nearest_point::extend_interface_values"));

    if (not input_vector_is_cut)
      AssertThrow(n_components == dof_handler_src->get_fe().n_components(),
                  dealii::ExcMessage("There is a mismatch in the number of components "
                                     "between your passed DoFHandler and the template parameter."));
    else
      {
        AssertThrow(dof_handler_dst != nullptr, dealii::ExcInternalError());

        unsigned int n_comp = dof_handler_src->get_fe().n_components();
        // for two phase cut, dof_handler_src->get_fe().n_components() returns twice the number
        // of components
        if (CutUtil::get_cut_type(*dof_handler_src) == CutUtil::CutPhaseType::two_phase_cut)
          n_comp /= 2;
        AssertThrow(n_components == n_comp,
                    dealii::ExcMessage(
                      "There is a mismatch in the number of components "
                      "between your passed DoFHandler and the template parameter."));
        AssertThrow(n_components == dof_handler_dst->get_fe().n_components(),
                    dealii::ExcMessage(
                      "There is a mismatch in the number of components "
                      "between your passed DoFHandler and the template parameter."));
      }

    AssertThrow(is_reinit_called, dealii::ExcMessage("You need to call reinit() first."));
    AssertThrow(not interface_points_from_user,
                dealii::ExcMessage("reinit() was called with interface points. Use "
                                   "extend_interface_values() with interface values instead."));

    if (nearest_point_data.type == NearestPointType::nearest_point_fast)
      return extend_interface_values_nearest_point_fast<n_components>(solution_out,
                                                                      solution_in,
                                                                      zero_out,
                                                                      operation);
    const bool update_ghosts = not solution_in.has_ghost_elements();
    if (update_ghosts)
      solution_in.update_ghost_values();

    const auto vals = dealii::VectorTools::point_values<n_components>(remote_point_evaluation,
                                                                      *dof_handler_src,
                                                                      solution_in);
    // store interface values to vector
    if (zero_out)
      solution_out = 0.0;

    for (unsigned int i = 0; i < projected_points_at_interface.size(); ++i)
      {
        if constexpr (n_components > 1)
          {
            for (unsigned int d = 0; d < dof_indices[i].size(); ++d)
              if (solution_out.locally_owned_elements().is_element(dof_indices[i][d]))
                solution_out[dof_indices[i][d]] =
                  not operation ? vals[i][d] : operation(vals[i][d]);
          }
        else if (solution_out.locally_owned_elements().is_element(dof_indices[i][0]))
          solution_out[dof_indices[i][0]] = not operation ? vals[i] : operation(vals[i]);
      }

    if (update_ghosts)
      solution_in.zero_out_ghost_values();
  }

  template <int dim, typename number>
  template <int n_components>
  void
  NearestPoint<dim, number>::extend_interface_values(
    VectorType                                &solution_out,
    const dealii::ArrayView<const number>     &interface_values,
    const bool                                 zero_out,
    const std::function<number(const number)> &operation) const
  {
    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
        timer_output.value(), ScopedName("nearest_point::extend_interface_values"));

    AssertThrow(is_reinit_called, dealii::ExcMessage("You need to call reinit() first."));
    AssertThrow(interface_points_from_user,
                dealii::ExcMessage("You need to call reinit() with interface points first."));
    AssertThrow(n_components == dof_handler_src->get_fe().n_components(),
                dealii::ExcMessage("There is a mismatch in the number of components "
                                   "between your passed DoFHandler and the template parameter."));
    AssertThrow(interface_values.size() == locally_owned_surface_indices.size() * n_components,
                dealii::ExcDimensionMismatch(interface_values.size(),
                                             locally_owned_surface_indices.size() * n_components));

    if (zero_out)
      solution_out = 0.0;

    scatter_interface_values_to_stencil<n_components>(solution_out, interface_values, operation);
  }

  template <int dim, typename number>
  template <int n_components>
  void
  NearestPoint<dim, number>::extend_interface_values_nearest_point_fast(
    VectorType                                &solution_out,
    const VectorType                          &solution_in,
    const bool                                 zero_out,
    const std::function<number(const number)> &operation) const
  {
    // Ensure that the nearest point evaluation type is the expected fast version.
    Assert(nearest_point_data.type == NearestPointType::nearest_point_fast,
           dealii::ExcNotImplemented());

    // store interface values to vector
    if (zero_out)
      solution_out = 0.0;

    const bool update_ghosts = !solution_in.has_ghost_elements();
    if (update_ghosts)
      solution_in.update_ghost_values();

    // Create a finite element point evaluator with the base (non-active) FE.
    std::unique_ptr<dealii::FEPointEvaluation<n_components, dim>> fep =
      std::make_unique<dealii::FEPointEvaluation<n_components, dim>>(mapping,
                                                                     dof_handler_src->get_fe(),
                                                                     dealii::update_values);

    // Reserve space for values extracted from source DoFHandler.
    std::vector<number> values_src;
    values_src.reserve(locally_owned_surface_indices.size() * n_components);

    // Temporary vector for DoF values per cell.
    dealii::Vector<number> solution_values(dof_handler_src->get_fe().n_dofs_per_cell());

    // Loop over all surface cells and corresponding unit points on them.
    for (const auto &cell_and_points : surface_cells_and_unit_points)
      {
        const auto                                                    &cell = cell_and_points.first;
        dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>> dealii_cell(
          &dof_handler_src->get_triangulation(), cell.first, cell.second, dof_handler_src);

        // If we are using an hpDoFHandler for cut cells, use the correct active FE.
        if (input_vector_is_cut)
          {
            const dealii::types::fe_index active_fe_index =
              *dealii_cell->get_active_fe_indices().begin();
            fep = std::make_unique<dealii::FEPointEvaluation<n_components, dim>>(
              mapping,
              dof_handler_src->get_fe(active_fe_index),
              dealii::update_values,
              0 /*first selected component*/);
            solution_values.reinit(dof_handler_src->get_fe(active_fe_index).n_dofs_per_cell());
          }

        dealii_cell->get_dof_values(solution_in, solution_values);

        fep->reinit(dealii_cell, cell_and_points.second);
        fep->evaluate(solution_values, dealii::EvaluationFlags::values);

        for (const auto q : fep->quadrature_point_indices())
          {
            const auto val = fep->get_value(q);
            if constexpr (n_components > 1)
              for (unsigned int c = 0; c < n_components; ++c)
                values_src.push_back(val[c]);
            else
              values_src.push_back(val);
          }
      }

    AssertDimension(locally_owned_surface_indices.size() * n_components, values_src.size());

    scatter_interface_values_to_stencil<n_components>(solution_out,
                                                      dealii::make_array_view(values_src),
                                                      operation);

    if (update_ghosts)
      solution_in.zero_out_ghost_values();
  }

  template <int dim, typename number>
  template <int n_components>
  void
  NearestPoint<dim, number>::scatter_interface_values_to_stencil(
    VectorType                                &solution_out,
    const dealii::ArrayView<const number>     &values_src,
    const std::function<number(const number)> &operation) const
  {
    AssertDimension(ghost_surface_indices.size(), dof_indices.size());

    // Prepare buffer for ghost values on destination side.
    std::vector<number> values_dst(ghost_surface_indices.size() * n_components);

    // Export interpolated values from local to ghosted layout.
    partitioner.export_to_ghosted_array<number, n_components>(values_src,
                                                              dealii::make_array_view(values_dst));

    // Scatter the interpolated values into the output vector.
    // Optionally apply an operation to the values before assigning.
    for (unsigned int i = 0; i < ghost_surface_indices.size(); ++i)
      for (unsigned int d = 0; d < dof_indices[i].size(); ++d)
        if (solution_out.locally_owned_elements().is_element(dof_indices[i][d]))
          solution_out[dof_indices[i][d]] = not operation ?
                                              values_dst[i * n_components + d] :
                                              operation(values_dst[i * n_components + d]);
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::write_to_file(const std::string filename) const
  {
    const auto global_points_normal_to_interface_all =
      dealii::Utilities::MPI::reduce<std::vector<dealii::Point<dim>>>(
        projected_points_at_interface, mpi_comm, [](const auto &a, const auto &b) {
          auto result = a;
          result.insert(result.end(), b.begin(), b.end());
          return result;
        });

    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        std::ofstream myfile;
        myfile.open(filename + ".txt");

        for (const auto &p : global_points_normal_to_interface_all)
          {
            for (unsigned int d = 0; d < dim; ++d)
              myfile << p[d] << " ";
            myfile << std::endl;
          }

        myfile.close();
      }
    {
      const auto all_points_not_found =
        dealii::Utilities::MPI::reduce<std::vector<dealii::Point<dim>>>(
          points_not_found, mpi_comm, [](const auto &a, const auto &b) {
            auto result = a;
            result.insert(result.end(), b.begin(), b.end());
            return result;
          });

      if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0 and
          all_points_not_found.size() > 0)
        {
          std::ofstream myfile;
          myfile.open(filename + "_not_found.txt");

          for (const auto &p : all_points_not_found)
            {
              for (unsigned int d = 0; d < dim; ++d)
                myfile << p[d] << " ";

              myfile << std::endl;
            }

          myfile.close();
        }
    }
  }

  template <int dim, typename number>
  std::vector<dealii::Point<dim>>
  NearestPoint<dim, number>::evaluate_unit_normals() const
  {
    std::vector<dealii::Point<dim>> unit_normals;

    if (normal_vector)
      {
        std::array<std::vector<number>, dim> evaluation_values_normal;

        const bool normal_vector_update_ghosts = not normal_vector->has_ghost_elements();
        if (normal_vector_update_ghosts)
          normal_vector->update_ghost_values();
        for (int comp = 0; comp < dim; ++comp)
          {
            evaluation_values_normal[comp] =
              dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                   dof_handler_ls,
                                                   normal_vector->block(comp));
          }

        if (normal_vector_update_ghosts)
          normal_vector->zero_out_ghost_values();

        unit_normals.resize(evaluation_values_normal[0].size());
        for (unsigned int counter = 0; counter < unit_normals.size(); ++counter)
          for (unsigned int comp = 0; comp < dim; ++comp)
            unit_normals[counter][comp] = evaluation_values_normal[comp][counter];
      }
    else
      {
        // fall back to the gradient of the signed distance function
        const auto evaluation_values_gradient =
          dealii::VectorTools::point_gradients<1>(remote_point_evaluation,
                                                  dof_handler_ls,
                                                  signed_distance);

        unit_normals.reserve(evaluation_values_gradient.size());
        for (const auto &gradient : evaluation_values_gradient)
          unit_normals.emplace_back(gradient);
      }

    for (auto &unit_normal : unit_normals)
      {
        const auto n_norm = unit_normal.norm();
        unit_normal =
          n_norm > tolerance_normal_vector ? unit_normal / n_norm : dealii::Point<dim>();
      }

    return unit_normals;
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::reinit_remote_point_evaluation(
    const std::vector<dealii::Point<dim>> &points)
  {
    total_points_rpe += points.size();

    remote_point_evaluation.reinit(points, dof_handler_ls.get_triangulation(), mapping);

    if (not remote_point_evaluation.all_points_found())
      {
        for (unsigned int i = 0; i < points.size(); ++i)
          if (not remote_point_evaluation.point_found(i))
            points_not_found.emplace_back(points[i]);

        write_to_file("unmatched_points");

        AssertThrow(false, dealii::ExcMessage("Processed point is outside domain."));
      }
  }

  template <int dim, typename number>
  std::vector<number>
  NearestPoint<dim, number>::evaluate_distances() const
  {
    std::vector<number> distances = dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                                         dof_handler_ls,
                                                                         signed_distance);

    // shift isocontour
    for (auto &e : distances)
      e -= nearest_point_data.isocontour;

    return distances;
  }

  template <int dim, typename number>
  bool
  NearestPoint<dim, number>::local_compute_normal_correction(std::vector<dealii::Point<dim>> &y)
  {
    // points that are not (yet) at the interface and still needs to be processed
    std::vector<unsigned int> unmatched_points_idx(y.size());
    std::iota(unmatched_points_idx.begin(), unmatched_points_idx.end(), 0);
    int n_unmatched_points = dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

    // temporary variable for signed distance
    std::vector<number> evaluation_values_distance;

    for (int j = 0; j < nearest_point_data.max_iter; ++j)
      {
        std::vector<dealii::Point<dim>> unmatched_points(unmatched_points_idx.size());

        for (unsigned int counter = 0; counter < unmatched_points_idx.size(); ++counter)
          unmatched_points[counter] = y[unmatched_points_idx[counter]];

        std::ostringstream str;
        str << "     j=" << j << " (normal) " << n_unmatched_points;

        reinit_remote_point_evaluation(unmatched_points);

        evaluation_values_distance = evaluate_distances();

        const std::vector<dealii::Point<dim>> evaluation_values_unit_normal =
          evaluate_unit_normals();

        std::vector<unsigned int> unmatched_points_idx_next;

        for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
          {
            // skip point where the distance to the interface is already close enough
            if (std::abs(evaluation_values_distance[counter]) <= tol_distance)
              continue;

            // compute closest point and update value in global vector
            unmatched_points[counter] -=
              evaluation_values_distance[counter] * evaluation_values_unit_normal[counter];

            y[unmatched_points_idx[counter]] = unmatched_points[counter];

            unmatched_points_idx_next.emplace_back(unmatched_points_idx[counter]);
          }

        // remove points from processing that are already at the interface
        unmatched_points_idx.swap(unmatched_points_idx_next);

        // if every point is close enough to the interface, we are finished
        n_unmatched_points = dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

        str << " -> " << n_unmatched_points << " (✗)";
        if (nearest_point_data.verbosity_level > 1)
          Journal::print_line(pcout, str.str(), "nearest_point", 2);

        if (n_unmatched_points == 0)
          break;
      }

    if (n_unmatched_points > 0)
      {
        pcout << "WARNING: The tolerance of " << n_unmatched_points
              << " points is not yet attained. Max distance value: "
              << dealii::Utilities::MPI::max(internal::max_abs(evaluation_values_distance),
                                             mpi_comm)
              << std::endl;
        return false;
      }
    return true;
  }

  template <int dim, typename number>
  bool
  NearestPoint<dim, number>::local_compute_normal_and_tangential_correction_coquerelle(
    std::vector<dealii::Point<dim>> &y)
  {
    AssertThrow(dim == 2 or dim == 3,
                dealii::ExcMessage("Use local_compute_normal_correction for dim==1."));

    total_points_rpe = 0;

    // 0) Perform correction in normal direction
    local_compute_normal_correction(y);

    // points that are not (yet) at the interface and still need to be processed
    std::vector<unsigned int> unmatched_points_idx(y.size());
    std::iota(unmatched_points_idx.begin(), unmatched_points_idx.end(), 0);

    int n_unmatched_points = dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

    number     max_tangential_distance = 0.0;
    const bool normal_vector_update_ghosts =
      normal_vector != nullptr and not normal_vector->has_ghost_elements();
    if (normal_vector_update_ghosts)
      normal_vector->update_ghost_values();

    // 1) Perform correction in tangential direction
    for (int it = 0; it < nearest_point_data.max_iter and n_unmatched_points > 0; ++it)
      {
        std::vector<dealii::Point<dim>> unmatched_points(unmatched_points_idx.size());
        for (unsigned int i = 0; i < unmatched_points_idx.size(); ++i)
          unmatched_points[i] = y[unmatched_points_idx[i]];

        std::ostringstream str;
        str << " i=" << it << " (tangent) ";
        str << n_unmatched_points << " -> ";
        if (nearest_point_data.verbosity_level > 1)
          Journal::print_line(pcout, str.str(), "nearest_point");

        reinit_remote_point_evaluation(unmatched_points);

        const std::vector<dealii::Point<dim>> evaluation_values_unit_normal =
          evaluate_unit_normals();

        // compute the tangential correction for each point
        std::vector<unsigned int> unmatched_points_idx_next;

        for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
          {
            const auto unit_tangents =
              internal::compute_unit_tangents(evaluation_values_unit_normal[counter]);

            // determine tangential offset for each direction
            const auto omega =
              internal::compute_tangential_offsets(unmatched_points[counter],
                                                   stencil[unmatched_points_idx[counter]],
                                                   unit_tangents);

            const number max_omega = internal::max_abs(omega);

            if (max_omega <= tol_distance)
              continue; // no need to perform a tangential corection

            max_tangential_distance = std::max(max_omega, max_tangential_distance);

            // correct point by tangential offset
            for (unsigned int d = 0; d < dim - 1; ++d)
              unmatched_points[counter] -= omega[d] * unit_tangents[d];

            unmatched_points_idx_next.emplace_back(unmatched_points_idx[counter]);
          }
        // 2) Proceed with the correction in normal direction
        local_compute_normal_correction(unmatched_points);

        // update points in global vector
        for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
          y[unmatched_points_idx[counter]] = unmatched_points[counter];

        // remove points from processing that are already at the interface
        unmatched_points_idx.swap(unmatched_points_idx_next);

        // if every point is close enough to the interface, we are finished
        n_unmatched_points = dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

        if (n_unmatched_points == 0)
          break;
      }

    if (normal_vector_update_ghosts)
      normal_vector->zero_out_ghost_values();

    max_tangential_distance = dealii::Utilities::MPI::max(max_tangential_distance, mpi_comm);

    if (n_unmatched_points > 0)
      {
        pcout << "WARNING: The tolerance of the tangential correction of " << n_unmatched_points
              << " points is not yet attained. Max tangential distance value: "
              << max_tangential_distance << " max tolerance: " << tol_distance << std::endl;
        return false;
      }
    return true;
  }

  template <int dim, typename number>
  bool
  NearestPoint<dim, number>::local_compute_normal_and_tangential_correction(
    std::vector<dealii::Point<dim>> &y)
  {
    AssertThrow(dim == 2 or dim == 3,
                dealii::ExcMessage("Use local_compute_normal_correction for dim==1."));

    total_points_rpe = 0;

    // points that are not (yet) at the interface and still needs to be processed
    std::vector<unsigned int> unmatched_points_idx(y.size());
    std::iota(unmatched_points_idx.begin(), unmatched_points_idx.end(), 0);

    // temporary variable for signed distance
    std::vector<number>       evaluation_values_distance;
    std::vector<unsigned int> unmatched_points_normal_idx_next;

    for (int k = 0; k < nearest_point_data.max_iter; ++k)
      {
        // correct entire points
        number max_tangential_distance = 0;
        int n_unmatched_points = dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

        if (n_unmatched_points == 0)
          break;

        std::vector<unsigned int> unmatched_points_normal_and_tangential_idx_next;

        {
          std::ostringstream str;
          str << " k=" << k << " -> " << n_unmatched_points;
          if (nearest_point_data.verbosity_level > 1)
            Journal::print_line(pcout, str.str(), "nearest_point");
        }

        // correct only unmatched points
        for (int j = 0; j < nearest_point_data.max_iter; ++j)
          {
            std::vector<dealii::Point<dim>> unmatched_points(unmatched_points_idx.size());
            for (unsigned int i = 0; i < unmatched_points_idx.size(); ++i)
              unmatched_points[i] = y[unmatched_points_idx[i]];

            // just for output purposes
            std::ostringstream str;
            str << "   j=" << j << " " << n_unmatched_points << " -> ";

            reinit_remote_point_evaluation(unmatched_points);

            evaluation_values_distance = evaluate_distances();

            const std::vector<dealii::Point<dim>> evaluation_values_unit_normal =
              evaluate_unit_normals();

            unmatched_points_normal_idx_next.clear();

            unsigned int n_complete = 0;

            for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
              {
                // perform tangential correction where normal correction was successful
                if (std::abs(evaluation_values_distance[counter]) <= tol_distance)
                  {
                    const auto unit_tangents =
                      internal::compute_unit_tangents(evaluation_values_unit_normal[counter]);

                    // determine tangential offset for each direction
                    const auto omega =
                      internal::compute_tangential_offsets(unmatched_points[counter],
                                                           stencil[unmatched_points_idx[counter]],
                                                           unit_tangents);

                    const number max_omega = internal::max_abs(omega);

                    max_tangential_distance = std::max(max_omega, max_tangential_distance);

                    if (max_omega > tol_distance)
                      {
                        unmatched_points_normal_and_tangential_idx_next.emplace_back(
                          unmatched_points_idx[counter]);

                        // correct point by tangential offset
                        for (unsigned int d = 0; d < dim - 1; ++d)
                          y[unmatched_points_idx[counter]] -= omega[d] * unit_tangents[d];
                      }
                    else
                      n_complete += 1;
                  }
                else
                  {
                    unmatched_points[counter] -=
                      evaluation_values_distance[counter] * evaluation_values_unit_normal[counter];

                    y[unmatched_points_idx[counter]] = unmatched_points[counter];

                    unmatched_points_normal_idx_next.emplace_back(unmatched_points_idx[counter]);
                  }
              }

            n_complete = dealii::Utilities::MPI::sum(n_complete, mpi_comm);

            // remove points from processing that are already at the interface
            unmatched_points_idx.swap(unmatched_points_normal_idx_next);

            // if every point is close enough to the interface, we are finished
            n_unmatched_points = dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

            str << n_unmatched_points << " (n ✗) "
                << dealii::Utilities::MPI::sum(
                     unmatched_points_normal_and_tangential_idx_next.size(), mpi_comm)
                << " (n ✓ | t ✗) " << n_complete << " (n ✓ | t ✓) ";
            if (nearest_point_data.verbosity_level > 1)
              Journal::print_line(pcout, str.str(), "nearest_point", 10 /*special characters*/);

            if (n_unmatched_points == 0)
              break;
          }

        std::copy(unmatched_points_normal_idx_next.begin(),
                  unmatched_points_normal_idx_next.end(),
                  std::back_inserter(unmatched_points_normal_and_tangential_idx_next));
        std::sort(unmatched_points_normal_and_tangential_idx_next.begin(),
                  unmatched_points_normal_and_tangential_idx_next.end());

        unmatched_points_normal_and_tangential_idx_next.erase(
          std::unique(unmatched_points_normal_and_tangential_idx_next.begin(),
                      unmatched_points_normal_and_tangential_idx_next.end()),
          unmatched_points_normal_and_tangential_idx_next.end());

        unmatched_points_idx.swap(unmatched_points_normal_and_tangential_idx_next);

        if (k < nearest_point_data.max_iter - 1)
          continue;

        if (n_unmatched_points > 0)
          {
            pcout << "WARNING: The tolerance of " << n_unmatched_points
                  << " points is not yet attained. Max distance value: "
                  << dealii::Utilities::MPI::max(internal::max_abs(evaluation_values_distance),
                                                 mpi_comm)
                  << std::endl;
            return false;
          }

        max_tangential_distance = dealii::Utilities::MPI::max(max_tangential_distance, mpi_comm);

        if (max_tangential_distance > tol_distance)
          {
            pcout << "WARNING: The tolerance of the tangential correction of "
                  << dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm)
                  << " points is not yet attained. Max tangential distance value: "
                  << max_tangential_distance << " max tolerance: " << tol_distance << std::endl;
            return false;
          }
      }

    return true;
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::local_compute_nearest_point()
  {
    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
        timer_output.value(), ScopedName("nearest_point::reinit::project::mca"));
    // create a point cloud of the surface
    dealii::GridTools::MarchingCubeAlgorithm<dim, VectorType> mc(
      mapping,
      dof_handler_ls.get_fe(),
      nearest_point_data.mca.n_subdivisions, /*n subdivisions*/
      nearest_point_data.mca.tolerance /*tolerance*/);

    std::vector<dealii::Point<dim>> surface_points;
    mc.process(dof_handler_ls, signed_distance, nearest_point_data.isocontour, surface_points);

    // all gather surface points
    const auto surface_points_global = dealii::Utilities::MPI::all_gather(mpi_comm, surface_points);

    // TODO: find a faster way to get a single vector for all processes
    surface_points.clear();
    for (unsigned int i = 0; i < surface_points_global.size(); ++i)
      for (unsigned int j = 0; j < surface_points_global[i].size(); ++j)
        surface_points.emplace_back(surface_points_global[i][j]);

    if (timer_scope)
      timer_scope->stop();

    if (timer_output)
      timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
        timer_output.value(), ScopedName("nearest_point::reinit::project::search"));

    const auto used_vertices_rtree = pack_rtree(surface_points);

    if (not used_vertices_rtree.empty())
      {
        // search for nearest point
        for (unsigned int i = 0; i < stencil.size(); ++i)
          {
            std::vector<dealii::Point<dim>> closest_vertex_in_domain;
            used_vertices_rtree.query(boost::geometry::index::nearest(stencil[i], 1),
                                      std::back_inserter(closest_vertex_in_domain));

            AssertThrow(closest_vertex_in_domain.size() == 1,
                        dealii::ExcMessage("The number of nearest points is wrong."));

            projected_points_at_interface[i] = closest_vertex_in_domain[0];
          }
      }
    if (timer_scope)
      timer_scope->stop();
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::local_compute_nearest_point_fast()
  {
    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
        timer_output.value(), ScopedName("nearest_point::reinit::project::mca"));

    // Storage for physical interface points extracted from the level set isocontour
    std::vector<dealii::Point<dim>> surface_points;

    collect_interface_cells_and_intersection_points<dim, number>(
      surface_cells_and_unit_points,
      surface_points,
      dof_handler_ls,
      mapping,
      signed_distance,
      nearest_point_data.isocontour,
      nearest_point_data.mca.n_subdivisions,
      nearest_point_data.mca.tolerance);

    if (timer_scope)
      timer_scope->stop();

    setup_nearest_point_fast(surface_points);
  }

  template <int dim, typename number>
  void
  NearestPoint<dim, number>::setup_nearest_point_fast(
    const std::vector<dealii::Point<dim>> &surface_points)
  {
#ifdef DEAL_II_WITH_ARBORX
    std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
    if (timer_output)
      timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
        timer_output.value(), ScopedName("nearest_point::reinit::project::search"));

    // Build ArborX distributed search tree using collected surface points
    dealii::ArborXWrappers::DistributedTree distributed_tree(mpi_comm, surface_points);

    // Construct a nearest-neighbor search for each stencil point, looking for 1 nearest point
    dealii::ArborXWrappers::PointNearestPredicate bb_near(stencil, 1);
    // Perform nearest neighbor search; returns matching local indices of surface points and owning
    // ranks
    const auto &[indices_and_ranks, offsets_stencil] = distributed_tree.query(bb_near);

    // Sanity check: ArborX gives 1 match per stencil point
    for (unsigned int i = 1; i < offsets_stencil.size(); ++i)
      Assert(offsets_stencil[i] - offsets_stencil[i - 1] == 1,
             dealii::ExcMessage("The offset needs to be one"));

    const auto [offset, n_global_surface_points] =
      dealii::Utilities::MPI::partial_and_total_sum(surface_points.size(), mpi_comm);
    const auto offsets = dealii::Utilities::MPI::all_gather(mpi_comm, offset);

    // Locally owned indices (global) of surface points owned by this process
    locally_owned_surface_indices.clear();
    for (auto i = offset; i < offset + surface_points.size(); ++i)
      locally_owned_surface_indices.push_back(i);

    // Ghost indices (global) corresponding to matched surface points for each locally owned stencil
    // point
    ghost_surface_indices.clear();
    for (const auto &[index, rank] : indices_and_ranks)
      ghost_surface_indices.push_back(offsets[rank] + index);

    // Set up communication pattern between local and ghost surface point data
    partitioner.reinit(locally_owned_surface_indices, ghost_surface_indices, mpi_comm);

    // Prepare data to export surface point coordinates to ghost processes
    // and fill projected_points_at_interface

    // Break surface_points into separate x/y/z (or dim-wise) components
    std::vector<std::vector<double>> components(dim, std::vector<double>(surface_points.size()));
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < surface_points.size(); ++i)
        components[d][i] = surface_points[i][d];

    // Communicate each component array using the partitioner
    std::vector<std::vector<double>> ghosted_components(
      dim, std::vector<double>(ghost_surface_indices.size()));
    for (unsigned int d = 0; d < dim; ++d)
      partitioner.export_to_ghosted_array<double>(dealii::make_array_view(components[d]),
                                                  dealii::make_array_view(ghosted_components[d]));

    // Reconstruct Point<dim> structures from ghosted components
    std::vector<dealii::Point<dim>> points_dst(ghost_surface_indices.size());
    for (unsigned int i = 0; i < ghost_surface_indices.size(); ++i)
      for (unsigned int d = 0; d < dim; ++d)
        points_dst[i][d] = ghosted_components[d][i];

    // Store the projected interface points in the class member variable
    for (unsigned int i = 0; i < ghost_surface_indices.size(); ++i)
      projected_points_at_interface[i] = points_dst[i];

    if (timer_scope)
      timer_scope->stop();
#else
    (void)surface_points;
    AssertThrow(
      false,
      dealii::ExcMessage(
        "deal.II is not set up with Arborx, which is a prerequisite for using the \"nearest_point_fast\" algorithm!"));
#endif
  }

  template class NearestPoint<1, double>;
  template class NearestPoint<2, double>;
  template class NearestPoint<3, double>;

  template void
  NearestPoint<1, double>::extend_interface_values<1>(
    VectorType &,
    const VectorType &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<2, double>::extend_interface_values<1>(
    VectorType &,
    const VectorType &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<2, double>::extend_interface_values<2>(
    VectorType &,
    const VectorType &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<3, double>::extend_interface_values<1>(
    VectorType &,
    const VectorType &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<3, double>::extend_interface_values<2>(
    VectorType &,
    const VectorType &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<3, double>::extend_interface_values<3>(
    VectorType &,
    const VectorType &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<1, double>::extend_interface_values<1>(
    VectorType &,
    const dealii::ArrayView<const double> &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<2, double>::extend_interface_values<1>(
    VectorType &,
    const dealii::ArrayView<const double> &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<2, double>::extend_interface_values<2>(
    VectorType &,
    const dealii::ArrayView<const double> &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<3, double>::extend_interface_values<1>(
    VectorType &,
    const dealii::ArrayView<const double> &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<3, double>::extend_interface_values<2>(
    VectorType &,
    const dealii::ArrayView<const double> &,
    const bool,
    const std::function<double(const double)> &) const;
  template void
  NearestPoint<3, double>::extend_interface_values<3>(
    VectorType &,
    const dealii::ArrayView<const double> &,
    const bool,
    const std::function<double(const double)> &) const;
} // namespace MeltPoolDG::LevelSet::Tools
