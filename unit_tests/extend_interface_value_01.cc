#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// uncomment to activate debug output
// #define MPDG_TEST_ENABLE_DEBUG

using namespace dealii;
using namespace MeltPoolDG;

using VectorType      = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;


// Initial fields
template <int dim, unsigned int n_components = 1>
class InitializeTemperature : public Function<dim>
{
public:
  InitializeTemperature()
    : Function<dim>(n_components)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    const double T_max = 500;
    const double T_min = 0;
    return T_max - (T_max - T_min) * p[component];
  }
};

template <int dim>
Point<dim>
compute_projected_point_analytically(const Point<dim> &p,
                                     const Point<dim> &center,
                                     const double      radius)
{
  const Tensor<1, dim> vec = p - center;
  const double         r   = vec.norm();

  if (std::abs(r) < 1e-10)
    return Point<dim>(0);

  Point<dim> p_proj;

  if constexpr (dim == 2 || dim == 3)
    {
      for (unsigned int d = 0; d < dim; ++d)
        p_proj[d] = radius * vec[d] / r;
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  return p_proj + center;
}

template <int n_components, int n_refinements, int n_iter, int dim = 2>
void
run_test(const LevelSet::NearestPointType type = LevelSet::NearestPointType::closest_point_normal)
{
  std::cout.precision(8);
  MPI_Comm                   mpi_comm(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  pcout << "------------------------------------------------------------------" << std::endl;
  pcout << " START TEST: n_iter=" << n_iter << " type " << type << " #comp " << n_components
        << " #refine " << n_refinements << " dim " << dim << std::endl;
  pcout << "------------------------------------------------------------------" << std::endl;

  Timer      timer_total(mpi_comm);
  const int  degree      = 1;
  const int  degree_temp = 2;
  Point<dim> center;
  for (unsigned int d = 0; d < dim; ++d)
    center[d] = 0.5;

  const double radius = 0.25;


  parallel::distributed::Triangulation<dim> triangulation(mpi_comm);

  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refinements);

  // Output mesh per processor
  MappingQGeneric<dim> mapping(degree);
  FE_Q<dim>            fe(degree);
  DoFHandler<dim>      dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // setup level set related vectors
  IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  VectorType solution_distance;
  solution_distance.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

  // compute normals
  BlockVectorType solution_normal(dim);

  for (unsigned int d = 0; d < dim; ++d)
    solution_normal.block(d).reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

  FEValues<dim> normal_eval(mapping,
                            fe,
                            Quadrature<dim>(fe.get_unit_support_points()),
                            update_quadrature_points);

  const unsigned int                   dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);

          normal_eval.reinit(cell);

          for (const auto q : normal_eval.quadrature_point_indices())
            {
              Point<dim> normal_vec;
              normal_vec = normal_eval.quadrature_point(q) - center;
              normal_vec /= normal_vec.norm();

              // auto idx = fe_dim.system_to_component_index(q);
              for (unsigned int d = 0; d < dim; ++d)
                solution_normal.block(d)[local_dof_indices[q]] = normal_vec[d];
            }
        }
    }
  // interpolate distance onto quadrature points
  dealii::VectorTools::interpolate(mapping,
                                   dof_handler,
                                   Functions::SignedDistance::Sphere<dim>(center, radius),
                                   solution_distance);

  // setup temperature field
  DoFHandler<dim> dof_handler_temp(triangulation);
  if constexpr (n_components == 1)
    {
      FE_Q<dim> fe_temp(degree_temp);
      dof_handler_temp.distribute_dofs(fe_temp);
    }
  else
    {
      FESystem<dim> fe_temp(FE_Q<dim>(degree_temp), n_components);
      dof_handler_temp.distribute_dofs(fe_temp);
    }

  VectorType solution_temp, solution_temp_interface;
  locally_owned_dofs    = dof_handler_temp.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_temp);

  solution_temp.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  solution_temp_interface.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  // interpolate temperature onto quadrature points
  dealii::VectorTools::interpolate(mapping,
                                   dof_handler_temp,
                                   InitializeTemperature<dim, n_components>(),
                                   solution_temp);
  // ------------------------------------------------------------------------------------
  // ------------------------------------------------------------------------------------
  //      START OF ACTUAL ALGORITHM
  // ------------------------------------------------------------------------------------
  // ------------------------------------------------------------------------------------
  LevelSet::NearestPointData<double> nearest_point_data;

  nearest_point_data.rel_tol               = 1e-10;
  nearest_point_data.narrow_band_threshold = 0.1;
  nearest_point_data.max_iter              = n_iter;
  nearest_point_data.type                  = type;
  nearest_point_data.verbosity_level       = 3;

  TimerOutput timer(pcout, TimerOutput::never, TimerOutput::wall_times);

  std::vector<Point<dim>> all_marked_vertices;

  // restrict RemotePointEvaluation to cells which are adjacent to the marked
  // vertices
  const auto marked_vertices = [&]() -> std::vector<bool> {
    all_marked_vertices.clear();
    std::vector<bool> marked_vertices(triangulation.n_vertices(), false);

    Vector<double> local_signed_distance;
    local_signed_distance.reinit(dof_handler.get_fe().n_dofs_per_cell());

    VectorType mark_cells;
    mark_cells.reinit(triangulation.global_active_cell_index_partitioner().lock());
    mark_cells = 0.0;

    const bool update_ghosts = !solution_distance.has_ghost_elements();

    if (update_ghosts)
      solution_distance.update_ghost_values();

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        cell->get_dof_values(solution_distance, local_signed_distance);

        std::sort(local_signed_distance.begin(), local_signed_distance.end());

        // check if level-set values are below narrow_band_threshold
        if ((std::abs(local_signed_distance[local_signed_distance.size() - 1]) <=
             nearest_point_data.narrow_band_threshold) &&
            (std::abs(local_signed_distance[0]) <= nearest_point_data.narrow_band_threshold))
          mark_cells[cell->global_active_cell_index()] = 1;
      }

    if (update_ghosts)
      solution_distance.zero_out_ghost_values();

    mark_cells.update_ghost_values();

    for (const auto &cell : triangulation.active_cell_iterators())
      if (!cell->is_artificial())
        {
          if (mark_cells[cell->global_active_cell_index()] > 0)
            for (const auto v : cell->vertex_indices())
              {
                marked_vertices[cell->vertex_index(v)] = true;
                all_marked_vertices.emplace_back(cell->vertex(v));
              }
        }

    all_marked_vertices = Utilities::MPI::reduce<std::vector<Point<dim>>>(
      all_marked_vertices, mpi_comm, [](const auto &a, const auto &b) {
        auto result = a;
        result.insert(result.end(), b.begin(), b.end());
        return result;
      });

    return marked_vertices;
  };

  const typename Utilities::MPI::RemotePointEvaluation<dim, dim>::AdditionalData additional_data(
    1e-6 /*tolerance*/, true /*unique mapping*/, 0 /*rtree_level=*/, marked_vertices);
  Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(additional_data);

  const double min_cell_size = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(dim);

  timer_total.start();

  // 1) compute accuracy of projected points
  LevelSet::Tools::NearestPoint<dim, double> cpp(
    mapping,
    dof_handler,
    solution_distance,
    solution_normal,
    remote_point_evaluation,
#ifdef MPDG_TEST_ENABLE_DEBUG
    nearest_point_data,
    std::optional<std::reference_wrapper<dealii::TimerOutput>>{timer});
#else
    nearest_point_data);
#endif

  cpp.reinit(&dof_handler_temp);

#ifdef MPDG_TEST_ENABLE_DEBUG
  {
    std::string output = "n_components=" + std::to_string(n_components) +
                         "_n_iter=" + std::to_string(n_iter) + "_" + type._to_string() +
                         "_dim=" + std::to_string(dim) + "_output";
    cpp.write_to_file("interface_points_" + output);
  }
#endif

  const auto evaluation_points = cpp.get_projected_points();

  const auto dof_indices = cpp.get_projected_points_dof_indices();

  FEValues<dim> req_values(mapping,
                           dof_handler_temp.get_fe(),
                           Quadrature<dim>(dof_handler_temp.get_fe().get_unit_support_points()),
                           update_quadrature_points);

  const unsigned int n_q_points = req_values.get_quadrature().size();

  std::vector<types::global_dof_index> temp_local_dof_indices(n_q_points);

  std::map<types::global_dof_index, Point<dim>> quad_points;

  for (const auto &cell : dof_handler_temp.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          req_values.reinit(cell);
          cell->get_dof_indices(temp_local_dof_indices);

          for (const auto q : req_values.quadrature_point_indices())
            quad_points[temp_local_dof_indices[q]] = req_values.quadrature_point(q);
        }
    }

  double max_dist = 0;
  for (unsigned int i = 0; i < evaluation_points.size(); ++i)
    {
      max_dist = std::max(max_dist,
                          compute_projected_point_analytically<dim>(quad_points[dof_indices[i][0]],
                                                                    center,
                                                                    0.25)
                            .distance(evaluation_points[i]));
      if (false)
        {
          std::cout << "Point: " << quad_points[dof_indices[i][0]] << std::endl;
          std::cout << "analytical project "
                    << compute_projected_point_analytically<dim>(quad_points[dof_indices[i][0]],
                                                                 center,
                                                                 0.25)
                    << std::endl;
          std::cout << "numerical project" << evaluation_points[i] << std::endl;
        }
    }

  max_dist = Utilities::MPI::max(max_dist, mpi_comm);
  pcout << "max distance to analytical solution (relative): " << max_dist / min_cell_size
        << std::endl;
  {
    if (nearest_point_data.type == LevelSet::NearestPointType::nearest_point_fast)
      remote_point_evaluation.reinit(evaluation_points, dof_handler.get_triangulation(), mapping);

    solution_distance.update_ghost_values();
    const auto evaluation_values_distance =
      dealii::VectorTools::point_values<1>(remote_point_evaluation, dof_handler, solution_distance);
    solution_distance.zero_out_ghost_values();
    double max_distance =
      evaluation_values_distance.size() == 0 ?
        0.0 :
        *std::max_element(evaluation_values_distance.begin(), evaluation_values_distance.end());

    max_distance = Utilities::MPI::max(max_distance, mpi_comm);
    pcout << "max distance to numerical distance solution (relative): "
          << max_distance / min_cell_size << std::endl;
  }

  cpp.template extend_interface_values<n_components>(solution_temp_interface, solution_temp, false);

#ifdef MPDG_TEST_ENABLE_DEBUG
  std::ofstream myfile;
  myfile.open("marked_vertices.dat");

  for (const auto &p : all_marked_vertices)
    {
      for (unsigned int d = 0; d < dim; ++d)
        myfile << p[d] << " ";
      myfile << std::endl;
    }

  myfile.close();
#endif

  timer_total.stop();

  // ------------------------------------------------------------------------------------
  // ------------------------------------------------------------------------------------
  //      END OF ACTUAL ALGORITHM
  // ------------------------------------------------------------------------------------
  // ------------------------------------------------------------------------------------
  const auto norm =
    MeltPoolDG::VectorTools::compute_norm<dim, double>(solution_temp_interface,
                                                       triangulation,
                                                       mapping,
                                                       dof_handler_temp,
                                                       QGauss<dim>(degree_temp + 1),
                                                       dealii::VectorTools::L2_norm);
  pcout << "norm of interface vals: " << norm << std::endl;


#ifdef MPDG_TEST_ENABLE_DEBUG
  timer.print_wall_time_statistics(MPI_COMM_WORLD);
  // add output vectors
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;

  DataOut<dim> data_out;
  data_out.set_flags(flags);

  data_out.add_data_vector(dof_handler, solution_distance, "solution_distance");
  for (unsigned int d = 0; d < dim; ++d)
    data_out.add_data_vector(dof_handler, solution_normal.block(d), "normal_" + std::to_string(d));

  if constexpr (n_components > 1)
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(n_components,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(dof_handler_temp,
                               solution_temp_interface,
                               std::vector<std::string>(n_components, "solution_temp_interface"),
                               vector_component_interpretation);

      data_out.add_data_vector(dof_handler_temp,
                               solution_temp,
                               std::vector<std::string>(n_components, "solution_temp"),
                               vector_component_interpretation);
    }
  else
    {
      data_out.add_data_vector(dof_handler_temp,
                               solution_temp_interface,
                               "solution_temp_interface");
      data_out.add_data_vector(dof_handler_temp, solution_temp, "solution_temp");
    }

  pcout << "Elapsed wall time " << timer_total.wall_time() << std::endl;
  data_out.build_patches(mapping);
  std::string output = "n_components=" + std::to_string(n_components) +
                       "_n_iter=" + std::to_string(n_iter) + "_" + type._to_string() +
                       "_dim=" + std::to_string(dim) + "_output.vtu";
  data_out.write_vtu_in_parallel(output, mpi_comm);
#endif
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  run_test<1, 6, 1>();
  // TODO: disabled due to unstable test results. Inspect in the future
  // run_test<1, 5, 1, 3>(LevelSet::NearestPointType::nearest_point);
#ifdef DEAL_II_WITH_ARBORX
  run_test<1, 5, 1, 3>(LevelSet::NearestPointType::nearest_point_fast);
#endif
  run_test<1, 6, 16>();
  run_test<2, 6, 5>();
  run_test<2, 6, 5>(LevelSet::NearestPointType::closest_point_normal_collinear);
  run_test<2, 6, 1>(LevelSet::NearestPointType::nearest_point);
#ifdef DEAL_II_WITH_ARBORX
  run_test<2, 6, 1>(LevelSet::NearestPointType::nearest_point_fast);
#endif
  run_test<2, 6, 5>(LevelSet::NearestPointType::closest_point_normal_collinear_coquerelle);

  return 0;
}
