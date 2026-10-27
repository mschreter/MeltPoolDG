// Test the closest point projection of NearestPoint with and without a provided normal vector
// field. If no normal vector field is provided, the normal is computed from the gradient of the
// signed distance function.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <vector>

using namespace dealii;
using namespace MeltPoolDG;

using VectorType      = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

template <int dim>
class LinearFunction : public Function<dim>
{
public:
  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    return 500. - 500. * p[0];
  }
};

template <int dim>
Point<dim>
compute_projected_point_analytically(const Point<dim> &p,
                                     const Point<dim> &center,
                                     const double      radius)
{
  const Tensor<1, dim> vec = p - center;
  return center + radius * vec / vec.norm();
}

template <int dim>
void
run_test(const LevelSet::NearestPointType type,
         const unsigned int               n_refinements,
         const unsigned int               degree)
{
  std::cout.precision(4);
  MPI_Comm                   mpi_comm(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  pcout << "------------------------------------------------------------------" << std::endl;
  pcout << " START TEST: type " << type << " #refine " << n_refinements << " degree " << degree
        << " dim " << dim << std::endl;
  pcout << "------------------------------------------------------------------" << std::endl;

  Point<dim> center;
  for (unsigned int d = 0; d < dim; ++d)
    center[d] = 0.5;
  const double radius = 0.25;

  parallel::distributed::Triangulation<dim> triangulation(mpi_comm);
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refinements);

  MappingQ<dim>   mapping(1);
  FE_Q<dim>       fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  VectorType signed_distance(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  dealii::VectorTools::interpolate(mapping,
                                   dof_handler,
                                   Functions::SignedDistance::Sphere<dim>(center, radius),
                                   signed_distance);

  // exact normal vector at the nodes
  BlockVectorType normal_vector(dim);
  for (unsigned int d = 0; d < dim; ++d)
    normal_vector.block(d).reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

  std::map<types::global_dof_index, Point<dim>> support_points;
  {
    FEValues<dim> fe_values(mapping,
                            fe,
                            Quadrature<dim>(fe.get_unit_support_points()),
                            update_quadrature_points);

    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(local_dof_indices);

          for (const auto q : fe_values.quadrature_point_indices())
            {
              const auto &p                        = fe_values.quadrature_point(q);
              support_points[local_dof_indices[q]] = p;

              const Tensor<1, dim> n = (p - center) / (p - center).norm();
              for (unsigned int d = 0; d < dim; ++d)
                normal_vector.block(d)[local_dof_indices[q]] = n[d];
            }
        }
  }

  VectorType solution(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  dealii::VectorTools::interpolate(mapping, dof_handler, LinearFunction<dim>(), solution);

  LevelSet::NearestPointData<double> nearest_point_data;
  nearest_point_data.rel_tol               = 1e-8;
  nearest_point_data.narrow_band_threshold = 0.1;
  nearest_point_data.max_iter              = 20;
  nearest_point_data.type                  = type;

  const double min_cell_size = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(dim);

  std::vector<std::vector<Point<dim>>> all_projected_points;

  for (const bool use_normal_vector : {true, false})
    {
      Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation;

      std::optional<LevelSet::Tools::NearestPoint<dim, double>> nearest_point;
      if (use_normal_vector)
        nearest_point.emplace(mapping,
                              dof_handler,
                              signed_distance,
                              normal_vector,
                              remote_point_evaluation,
                              nearest_point_data);
      else
        nearest_point.emplace(
          mapping, dof_handler, signed_distance, remote_point_evaluation, nearest_point_data);

      nearest_point->reinit(&dof_handler);

      const auto &projected_points = nearest_point->get_projected_points();
      const auto &dof_indices      = nearest_point->get_projected_points_dof_indices();

      double max_dist = 0;
      for (unsigned int i = 0; i < projected_points.size(); ++i)
        max_dist = std::max(
          max_dist,
          compute_projected_point_analytically(support_points.at(dof_indices[i][0]), center, radius)
            .distance(projected_points[i]));

      // extend values from the interface
      VectorType solution_interface(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
      nearest_point->extend_interface_values(solution_interface, solution);

      // the linear function is exactly represented by the FE space
      double max_error = 0;
      for (unsigned int i = 0; i < projected_points.size(); ++i)
        max_error = std::max(max_error,
                             std::abs(solution_interface[dof_indices[i][0]] -
                                      LinearFunction<dim>().value(projected_points[i], 0)));

      pcout << (use_normal_vector ? "with normal vector" : "without normal vector") << std::endl;
      pcout << "  max distance to analytical solution (relative): "
            << Utilities::MPI::max(max_dist, mpi_comm) / min_cell_size << std::endl;
      pcout << "  extended values exact: "
            << (Utilities::MPI::max(max_error, mpi_comm) < 1e-10 ? "yes" : "no") << std::endl;

      all_projected_points.emplace_back(projected_points);
    }

  AssertDimension(all_projected_points[0].size(), all_projected_points[1].size());

  double max_dist_between = 0;
  for (unsigned int i = 0; i < all_projected_points[0].size(); ++i)
    max_dist_between =
      std::max(max_dist_between, all_projected_points[0][i].distance(all_projected_points[1][i]));

  pcout << "max distance between projected points with/without normal vector (relative): "
        << Utilities::MPI::max(max_dist_between, mpi_comm) / min_cell_size << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  run_test<2>(LevelSet::NearestPointType::closest_point_normal, 6, 1);
  // for a linear level set function, the gradient is too inaccurate to obtain convergence of the
  // tangential correction
  run_test<2>(LevelSet::NearestPointType::closest_point_normal_collinear, 6, 2);
  run_test<2>(LevelSet::NearestPointType::closest_point_normal_collinear_coquerelle, 6, 2);
  run_test<3>(LevelSet::NearestPointType::closest_point_normal, 4, 1);
  run_test<3>(LevelSet::NearestPointType::closest_point_normal_collinear_coquerelle, 4, 2);

  return 0;
}
