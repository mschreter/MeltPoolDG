// Test the nearest_point_fast algorithm of NearestPoint with a user-provided point cloud
// representing the interface and user-provided values at these points, which are extended to the
// narrow band around the interface.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// uncomment to write the extended DoF vector to a vtu file
#define MPDG_TEST_ENABLE_DEBUG

using namespace dealii;
using namespace MeltPoolDG;

using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int dim>
Point<dim>
compute_projected_point_analytically(const Point<dim> &p,
                                     const Point<dim> &center,
                                     const double      radius)
{
  const Tensor<1, dim> vec = p - center;
  return center + radius * vec / vec.norm();
}

// value of the i-th component at the interface point p
template <int dim>
double
interface_value(const Point<dim> &p, const unsigned int component)
{
  return 100. * (component + 1) + p[component];
}

// (quasi-)uniformly distributed points on a circle/sphere; the points are distributed among the
// processes, while process 0 does not provide any point
template <int dim>
std::vector<Point<dim>>
create_interface_points(const Point<dim>  &center,
                        const double       radius,
                        const unsigned int n_points,
                        const MPI_Comm     mpi_comm)
{
  const unsigned int n_procs = Utilities::MPI::n_mpi_processes(mpi_comm);
  const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_comm);

  std::vector<Point<dim>> points;

  for (unsigned int i = 0; i < n_points; ++i)
    {
      if (n_procs > 1 and (rank == 0 or i % (n_procs - 1) != rank - 1))
        continue;

      Tensor<1, dim> direction;
      if constexpr (dim == 2)
        {
          const double phi = 2. * numbers::PI * i / n_points;
          direction[0]     = std::cos(phi);
          direction[1]     = std::sin(phi);
        }
      else if constexpr (dim == 3)
        {
          // Fibonacci sphere
          const double golden_angle = numbers::PI * (3. - std::sqrt(5.));
          const double z            = 1. - 2. * (i + 0.5) / n_points;
          const double r            = std::sqrt(1. - z * z);
          direction[0]              = r * std::cos(golden_angle * i);
          direction[1]              = r * std::sin(golden_angle * i);
          direction[2]              = z;
        }
      points.emplace_back(center + radius * direction);
    }

  return points;
}

// write the interface points together with the corresponding values to a vtu file
template <int dim>
void
write_interface_points(const std::string              &filename,
                       const std::vector<Point<dim>>  &points,
                       const std::vector<double>      &values,
                       const std::vector<std::string> &names,
                       const std::vector<DataComponentInterpretation::DataComponentInterpretation>
                                     &component_interpretation,
                       const MPI_Comm mpi_comm)
{
  const auto all_points = Utilities::MPI::gather(mpi_comm, points);
  const auto all_values = Utilities::MPI::gather(mpi_comm, values);

  if (Utilities::MPI::this_mpi_process(mpi_comm) != 0)
    return;

  const unsigned int n_components = names.size();

  std::vector<DataOutBase::Patch<0, dim>> patches;
  for (unsigned int rank = 0; rank < all_points.size(); ++rank)
    for (unsigned int i = 0; i < all_points[rank].size(); ++i)
      {
        DataOutBase::Patch<0, dim> patch;
        patch.vertices[0] = all_points[rank][i];
        patch.patch_index = patches.size();
        patch.data.reinit(n_components, 1);
        for (unsigned int c = 0; c < n_components; ++c)
          patch.data(c, 0) = all_values[rank][i * n_components + c];
        patches.emplace_back(patch);
      }

  std::vector<std::tuple<unsigned int,
                         unsigned int,
                         std::string,
                         DataComponentInterpretation::DataComponentInterpretation>>
    nonscalar_data_ranges;
  if (n_components == dim and
      component_interpretation[0] == DataComponentInterpretation::component_is_part_of_vector)
    nonscalar_data_ranges.emplace_back(0, dim - 1, names[0], component_interpretation[0]);

  std::ofstream out(filename);
  DataOutBase::write_vtu(patches, names, nonscalar_data_ranges, DataOutBase::VtkFlags(), out);
}

template <int dim, int n_components>
void
run_test(const unsigned int n_refinements, const unsigned int n_interface_points)
{
  std::cout.precision(4);
  MPI_Comm                   mpi_comm(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  pcout << "------------------------------------------------------------------" << std::endl;
  pcout << " START TEST: #comp " << n_components << " #refine " << n_refinements << " dim " << dim
        << " #interface points " << n_interface_points << std::endl;
  pcout << "------------------------------------------------------------------" << std::endl;

  Point<dim> center;
  for (unsigned int d = 0; d < dim; ++d)
    center[d] = 0.5;
  const double radius = 0.25;

  parallel::distributed::Triangulation<dim> triangulation(mpi_comm);
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refinements);

  MappingQ<dim> mapping(1);

  // level set function, used to determine the narrow band
  FE_Q<dim>       fe_ls(1);
  DoFHandler<dim> dof_handler_ls(triangulation);
  dof_handler_ls.distribute_dofs(fe_ls);

  VectorType signed_distance(dof_handler_ls.locally_owned_dofs(),
                             DoFTools::extract_locally_relevant_dofs(dof_handler_ls),
                             mpi_comm);
  dealii::VectorTools::interpolate(mapping,
                                   dof_handler_ls,
                                   Functions::SignedDistance::Sphere<dim>(center, radius),
                                   signed_distance);

  // field to which the interface values are extended
  FESystem<dim>   fe(FE_Q<dim>(2), n_components);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  VectorType solution(dof_handler.locally_owned_dofs(),
                      DoFTools::extract_locally_relevant_dofs(dof_handler),
                      mpi_comm);

  std::map<types::global_dof_index, Point<dim>> support_points;
  {
    FEValues<dim> fe_values(mapping,
                            fe,
                            Quadrature<dim>(fe.base_element(0).get_unit_support_points()),
                            update_quadrature_points);

    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(local_dof_indices);

          for (const auto q : fe_values.quadrature_point_indices())
            support_points[local_dof_indices[fe.component_to_system_index(0, q)]] =
              fe_values.quadrature_point(q);
        }
  }

  // interface points and values
  const auto interface_points =
    create_interface_points<dim>(center, radius, n_interface_points, mpi_comm);

  std::vector<double> interface_values;
  for (const auto &p : interface_points)
    for (unsigned int c = 0; c < n_components; ++c)
      interface_values.emplace_back(interface_value(p, c));

  LevelSet::NearestPointData<double> nearest_point_data;
  nearest_point_data.narrow_band_threshold = 0.1;
  nearest_point_data.type                  = LevelSet::NearestPointType::nearest_point_fast;

  Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation;

  LevelSet::Tools::NearestPoint<dim, double> nearest_point(
    mapping, dof_handler_ls, signed_distance, remote_point_evaluation, nearest_point_data);

  nearest_point.reinit(dof_handler, interface_points);

  nearest_point.template extend_interface_values<n_components>(solution, interface_values);

  const auto &projected_points = nearest_point.get_projected_points();
  const auto &dof_indices      = nearest_point.get_projected_points_dof_indices();

  double max_dist  = 0;
  double max_error = 0;
  for (unsigned int i = 0; i < projected_points.size(); ++i)
    {
      max_dist = std::max(max_dist,
                          compute_projected_point_analytically(support_points.at(dof_indices[i][0]),
                                                               center,
                                                               radius)
                            .distance(projected_points[i]));

      // the extended values must coincide with the values at the nearest interface point
      for (unsigned int c = 0; c < n_components; ++c)
        max_error =
          std::max(max_error,
                   std::abs(solution[dof_indices[i][c]] - interface_value(projected_points[i], c)));
    }

  const double min_cell_size = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(dim);

  pcout << "no. stencil points: " << Utilities::MPI::sum(projected_points.size(), mpi_comm)
        << std::endl;
  pcout << "max distance to analytical solution (relative): "
        << Utilities::MPI::max(max_dist, mpi_comm) / min_cell_size << std::endl;
  pcout << "max error of extended values: " << Utilities::MPI::max(max_error, mpi_comm)
        << std::endl;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;

  DataOut<dim> data_out;
  data_out.set_flags(flags);

  // interpret the components as vector if they match the spatial dimension
  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(n_components,
                             n_components == dim ?
                               DataComponentInterpretation::component_is_part_of_vector :
                               DataComponentInterpretation::component_is_scalar);

  std::vector<std::string> names(n_components, "extended_values");
  if (n_components > 1 and n_components != dim)
    for (unsigned int c = 0; c < n_components; ++c)
      names[c] += "_" + std::to_string(c);

  solution.update_ghost_values();
  signed_distance.update_ghost_values();

  data_out.add_data_vector(dof_handler, solution, names, component_interpretation);
  data_out.add_data_vector(dof_handler_ls, signed_distance, "signed_distance");

  const std::string filename_suffix =
    "_dim=" + std::to_string(dim) + "_n_components=" + std::to_string(n_components) + ".vtu";

  data_out.build_patches(mapping, fe.degree);
  data_out.write_vtu_in_parallel("extend_interface_value_03" + filename_suffix, mpi_comm);

  write_interface_points<dim>("extend_interface_value_03_interface_points" + filename_suffix,
                              interface_points,
                              interface_values,
                              names,
                              component_interpretation,
                              mpi_comm);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  run_test<2, 1>(6, 2048);
  run_test<2, 2>(6, 2048);
  run_test<3, 3>(4, 20000);

  return 0;
}
