#include "radiative_transport.hpp"
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <algorithm>
#include <cmath>

#include "../../mp-heat-transfer/heat_transfer_case.hpp"
#include "../radiative_transport_case.hpp"


namespace MeltPoolDG::Simulation::RadiativeTransport
{
  template <int dim, typename number>
  class LevelSetHeaviside : public dealii::Function<dim, number>
  {
  public:
    LevelSetHeaviside(const InterfaceCase              interface_case_in,
                      const std::pair<number, number> &interface_case_info_in,
                      const number                     epsilon_cell_in)
      : dealii::Function<dim, number>(1)
      , eps(epsilon_cell_in)
      , interface_case(interface_case_in)
      , interface_case_info(interface_case_info_in)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int) const override
    {
      if (interface_case == InterfaceCase::straight)
        {
          const auto y            = p[dim - 1];
          const auto current_time = this->get_time();
          return CharacteristicFunctions::smoothed_heaviside(
            level +
              interface_case_info.first *
                (current_time < interface_case_info.second ? current_time :
                                                             interface_case_info.second) /
                interface_case_info.second -
              y,
            eps); // 0 side of H() stands for gas, 1 side of H() stands for liquid
        }
      else if (interface_case == InterfaceCase::single_powder_particle)
        {
          // say we inherit from a straight interface case
          const auto y = p[dim - 1];

          number straight_value = CharacteristicFunctions::smoothed_heaviside(
            level - y,
            eps); // 0 side of H() stands for gas, 1 side of H() stands for liquid

          dealii::Point<dim, number> sphere_center;
          sphere_center[dim - 1] = interface_case_info.first;

          // now add a powder particle with gradient:
          const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere(
            sphere_center, interface_case_info.second);
          number powder_particle_value =
            CharacteristicFunctions::smoothed_heaviside(-distance_sphere.value(p), eps);
          return std::max(straight_value, powder_particle_value);
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
          return 0.0;
        }
      return 0.0;
    }

  private:
    const number              eps                 = 0.1;
    const number              level               = 0.0;
    InterfaceCase             interface_case      = InterfaceCase::straight;
    std::pair<number, number> interface_case_info = std::pair<number, number>(0., 0.);
  };


  template <int dim, typename number, typename Problem>
  SimulationRadTrans<dim, number, Problem>::SimulationRadTrans(std::string    parameter_file,
                                                               const MPI_Comm mpi_communicator)
    : Problem(parameter_file, mpi_communicator)
    , cell_repetitions(dim, 1)
  {}


  template <int dim, typename number, typename Problem>
  bool
  SimulationRadTrans<dim, number, Problem>::add_case_specific_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.add_parameter("domain x min", domain_x_min, "minimum x coordinate of simulation domain");
    prm.add_parameter("domain y min", domain_y_min, "minimum y coordinate of simulation domain");
    prm.add_parameter("domain x max", domain_x_max, "maximum x coordinate of simulation domain");
    prm.add_parameter("domain y max", domain_y_max, "maximum y coordinate of simulation domain");
    prm.add_parameter("cell repetitions",
                      cell_repetitions,
                      "cell repetitions per dim applied before global refinement or amr");

    prm.add_parameter("power",
                      power_in,
                      "Sets the intensity scale of the laser source. Is a scalar value");
    prm.add_parameter("source center", center_in, "location of the heat source center");
    prm.add_parameter("source radius", radius_in, "heat source radius");

    prm.add_parameter(
      "interface case",
      interface_case,
      "kind of interface for this simulation. "
      "straight: straigt interface that moves upwards; "
      "single_powder_particle: a single hanging powder particle above a static straight interface; "
      "powder bed: ");

    prm.add_parameter("straight interface upward speed", speed, "straight interface upward speed");
    prm.add_parameter("straight interface movement end time",
                      end_time,
                      "end time of the straight interface movement");

    prm.add_parameter("powder particle radius",
                      powder_particle_radius,
                      "hanging powder particle radius");
    prm.add_parameter("powder particle offset",
                      powder_particle_offset,
                      "hanging powder particle offset from [dim-1] = 0 plane");

    return this->parameters.base.do_print_parameters;
  }


  template <int dim, typename number, typename Problem>
  void
  SimulationRadTrans<dim, number, Problem>::create_spatial_discretization()
  {
    if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
      {
#ifdef DEAL_II_WITH_METIS
        this->triangulation = std::make_shared<dealii::parallel::shared::Triangulation<dim>>(
          this->mpi_communicator,
          dealii::Triangulation<dim>::MeshSmoothing::none,
          true,
          dealii::parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Missing Metis support of the deal.II installation. "
            "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
      }
    else
      {
        this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
          this->mpi_communicator);
      }

    const dealii::Point<dim, number> bottom_left =
      dim == 1 ? dealii::Point<dim, number>(domain_y_min) :
      dim == 2 ? dealii::Point<dim, number>(domain_x_min, domain_y_min) :
                 dealii::Point<dim, number>(domain_x_min, domain_x_min, domain_y_min);
    const dealii::Point<dim, number> top_right =
      dim == 1 ? dealii::Point<dim, number>(domain_y_max) :
      dim == 2 ? dealii::Point<dim, number>(domain_x_max, domain_y_max) :
                 dealii::Point<dim, number>(domain_x_max, domain_x_max, domain_y_max);
    if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
      {
        std::vector<unsigned int> subdivisions(
          dim, 5 * dealii::Utilities::pow(2, this->parameters.base.global_refinements));
        subdivisions[dim - 1] *= 2;
        for (int d = 0; d < dim; d++)
          subdivisions[d] *= cell_repetitions[d];

        dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(
          *this->triangulation, subdivisions, bottom_left, top_right, true /*colorize*/);
      }
    else
      {
        dealii::GridGenerator::subdivided_hyper_rectangle(
          *this->triangulation, cell_repetitions, bottom_left, top_right, true /*colorize*/);
      }
  }


  template <int dim, typename number, typename Problem>
  void
  SimulationRadTrans<dim, number, Problem>::set_boundary_conditions()
  {
    // face numbering according to the deal.II colorize flag
    [[maybe_unused]] const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
      get_colorized_rectangle_boundary_ids<dim>();

    if (auto temp_ptr =
          dynamic_cast<MeltPoolDG::RadiativeTransport::RadiativeTransportCase<dim, number> *>(this))
      this->attach_boundary_condition(
        {upper_bc,
         std::make_shared<Heat::GaussProjectionIntensityProfile<dim, number>>(
           power_in, radius_in, center_in, -dealii::Point<dim, number>::unit_vector(dim - 1))},
        "dirichlet",
        "intensity");
    else
      this->attach_boundary_condition(upper_bc, "laser_inflow", "intensity");

    if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
      this->triangulation->refine_global(this->parameters.base.global_refinements);
  }


  template <int dim, typename number, typename Problem>
  void
  SimulationRadTrans<dim, number, Problem>::set_field_conditions()
  {
    // pass simulation-specific parameters to the simulation class.
    // Done after json parsing, is relevant for heaviside
    if (interface_case == InterfaceCase::straight)
      interface_case_info = std::pair<number, number>(speed, end_time);
    else if (interface_case == InterfaceCase::single_powder_particle)
      interface_case_info =
        std::pair<number, number>(powder_particle_offset, powder_particle_radius);

    // determine the interface epsilon parameter from minimum mesh size
    number       thickness_scale_factor = 2.5;
    const number epsilon_cell = dealii::GridTools::minimal_cell_diameter(*this->triangulation) /
                                std::sqrt(dim) * thickness_scale_factor;

    // attach the prescribed heaviside function field
    this->attach_initial_condition(std::make_shared<LevelSetHeaviside<dim, number>>(
                                     interface_case, interface_case_info, epsilon_cell),
                                   "prescribed_heaviside");

    if (auto temp_ptr =
          dynamic_cast<MeltPoolDG::RadiativeTransport::RadiativeTransportCase<dim, number> *>(this))
      this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(),
                                     "intensity");

    // this is only used to TODO generate the comparison solution with the gaussian laser
    if (auto temp_ptr = dynamic_cast<Heat::HeatTransferCase<dim, number> *>(this))
      // attach dummy initial conditions for the heat transfer operation
      this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(),
                                     "heat_transfer");
  }
} // namespace MeltPoolDG::Simulation::RadiativeTransport
