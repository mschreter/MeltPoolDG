#pragma once

#include "powder_bed.hpp"
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/case_registration.hpp>
#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>

#include <cmath>
#include <memory>
#include <type_traits>

#include "../../../mp-melt-pool/melt_pool_case.hpp"
#include "../../heat_transfer_case.hpp"

#ifdef MELT_POOL_DG_WITH_RTE
#  include "../../../mp-radiative-transport/radiative_transport_case.hpp"
#endif


namespace MeltPoolDG::Simulation::PowderBed
{
  template <int dim, typename number, typename CaseClass>
  SimulationPowderBed<dim, number, CaseClass>::SimulationPowderBed(std::string    parameter_file,
                                                                   const MPI_Comm mpi_communicator)
    : CaseClass(parameter_file, mpi_communicator)
    , cell_repetitions(dim, 1)
  {}


  template <int dim, typename number, typename CaseClass>
  bool
  SimulationPowderBed<dim, number, CaseClass>::add_case_specific_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.add_parameter("domain x min", domain_x_min, "minimum x coordinate of simulation domain");
    prm.add_parameter("domain x max", domain_x_max, "maximum x coordinate of simulation domain");
    prm.add_parameter("domain y min", domain_y_min, "minimum y coordinate of simulation domain");
    prm.add_parameter("domain y max", domain_y_max, "maximum y coordinate of simulation domain");
    prm.add_parameter("domain z min", domain_z_min, "minimum z coordinate of simulation domain");
    prm.add_parameter("domain z max", domain_z_max, "maximum z coordinate of simulation domain");
    prm.add_parameter("cell repetitions",
                      cell_repetitions,
                      "cell repetitions per dim applied before global refinement or amr");
    prm.add_parameter("initial temperature", T_initial, "Set the initial temperature.");
    powder_bed_data.add_parameters(prm);

    return this->parameters.base.do_print_parameters;
  }


  template <int dim, typename number, typename CaseClass>
  void
  SimulationPowderBed<dim, number, CaseClass>::create_spatial_discretization()
  {
    if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
      {
#ifdef DEAL_II_WITH_METIS
        this->triangulation = std::make_shared<dealii::parallel::shared::Triangulation<dim>>(
          this->mpi_communicator,
          dealii::Triangulation<dim>::none,
          false,
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
      dim == 1 ? dealii::Point<dim, number>(domain_x_min) :
      dim == 2 ? dealii::Point<dim, number>(domain_x_min, domain_y_min) :
                 dealii::Point<dim, number>(domain_x_min, domain_y_min, domain_z_min);
    const dealii::Point<dim, number> top_right =
      dim == 1 ? dealii::Point<dim, number>(domain_x_max) :
      dim == 2 ? dealii::Point<dim, number>(domain_x_max, domain_y_max) :
                 dealii::Point<dim, number>(domain_x_max, domain_y_max, domain_z_max);

    if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
      {
        dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                          cell_repetitions,
                                                          bottom_left,
                                                          top_right,
                                                          /* colorize */ true);
      }
    else // do simplex
      {
        std::vector<unsigned int> subdivisions(
          dim, 5 * dealii::Utilities::pow(2, this->parameters.base.global_refinements));
        subdivisions[dim - 1] *= 2;
        for (int d = 0; d < dim; d++)
          subdivisions[d] *= cell_repetitions[d];

        dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                         subdivisions,
                                                                         bottom_left,
                                                                         top_right,
                                                                         /* colorize */ true);
      }
  }


  template <int dim, typename number, typename CaseClass>
  void
  SimulationPowderBed<dim, number, CaseClass>::set_boundary_conditions()
  {
    // face numbering according to the deal.II colorize flag
    const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
      get_colorized_rectangle_boundary_ids<dim>();

    // BC for heat transfer
    this->attach_boundary_condition(
      {lower_bc, std::make_shared<dealii::Functions::ConstantFunction<dim>>(T_initial)},
      "dirichlet",
      "heat_transfer");

    // BC for two-phase flow
    if constexpr (std::is_same_v<CaseClass, MeltPoolCase<dim, number>>)
      {
        // slip BC on all boundaries
        this->attach_boundary_condition(lower_bc, "symmetry", "navier_stokes_u");
        this->attach_boundary_condition(upper_bc, "symmetry", "navier_stokes_u");
        if (dim >= 2)
          {
            this->attach_boundary_condition(left_bc, "symmetry", "navier_stokes_u");
            this->attach_boundary_condition(right_bc, "symmetry", "navier_stokes_u");
          }
        if (dim >= 3)
          {
            this->attach_boundary_condition(front_bc, "symmetry", "navier_stokes_u");
            this->attach_boundary_condition(back_bc, "symmetry", "navier_stokes_u");
          }

        // fix pressure condition can be set on any (non-periodic) boundary
        this->attach_boundary_condition(lower_bc, "fix_pressure_constant", "navier_stokes_p");
      }

      // BC for RTE
#ifdef MELT_POOL_DG_WITH_RTE
    if constexpr (std::is_same_v<CaseClass,
                                 RadiativeTransport::RadiativeTransportCase<dim, number>>)
      {
        this->attach_boundary_condition(
          {upper_bc,
           std::make_shared<Heat::GaussProjectionIntensityProfile<dim, number>>(
             this->parameters.laser.power,
             this->parameters.laser.radius,
             this->parameters.laser.template get_starting_position<dim>(),
             this->parameters.laser.template get_beam_direction<dim>())},
          "dirichlet",
          "intensity");
      }
    else
#endif
      if (this->parameters.laser.model == Heat::LaserModelType::RTE)
      {
        this->attach_boundary_condition(upper_bc, "laser_inflow", "intensity");
      }

    if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
      this->triangulation->refine_global(this->parameters.base.global_refinements);
  }


  template <int dim, typename number, typename CaseClass>
  void
  SimulationPowderBed<dim, number, CaseClass>::set_field_conditions()
  {
    // attach initial temperature
    this->attach_initial_condition(
      std::make_shared<dealii::Functions::ConstantFunction<dim>>(T_initial), "heat_transfer");

    // only for heat transfer case
    if constexpr (std::is_same_v<CaseClass, Heat::HeatTransferCase<dim, number>>)
      {
        // create a temporary reinit data instance since reinit data is not available for the heat
        // problem
        LevelSet::ReinitializationData<number> reinit_data;
        reinit_data.interface_thickness_parameter.type =
          LevelSet::InterfaceThicknessParameterType::proportional_to_cell_size;
        reinit_data.interface_thickness_parameter.value = 1.5;

        // attach prescribed heaviside
        this->attach_initial_condition(
          std::make_shared<MeltPool::PowderBedLevelSet<dim, double>>(
            powder_bed_data,
            LevelSet::LevelSetType::smoothed_heaviside,
            reinit_data.compute_interface_thickness_parameter_epsilon(
              dealii::GridTools::minimal_cell_diameter(*this->triangulation) / std::sqrt(dim))),
          "prescribed_heaviside");
      }

    // only for melt pool case
    if constexpr (std::is_same_v<CaseClass, MeltPoolCase<dim, number>>)
      {
        auto mp_case = dynamic_cast<MeltPoolCase<dim, number> *>(this);

        this->attach_initial_condition(
          std::make_shared<MeltPool::PowderBedLevelSet<dim, double>>(
            powder_bed_data,
            LevelSet::LevelSetType::signed_distance,
            mp_case->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
              dealii::GridTools::minimal_cell_diameter(*this->triangulation) / std::sqrt(dim))),
          "signed_distance");

        this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                       "navier_stokes_u");
      }

    if (this->parameters.laser.model == Heat::LaserModelType::RTE)
      {
        this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(),
                                       "intensity");
      }
  }
} // namespace MeltPoolDG::Simulation::PowderBed
