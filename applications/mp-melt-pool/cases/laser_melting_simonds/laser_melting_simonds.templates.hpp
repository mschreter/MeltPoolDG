#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>

#include <cassert>

#include "laser_melting_simonds.hpp"
//

#include "../../../mp-heat-transfer/heat_transfer_case.hpp"
#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::LaserMeltingSimonds
{
  template <int dim>
  InflowVelocity<dim>::InflowVelocity()
    : Function<dim>(dim)
  {
    AssertThrow(dim > 1, ExcMessage("Not implemented for dim=1"));
  }

  template <int dim>
  double
  InflowVelocity<dim>::value(const Point<dim> &p, const unsigned int component) const
  {
    // only the 0 component is relevant
    if (component > 0)
      return 0;

    // not at the inflow wall
    if (std::abs(p[0] + width * 0.5) > 1e-12)
      return 0;

    const double y  = p[dim - 1];
    const double y0 = 0.5 * delta_h;
    const double y1 = 1.5 * delta_h;

    if (y >= y1)
      return inflow_velocity;
    else if (y >= y0)
      {
        const double s    = (y - y0) / (y1 - y0);
        const double ramp = 2.0 * s * s - s * s * s * s;

        return inflow_velocity * ramp;
      }
    else
      return 0.0;
  }

  template <int dim>
  InitialLevelSet<dim>::InitialLevelSet(const double                 eps,
                                        const LevelSet::LevelSetType level_set_type)
    : dealii::Function<dim>()
    , distance_plane(dealii::Point<dim>(), -dealii::Point<dim>::unit_vector(dim - 1))
    , eps(eps)
    , level_set_type(level_set_type)
  {}

  template <int dim>
  double
  InitialLevelSet<dim>::value(const dealii::Point<dim> &p, const unsigned int /*component*/) const
  {
    const auto signed_distance = distance_plane.value(p);

    switch (level_set_type)
      {
        case LevelSet::LevelSetType::tanh:
          return CharacteristicFunctions::tanh_characteristic_function(signed_distance, eps);
        case LevelSet::LevelSetType::smoothed_heaviside:
          return CharacteristicFunctions::smoothed_heaviside(signed_distance, eps);
        case LevelSet::LevelSetType::signed_distance:
          return signed_distance;
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }
    // unreachable dummy return
    return 0.0;
  }

  template <int dim>
  InitialConditionTemperature<dim>::InitialConditionTemperature(const double T_initial_bottom,
                                                                const double T_initial_top,
                                                                const double y_min,
                                                                const double y_max)
    : Function<dim>()
    , T_initial_bottom(T_initial_bottom)
    , T_initial_top(T_initial_top)
    , y_min(y_min)
    , grad_T((T_initial_top - T_initial_bottom) / (y_max - y_min))
  {}

  template <int dim>
  double
  InitialConditionTemperature<dim>::value(const Point<dim> &p,
                                          const unsigned int /*component*/) const
  {
    if (T_initial_top == T_initial_bottom)
      return T_initial_top;
    else
      return T_initial_bottom + grad_T * (p[dim - 1] - y_min);
  }

  template <int dim, typename Number, typename CaseClass>
  SimulationLaserMeltingSimonds<dim, Number, CaseClass>::SimulationLaserMeltingSimonds(
    std::string    parameter_file,
    const MPI_Comm mpi_communicator)
    : CaseClass(parameter_file, mpi_communicator)
    , cell_repetitions(dim, 1)
  {}

  template <int dim, typename Number, typename CaseClass>
  bool
  SimulationLaserMeltingSimonds<dim, Number, CaseClass>::add_case_specific_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("domain");
    {
      prm.add_parameter("width", width, "Width of the box. Only relevant for dim > 1.");
      prm.add_parameter("height substrate", height_substrate, "Height of the metal substrate.");
      prm.add_parameter("height gas", height_gas, "Height of the ambient gas.");
      prm.add_parameter("delta h", delta_h, "Distance between interface and gas inflow domain.");
    }
    prm.leave_subsection();

    prm.enter_subsection("mesh");
    {
      prm.add_parameter("cell repetitions",
                        cell_repetitions,
                        "cell repetitions per dim applied before global refinement or amr");
      prm.add_parameter("n local refinement",
                        n_local_refinement,
                        "number of (additional to the global) refinements for local region.");
      prm.add_parameter("local refinement 1 bottom left",
                        local_refinement_1_bottom_left,
                        "Bottom left point of locally refined region.");
      prm.add_parameter("local refinement 1 top right",
                        local_refinement_1_top_right,
                        "Top right point of locally refined region.");
      prm.add_parameter("local refinement 2 bottom left",
                        local_refinement_2_bottom_left,
                        "Bottom left point of locally refined region.");
      prm.add_parameter("local refinement 2 top right",
                        local_refinement_2_top_right,
                        "Top right point of locally refined region.");
    }
    prm.leave_subsection();

    prm.enter_subsection("initial temperature");
    {
      prm.add_parameter("top", T_initial_top, "Set the initial temperature on the top boundary.");
      prm.add_parameter("bottom",
                        T_initial_bottom,
                        "Set the initial temperature on the bottom boundary.");
    }
    prm.leave_subsection();

    prm.enter_subsection("prescribed level set");
    {
      prm.add_parameter(
        "eps prefactor",
        eps_prefactor,
        "Factor multiplied by the minimum mesh size "
        "to compute the interface thickness parameter. Used for the prescribed level-set function "
        "for the HeatTransferCase with a diffuse-interface operator");
    }
    prm.leave_subsection();

    prm.enter_subsection("bc");
    {
      prm.add_parameter("inflow velocity",
                        inflow_velocity,
                        "Set the inflow velocity at the vertical domain boundary.");
      prm.add_parameter("inflow temperature",
                        inflow_temperature,
                        "Set the inflow temperature at the vertical domain boundary.");
      prm.add_parameter("outlet pressure",
                        outlet_pressure,
                        "Set the pressure at the vertical outlet boundary.");
    }
    prm.leave_subsection();

    return this->parameters.base.do_print_parameters;
  }

  template <int dim, typename Number, typename CaseClass>
  void
  SimulationLaserMeltingSimonds<dim, Number, CaseClass>::create_spatial_discretization()
  {
    // NOTE: Simulation specific parameters that are read via the input file are only
    // available here and not in the constructor
    const Number half_width = width * 0.5;
    bottom_left             = (dim == 1) ? Point<dim>(-height_substrate) :
                              (dim == 2) ? Point<dim>(-half_width, -height_substrate) :
                                           Point<dim>(-half_width, -half_width, -height_substrate);
    top_right               = (dim == 1) ? Point<dim>(height_gas) :
                              (dim == 2) ? Point<dim>(half_width, height_gas) :
                                           Point<dim>(half_width, half_width, height_gas);

    if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP or dim == 1)
      {
#ifdef DEAL_II_WITH_METIS
        this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
          this->mpi_communicator,
          (Triangulation<dim>::MeshSmoothing::none),
          true,
          parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
        AssertThrow(
          false,
          ExcMessage("Missing Metis support of the deal.II installation. "
                     "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
      }
    else
      {
        this->triangulation =
          std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
      }

    // create mesh
    if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
      {
        std::vector<unsigned int> subdivisions(
          dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
        subdivisions[dim - 1] *= 2;
        for (int d = 0; d < dim; d++)
          subdivisions[d] *= cell_repetitions[d];

        GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                 subdivisions,
                                                                 bottom_left,
                                                                 top_right);
      }
    else
      {
        GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                  cell_repetitions,
                                                  bottom_left,
                                                  top_right);
      }
  }

  template <int dim, typename Number, typename CaseClass>
  void
  SimulationLaserMeltingSimonds<dim, Number, CaseClass>::set_boundary_conditions()
  {
    // refine global before setting IDs as we consider dimension-dependent
    // splitting of the boundaries
    this->triangulation->refine_global(this->parameters.base.global_refinements);

    //  Note: For 1D we consider the constraints along the z-axis, i.e. bottom_bc and top_bc.
    const types::boundary_id substrate_bottom_bc       = 1;
    const types::boundary_id substrate_outflow_wall_bc = 2;
    const types::boundary_id substrate_inflow_wall_bc  = 3;
    const types::boundary_id gas_top_bc                = 4;
    const types::boundary_id gas_inflow_bc             = 5;
    const types::boundary_id gas_outlet_bc             = 6;

    std::vector<types::boundary_id> all_bcs = {substrate_bottom_bc, substrate_outflow_wall_bc, substrate_inflow_wall_bc, gas_top_bc, gas_inflow_bc, gas_outlet_bc};

    const auto double_eq = [](const Number a, const Number b) { return std::abs(a - b) <= 1e-10; };

    // set boundary ids
    for (const auto &cell : this->triangulation->cell_iterators())
      for (const auto &face : cell->face_iterators())
        if ((face->at_boundary()))
          {
            // set boundaries at faces where z=const
            if (double_eq(face->center()[dim - 1], bottom_left[dim - 1]))
              face->set_boundary_id(substrate_bottom_bc);
            else if (double_eq(face->center()[dim - 1], top_right[dim - 1]))
              face->set_boundary_id(gas_top_bc);
            // set wall boundaries
            else if (dim > 1)
              {
                // substrate: walls (all points with z<delta_h)
                if (double_eq(face->center()[0], bottom_left[0]) &&
                    (face->center()[dim - 1] <= delta_h))
                  face->set_boundary_id(substrate_inflow_wall_bc);
                else if (face->center()[dim - 1] <= delta_h)
                  face->set_boundary_id(substrate_outflow_wall_bc);
                // gas inflow: all points with x=0 and z>=delta_h
                else if (double_eq(face->center()[0], bottom_left[0]) &&
                         (face->center()[dim - 1] > delta_h))
                  face->set_boundary_id(gas_inflow_bc);
                else if (face->center()[dim - 1] > delta_h)
                  face->set_boundary_id(gas_outlet_bc);
                else
                  AssertThrow(false, ExcNotImplemented());
              }
          }

    this->attach_boundary_condition(
      {substrate_bottom_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_initial_bottom)},
      "dirichlet",
      "heat_transfer");

    if constexpr (std::is_same_v<CaseClass, MeltPoolCase<dim, Number>>)
      {
        if (inflow_velocity == 0 or dim == 1)
          {
            // NS
            for (const auto bc: all_bcs)
            {
              if (true)
              {
                  if (bc == gas_top_bc)
                  {
                    this->attach_boundary_condition(
                      {bc, std::make_shared<Functions::ConstantFunction<dim>>(outlet_pressure)},
                      "open",
                      "navier_stokes_u");
                  }
                  else
                  {
                  this->attach_boundary_condition(bc, "no_slip", "navier_stokes_u");
                  //this->attach_boundary_condition(
                      //{bc, std::make_shared<dealii::Functions::ZeroFunction<dim>>()},
                      //"dirichlet",
                      //"reinitialization");

                  //const Number height = height_gas + height_substrate;

                    //AssertThrow((height - width) <= 1e-10 || dim == 1,
                                //ExcMessage("Epsilon is only defined if a square domain is considered."));

                    //const auto min_mesh_size =
                      //this->parameters.amr.do_amr ?
                        //height / std::pow(2, this->parameters.amr.max_grid_refinement_level) :
                        //height / std::pow(2, this->parameters.base.global_refinements);
                    //const auto eps = eps_prefactor * min_mesh_size;


                   //auto ls = std::make_shared<InitialLevelSet<dim>>(
                                             //eps, LevelSet::LevelSetType::smoothed_heaviside);

                    //this->attach_boundary_condition(
                        //{bc, ls},
                        //"dirichlet",
                        //"level_set");
                  }

              }
              //else
                  //this->attach_boundary_condition(bc, "symmetry", "navier_stokes_u");
            }

            //this->attach_boundary_condition(gas_top_bc, "fix_pressure_constant", "navier_stokes_p");
          }
        else
          {
            // TODO: fix or add assert that this is only valid for eps = absolute value
            const double eps =
              this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
                dealii::GridTools::minimal_cell_diameter(*this->triangulation) /
                this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

            AssertThrow(eps > 0, dealii::ExcNotImplemented());

            // bottom
            this->attach_boundary_condition(substrate_bottom_bc, "no_slip", "navier_stokes_u");

            // top
            this->attach_boundary_condition(gas_top_bc, "symmetry", "navier_stokes_u");

            // outflow (gas + substrate)
            //
            // !!! Note: there is a switch in the BC between no-slip and pressure along the boundary
            this->attach_boundary_condition(
              {gas_outlet_bc, std::make_shared<Functions::ConstantFunction<dim>>(outlet_pressure)},
              "open",
              "navier_stokes_u");

            // inflow (gas + substrate)
            this->attach_boundary_condition({gas_inflow_bc,
                                             std::make_shared<InflowVelocity<dim>>()},
                                            "dirichlet",
                                            "navier_stokes_u");
            this->attach_boundary_condition({substrate_inflow_wall_bc,
                                             std::make_shared<InflowVelocity<dim>>()},
                                            "dirichlet",
                                            "navier_stokes_u");
            this->attach_boundary_condition(
              {gas_inflow_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_initial_bottom)},
              "dirichlet",
              "heat_transfer");
            this->attach_boundary_condition({substrate_inflow_wall_bc,
                                             std::make_shared<Functions::ConstantFunction<dim>>(
                                               T_initial_bottom)},
                                            "dirichlet",
                                            "heat_transfer");
            // !!! Note: there is a switch in the BC between dirichlet and hom. Neumann for the
            // level set
            this->attach_boundary_condition({gas_inflow_bc,
                                             std::make_shared<InitialLevelSet<dim>>(eps)},
                                            "dirichlet",
                                            "level_set");
            this->attach_boundary_condition(
              {gas_inflow_bc, std::make_shared<dealii::Functions::ZeroFunction<dim>>()},
              "dirichlet",
              "reinitialization");
          }
      }

    /*
     * locally refined region described by max. 2 bounding boxes
     */
    if (n_local_refinement > 0)
      {
        if constexpr (dim == 2)
          {
            const auto refinement_region =
              BoundingBox<dim>({local_refinement_1_bottom_left, local_refinement_1_top_right});

            const auto refinement_region_2 =
              BoundingBox<dim>({local_refinement_2_bottom_left, local_refinement_2_top_right});

            for (unsigned int j = 0; j < n_local_refinement; ++j)
              {
                for (auto &cell : this->triangulation->active_cell_iterators() |
                                    IteratorFilters::LocallyOwnedCell())
                  {
                    for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                      if (refinement_region.point_inside(cell->vertex(i)) or
                          refinement_region_2.point_inside(cell->vertex(i)))
                        {
                          cell->set_refine_flag();
                          break;
                        }
                  }
                this->triangulation->execute_coarsening_and_refinement();
              }
          }
        else
          AssertThrow(false, ExcNotImplemented());
      }
  }

  template <int dim, typename Number, typename CaseClass>
  void
  SimulationLaserMeltingSimonds<dim, Number, CaseClass>::set_field_conditions()
  {
    if constexpr (std::is_same_v<CaseClass, Heat::HeatTransferCase<dim, Number>>)
      {
        if (this->parameters.laser.model == Heat::LaserModelType::interface_projection_regularized)
          {
            // attach prescribed heaviside

            const Number height = height_gas + height_substrate;

            AssertThrow((height - width) <= 1e-10 || dim == 1,
                        ExcMessage("Epsilon is only defined if a square domain is considered."));

            const auto min_mesh_size =
              this->parameters.amr.do_amr ?
                height / std::pow(2, this->parameters.amr.max_grid_refinement_level) :
                height / std::pow(2, this->parameters.base.global_refinements);
            const auto eps = eps_prefactor * min_mesh_size;

            this->attach_initial_condition(std::make_shared<InitialLevelSet<dim>>(
                                             eps, LevelSet::LevelSetType::smoothed_heaviside),
                                           "prescribed_heaviside");
          }
        else if (this->parameters.laser.model == Heat::LaserModelType::interface_projection_sharp)
          {
            this->attach_initial_condition(std::make_shared<Functions::SignedDistance::Plane<dim>>(
                                             Point<dim>(), -Point<dim>::unit_vector(dim - 1)),
                                           "prescribed_signed_distance");
          }
        else
          {
            AssertThrow(false, ExcMessage("Laser model not supported."));
          }
      }
    if constexpr (std::is_same_v<CaseClass, MeltPoolCase<dim, Number>>)
      {
        this->attach_initial_condition(std::make_shared<Functions::SignedDistance::Plane<dim>>(
                                         Point<dim>(), -Point<dim>::unit_vector(dim - 1)),
                                       "signed_distance");
        if (inflow_velocity > 0 or dim > 1)
          this->attach_initial_condition(std::make_shared<InflowVelocity<dim>>(),
                                         "navier_stokes_u");
        else
          this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(),
                                         "navier_stokes_u");
      }


    this->attach_initial_condition(
      std::make_shared<InitialConditionTemperature<dim>>(
        T_initial_bottom, T_initial_top, bottom_left[dim - 1], top_right[dim - 1]),
      "heat_transfer");
  }

  template <int dim, typename Number, typename CaseClass>
  void
  SimulationLaserMeltingSimonds<dim, Number, CaseClass>::do_postprocessing(
    [[maybe_unused]] const GenericDataOut<dim, Number> &generic_data_out) const
  {
    if (this->parameters.output.paraview.enable == false)
      return;

    n_time_step += 1;

    if ((n_time_step % this->parameters.output.write_frequency) &&
        generic_data_out.get_time() != this->parameters.time_stepping.end_time)
      return;
    if constexpr (std::is_same_v<CaseClass, MeltPoolCase<dim, Number>>)
      {
        generic_data_out.get_vector("velocity").update_ghost_values();
        generic_data_out.get_vector("density").update_ghost_values();
        generic_data_out.get_vector("rho_cp").update_ghost_values();
        generic_data_out.get_vector("temperature").update_ghost_values();

        QGauss<dim> quad(generic_data_out.get_dof_handler("density").get_fe().tensor_degree() + 2);

        FEValues<dim> dens_values(generic_data_out.get_mapping(),
                                  generic_data_out.get_dof_handler("density").get_fe(),
                                  quad,
                                  update_values | update_JxW_values);

        FEValues<dim> rho_cp_values(generic_data_out.get_mapping(),
                                    generic_data_out.get_dof_handler("rho_cp").get_fe(),
                                    quad,
                                    update_values);

        FEValues<dim> vel_values(generic_data_out.get_mapping(),
                                 generic_data_out.get_dof_handler("velocity").get_fe(),
                                 quad,
                                 update_values);

        FEValues<dim> temp_values(generic_data_out.get_mapping(),
                                  generic_data_out.get_dof_handler("temperature").get_fe(),
                                  quad,
                                  update_values);

        std::vector<Number>              density(dens_values.get_quadrature().size());
        std::vector<Number>              rho_cp(dens_values.get_quadrature().size());
        std::vector<Number>              temperature(dens_values.get_quadrature().size());
        std::vector<Tensor<1, dim>>      velocity(density.size(), Tensor<1, dim>());
        const FEValuesExtractors::Vector velocities(0);

        Number              mass = 0;
        std::vector<Number> momentum(dim, 0);
        Number              thermal_energy = 0;
        Number              kinetic_energy = 0;

        for (const auto &cell :
             this->triangulation->active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
          {
            {
              TriaIterator<DoFCellAccessor<dim, dim, false>> dof_cell(
                &generic_data_out.get_dof_handler("density").get_triangulation(),
                cell->level(),
                cell->index(),
                &generic_data_out.get_dof_handler("density"));
              dens_values.reinit(dof_cell);
              dens_values.get_function_values(generic_data_out.get_vector("density"), density);
            }

            {
              TriaIterator<DoFCellAccessor<dim, dim, false>> dof_cell(
                &generic_data_out.get_dof_handler("density").get_triangulation(),
                cell->level(),
                cell->index(),
                &generic_data_out.get_dof_handler("rho_cp"));
              rho_cp_values.reinit(dof_cell);
              rho_cp_values.get_function_values(generic_data_out.get_vector("rho_cp"), rho_cp);
              temp_values.reinit(dof_cell);
              temp_values.get_function_values(generic_data_out.get_vector("temperature"),
                                              temperature);
            }

            {
              TriaIterator<DoFCellAccessor<dim, dim, false>> dof_cell(
                &generic_data_out.get_dof_handler("density").get_triangulation(),
                cell->level(),
                cell->index(),
                &generic_data_out.get_dof_handler("velocity"));
              vel_values.reinit(dof_cell);
              vel_values[velocities].get_function_values(generic_data_out.get_vector("velocity"),
                                                         velocity);
            }

            for (const unsigned int q : dens_values.quadrature_point_indices())
              {
                mass += density[q] * dens_values.JxW(q);
                thermal_energy += rho_cp[q] * temperature[q] * dens_values.JxW(q);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    momentum[d] += density[q] * velocity[q][d] * dens_values.JxW(q);
                    kinetic_energy +=
                      0.5 * density[q] * velocity[q][d] * velocity[q][d] * dens_values.JxW(q);
                  }
              }
          }

        generic_data_out.get_vector("velocity").zero_out_ghost_values();
        generic_data_out.get_vector("density").zero_out_ghost_values();
        generic_data_out.get_vector("rho_cp").zero_out_ghost_values();
        generic_data_out.get_vector("temperature").zero_out_ghost_values();

        mass =
          Utilities::MPI::sum(mass, generic_data_out.get_vector("density").get_mpi_communicator());

        for (unsigned int d = 0; d < dim; ++d)
          momentum[d] =
            Utilities::MPI::sum(momentum[d],
                                generic_data_out.get_vector("density").get_mpi_communicator());

        kinetic_energy =
          Utilities::MPI::sum(kinetic_energy,
                              generic_data_out.get_vector("density").get_mpi_communicator());
        thermal_energy =
          Utilities::MPI::sum(thermal_energy,
                              generic_data_out.get_vector("density").get_mpi_communicator());

        output_table.add_value("time", generic_data_out.get_time());
        output_table.add_value("mass", mass);
        output_table.add_value("kinetic_energy", kinetic_energy);
        output_table.add_value("thermal_energy", thermal_energy);

        for (unsigned int d = 0; d < dim; ++d)
          output_table.add_value("momentum_" + std::to_string(d), momentum[d]);

        if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
          {
            namespace fs = std::filesystem;
            std::ofstream output(
              fs::path(this->parameters.output.directory) /
              fs::path(this->parameters.output.paraview.filename + "_conservation_variables.txt"));

            output_table.write_text(output);
          }
      }
  }
} // namespace MeltPoolDG::Simulation::LaserMeltingSimonds
