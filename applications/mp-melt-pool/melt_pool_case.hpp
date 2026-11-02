#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#include <meltpooldg/flow/flow_data.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/level_set/level_set_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/melt_front_propagation_data.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/amr_data.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/restart_data.hpp>

#include <iostream>
#include <string>
#include <type_traits>

namespace MeltPoolDG
{
  BETTER_ENUM(AMRStrategy, char, generic, adaflo, KellyErrorEstimator)
  BETTER_ENUM(AutomaticGridRefinementType, char, fixed_fraction, fixed_number)

  /**
   * Collection of all parameters of MeltPoolDG.
   *
   * @warning MeltPoolCaseParameters are read in order they are specified in the parameter
   * files. We exploit this behavior, e.g, in MaterialData, to allow to set
   * default parameters based on the user input and to override individual
   * entries subsequently. Please don't sort your input files and if you do be
   * aware that you might change the behavior!
   */
  template <typename number>
  struct MeltPoolCaseParameters : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      amr.add_parameters(prm);
      ls.add_parameters(prm);
      heat.add_parameters(prm);
      laser.add_parameters(prm);
      rte.add_parameters(prm);
      flow.add_parameters(prm);
      evapor.add_parameters(prm);
      material.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
      restart.add_parameters(prm);
      melt_front.add_parameters(prm);
#ifdef MPDG_ENABLE_ADAFLO
      adaflo_params.add_parameters(prm);
#endif

      prm.enter_subsection("application specific");
      {
        prm.add_parameter(
          "do heat transfer",
          application_specific_parameters.do_heat_transfer,
          "Set this parameter to true if you want to consider a coupling with heat transfer.");
        prm.add_parameter(
          "do solidification",
          application_specific_parameters.do_solidification,
          "Set this parameter to true if you want to consider melting/solidification effects.");
        prm.add_parameter(
          "do advect level set",
          application_specific_parameters.do_advect_level_set,
          "Set this parameter to true if you want to advect the level set with the fluid velocity.");
        prm.add_parameter(
          "do extrapolate coupling terms",
          application_specific_parameters.do_extrapolate_coupling_terms,
          "Set this parameter to true if you want to extrapolate the solution vectors for semi-explicit "
          "treatment of coupling terms.");
        prm.enter_subsection("amr");
        {
          prm.add_parameter("strategy",
                            application_specific_parameters.amr.strategy,
                            "Select the AMR strategy.");
          prm.add_parameter(
            "do auto detect frequency",
            application_specific_parameters.amr.do_auto_detect_frequency,
            "Automatically determine the frequency of remeshing. If this parameter is set, the parameter "
            "`amr: every n step` is ignored.");
          prm.add_parameter(
            "automatic grid refinement type",
            application_specific_parameters.amr.automatic_grid_refinement_type,
            "If the cells are refined automatically (strategy generic/KellyErrorEstimator), choose between "
            "refine_and_coarsen_fixed_number and refine_and_coarsen_fixed_fraction.");
          prm.add_parameter(
            "do refine all interface cells",
            application_specific_parameters.amr.do_refine_all_interface_cells,
            "Enforce all cells with level set values between -0.975 and 0.975 to be refined.");
          prm.add_parameter("refine gas domain",
                            application_specific_parameters.amr.refine_gas_domain,
                            "Refine the gas domain.");
          prm.add_parameter(
            "fraction of melting point refined in solid",
            application_specific_parameters.amr.fraction_of_melting_point_refined_in_solid,
            "Define a fraction of the melting point. Cells in the solid with a higher temperature are enforced "
            "to be refined.");
        }
        prm.leave_subsection();
        prm.enter_subsection("coupling ls evapor");
        {
          prm.add_parameter("n max iter",
                            application_specific_parameters.level_set_evapor_coupling.n_max_iter,
                            "Maximum number of iterations for nonlinear solution.");
          prm.add_parameter("tol",
                            application_specific_parameters.level_set_evapor_coupling.tol,
                            "If the change of the l2-norm of the level set is smaller than 'tol', "
                            "the iteration is stopped.");
        }
        prm.leave_subsection();
        prm.enter_subsection("mp heat up");
        {
          prm.add_parameter("time step size",
                            application_specific_parameters.mp_heat_up.time_step_size,
                            "Time step size until heat up is finished.");
          prm.add_parameter(
            "max change factor time step size",
            application_specific_parameters.mp_heat_up.max_change_factor_time_step_size,
            "Maximum allowed factor of changing the time step size between two time steps.");
          prm.add_parameter("max temperature",
                            application_specific_parameters.mp_heat_up.max_temperature,
                            "Temperature at which heat up is finished.");
        }
        prm.leave_subsection();
        prm.enter_subsection("coupling heat evapor");
        {
          prm.add_parameter("n max iter",
                            application_specific_parameters.heat_evapor_coupling.n_max_iter,
                            "Maximum number of iterations for nonlinear solution.");
          prm.add_parameter("tol",
                            application_specific_parameters.heat_evapor_coupling.tol,
                            "If the change of the l2-norm of the level set is smaller than 'tol', "
                            "the iteration is stopped.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
    void
    post(const std::string &parameter_filename) final
    {
      /************************************************************************************
       * set input-file-dependent default parameters
       ************************************************************************************/
      amr.post(base.global_refinements, restart.load >= 0);
      heat.post(base.fe, base.verbosity_level);
      laser.post(base.dimension,
                 heat.diffuse.use_volume_specific_thermal_capacity_for_phase_interpolation,
                 material);
      rte.post(base.fe, base.verbosity_level);
      ls.post(base.fe, base.verbosity_level);
      evapor.post(material,
                  heat.diffuse.use_volume_specific_thermal_capacity_for_phase_interpolation);
      flow.post(material);
      output.post(time_stepping.time_step_size, parameter_filename);
      restart.post(output.directory);
#ifdef MPDG_ENABLE_ADAFLO
      adaflo_params.post(material, base.fe.type, time_stepping);
#endif
      /************************************************************************************
       * check input parameters for validity
       ************************************************************************************/
      check_input_parameters();

      parameters_read = true;
    }


  private:
    void
    check_input_parameters() const
    {
      base.check_input_parameters();
      heat.check_input_parameters(base.fe);
      laser.check_input_parameters();
      rte.check_input_parameters(base.fe);
      ls.check_input_parameters(base.fe);
      evapor.check_input_parameters(material);
      flow.check_input_parameters(ls.curv.enable);
      profiling.check_input_parameters(time_stepping.time_step_size);
      restart.check_input_parameters(time_stepping.time_step_size);

      if (application_specific_parameters.do_solidification and
          not application_specific_parameters.do_heat_transfer)
        AssertThrow(false,
                    dealii::ExcMessage("In case of solidification flag >>> do solidification <<< "
                                       "and >>> do heat transfer <<< have to be set to true."));

      AssertThrow(
        application_specific_parameters.amr.fraction_of_melting_point_refined_in_solid <= 1 and
          application_specific_parameters.amr.fraction_of_melting_point_refined_in_solid >= 0,
        dealii::ExcMessage(
          ">>>fraction of melting point refined in solid<<< must be between 0 and 1."));

#ifdef MPDG_ENABLE_ADAFLO
      adaflo_params.check_input_parameters(evapor.evaporative_dilation_rate.enable);
#endif
    }


  public:
    BaseData                                           base;
    TimeIntegration::TimeSteppingData<number>          time_stepping;
    AdaptiveMeshingData<number>                        amr;
    LevelSet::LevelSetData<number>                     ls;
    Heat::HeatData<number>                             heat;
    Heat::LaserData<number>                            laser;
    RadiativeTransport::RadiativeTransportData<number> rte;
    Flow::FlowData<number>                             flow;
    Evaporation::EvaporationData<number>               evapor;
    MaterialData<number>                               material;
    OutputData<number>                                 output;
    Profiling::ProfilingData<number>                   profiling;
    Restart::RestartData<number>                       restart;
#ifdef MPDG_ENABLE_ADAFLO
    Flow::AdafloWrapperParameters<number> adaflo_params;
#endif
    MeltFrontPropagationData<number> melt_front;

    struct
    {
      bool do_heat_transfer              = false;
      bool do_solidification             = false;
      bool do_advect_level_set           = true;
      bool do_extrapolate_coupling_terms = false;

      struct
      {
        // number of iterations to balance nonlinearity in advection diffusion equation with
        // evaporation
        int    n_max_iter = 1;
        number tol        = 1e-10;
      } level_set_evapor_coupling;

      struct
      {
        // number of iterations to balance nonlinearity in heat equation with
        // evaporation
        int    n_max_iter = 1;
        number tol        = 1e-10;
      } heat_evapor_coupling;

      struct
      {
        AMRStrategy                 strategy = AMRStrategy::generic;
        AutomaticGridRefinementType automatic_grid_refinement_type =
          AutomaticGridRefinementType::fixed_number;
        bool   do_auto_detect_frequency                   = false;
        bool   do_refine_all_interface_cells              = false;
        number fraction_of_melting_point_refined_in_solid = 1.0;
        bool   refine_gas_domain                          = false;
      } amr;

      struct
      {
        number time_step_size                   = -1;
        number max_temperature                  = -1;
        number max_change_factor_time_step_size = 1.5;
      } mp_heat_up;

    } application_specific_parameters;
  };

  template <int dim, typename number>
  class MeltPoolCase : public SimulationCaseBase<dim, number>
  {
  public:
    MeltPoolCaseParameters<number> parameters;

    MeltPoolCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG
