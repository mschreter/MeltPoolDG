#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_source_contribution.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <vector>

namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  class HeatOperationBase
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    virtual ~HeatOperationBase() = default;

    virtual void
    reinit() = 0;

    virtual void
    distribute_dofs(ScratchData<dim, dim, number> &scratch_data) const = 0;

    virtual void
    setup_constraints(ScratchData<dim, dim, number> &scratch_data) const = 0;

    virtual void
    set_initial_condition(const dealii::Function<dim> &initial_temperature) = 0;

    virtual void
    distribute_constraints() = 0;

    virtual void
    init_time_advance() = 0;

    virtual void
    solve() = 0;

    virtual void
    register_interface_projection_data(
      const VectorType                         &distance,
      const BlockVectorType                    &normal_vector,
      const LevelSet::NearestPointData<double> &nearest_point_data) = 0;

    virtual void
    compute_interface_temperature() = 0;

    /**
     * Zero out all heat source contributions. This function should be called once per time step
     * before any heat source is added via add_heat_source().
     */
    virtual void
    reset_heat_sources() = 0;

    /**
     * Add the heat source contribution @p heat_source, e.g. from a laser, for the current time
     * step. An exception is thrown if the type of contribution is not supported by the heat
     * operator.
     */
    virtual void
    add_heat_source(const HeatSourceContribution<dim, number> &heat_source) = 0;

    /**
     * register vectors for adaptive mesh refinement solution transfer
     */
    virtual void
    attach_vectors(std::vector<VectorType *> &vectors) = 0;

    /**
     * attach vectors for output
     */
    virtual void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const = 0;

    virtual void
    attach_output_vectors_failed_step(GenericDataOut<dim, number> &data_out) const = 0;

    /*
     * getters
     */
    virtual const VectorType &
    get_temperature() const = 0;

    virtual VectorType &
    get_temperature() = 0;

    virtual const VectorType &
    get_interface_temperature() const = 0;

    virtual VectorType &
    get_interface_temperature() = 0;

    virtual const VectorType &
    get_heat_source() const = 0;

    virtual VectorType &
    get_heat_source() = 0;
  };
} // namespace MeltPoolDG::Heat
