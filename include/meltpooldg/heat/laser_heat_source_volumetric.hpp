#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_source_receiver.hpp>
#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>

#include <memory>

namespace MeltPoolDG::Heat
{
  /**
   * Volumetric laser heat source model: the heat source is given by the intensity profile.
   */
  template <int dim, typename number>
  class LaserHeatSourceVolumetric final : public LaserHeatSourceBase<dim, number>
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    LaserHeatSourceVolumetric(const ScratchData<dim, dim, number> &scratch_data_in,
                              const LaserData<number>             &laser_data,
                              const LaserBeam<dim, number>        &beam,
                              const unsigned int                   heat_dof_idx_in,
                              HeatSourceReceiver<dim, number>     &heat_operation_in);

    void
    apply() override;

  private:
    /**
     * Compute a DoF vector of the volumetric heat source by interpolating the intensity profile.
     */
    void
    compute_volumetric_heat_source(VectorType &heat_source_vector) const;

    const ScratchData<dim, dim, number>                       &scratch_data;
    const std::shared_ptr<const dealii::Function<dim, number>> intensity_profile;
    const unsigned int                                         heat_dof_idx;
    HeatSourceReceiver<dim, number>                           &heat_operation;

    VectorType heat_source;
  };
} // namespace MeltPoolDG::Heat
