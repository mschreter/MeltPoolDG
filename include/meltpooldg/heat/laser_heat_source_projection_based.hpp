#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_source_receiver.hpp>
#include <meltpooldg/heat/laser_beam.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>

#include <memory>

namespace MeltPoolDG::Heat
{
  /**
   * Projection based laser heat source model: the intensity profile is projected in beam
   * direction onto the phase interface. Depending on the laser model, the resulting
   * interfacial heat flux is
   *
   * - LaserModelType::interface_projection_regularized: regularized over the diffuse interface
   *   and added to the volumetric heat source,
   * - LaserModelType::interface_projection_sharp(_conforming): integrated along the sharp
   *   interface and added to the right-hand side of the heat operation.
   *
   * For heat operations that evaluate the interfacial heat flux themselves (CutFEM), see
   * LaserInterfaceHeatFlux.
   */
  template <int dim, typename number>
  class LaserHeatSourceProjectionBased final : public LaserHeatSourceBase<dim, number>
  {
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    LaserHeatSourceProjectionBased(const ScratchData<dim, dim, number>  &scratch_data_in,
                                   const LaserData<number>              &laser_data_in,
                                   const LaserBeam<dim, number>         &beam,
                                   const LaserCouplingData<dim, number> &coupling_data_in,
                                   HeatSourceReceiver<dim, number>      &heat_operation_in);

    bool
    requires_inner_face_loops() const override;

    void
    apply() override;

  private:
    /**
     * Compute a DoF vector of the interfacial heat source, regularized over the diffuse
     * interface by a delta function. If no @p normal_vector is given, the normal vector is
     * computed from the gradient of the heaviside.
     */
    void
    compute_regularized_heat_source(VectorType            &heat_source_vector,
                                    const BlockVectorType *normal_vector) const;

    /**
     * Add the weak form of the interfacial heat source to @p heat_rhs by evaluating the surface
     * integral along the sharp interface, determined by the marching cube algorithm.
     */
    void
    add_sharp_heat_source(VectorType &heat_rhs, const BlockVectorType *normal_vector) const;

    /**
     * Same as add_sharp_heat_source(), but the surface integral is evaluated over the element
     * faces along the phase boundary.
     *
     * @note This function should only be used, if the level-set isosurface
     * is aligned with element faces and does not cross the cells.
     */
    void
    add_sharp_conforming_heat_source(VectorType            &heat_rhs,
                                     const BlockVectorType *normal_vector) const;

    /**
     * Local interfacial heat source
     *
     * For the sharp interfacial heat source, set @param delta_value to 1.
     */
    number
    local_compute_interfacial_heat_source(const dealii::Point<dim>             &p,
                                          const dealii::Tensor<1, dim, number> &normal_vector,
                                          const number                          delta_value,
                                          const number                          heaviside) const;

    const ScratchData<dim, dim, number> &scratch_data;

    const LaserData<number> &laser_data;

    const LaserCouplingData<dim, number> coupling_data;

    HeatSourceReceiver<dim, number> &heat_operation;

    const std::shared_ptr<const dealii::Function<dim, number>> intensity_profile;

    const dealii::Tensor<1, dim, number> laser_direction;

    const bool variable_properties_over_interface;

    std::unique_ptr<const LevelSet::DeltaApproximationBase<number>> delta_phase_weighted;

    VectorType heat_source;
  };
} // namespace MeltPoolDG::Heat
