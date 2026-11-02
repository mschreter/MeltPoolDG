#pragma once

#include <deal.II/base/numbers.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>

#include <functional>
#include <vector>

namespace MeltPoolDG::Heat
{
  /**
   * Data, which is required to couple a laser heat source to the heat operation and to the
   * phase interface.
   */
  template <int dim, typename number>
  struct LaserCouplingData
  {
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    // DoFHandler index of the heat source vectors of the heat operation
    unsigned int heat_no_bc_dof_idx = dealii::numbers::invalid_unsigned_int;
    unsigned int heat_quad_idx      = dealii::numbers::invalid_unsigned_int;

    // phase indicator (0: gas, 1: liquid) and the corresponding DoFHandler index
    const VectorType *level_set_as_heaviside = nullptr;
    unsigned int      level_set_dof_idx      = dealii::numbers::invalid_unsigned_int;

    // optional normal vector; if not provided, it is computed from the gradient of the heaviside.
    // It is requested on each evaluation, since the vector might not exist yet during the setup
    // and might be recreated during the simulation.
    std::function<const BlockVectorType &()> get_normal_vector;
    unsigned int                             normal_dof_idx = 0;

    // true if material properties vary continuously over the interface
    bool variable_properties_over_interface = false;

    // only required for LaserModelType::RTE: boundaries, at which the laser enters the domain
    const RadiativeTransport::RadiativeTransportData<number> *rte_data = nullptr;
    std::vector<dealii::types::boundary_id>                   rte_inflow_boundary_ids;
    bool                                                      print_boundary_ids = false;
  };

  /**
   * Interface of a laser heat source model for heat operations, which accept heat sources as DoF
   * vectors (HeatSourceReceiver, i.e. the diffuse heat operator), see create_laser_heat_source().
   * For heat operations that evaluate the interfacial heat flux themselves (CutFEM), see
   * LaserInterfaceHeatFlux.
   *
   * A heat source model is bound to the heat operation during construction and adds the heat
   * input of the laser to it once per time step. It follows the state of the laser beam. Models
   * that require their own degrees of freedom (e.g. the RTE) may override the setup functions,
   * which do nothing by default.
   */
  template <int dim, typename number>
  class LaserHeatSourceBase
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    virtual ~LaserHeatSourceBase() = default;

    virtual void
    distribute_dofs(const FiniteElementData & /*fe_data*/)
    {}

    virtual void
    setup_constraints()
    {}

    virtual void
    distribute_constraints()
    {}

    virtual void
    reinit()
    {}

    virtual void
    attach_vectors(DoFHandlerAndVectorDataType<dim, VectorType> & /*data*/)
    {}

    virtual void
    attach_output_vectors(GenericDataOut<dim, number> & /*data_out*/) const
    {}

    /**
     * Return true if the evaluation of the heat source requires loops over inner faces.
     */
    virtual bool
    requires_inner_face_loops() const
    {
      return false;
    }

    /**
     * Add the laser heat source of the current time to the heat operation.
     */
    virtual void
    apply() = 0;
  };
} // namespace MeltPoolDG::Heat
