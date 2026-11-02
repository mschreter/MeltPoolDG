#include <deal.II/base/exceptions.h>

#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/heat/laser_interface_heat_flux.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  LaserInterfaceHeatFlux<dim, number>::LaserInterfaceHeatFlux(const LaserData<number> &laser_data,
                                                              const LaserBeam<dim, number> &beam)
    : intensity_profile(create_intensity_profile(laser_data, beam, false /*is_volumetric*/))
    , laser_direction(beam.get_direction())
    , absorptivity(laser_data.absorptivity_liquid)
  {
    AssertThrow(laser_data.model == LaserModelType::interface_projection_sharp or
                  laser_data.model == LaserModelType::interface_projection_sharp_conforming,
                ExcMessage("The CutFEM heat operator requires the laser model "
                           "interface_projection_sharp or interface_projection_sharp_conforming."));
    AssertThrow(laser_data.absorptivity_gas == laser_data.absorptivity_liquid,
                ExcMessage("The CutFEM heat operator requires equal absorptivities of the gas "
                           "and the liquid."));
  }

  template <int dim, typename number>
  typename LaserInterfaceHeatFlux<dim, number>::VectorizedArrayType
  LaserInterfaceHeatFlux<dim, number>::value(
    const Point<dim, VectorizedArrayType>     &p,
    const Tensor<1, dim, VectorizedArrayType> &normal) const
  {
    return VectorTools::evaluate_function_at_vectorized_points<dim, number>(*intensity_profile,
                                                                            p,
                                                                            0 /*component*/) *
           compute_projection_factor(laser_direction, normal) * absorptivity;
  }

  template class LaserInterfaceHeatFlux<1, double>;
  template class LaserInterfaceHeatFlux<2, double>;
  template class LaserInterfaceHeatFlux<3, double>;
} // namespace MeltPoolDG::Heat
