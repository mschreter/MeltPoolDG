## [Main] - 2026/11/02

### Changed

- MINOR Replace the `LaserOperation` by separate components with one-way dependencies:
  - `LaserBeam` holds the power, position and direction of the beam over time; the intensity profiles evaluate the current beam state directly.
  - For the diffuse heat operator, `create_laser_heat_source()` creates one of the heat source models (`LaserHeatSourceVolumetric`, `LaserHeatSourceProjectionBased`, `LaserHeatSourceRTE`). The model adds its heat source once per time step to the heat operation via the interface `HeatSourceReceiver`.
  - For the CutFEM heat operator, `LaserInterfaceHeatFlux` is registered as interfacial heat flux (`InterfaceHeatFlux`), which the operator evaluates along the cut interface. The CutFEM heat operator no longer knows about lasers. It throws an exception if it is combined with a laser model other than `interface_projection_sharp(_conforming)`, which was silently ignored before.

- MINOR Simulation cases specify the inflow boundary of the RTE laser model by attaching the boundary condition `"laser_inflow"` for `"intensity"` instead of setting `LaserData::rte_boundary_id`.

- MINOR `LaserData::post()` no longer scales the laser power by the absorptivity for the CutFEM heat operator; the absorptivity is applied by the interfacial heat flux.

### Fixed

- MINOR `mp-heat-transfer`: fix the heat source of the laser models `interface_projection_sharp(_conforming)` in combination with the diffuse heat operator, which was not set up due to an inverted flag and accumulated over time steps.
- MINOR `mp-heat-transfer`: the laser heat source of the models `volumetric` and `interface_projection_regularized` no longer overwrites a `"prescribed_heat_source"`, but is added to it.
