# Turbomachinery Design Code — Core Solver Roadmap

## Nomenclature & Assumptions

* **Units System:** SI units are strictly enforced across the entire codebase (Temperatures in K, Pressures in Pa, Mass Flow in kg/s, velocities in m/s, radii in m). The validation benchmark suite converts OTAC reference data (US Customary) to SI before comparison.
* **Coordinate System:** Cylindrical coordinates `(z, r, theta)` where `z` is the machine axis, `r` is the radial distance from the axis, and `theta` is the tangential direction. The meridional plane combines `z` and `r`. The meridional angle `phi` is defined as the angle between the meridional velocity vector and the machine axis `z`. In purely axial configurations `phi = 0 deg`; in purely radial configurations `phi = 90 deg`.
* **Core Station Variables** *(7 inputs required per station to uniquely define the fluid state)*:
  * `m_dot`: Mass flow rate [kg/s]
  * `P_t`: Total Pressure [Pa]
  * `h_t`: Total Enthalpy [J/kg]
  * `MN`: Mach Number [-]
  * `alpha`: Absolute swirl angle — angle between absolute velocity vector and meridional plane [deg]
  * `phi`: Meridional slope angle — angle between meridional velocity and machine axis [deg]
  * `radius`: Local flow radius from machine axis [m]
* **Derived Station Variables** *(computed from the 7 inputs plus rotational speed `omega`)*:
  * `T_t`: Total Temperature [K]
  * `T_s`, `P_s`, `h_s`, `rho`: Static state properties
  * `Cp`, `gamma`: Thermodynamic properties
  * `V_m`: Meridional velocity [m/s]
  * `V_theta`: Tangential velocity [m/s]
  * `V_r`: Radial velocity [m/s]
  * `V_z`: Axial velocity [m/s]
  * `V_rel`: Relative velocity magnitude [m/s]
  * `beta`: Relative flow angle — angle between relative velocity and meridional plane [deg]
  * `MN_rel`: Relative Mach number [-]
  * `P_t_rel`: Relative total pressure [Pa]
  * `rothalpy`: `h_t - omega * r * V_theta` [J/kg] — conserved across rotating blade rows
  * `U`: Blade speed `omega * r` [m/s]
  * `r_inner`, `r_outer`: Annulus inner/outer radii at station [m]
  * `span`: Local blade span [m]
* **Sign Convention:**
  * Deviation `delta`: Positive value indicates the fluid exit angle is turned **less** than the blade metal angle (fluid exits further from axial than the blade trailing edge).
  * Incidence `i`: Calculated relative to design incidence for off-design loss corrections.
* **Future Interface Variables** *(reserved for map generation and pyCycle integration — not implemented in this roadmap)*:
  * `N_c`: Corrected Shaft Speed [rpm / sqrt(T_t_in / T_ref)]
  * `m_dot_c`: Corrected Mass Flow [kg/s * sqrt(T_t_in / T_ref) / (P_t_in / P_ref)]
  * `PR`: Total-to-Total Pressure Ratio [-]
  * `eta_a`: Adiabatic Efficiency [-]
  * `beta_map`: pyCycle auxiliary map interpolation parameter [-] *(distinct from flow angle `beta`)*
* **Dependencies:** `cantera`, `numpy`, `scipy`, `numba`, `pytest`.

---

## Architecture Decisions Log

*(Locked decisions. Do not deviate from these without explicit instruction.)*

* **Decision 1:** The core solver is deliberately framework-agnostic. `BladeRow.solve()` must be callable as a standalone Python function with no OpenMDAO or pyCycle imports anywhere in `src/`. This is the primary architectural constraint enabling future map generation and pyCycle integration. The public interface boundary — `FlowStation` in, `FlowStation` out — is the intended future pyCycle coupling point.

* **Decision 2:** Thermodynamic backend is a standalone Cantera module. Cantera uses the same NASA polynomial fits underlying pyCycle's JANAF tables, guaranteeing enthalpy and entropy reference state consistency at any future integration boundary. A pre-computed `scipy.interpolate.RectBivariateSpline` table over the expected operating range wraps all Cantera calls inside the solver to eliminate API overhead during nested iterations. Nothing in `src/solver/` may call Cantera directly; all thermodynamic evaluation goes through the interpolation table.

* **Decision 3:** Array-First I/O structures are mandatory. All `FlowStation`, `BladeRowGeometry`, and `SegmentGeometry` dataclasses must type their fields to accept numpy 1D arrays, even when the Phase 2 meanline solver only reads index `[0]`. This ensures zero interface changes when transitioning to the Phase 4 streamline matrix solver.

* **Decision 4:** The solver backend is abstracted behind a common interface in `src/solver/backend.py`. Residual functions in `src/solver/residuals.py` are pure, framework-agnostic, and `@numba.njit` compiled. The default backend dispatches to `scipy.optimize.root` with `method='lm'`. An OpenMDAO-compatible backend wrapper is implemented in parallel, wrapping the same residual functions in a structure that could be expressed as an `ImplicitComponent` with finite-difference partials. The calling code selects the backend via a single argument; no physics code changes between backends. OpenMDAO's Newton solver is excluded from the core because its convergence advantage only materialises when analytic derivatives are available — which they are not for empirical loss correlations.

* **Decision 5:** The meanline solver implements the OTAC 5-equation governing set per blade row. The active residuals are: (1) Euler turbomachinery equation, (2) total pressure loss condition, (3) turning/deviation condition, (4) effective radius constraint, (5) effective area constraint. Mass conservation and slope are trivially satisfied and are not iterated. The 5 solver unknowns are `{h_t2, P_t2, alpha_2, radius_2, MN_2}`. For stator rows the Euler equation is trivially satisfied, reducing to 4 unknowns and 4 residuals.

* **Decision 6:** Loss, deviation, and blockage are implemented as **stackable, interchangeable socket modules** following the OTAC subelement pattern. Each socket is a pure callable with a fixed signature. The default loss socket is Ainley-Mathieson, the default deviation socket applies the standard rule `beta_2 = beta_blade_te - delta`, and the default blockage socket applies fixed hub/tip/wake fractional blockage values. New correlations can be substituted without modifying `BladeRow` or `residuals.py`. All blade parameters required by any correlation that are not present in `SegmentGeometry` must be derived internally within that socket module.

* **Decision 7:** Two geometry dataclasses are required, matching OTAC's separation of concerns. `BladeRowGeometry` holds row-level fields (blade count, chord, hub hade angle, hub/tip radii, machine area, blockage factors). `SegmentGeometry` holds streamtube-level fields (local radius, local area, leading/trailing edge metal angles, meridional slope, wake blockage, pitch, stagger angle, blade thickness). `SegmentGeometry` fields must satisfy all Ainley-Mathieson input requirements without modification.

---

## Package Structure

```
turbomachinery/
├── src/
│   ├── core/
│   │   └── data_structures.py         # FlowStation, BladeRowGeometry, SegmentGeometry dataclasses
│   ├── thermo/
│   │   ├── cantera_wrapper.py          # Standalone Cantera evaluation functions
│   │   └── interpolation_table.py     # Pre-computed RectBivariateSpline lookup
│   └── solver/
│       ├── residuals.py               # Pure @numba.njit residual functions
│       ├── backend.py                 # Solver abstraction layer (scipy / openmdao-compatible dispatch)
│       ├── kinematics.py              # Velocity triangles, Euler equation, rothalpy
│       ├── blade_row.py               # BladeRow solver orchestration (public API)
│       └── sockets/
│           ├── loss_ainley_mathieson.py
│           ├── deviation_standard.py
│           └── blockage_fixed.py
├── tests/
│   ├── unit/
│   │   ├── test_data_structures.py
│   │   ├── test_thermo.py
│   │   ├── test_kinematics.py
│   │   ├── test_sockets.py
│   │   └── test_residuals.py
│   └── integration/
│       ├── test_smith_chart.py        # Single-stage turbine Smith chart (primary Phase 2 gate)
│       ├── test_1d_solver.py          # Additional meanline verification cases
│       └── test_streamline_solver.py  # Phase 4 streamline verification
└── scripts/
    └── run_solver.py                  # Top-level entry point for standalone solver runs
```

---

## Phase 0: Environment & Package Foundation
**Status:** `[NOT STARTED]`
**Objective:** Pin all dependencies, establish the package skeleton, and verify the development environment before any solver code is written.

* **Milestone 0.1: Dependency Pinning**
    * **Target Files:** `requirements.txt`, `environment.yml`
    * [ ] Pin exact versions of `cantera`, `numpy`, `scipy`, `numba`, `pytest`.
    * [ ] Verify that the pinned Cantera version supports the `Solution('air.yaml')` interface used in Phase 1.
    * [ ] Document the verified Python version in `README.md`.

* **Milestone 0.2: Package Skeleton**
    * **Target Files:** All `__init__.py` files per the Package Structure above.
    * [ ] Create the full directory tree and empty `__init__.py` files.
    * [ ] Confirm `pytest` discovers the `tests/` tree correctly with a passing smoke test.
    * [ ] Confirm all top-level imports resolve without errors.

---

## Phase 1: Data Structures & Thermodynamic Foundation
**Status:** `[NOT STARTED]`
**Objective:** Establish array-compatible I/O contracts and the thermodynamic lookup layer before any aerodynamic code is written. Nothing in Phase 2 onward may make direct Cantera calls; all thermodynamic evaluation goes through the interpolation table in `src/thermo/interpolation_table.py`.

* **Milestone 1.1: FlowStation & Geometry Dataclasses**
    * **Target Files:** `src/core/data_structures.py`, `tests/unit/test_data_structures.py`
    * [ ] Write `pytest` assertions verifying correct instantiation with scalar inputs (shape `(1,)`) and vector inputs (shape `(N,)`).
    * [ ] Define `FlowStation` dataclass. Primary input fields (all `np.ndarray`): `m_dot`, `P_t`, `h_t`, `MN`, `alpha`, `phi`, `radius`, `omega`. Computed output fields (all `np.ndarray`, populated by the thermo module and kinematics functions): `T_t`, `T_s`, `P_s`, `h_s`, `rho`, `Cp`, `gamma`, `V_m`, `V_theta`, `V_r`, `V_z`, `V_rel`, `beta`, `MN_rel`, `P_t_rel`, `rothalpy`, `U`, `r_inner`, `r_outer`, `span`. Include an `is_rotating: bool` flag.
    * [ ] Define `BladeRowGeometry` dataclass (all `np.ndarray`): `blade_count`, `chord`, `hub_hade_angle`, `hub_radius`, `tip_radius`, `machine_area`, `hub_blockage_fraction`, `tip_blockage_fraction`.
    * [ ] Define `SegmentGeometry` dataclass (all `np.ndarray`): `radius`, `area`, `metal_angle_le`, `metal_angle_te`, `phi_machine`, `wake_blockage_fraction`, `pitch`, `stagger_angle`, `blade_thickness`. These fields must satisfy all Ainley-Mathieson input requirements (Decision 7) without modification.
    * [ ] Write a `validate()` method on each dataclass asserting array shape consistency across all fields.
    * [ ] Write a `compute_velocity_triangles(station: FlowStation, thermo_table)` function that populates all derived velocity and relative-frame fields from the 7 primary inputs plus `omega`. Rothalpy must be computed as `h_t - omega * radius * V_theta`.

* **Milestone 1.2: Thermodynamic Module**
    * **Target Files:** `src/thermo/cantera_wrapper.py`, `src/thermo/interpolation_table.py`, `tests/unit/test_thermo.py`
    * [ ] Write `pytest` assertions verifying `Cp`, `gamma`, `h`, and `s` for standard air at 288.15 K and 101325 Pa against known NASA polynomial reference values.
    * [ ] Implement `get_thermo_state(T, P)` using Cantera's `Solution('air.yaml')` interface, returning a dict of `{Cp, gamma, h, s, rho}`.
    * [ ] Pre-compute a 2D `RectBivariateSpline` interpolation table for `Cp`, `gamma`, `h`, and `s` over the expected operating range. Define `T_MIN`, `T_MAX`, `P_MIN`, `P_MAX` as module-level constants at the top of `interpolation_table.py`.
    * [ ] Write a test asserting that the spline table and the direct Cantera call agree to within a defined tolerance across the full operating range grid.

* **Milestone 1.3: Dummy Solver Smoke Test**
    * **Target Files:** `src/solver/dummy_1d.py`, `tests/unit/test_data_structures.py`
    * [ ] Implement a trivial pass-through function that accepts `FlowStation` and `SegmentGeometry` inputs, applies a fixed scaling factor to `P_t`, and returns a modified `FlowStation`.
    * [ ] Write tests confirming the function handles both scalar `(1,)` and vector `(N,)` array inputs without shape errors across repeated calls.
    * [ ] Delete `dummy_1d.py` once Phase 2 is complete.

---

## Phase 2: 1D Meanline Core
**Status:** `[NOT STARTED]`
**Objective:** Build a mathematically robust, framework-independent 1D meanline aerodynamic solver implementing the OTAC 5-equation governing set at the design point (index `[0]` of all arrays).

* **Milestone 2.1: Kinematics, Velocity Triangles & Euler Equation**
    * **Target Files:** `src/solver/kinematics.py`, `tests/unit/test_kinematics.py`
    * [ ] Write unit tests for ideal (lossless) velocity triangle calculations with analytically known outputs, covering both rotating and non-rotating cases.
    * [ ] Implement the Euler turbomachinery equation: `h_t2 - h_t1 = omega * (r2 * V_theta2 - r1 * V_theta1)`. For stator rows (`omega = 0`) this is trivially satisfied and must not be added to the active residual vector.
    * [ ] Implement functions to compute all derived `FlowStation` fields from the 7 primary inputs plus `omega`, using the cylindrical coordinate system defined in the Nomenclature section.
    * [ ] Write a test asserting rothalpy conservation across an isentropic rotating blade row: `rothalpy_in == rothalpy_out`.
    * [ ] All functions must be pure (no side effects) to satisfy the `@numba.njit` requirement in Milestone 2.3.

* **Milestone 2.2: Loss, Deviation & Blockage Sockets**
    * **Target Files:** `src/solver/sockets/loss_ainley_mathieson.py`, `src/solver/sockets/deviation_standard.py`, `src/solver/sockets/blockage_fixed.py`, `tests/unit/test_sockets.py`
    * [ ] Define the socket callable interface. Loss sockets: `loss(inlet: FlowStation, geometry: SegmentGeometry) -> float`, returning `delta_P_t_loss`. Deviation sockets: `deviation(inlet: FlowStation, geometry: SegmentGeometry) -> float`, returning deviation angle `delta`. Blockage sockets: `blockage(geometry: SegmentGeometry) -> tuple[float, float]`, returning `(A_blockage, r_blockage)`.
    * [ ] Write unit tests asserting that the Ainley-Mathieson socket returns a positive `delta_P_t_loss` for non-zero incidence and that applying it to an ideal state produces a measurable drop in efficiency.
    * [ ] Implement `loss_ainley_mathieson`. Incidence is computed internally as `i = beta_1 - metal_angle_le`. All required parameters not directly present in `SegmentGeometry` must be derived internally.
    * [ ] Implement `deviation_standard`: returns `delta` such that the residual can apply `beta_2 = metal_angle_te - delta`. Positive `delta` means the fluid exits further from axial than the blade trailing edge, per the sign convention in Nomenclature.
    * [ ] Implement `blockage_fixed`: applies fixed fractional reductions. Effective flow area: `A_flow = A_machine - A_blockage`. Effective radius: `r_flow = r_machine - r_blockage`.
    * [ ] Write a test confirming that a zero-blockage socket returns `(0.0, 0.0)` and has no effect on the effective area or radius constraints.

* **Milestone 2.3: Governing Equation Residuals & Solver Backend**
    * **Target Files:** `src/solver/residuals.py`, `src/solver/backend.py`, `tests/unit/test_residuals.py`
    * [ ] Write convergence tests using known solvable configurations with analytically verifiable residuals, covering both rotor (5 residuals) and stator (4 residuals) cases.
    * [ ] Implement the OTAC governing equation residuals in `residuals.py` as a pure function decorated with `@numba.njit`. The residuals for a rotor are:
        ```
        R1 = h_t2 - h_t1 - omega * (r2 * V_theta2 - r1 * V_theta1)   [Euler / energy]
        R2 = P_t2 - (P_t2_ideal - delta_P_t_loss)                      [total pressure loss]
        R3 = beta_2 - (metal_angle_te - delta)                          [turning / deviation]
        R4 = r2 - (r_machine2 - r_blockage2)                           [effective radius]
        R5 = A2 - (A_machine2 - A_blockage2)                           [effective area]
        ```
        For stator rows, omit R1. Solver unknowns are `{h_t2, P_t2, alpha_2, radius_2, MN_2}`.
    * [ ] Verify JIT compilation of residual functions succeeds in isolation before wiring into the backend.
    * [ ] Implement `backend.py` with the interface:
        ```python
        def solve(residual_fn, x0, inlet: FlowStation, row_geom: BladeRowGeometry,
                  seg_geom: SegmentGeometry, sockets: dict, backend='scipy')
        ```
        The `scipy` backend dispatches to `scipy.optimize.root` with `method='lm'`. The `openmdao` backend wraps the same residual function in a structure compatible with an OpenMDAO `ImplicitComponent` with finite-difference partials — no OpenMDAO import is required at this stage; the wrapper is written to be drop-in compatible when OpenMDAO is added at a later phase.
    * [ ] The OpenMDAO-compatible backend wrapper must maintain a cached problem state, rebuilt only when problem dimensions change, to avoid setup overhead during repeated calls.
    * [ ] The solver must raise a descriptive `ConvergenceError` (not silently return NaN) if the backend fails to converge within tolerance.

* **Milestone 2.4: BladeRow Orchestration**
    * **Target Files:** `src/solver/blade_row.py`
    * [ ] Implement `BladeRow.solve(inlet: FlowStation, row_geom: BladeRowGeometry, seg_geom: SegmentGeometry, sockets: dict, omega: float) -> FlowStation` as the public-facing entry point for a single blade row solution. This is the intended future pyCycle coupling boundary — its signature must not be changed without explicit instruction.
    * [ ] The `sockets` dict accepts keys `'loss'`, `'deviation'`, `'blockage'` mapping to any callable satisfying the socket interface from Milestone 2.2. Default to the Phase 2.2 implementations if not supplied.
    * [ ] Write a test verifying that substituting a zero-loss socket stub produces isentropic output (`P_t2 == P_t1`).
    * [ ] Write a test verifying that substituting a zero-deviation socket stub produces `beta_2 == metal_angle_te` exactly.
    * [ ] Write a test verifying that `BladeRow.solve()` has no imports from `openmdao` or `pycycle` anywhere in its call stack.

* **Milestone 2.5: Standalone Verification — Smith Chart**
    * **Target Files:** `tests/integration/test_smith_chart.py`
    * [ ] Reproduce the OTAC single-stage axial turbine Smith chart: sweep flow coefficient `phi_coeff = V_m / U` and loading coefficient `psi = delta_h_t / U^2` across the design space at meanline conditions using the Ainley-Mathieson loss socket.
    * [ ] Assert that computed peak adiabatic efficiency contours agree with the OTAC reference chart to within 1 percentage point.
    * [ ] As a secondary check, verify the non-dimensional loss parameter vs. rotor incidence curve matches the OTAC/Ainley-Mathieson reference shape: bowl-shaped with minimum near design incidence.
    * [ ] **Hard gate:** Phase 3 must not begin until this test passes.

---

## Phase 3: Off-Design Analysis
**Status:** `[NOT STARTED]`
**Objective:** Extend the design-point solver to off-design operating conditions. Blade geometry is fixed from the design solution; the solver finds performance at arbitrary `N_c` and `m_dot_c`.

* **Milestone 3.1: Off-Design Mode & Metal Angle Interpolation**
    * **Target Files:** `src/solver/blade_row.py`
    * [ ] Implement off-design mode in `BladeRow`: accept a fixed `SegmentGeometry` from a prior design solve. Incidence is computed as `i = beta_1 - metal_angle_le` at the current operating point and passed to the loss and deviation sockets.
    * [ ] Implement metal angle interpolation: as the flow radius of a segment changes during off-design, interpolate `metal_angle_le` and `metal_angle_te` from the design values saved at each radial station. For meanline analysis this is a single-station lookup; the interpolation logic must be written to be extensible for Phase 4 multi-stream analysis without interface changes.

* **Milestone 3.2: Off-Design Verification**
    * **Target Files:** `tests/integration/test_1d_solver.py`
    * [ ] Using the NASA 23B-20 single-stage axial compressor as the reference case, run a sweep of operating points across multiple corrected speeds.
    * [ ] Assert that the computed speed-line shape (pressure ratio and efficiency vs. corrected mass flow) is qualitatively correct: monotonically decreasing pressure ratio with increasing mass flow per speed line, with efficiency peaking near design mass flow.
    * [ ] Assert that the design point result is recovered when the design-point boundary conditions are re-applied in off-design mode.

---

## Phase 4: Streamline Analysis
**Status:** `[NOT STARTED]`
**Objective:** Upgrade solver fidelity to hub-to-tip distributions, exploiting the array-based architecture from Phase 1. Expands the meanline solver to `n` streamtubes per blade row, requiring the full OTAC `7n`-equation system.

* **Milestone 4.1: Expander & Reducer**
    * **Target Files:** `src/solver/blade_row.py`, `tests/unit/test_data_structures.py`
    * [ ] Implement `Expander`: partitions a single inlet `FlowStation` into `n` streamtube `FlowStation` objects. Radial stations are distributed to fill the annulus without gaps: `r_eff_inner[i+1] = r_eff_outer[i]`.
    * [ ] Implement `Reducer`: aggregates `n` exit `FlowStation` objects back to a single mass-flow-averaged `FlowStation`.
    * [ ] Write tests asserting round-trip conservation: `Reducer(Expander(station, n))` recovers the original `m_dot`, `h_t`, and `P_t` to within numerical tolerance.

* **Milestone 4.2: Radial Equilibrium & Full Streamline Residuals**
    * **Target Files:** `src/solver/residuals.py`
    * [ ] Expand the residual vector to the full `7n` streamline system. The `n-1` radial equilibrium equations replace the single aggregate area equation:
        ```
        (1/rho) * dP/dr = (V_theta^2 / r) - (V_m^2 / r_s) * cos(phi)
        ```
        where `r_s` is the streamline radius of curvature.
    * [ ] Retain 1 aggregate radius constraint and 1 aggregate area constraint to anchor the hub-to-tip extent against blockage.
    * [ ] Populate the `n` slope equations `phi_2i, r_si = f_spline` using a `scipy` cubic spline fit to the current streamline positions at each iteration.

* **Milestone 4.3: Streamline Matrix Solver & Verification**
    * **Target Files:** `src/solver/blade_row.py`, `tests/integration/test_streamline_solver.py`
    * [ ] Upgrade `BladeRow.solve()` to assemble and solve the `7n x 7n` system. The backend interface from Phase 2.3 must be unchanged.
    * [ ] Write a test verifying that at `n=1` the streamline solver produces results identical to the meanline solver to within numerical tolerance.
    * [ ] Write a test verifying that spanwise-averaged outputs from an `n=5` run agree with the `n=1` meanline result to within acceptable engineering tolerance.
    * [ ] Validate against the OTAC E3 HPC or NASA 74A compressor streamline reference data: assert spanwise velocity distributions agree with the published results to within an acceptable tolerance.

---

## Known Gaps & Deferred Items
*(Identified in OTAC source documentation as unresolved or out of scope. An LLM must not attempt to implement any of these without a referenced methodology and explicit instruction.)*

* **Map generation and pyCycle integration:** Deferred. The `BladeRow.solve()` public interface is the intended future coupling boundary. No changes to its signature are permitted without explicit instruction.
* **Compressor choking:** OTAC documents identify a robust compressor choking mechanism as an unresolved gap. Turbine choking (throat area constraint, exit static pressure as independent variable) is documented and may be added to Phase 3 if required. Compressor choking must not be attempted.
* **Entropy-based compressor loss models:** Under active development at NASA Glenn. Ainley-Mathieson is the placeholder; no replacement is to be implemented without a referenced source equation set.
* **Secondary flow and cooling flow:** Not implemented in OTAC. Out of scope for all phases.
* **Radial mixing between blade rows:** Out of scope for Phase 4.
* **Supersonic and free-vortex convergence:** OTAC documents known solver instability in these regimes. Raising `ConvergenceError` is the correct response; no special handling is to be attempted.
* **B-spline slope and curvature:** Phase 4 uses `scipy` cubic splines as the initial implementation. B-spline upgrade is deferred.
* **Tandem stators and counter-rotating stages:** Out of scope.
* **Expander/Reducer partitioning logic:** The Phase 4.1 implementation uses equal mass flux division as an approximation. The correct partitioning algorithm should be sourced from turbomachinery literature before Phase 4 begins.