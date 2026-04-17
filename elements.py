import numpy as np
from scipy.optimize import fsolve

from flowstation import FlowStation, link_ports
from streamsegment import StreamSegment
from subelements import LossModel, DeviationModel, BlockageModel
from thermo import CP, GAMMA, R, entropy_from_pt_tt


# =============================================================================
# Base class (unchanged)
# =============================================================================
class Element:
    def __init__(self, name: str, mode: str = "des"):
        self.name  = name
        self.mode  = mode
        self.Fl_I  = FlowStation(f"{name}.Fl_I")
        self.Fl_O  = FlowStation(f"{name}.Fl_O")

    def residuals(self, vars, **kwargs):
        raise NotImplementedError

    def execute(self, **kwargs):
        raise NotImplementedError


# =============================================================================
# Expander — splits one aggregate flow into n streamtube flows
# =============================================================================
class Expander(Element):
    """
    Divides the aggregate inlet state into n equal-mass, equal-area
    streamtube outlet states.  Each outlet FlowStation is tagged with
    the centroid radius of its annular slice.

    In a full implementation the mass split could be non-uniform (e.g.
    specified via a spanwise mass-flux profile); equal split is the
    correct starting point for a uniform inlet.
    """

    def __init__(self, name: str, n_streams: int,
                 r_hub: float, r_tip: float, mode: str = "des"):
        super().__init__(name, mode)
        self.n     = n_streams
        self.r_hub = r_hub
        self.r_tip = r_tip
        # One outlet port per streamtube
        self.stream_outlets = [
            FlowStation(f"{name}.Fl_O_{i}") for i in range(n_streams)
        ]

    def residuals(self, vars, **kwargs):
        return []

    def execute(self, **kwargs):
        inlet    = self.Fl_I
        m_per    = inlet.m / self.n
        r_sq_hub = self.r_hub**2
        r_sq_tip = self.r_tip**2

        for i, port in enumerate(self.stream_outlets):
            # Centroid of the i-th equal-area annular slice
            r_in_sq  = r_sq_hub + i       * (r_sq_tip - r_sq_hub) / self.n
            r_out_sq = r_sq_hub + (i + 1) * (r_sq_tip - r_sq_hub) / self.n
            r_c      = np.sqrt(0.5 * (r_in_sq + r_out_sq))

            port.copy_from(inlet)
            port.m = m_per
            port.r = r_c

        # Aggregate outlet mirrors the inlet (for chaining to BladeRow.Fl_I)
        self.Fl_O.copy_from(inlet)


# =============================================================================
# Reducer — sums n streamtube flows into one aggregate flow
# =============================================================================
class Reducer(Element):
    """
    Mass-flow-weighted average of n StreamSegment exit states.
    Total enthalpy and entropy are averaged (not Pt, which is non-linear
    with entropy — the aggregate Pt is recomputed from the averaged ht/s).
    """

    def __init__(self, name: str, n_streams: int, mode: str = "des"):
        super().__init__(name, mode)
        self.n = n_streams
        self.stream_inlets = [
            FlowStation(f"{name}.Fl_I_{i}") for i in range(n_streams)
        ]

    def residuals(self, vars, **kwargs):
        return []

    def execute(self, **kwargs):
        ports   = self.stream_inlets
        total_m = sum(p.m for p in ports)
        if total_m < 1e-12:
            return

        def wavg(attr):
            return sum(p.m * getattr(p, attr) for p in ports) / total_m

        self.Fl_O.set_state_from_ht_s(
            m       = total_m,
            ht      = wavg("ht"),
            s       = wavg("s"),
            r       = wavg("r"),
            v_theta = wavg("v_theta"),
            v_m     = wavg("v_m"),
        )


# =============================================================================
# BladeRow — Assembly of n StreamSegments with a dedicated 5n-1 solver
# =============================================================================
class BladeRow(Element):
    """
    Core OTAC blade row object.

    Variable layout for the internal solver (flat vector, length 5n-1):
        [ht2_0 .. ht2_{n-1},        n  — Euler equation
         pt2_0 .. pt2_{n-1},        n  — loss equation
         beta2_0 .. beta2_{n-1},    n  — turning/deviation equation
         vm2_0 .. vm2_{n-1},        n  — continuity equation
         r_b_1 .. r_b_{n-1}]        n-1 — radial equilibrium

    The hub boundary r_b_0 = r_hub and tip boundary r_b_n = r_tip are
    fixed; only the n-1 interior boundaries are free.

    Mass flow per streamtube is fixed at inlet.m / n (continuity is
    trivially satisfied by the equal split from Expander), which removes
    ṁ from the unknowns and reduces 7n → 5n-1 active variables.

    Meridional slope phi and curvature rs are treated as parameters
    updated externally by SlopeAndCurvature before each BladeRow solve.
    """

    def __init__(self, name: str, n_streams: int,
                 r_hub: float, r_tip: float,
                 beta_le_dist: list, beta_te_dist: list,
                 loss_models: list, deviation_models: list,
                 blockage_models: list, mode: str = "des"):
        super().__init__(name, mode)
        self.n     = n_streams
        self.r_hub = r_hub
        self.r_tip = r_tip

        # Build StreamSegment children
        self.segments: list[StreamSegment] = []
        for i in range(n_streams):
            seg = StreamSegment(
                name            = f"{name}.SS_{i}",
                beta_le         = beta_le_dist[i],
                beta_te         = beta_te_dist[i],
                loss_model      = loss_models[i],
                deviation_model = deviation_models[i],
                blockage_model  = blockage_models[i],
                mode            = mode,
            )
            self.segments.append(seg)

        # Equal-area initial boundary radii (n+1 values)
        self._r_bounds = self._equal_area_bounds(r_hub, r_tip, n_streams)
        self._update_segment_geometry(self._r_bounds)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _equal_area_bounds(r_hub: float, r_tip: float, n: int) -> np.ndarray:
        r_sq_hub = r_hub**2
        r_sq_tip = r_tip**2
        return np.array([
            np.sqrt(r_sq_hub + k * (r_sq_tip - r_sq_hub) / n)
            for k in range(n + 1)
        ])

    def _update_segment_geometry(self, r_bounds: np.ndarray):
        for i, seg in enumerate(self.segments):
            seg.r_inner = r_bounds[i]
            seg.r_outer = r_bounds[i + 1]

    @staticmethod
    def _bounds_from_interior(r_hub: float, r_int: np.ndarray,
                               r_tip: float) -> np.ndarray:
        return np.concatenate([[r_hub], r_int, [r_tip]])

    # ------------------------------------------------------------------
    # Static thermodynamic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _static_pressure(pt: float, ht: float, vm: float,
                         v_theta: float) -> float:
        v_sq = vm**2 + v_theta**2
        t    = max((ht - 0.5 * v_sq) / CP, 1.0)
        tt   = max(ht / CP, 1.0)
        return pt * (t / tt) ** (GAMMA / (GAMMA - 1.0))

    @staticmethod
    def _density(pt: float, ht: float, vm: float, v_theta: float) -> float:
        v_sq = vm**2 + v_theta**2
        t    = max((ht - 0.5 * v_sq) / CP, 1.0)
        ps   = BladeRow._static_pressure(pt, ht, vm, v_theta)
        return max(ps / (R * t), 1e-6)

    # ------------------------------------------------------------------
    # Radial equilibrium residual (simplified, no slope/curvature)
    # ------------------------------------------------------------------
    def _re_residual(self, i: int, ht2: np.ndarray, pt2: np.ndarray,
                     beta2: np.ndarray, vm2: np.ndarray,
                     r_bounds: np.ndarray, omega: float) -> float:
        """
        Simplified radial equilibrium between streamtubes i and i+1.

        Integrates  dPs/dr = rho * Vth^2 / r  across the midpoint gap,
        using a first-order finite difference between adjacent centroids.

        When SlopeAndCurvature is active, add the curvature term:
            - Vm^2 * cos(phi) / rs
        inside the parentheses.
        """
        r2_i   = 0.5 * (r_bounds[i]   + r_bounds[i + 1])
        r2_ip1 = 0.5 * (r_bounds[i+1] + r_bounds[i + 2])

        vt_i   = omega * r2_i   - vm2[i]   * np.tan(beta2[i])
        vt_ip1 = omega * r2_ip1 - vm2[i+1] * np.tan(beta2[i+1])

        ps_i   = self._static_pressure(pt2[i],   ht2[i],   vm2[i],   vt_i)
        ps_ip1 = self._static_pressure(pt2[i+1], ht2[i+1], vm2[i+1], vt_ip1)
        rho_i  = self._density(pt2[i],   ht2[i],   vm2[i],   vt_i)
        rho_ip1= self._density(pt2[i+1], ht2[i+1], vm2[i+1], vt_ip1)

        rho_avg = 0.5 * (rho_i + rho_ip1)
        vt_avg  = 0.5 * (vt_i  + vt_ip1)
        r_avg   = 0.5 * (r2_i  + r2_ip1)
        dr      = r2_ip1 - r2_i

        re_rhs = rho_avg * vt_avg**2 / max(r_avg, 1e-6) * dr
        return (ps_ip1 - ps_i - re_rhs) / max(ps_i, 1.0)

    # ------------------------------------------------------------------
    # Full 5n-1 residual vector
    # ------------------------------------------------------------------
    def residuals_5n(self, vars_flat: np.ndarray, omega: float) -> np.ndarray:
        n = self.n

        ht2   = vars_flat[0*n : 1*n]
        pt2   = vars_flat[1*n : 2*n]
        beta2 = vars_flat[2*n : 3*n]
        vm2   = vars_flat[3*n : 4*n]
        r_int = vars_flat[4*n:]           # n-1 interior boundary radii

        r_bounds = self._bounds_from_interior(self.r_hub, r_int, self.r_tip)
        self._update_segment_geometry(r_bounds)

        resids = []

        # 4n per-tube residuals (Euler, loss, turning, continuity)
        for i, seg in enumerate(self.segments):
            r2_i = 0.5 * (r_bounds[i] + r_bounds[i + 1])
            r_euler, r_loss, r_turn, r_cont = seg.per_tube_residuals(
                ht2[i], pt2[i], beta2[i], r2_i, vm2[i], omega
            )
            resids.extend([r_euler, r_loss, r_turn, r_cont])

        # n-1 radial equilibrium residuals
        for i in range(n - 1):
            resids.append(
                self._re_residual(i, ht2, pt2, beta2, vm2, r_bounds, omega)
            )

        return np.array(resids)   # length 4n + (n-1) = 5n-1

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    def execute(self, omega: float):
        n = self.n

        # Propagate aggregate inlet to each segment (equal split)
        agg = self.Fl_I
        for seg in self.segments:
            seg.Fl_I.copy_from(agg)
            seg.Fl_I.m = agg.m / n

        # Initial guess
        ht0    = np.full(n, agg.ht * (1.02 if omega > 0 else 0.98))
        pt0    = np.full(n, agg.pt * (1.3  if omega > 0 else 0.97))
        beta0  = np.array([seg.beta_te for seg in self.segments])
        vm0    = np.full(n, agg.v_m)
        r_int0 = self._r_bounds[1:-1].copy()

        x0 = np.concatenate([ht0, pt0, beta0, vm0, r_int0])

        sol, _, ier, msg = fsolve(
            self.residuals_5n, x0, args=(omega,), full_output=True
        )
        if ier != 1:
            print(f"  WARNING [{self.name}]: 5n-1 solver did not converge — {msg}")

        # Unpack and write exit states
        ht2   = sol[0*n : 1*n]
        pt2   = sol[1*n : 2*n]
        beta2 = sol[2*n : 3*n]
        vm2   = sol[3*n : 4*n]
        r_int = sol[4*n:]
        r_b   = self._bounds_from_interior(self.r_hub, r_int, self.r_tip)

        for i, seg in enumerate(self.segments):
            r2_i = 0.5 * (r_b[i] + r_b[i + 1])
            seg.write_exit(ht2[i], pt2[i], beta2[i], r2_i, vm2[i], omega)

        self._write_aggregate_exit()

    def _write_aggregate_exit(self):
        """Mass-flow-weighted aggregate over all StreamSegment exits."""
        segs    = self.segments
        total_m = sum(s.Fl_O.m for s in segs)
        def wavg(attr):
            return sum(s.Fl_O.m * getattr(s.Fl_O, attr) for s in segs) / total_m

        self.Fl_O.set_state_from_ht_s(
            m=total_m, ht=wavg("ht"), s=wavg("s"),
            r=wavg("r"), v_theta=wavg("v_theta"), v_m=wavg("v_m"),
        )


# =============================================================================
# InletBoundary (unchanged interface, minor cleanup)
# =============================================================================
class InletBoundary(Element):
    def __init__(self, name: str, m: float, pt: float, tt: float,
                 r: float, v_theta: float, v_m: float, mode: str = "des"):
        super().__init__(name, mode)
        self._m = m;  self._pt = pt;  self._tt = tt
        self._r = r;  self._vt = v_theta;  self._vm = v_m

    def residuals(self, vars, **kwargs):
        return []

    def execute(self, **kwargs):
        self.Fl_O.set_state(self._m, self._pt, self._tt,
                            self._r, self._vt, self._vm)