import numpy as np
from scipy.optimize import fsolve

from flowstation import FlowStation, link_ports
from streamsegment import StreamSegment
from subelements import LossModel, DeviationModel, BlockageModel
from thermo import CP, GAMMA, R, entropy_from_pt_tt


# =============================================================================
# Base class
# =============================================================================
class Element:
    def __init__(self, name: str, mode: str = "des"):
        self.name = name
        self.mode = mode
        self.Fl_I = FlowStation(f"{name}.Fl_I")
        self.Fl_O = FlowStation(f"{name}.Fl_O")

    def residuals(self, vars, **kwargs):
        raise NotImplementedError

    def execute(self, **kwargs):
        raise NotImplementedError


# =============================================================================
# Expander
# =============================================================================
class Expander(Element):
    def __init__(self, name: str, n_streams: int,
                 r_hub: float, r_tip: float, mode: str = "des"):
        super().__init__(name, mode)
        self.n = n_streams
        self.r_hub = r_hub
        self.r_tip = r_tip
        self.stream_outlets = [
            FlowStation(f"{name}.Fl_O_{i}") for i in range(n_streams)
        ]

    def residuals(self, vars, **kwargs):
        return []

    def execute(self, **kwargs):
        inlet = self.Fl_I
        m_per = inlet.m / self.n
        r_sq_hub = self.r_hub ** 2
        r_sq_tip = self.r_tip ** 2

        for i, port in enumerate(self.stream_outlets):
            r_in_sq = r_sq_hub + i * (r_sq_tip - r_sq_hub) / self.n
            r_out_sq = r_sq_hub + (i + 1) * (r_sq_tip - r_sq_hub) / self.n
            r_c = np.sqrt(0.5 * (r_in_sq + r_out_sq))

            port.copy_from(inlet)
            port.m = m_per
            port.r = r_c

        self.Fl_O.copy_from(inlet)


# =============================================================================
# Reducer
# =============================================================================
class Reducer(Element):
    def __init__(self, name: str, n_streams: int, mode: str = "des"):
        super().__init__(name, mode)
        self.n = n_streams
        self.stream_inlets = [
            FlowStation(f"{name}.Fl_I_{i}") for i in range(n_streams)
        ]

    def residuals(self, vars, **kwargs):
        return []

    def execute(self, **kwargs):
        ports = self.stream_inlets
        total_m = sum(p.m for p in ports)
        if total_m < 1e-12:
            return

        def wavg(attr):
            return sum(p.m * getattr(p, attr) for p in ports) / total_m

        self.Fl_O.set_state_from_h_t_s(
            m=total_m,
            h_t=wavg("h_t"),
            s_s=wavg("s_s"),
            r=wavg("r"),
            v_u=wavg("v_u"),
            v_m=wavg("v_m"),
        )


# =============================================================================
# BladeRow
# =============================================================================
class BladeRow(Element):
    def __init__(self, name: str, n_streams: int,
                 r_hub: float, r_tip: float,
                 beta_le_dist: list, beta_te_dist: list,
                 loss_models: list, deviation_models: list,
                 blockage_models: list, mode: str = "des"):
        super().__init__(name, mode)
        self.n = n_streams
        self.r_hub = r_hub
        self.r_tip = r_tip

        self.segments: list[StreamSegment] = []
        for i in range(n_streams):
            seg = StreamSegment(
                name=f"{name}.SS_{i}",
                beta_le=beta_le_dist[i],
                beta_te=beta_te_dist[i],
                loss_model=loss_models[i],
                deviation_model=deviation_models[i],
                blockage_model=blockage_models[i],
                mode=mode,
            )
            self.segments.append(seg)

        self._r_bounds = self._equal_area_bounds(r_hub, r_tip, n_streams)
        self._update_segment_geometry(self._r_bounds)

    @staticmethod
    def _equal_area_bounds(r_hub: float, r_tip: float, n: int) -> np.ndarray:
        r_sq_hub = r_hub ** 2
        r_sq_tip = r_tip ** 2
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

    def _re_residual(self, i: int, h_t2: np.ndarray, p_t2: np.ndarray,
                     beta2: np.ndarray, v_m2: np.ndarray,
                     r_bounds: np.ndarray, omega: float) -> float:
        """
        Full Simple Radial Equilibrium (SRE) Equation solver.
        dht/dr - T·ds/dr = Vu · d(r·Vu)/dr / r + Vm · dVm/dr + Vm² · (sin φ / rs - cos φ · dφ/dr)
        """
        r2_i = 0.5 * (r_bounds[i] + r_bounds[i + 1])
        r2_ip1 = 0.5 * (r_bounds[i + 1] + r_bounds[i + 2])
        dr = r2_ip1 - r2_i

        # State i
        v_u_i = omega * r2_i - v_m2[i] * np.tan(beta2[i])
        t_t_i = max(h_t2[i] / CP, 1.0)
        s_s_i = entropy_from_pt_tt(p_t2[i], t_t_i)
        v_sq_i = v_m2[i] ** 2 + v_u_i ** 2
        t_s_i = max((h_t2[i] - 0.5 * v_sq_i) / CP, 1.0)

        # State i+1
        v_u_ip1 = omega * r2_ip1 - v_m2[i + 1] * np.tan(beta2[i + 1])
        t_t_ip1 = max(h_t2[i + 1] / CP, 1.0)
        s_s_ip1 = entropy_from_pt_tt(p_t2[i + 1], t_t_ip1)
        v_sq_ip1 = v_m2[i + 1] ** 2 + v_u_ip1 ** 2
        t_s_ip1 = max((h_t2[i + 1] - 0.5 * v_sq_ip1) / CP, 1.0)

        # Gradients & Averages
        t_avg = 0.5 * (t_s_i + t_s_ip1)
        r_v_u_i = r2_i * v_u_i
        r_v_u_ip1 = r2_ip1 * v_u_ip1

        dh_t_dr = (h_t2[i + 1] - h_t2[i]) / dr
        ds_s_dr = (s_s_ip1 - s_s_i) / dr
        d_rvu_dr = (r_v_u_ip1 - r_v_u_i) / dr
        dvm_dr = (v_m2[i + 1] - v_m2[i]) / dr

        v_u_avg = 0.5 * (v_u_i + v_u_ip1)
        v_m_avg = 0.5 * (v_m2[i] + v_m2[i + 1])
        r_avg = 0.5 * (r2_i + r2_ip1)

        # Curvature terms
        seg_i = self.segments[i]
        seg_ip1 = self.segments[i + 1]
        phi_avg = 0.5 * (seg_i.phi + seg_ip1.phi)
        rs_avg = 0.5 * (seg_i.rs + seg_ip1.rs)
        dphi_dr = (seg_ip1.phi - seg_i.phi) / dr if dr > 1e-6 else 0.0

        curv_term = v_m_avg ** 2 * (np.sin(phi_avg) / rs_avg - np.cos(phi_avg) * dphi_dr)

        # Balance Equation
        LHS = dh_t_dr - t_avg * ds_s_dr
        RHS = (v_u_avg / r_avg) * d_rvu_dr + v_m_avg * dvm_dr + curv_term

        # Normalised residual
        return (LHS - RHS) / max(abs(LHS), 1e-3)

    def residuals_5n(self, vars_flat: np.ndarray, omega: float) -> np.ndarray:
        n = self.n

        h_t2 = vars_flat[0 * n: 1 * n]
        p_t2 = vars_flat[1 * n: 2 * n]
        beta2 = vars_flat[2 * n: 3 * n]
        v_m2 = vars_flat[3 * n: 4 * n]
        r_int = vars_flat[4 * n:]

        r_bounds = self._bounds_from_interior(self.r_hub, r_int, self.r_tip)
        self._update_segment_geometry(r_bounds)

        resids = []

        for i, seg in enumerate(self.segments):
            r2_i = 0.5 * (r_bounds[i] + r_bounds[i + 1])
            r_euler, r_loss, r_turn, r_cont = seg.per_tube_residuals(
                h_t2[i], p_t2[i], beta2[i], r2_i, v_m2[i], omega
            )
            resids.extend([r_euler, r_loss, r_turn, r_cont])

        for i in range(n - 1):
            resids.append(
                self._re_residual(i, h_t2, p_t2, beta2, v_m2, r_bounds, omega)
            )

        return np.array(resids)

    def execute(self, omega: float):
        n = self.n

        # FIX: Rely on upstream component to push the profile. Do NOT override with aggregate average.
        h_t0 = np.array([seg.Fl_I.h_t * (1.02 if omega > 0 else 0.98) for seg in self.segments])
        p_t0 = np.array([seg.Fl_I.p_t * (1.3 if omega > 0 else 0.97) for seg in self.segments])
        beta0 = np.array([seg.beta_te for seg in self.segments])
        v_m0 = np.array([seg.Fl_I.v_m for seg in self.segments])
        r_int0 = self._r_bounds[1:-1].copy()

        x0 = np.concatenate([h_t0, p_t0, beta0, v_m0, r_int0])

        sol, _, ier, msg = fsolve(
            self.residuals_5n, x0, args=(omega,), full_output=True
        )
        if ier != 1:
            print(f"  WARNING [{self.name}]: 5n-1 solver did not converge — {msg}")

        h_t2 = sol[0 * n: 1 * n]
        p_t2 = sol[1 * n: 2 * n]
        beta2 = sol[2 * n: 3 * n]
        v_m2 = sol[3 * n: 4 * n]
        r_int = sol[4 * n:]
        r_b = self._bounds_from_interior(self.r_hub, r_int, self.r_tip)

        for i, seg in enumerate(self.segments):
            r2_i = 0.5 * (r_b[i] + r_b[i + 1])
            seg.write_exit(h_t2[i], p_t2[i], beta2[i], r2_i, v_m2[i], omega)

        self._write_aggregate_exit()

    def _write_aggregate_exit(self):
        segs = self.segments
        total_m = sum(s.Fl_O.m for s in segs)
        if total_m < 1e-12:
            return

        def wavg(attr):
            return sum(s.Fl_O.m * getattr(s.Fl_O, attr) for s in segs) / total_m

        # FIX: Average mass/enthalpy/pressure, deduce the mixed entropy from that
        h_t_agg = wavg("h_t")
        p_t_agg = wavg("p_t")
        t_t_agg = max(h_t_agg / CP, 1.0)
        s_s_agg = entropy_from_pt_tt(p_t_agg, t_t_agg)

        self.Fl_O.set_state_from_h_t_s(
            m=total_m,
            h_t=h_t_agg,
            s_s=s_s_agg,
            r=wavg("r"),
            v_u=wavg("v_u"),
            v_m=wavg("v_m")
        )


# =============================================================================
# InletBoundary
# =============================================================================
class InletBoundary(Element):
    def __init__(self, name: str, m: float, p_t: float, t_t: float,
                 r: float, v_u: float, v_m: float, mode: str = "des"):
        super().__init__(name, mode)
        self._m = m;
        self._p_t = p_t;
        self._t_t = t_t
        self._r = r;
        self._v_u = v_u;
        self._v_m = v_m

    def residuals(self, vars, **kwargs):
        return []

    def execute(self, **kwargs):
        self.Fl_O.set_state(self._m, self._p_t, self._t_t,
                            self._r, self._v_u, self._v_m)