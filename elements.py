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
        self.name  = name
        self.mode  = mode
        self.Fl_I  = FlowStation(f"{name}.Fl_I")
        self.Fl_O  = FlowStation(f"{name}.Fl_O")

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
        self.n     = n_streams
        self.r_hub = r_hub
        self.r_tip = r_tip
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
            r_in_sq  = r_sq_hub + i       * (r_sq_tip - r_sq_hub) / self.n
            r_out_sq = r_sq_hub + (i + 1) * (r_sq_tip - r_sq_hub) / self.n
            r_c      = np.sqrt(0.5 * (r_in_sq + r_out_sq))

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
        ports   = self.stream_inlets
        total_m = sum(p.m for p in ports)
        if total_m < 1e-12:
            return

        def wavg(attr):
            return sum(p.m * getattr(p, attr) for p in ports) / total_m

        # Method assumed updated in flowstation.py to match new variables
        self.Fl_O.set_state_from_h_t_s(
            m   = total_m,
            h_t = wavg("h_t"),
            s_s = wavg("s_s"),
            r   = wavg("r"),
            v_u = wavg("v_u"),
            v_m = wavg("v_m"),
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
        self.n     = n_streams
        self.r_hub = r_hub
        self.r_tip = r_tip

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

        self._r_bounds = self._equal_area_bounds(r_hub, r_tip, n_streams)
        self._update_segment_geometry(self._r_bounds)

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

    @staticmethod
    def _static_pressure(p_t: float, h_t: float, v_m: float,
                         v_u: float) -> float:
        v_sq = v_m**2 + v_u**2
        t    = max((h_t - 0.5 * v_sq) / CP, 1.0)
        t_t  = max(h_t / CP, 1.0)
        return p_t * (t / t_t) ** (GAMMA / (GAMMA - 1.0))

    @staticmethod
    def _density(p_t: float, h_t: float, v_m: float, v_u: float) -> float:
        v_sq = v_m**2 + v_u**2
        t    = max((h_t - 0.5 * v_sq) / CP, 1.0)
        p_s  = BladeRow._static_pressure(p_t, h_t, v_m, v_u)
        return max(p_s / (R * t), 1e-6)

    def _re_residual(self, i: int, h_t2: np.ndarray, p_t2: np.ndarray,
                     beta2: np.ndarray, v_m2: np.ndarray,
                     r_bounds: np.ndarray, omega: float) -> float:
        r2_i   = 0.5 * (r_bounds[i]   + r_bounds[i + 1])
        r2_ip1 = 0.5 * (r_bounds[i+1] + r_bounds[i + 2])

        v_u_i   = omega * r2_i   - v_m2[i]   * np.tan(beta2[i])
        v_u_ip1 = omega * r2_ip1 - v_m2[i+1] * np.tan(beta2[i+1])

        p_s_i   = self._static_pressure(p_t2[i],   h_t2[i],   v_m2[i],   v_u_i)
        p_s_ip1 = self._static_pressure(p_t2[i+1], h_t2[i+1], v_m2[i+1], v_u_ip1)
        rho_i   = self._density(p_t2[i],   h_t2[i],   v_m2[i],   v_u_i)
        rho_ip1 = self._density(p_t2[i+1], h_t2[i+1], v_m2[i+1], v_u_ip1)

        rho_avg = 0.5 * (rho_i + rho_ip1)
        v_u_avg = 0.5 * (v_u_i + v_u_ip1)
        r_avg   = 0.5 * (r2_i  + r2_ip1)
        dr      = r2_ip1 - r2_i

        re_rhs = rho_avg * v_u_avg**2 / max(r_avg, 1e-6) * dr
        return (p_s_ip1 - p_s_i - re_rhs) / max(p_s_i, 1.0)

    def residuals_5n(self, vars_flat: np.ndarray, omega: float) -> np.ndarray:
        n = self.n

        h_t2  = vars_flat[0*n : 1*n]
        p_t2  = vars_flat[1*n : 2*n]
        beta2 = vars_flat[2*n : 3*n]
        v_m2  = vars_flat[3*n : 4*n]
        r_int = vars_flat[4*n:]

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
        agg = self.Fl_I
        for seg in self.segments:
            seg.Fl_I.copy_from(agg)
            seg.Fl_I.m = agg.m / n

        # Extract values using updated names
        h_t0   = np.full(n, agg.h_t * (1.02 if omega > 0 else 0.98))
        p_t0   = np.full(n, agg.p_t * (1.3  if omega > 0 else 0.97))
        beta0  = np.array([seg.beta_te for seg in self.segments])
        v_m0   = np.full(n, agg.v_m)
        r_int0 = self._r_bounds[1:-1].copy()

        x0 = np.concatenate([h_t0, p_t0, beta0, v_m0, r_int0])

        sol, _, ier, msg = fsolve(
            self.residuals_5n, x0, args=(omega,), full_output=True
        )
        if ier != 1:
            print(f"  WARNING [{self.name}]: 5n-1 solver did not converge — {msg}")

        h_t2  = sol[0*n : 1*n]
        p_t2  = sol[1*n : 2*n]
        beta2 = sol[2*n : 3*n]
        v_m2  = sol[3*n : 4*n]
        r_int = sol[4*n:]
        r_b   = self._bounds_from_interior(self.r_hub, r_int, self.r_tip)

        for i, seg in enumerate(self.segments):
            r2_i = 0.5 * (r_b[i] + r_b[i + 1])
            seg.write_exit(h_t2[i], p_t2[i], beta2[i], r2_i, v_m2[i], omega)

        self._write_aggregate_exit()

    def _write_aggregate_exit(self):
        segs    = self.segments
        total_m = sum(s.Fl_O.m for s in segs)
        def wavg(attr):
            return sum(s.Fl_O.m * getattr(s.Fl_O, attr) for s in segs) / total_m

        self.Fl_O.set_state_from_h_t_s(
            m   = total_m,
            h_t = wavg("h_t"),
            s_s = wavg("s_s"),
            r   = wavg("r"),
            v_u = wavg("v_u"),
            v_m = wavg("v_m")
        )


# =============================================================================
# InletBoundary
# =============================================================================
class InletBoundary(Element):
    # Updated variables to match new FlowStation nomenclature
    def __init__(self, name: str, m: float, p_t: float, t_t: float,
                 r: float, v_u: float, v_m: float, mode: str = "des"):
        super().__init__(name, mode)
        self._m   = m;
        self._p_t = p_t;
        self._t_t = t_t
        self._r   = r;
        self._v_u = v_u;
        self._v_m = v_m

    def residuals(self, vars, **kwargs):
        return []

    def execute(self, **kwargs):
        self.Fl_O.set_state(self._m, self._p_t, self._t_t,
                            self._r, self._v_u, self._v_m)