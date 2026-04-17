import numpy as np
from flowstation import FlowStation, link_ports
from thermo import CP, GAMMA, R, entropy_from_pt_tt
from subelements import LossModel, DeviationModel, BlockageModel


class StreamSegment:
    """
    One streamtube across a single blade row.

    Geometry (r_inner, r_outer, phi, rs) is written by BladeRow before
    each residual evaluation.  The segment never drives iteration itself —
    it only computes residuals and writes a converged exit state when asked.
    """

    def __init__(self, name: str, beta_le: float, beta_te: float,
                 loss_model: LossModel, deviation_model: DeviationModel,
                 blockage_model: "BlockageModel", mode: str = "des"):
        self.name = name
        self.Fl_I = FlowStation(f"{name}.Fl_I")
        self.Fl_O = FlowStation(f"{name}.Fl_O")

        # Geometry — updated by BladeRow._update_segment_geometry()
        self.r_inner = 0.0
        self.r_outer = 0.0
        self.phi = 0.0  # meridional slope angle  [rad]
        self.rs = 1e9  # radius of curvature     [m]

        self.beta_le = beta_le
        self.beta_te = beta_te

        self.loss_model = loss_model
        self.deviation_model = deviation_model
        self.blockage_model = blockage_model
        self.mode = mode

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------
    @property
    def r_centroid(self) -> float:
        return 0.5 * (self.r_inner + self.r_outer)

    def annular_area(self, r_inner: float, r_outer: float,
                     phi: float = 0.0) -> float:
        """Annular flow area accounting for meridional slope [m²]."""
        return np.pi * (r_outer ** 2 - r_inner ** 2) * np.cos(phi)

    # ------------------------------------------------------------------
    # Per-tube residuals (called from BladeRow.residuals_5n)
    # ------------------------------------------------------------------
    def per_tube_residuals(self, h_t2: float, p_t2: float, beta2: float,
                           r2: float, v_m2: float, omega: float) -> tuple:
        """
        Returns (res_euler, res_loss, res_turn, res_cont).

        All four are normalised so they are O(1) near the solution,
        which helps the Newton solver's step-size heuristics.
        """
        inlet = self.Fl_I

        # Exit kinematics
        u2 = omega * r2
        w_u2 = v_m2 * np.tan(beta2)
        v_u2 = u2 - w_u2
        v_sq2 = v_m2 ** 2 + v_u2 ** 2

        # Exit thermodynamics
        h_t2 = max(h_t2, 1.0)
        h2 = h_t2 - 0.5 * v_sq2
        t_t2 = h_t2 / CP
        t2 = max(h2 / CP, 1.0)
        p2 = p_t2 * (t2 / max(t_t2, 1.0)) ** (GAMMA / (GAMMA - 1.0))
        rho2 = max(p2 / (R * t2), 1e-6)

        p_t2_ideal = inlet.p_t * (t_t2 / max(inlet.t_t, 1.0)) ** (GAMMA / (GAMMA - 1.0))

        # Subelement calls — implicit inside residual loop
        dp_loss = self.loss_model.compute(inlet, omega, r2, v_m2, beta2)
        delta = self.deviation_model.compute(
            inlet, self.beta_le, omega, r2, v_m2, beta2)
        a_block = self.blockage_model.compute(
            inlet, r2, self.r_inner, self.r_outer)

        # Effective flow area
        a_gross = self.annular_area(self.r_inner, self.r_outer, self.phi)
        a_eff = max(a_gross - a_block, 1e-8)

        # Residuals (normalised)
        euler_work = omega * (r2 * v_u2 - inlet.r * inlet.v_u)
        res_euler = ((h_t2 - inlet.h_t) - euler_work) / max(abs(inlet.h_t), 1.0)
        res_loss = (p_t2 - (p_t2_ideal - dp_loss)) / max(inlet.p_t, 1.0)
        res_turn = beta2 - (self.beta_te + delta)
        res_cont = (inlet.m - rho2 * v_m2 * a_eff) / max(inlet.m, 1e-6)

        return res_euler, res_loss, res_turn, res_cont

    # ------------------------------------------------------------------
    # Commit converged exit state
    # ------------------------------------------------------------------
    def write_exit(self, h_t2: float, p_t2: float, beta2: float,
                   r2: float, v_m2: float, omega: float):
        u2 = omega * r2
        w_u2 = v_m2 * np.tan(beta2)
        v_u2 = u2 - w_u2
        s_s2 = entropy_from_pt_tt(p_t2, max(h_t2 / CP, 1.0))

        # NOTE: If your FlowStation definition for `set_state_from_h_t_s`
        # doesn't take 'beta', you can remove it from kwargs below.
        self.Fl_O.set_state_from_h_t_s(
            m=self.Fl_I.m,
            h_t=h_t2,
            s_s=s_s2,
            r=r2,
            v_u=v_u2,
            v_m=v_m2,
            beta=beta2
        )