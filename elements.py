import numpy as np
from scipy.optimize import fsolve

from flowstation import FlowStation, link_ports
from subelements import LossModel, DeviationModel
from thermo import CP, GAMMA, R, entropy_from_pt_tt


# =============================================================================
# Element base class
# =============================================================================
class Element:
    def __init__(self, name: str, mode: str = "des"):
        self.name = name
        self.mode = mode
        self.Fl_I = FlowStation(f"{name}.Fl_I")
        self.Fl_O = FlowStation(f"{name}.Fl_O")

    def execute(self, **kwargs):
        raise NotImplementedError


# =============================================================================
# InletBoundary
# =============================================================================
class InletBoundary(Element):
    """Injects fixed boundary conditions into the model."""

    def __init__(self, name: str, m: float, p_t: float, t_t: float,
                 r: float, v_u: float, v_m: float):
        super().__init__(name)
        self._m   = m
        self._p_t = p_t
        self._t_t = t_t
        self._r   = r
        self._v_u = v_u
        self._v_m = v_m

    def execute(self, **kwargs):
        self.Fl_O.set_state(self._m, self._p_t, self._t_t,
                            self._r, self._v_u, self._v_m)


# =============================================================================
# BladeRow — single streamtube (meanline)
# =============================================================================
class BladeRow(Element):
    """
    Meanline blade row. Solves 5 equations for 5 unknowns.

    Solver variables
    ----------------
        h_t2  total enthalpy at exit          [J/kg]
        p_t2  total pressure at exit          [Pa]
        beta2 relative flow angle at exit     [rad]
        v_m2  meridional velocity at exit     [m/s]
        r2    exit radius (fixed constraint)  [m]

    Governing equations
    -------------------
        (1) Euler:      h_t2 - h_t1 = U2*Vu2 - U1*Vu1
        (2) Loss:       p_t2 = p_t2_ideal - delta_Pt
        (3) Turning:    beta2 = beta_te + delta
        (4) Continuity: m = rho2 * Vm2 * A_eff
        (5) Radius:     r2 = r_exit  (geometry constraint)

    Mass conservation (m2 = m1) is trivially satisfied for a single
    streamtube and drops out, reducing the 7-equation set to 5.

    BladeRow-specific state (not in FlowStation)
    ---------------------------------------------
        U_in      blade speed at inlet radius   [m/s]
        U_out     blade speed at exit radius    [m/s]
        incidence inlet incidence angle         [rad]
        PR        total-to-total pressure ratio [-]
        eta_tt    total-to-total isentropic eff [-]
    """

    def __init__(self, name: str,
                 r_exit:          float,
                 a_exit:          float,
                 a_blockage:      float,
                 beta_le:         float,
                 beta_te:         float,
                 loss_model:      LossModel,
                 deviation_model: DeviationModel,
                 phi:             float = 0.0,
                 mode:            str   = "des"):
        super().__init__(name, mode)

        self.r_exit     = r_exit
        self.a_exit     = a_exit
        self.a_blockage = a_blockage
        self.beta_le    = beta_le
        self.beta_te    = beta_te
        self.phi        = phi

        self.loss_model      = loss_model
        self.deviation_model = deviation_model

        # Blade-row performance state — set by execute()
        self.U_in      = 0.0
        self.U_out     = 0.0
        self.incidence = 0.0
        self.PR        = 0.0
        self.eta_tt    = 0.0

    # ------------------------------------------------------------------
    # Step 1 — resolve inlet relative frame
    # ------------------------------------------------------------------
    def _setup_inlet_frame(self, omega: float):
        """
        Compute blade speed at inlet and call _resolve_state on Fl_I
        so that Fl_I.beta, Fl_I.w, Fl_I.rho, Fl_I.ma_r etc. are
        all populated before the solver starts.
        """
        self.U_in      = omega * self.Fl_I.r
        w_u1           = self.U_in - self.Fl_I.v_u
        beta1          = np.arctan2(w_u1, self.Fl_I.v_m)
        self.incidence = beta1 - self.beta_le

        self.Fl_I._resolve_state(beta=beta1)

    # ------------------------------------------------------------------
    # Step 2 — write trial exit state into Fl_O
    # ------------------------------------------------------------------
    def _update_exit_station(self, h_t2, p_t2, beta2, v_m2, r2, omega):
        """
        Populate Fl_O from solver trial variables and resolve all
        derived quantities. Called on every residual evaluation so
        _residuals() reads directly from self.Fl_O.
        """
        self.U_out = omega * r2
        w_u2       = v_m2 * np.tan(beta2)
        v_u2       = self.U_out - w_u2

        s_s2 = entropy_from_pt_tt(p_t2, np.maximum(h_t2 / CP, 1.0))
        self.Fl_O.set_state_from_h_t_s(
            m=self.Fl_I.m, h_t=h_t2, s_s=s_s2,
            r=r2, v_u=v_u2, v_m=v_m2, beta=beta2
        )

    # ------------------------------------------------------------------
    # Step 3 — residuals, reading from self.Fl_I and self.Fl_O
    # ------------------------------------------------------------------
    def _residuals(self, vars: np.ndarray, omega: float) -> np.ndarray:
        h_t2, p_t2, beta2, v_m2, r2 = vars

        self._update_exit_station(h_t2, p_t2, beta2, v_m2, r2, omega)

        # Isentropic reference total pressure
        p_t2_ideal = (self.Fl_I.p_t
                      * (self.Fl_O.t_t / np.maximum(self.Fl_I.t_t, 1.0))
                      ** (self.Fl_I.gamma / (self.Fl_I.gamma - 1.0)))

        # Subelement calls — read from FlowStation objects
        dp_loss = self.loss_model.compute(self.Fl_I)
        delta   = self.deviation_model.compute(self.Fl_I, self.beta_le, self.Fl_O)

        # Effective flow area
        a_eff = np.maximum(
            np.pi * self.r_exit**2 * np.cos(self.phi) - self.a_blockage,
            1e-8
        )

        # (1) Euler: h_t2 - h_t1 = U2*Vu2 - U1*Vu1
        euler_work = self.U_out * self.Fl_O.v_u - self.U_in * self.Fl_I.v_u
        r_euler    = ((self.Fl_O.h_t - self.Fl_I.h_t) - euler_work) \
                     / np.maximum(np.abs(self.Fl_I.h_t), 1.0)

        # (2) Total pressure loss
        r_loss = (self.Fl_O.p_t - (p_t2_ideal - dp_loss)) \
                 / np.maximum(self.Fl_I.p_t, 1.0)

        # (3) Turning: exit angle = blade metal angle + deviation
        r_turn = self.Fl_O.beta - (self.beta_te + delta)

        # (4) Continuity: m = rho2 * Vm2 * A_eff
        r_cont = (self.Fl_I.m - self.Fl_O.rho * self.Fl_O.v_m * a_eff) \
                 / np.maximum(self.Fl_I.m, 1e-10)

        # (5) Radius constraint
        r_geom = (r2 - self.r_exit) / self.r_exit

        return np.array([r_euler, r_loss, r_turn, r_cont, r_geom])

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    def execute(self, omega: float = 0.0):
        self._setup_inlet_frame(omega)

        print(f"  [{self.name}] mode={self.mode} | "
              f"omega={omega:.1f} rad/s | "
              f"incidence={np.degrees(self.incidence):.2f} deg")

        # Initial guess
        x0 = np.array([
            self.Fl_I.h_t * (1.02 if omega > 0 else 0.98),
            self.Fl_I.p_t * (1.20 if omega > 0 else 0.97),
            self.beta_te,
            self.Fl_I.v_m,
            self.r_exit,
        ])

        # Scale each variable to O(1) for a well-conditioned Jacobian
        scales = np.array([
            np.maximum(np.abs(self.Fl_I.h_t), 1.0),
            np.maximum(self.Fl_I.p_t,         1.0),
            1.0,
            np.maximum(self.Fl_I.v_m,         1.0),
            np.maximum(self.r_exit,            1e-4),
        ], dtype=float)

        sol_norm, _, ier, msg = fsolve(
            lambda x: self._residuals(x * scales, omega),
            x0 / scales,
            full_output=True,
            epsfcn=1e-7,
        )
        if ier != 1:
            print(f"  WARNING [{self.name}]: did not converge — {msg}")

        # Fl_O already holds the converged state from the last residual call
        sol = sol_norm * scales
        self._update_exit_station(*sol, omega)

        # In design mode, freeze the solved area as the blade geometry
        if self.mode == "des":
            self.a_exit = (
                self.Fl_I.m
                / np.maximum(self.Fl_O.rho * self.Fl_O.v_m, 1e-10)
                + self.a_blockage
            )

        # Store performance metrics as BladeRow attributes
        self.PR    = self.Fl_O.p_t / np.maximum(self.Fl_I.p_t, 1.0)
        pr_exp     = (self.Fl_I.gamma - 1.0) / self.Fl_I.gamma
        self.eta_tt = (
            (self.Fl_O.t_t / np.maximum(self.Fl_I.t_t, 1.0)) ** (1.0 / pr_exp) - 1.0
        ) / np.maximum(self.PR ** pr_exp - 1.0, 1e-10)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def report(self):
        print(f"\n  [{self.name}] performance")
        print(f"    PR       : {float(self.PR):.4f}")
        print(f"    eta_tt   : {float(self.eta_tt):.4f}")
        print(f"    U_in     : {float(self.U_in):.2f} m/s")
        print(f"    U_out    : {float(self.U_out):.2f} m/s")
        print(f"    incidence: {float(np.degrees(self.incidence)):.2f} deg")
        print(f"  Inlet velocity triangle")
        print(f"    V={float(self.Fl_I.v):.1f}  Vu={float(self.Fl_I.v_u):.1f}  "
              f"Vm={float(self.Fl_I.v_m):.1f}  W={float(self.Fl_I.w):.1f}  "
              f"beta={float(np.degrees(self.Fl_I.beta)):.1f} deg  "
              f"Ma_r={float(self.Fl_I.ma_r):.3f}")
        print(f"  Exit velocity triangle")
        print(f"    V={float(self.Fl_O.v):.1f}  Vu={float(self.Fl_O.v_u):.1f}  "
              f"Vm={float(self.Fl_O.v_m):.1f}  W={float(self.Fl_O.w):.1f}  "
              f"beta={float(np.degrees(self.Fl_O.beta)):.1f} deg  "
              f"Ma_r={float(self.Fl_O.ma_r):.3f}")