from flowstation import FlowStation
from subelements import *

from scipy.optimize import fsolve
# =============================================================================
# 3. ELEMENT BASE CLASS
# =============================================================================
class Element:
    """
    Base class for all OTAC-style components.

    mode : str
        'des' — design mode: geometry is an *output*, solved for by the element.
        'od'  — off-design mode: geometry is *fixed*, performance is the output.

    All state updates must happen inside execute(), never in __init__.
    """

    def __init__(self, name: str, mode: str = "des"):
        self.name = name
        self.mode = mode
        self.Fl_I = FlowStation(name=f"{name}.Fl_I")
        self.Fl_O = FlowStation(name=f"{name}.Fl_O")

    def residuals(self, vars: list, **kwargs) -> list:
        raise NotImplementedError

    def execute(self, **kwargs):
        raise NotImplementedError

# =============================================================================
# 5. BLADE ROW ELEMENT
# =============================================================================
class BladeRow(Element):
    """
    Core meanline blade row.

    Subelements (LossModel, DeviationModel) are owned by the BladeRow and
    called from inside residuals() — fully implicit. The solver Jacobian
    therefore captures how loss and deviation respond to changes in the exit
    state on every Newton step.

    Design mode ('des') : a_exit is back-calculated after solve (area is output).
    Off-design  ('od')  : a_exit is fixed; solver finds matching performance.
    """

    def __init__(
        self,
        name:             str,
        r_exit:           float,
        a_exit:           float,
        a_blockage:       float,
        beta_le:          float,
        beta_te:          float,
        loss_model:       LossModel,
        deviation_model:  DeviationModel,
        mode:             str = "des",
    ):
        super().__init__(name, mode)
        self.r_exit          = r_exit
        self.a_exit          = a_exit
        self.a_blockage      = a_blockage
        self.beta_le         = beta_le
        self.beta_te         = beta_te
        self.loss_model      = loss_model
        self.deviation_model = deviation_model

    # ------------------------------------------------------------------
    # Kinematics helper
    # ------------------------------------------------------------------
    @staticmethod
    def velocity_triangle(omega: float, r: float, v_m: float,
                          v_theta: float = None,
                          beta_rel: float = None) -> dict:
        u = omega * r
        if v_theta is not None:
            w_theta  = u - v_theta
            beta_rel = np.arctan2(w_theta, v_m)
        elif beta_rel is not None:
            w_theta  = v_m * np.tan(beta_rel)
            v_theta  = u - w_theta
        else:
            raise ValueError("Must supply either v_theta or beta_rel.")
        return {"V_theta": v_theta, "V_sq": v_m**2 + v_theta**2, "beta": beta_rel}

    # ------------------------------------------------------------------
    # Residuals — subelements called HERE, implicit in the solver loop
    # ------------------------------------------------------------------
    def residuals(self, vars: list, omega: float) -> list:
        """
        7 residual equations.
        Solver variables: [m2, ht2, pt2, beta2, r2, a_flow2, v_m2]

        ht2 and pt2 are the primary thermodynamic variables (not tt2),
        consistent with OTAC's ht/s-anchored FlowStation.
        """
        m2, ht2, pt2, beta2, r2, a_flow2, v_m2 = vars
        inlet = self.Fl_I

        # Exit kinematics
        kine2    = self.velocity_triangle(omega, r2, v_m2, beta_rel=beta2)
        v_sq2    = kine2["V_sq"]
        v_theta2 = kine2["V_theta"]

        # Exit thermodynamics
        h2   = ht2 - 0.5 * v_sq2
        t2   = h2 / CP
        tt2  = ht2 / CP
        p2   = pt2 * (t2 / tt2) ** (GAMMA / (GAMMA - 1.0))
        rho2 = p2 / (R * t2)

        # Isentropic reference Pt (zero-loss baseline)
        pt2_ideal = inlet.pt * (tt2 / inlet.tt) ** (GAMMA / (GAMMA - 1.0))

        # ---- Implicit subelement calls ----
        dp_loss = self.loss_model.compute(inlet, omega, r2, v_m2, beta2)
        delta   = self.deviation_model.compute(inlet, self.beta_le, omega,
                                               r2, v_m2, beta2)

        # ---- Residuals ----
        # eq1 — mass conservation
        eq1 = (m2 - inlet.m) / inlet.m

        # eq2 — Euler turbine equation
        euler_work = omega * (r2 * v_theta2 - inlet.r * inlet.v_theta)
        eq2 = ((ht2 - inlet.ht) - euler_work) / abs(inlet.ht)

        # eq3 — total pressure loss
        eq3 = (pt2 - (pt2_ideal - dp_loss)) / inlet.pt

        # eq4 — exit flow angle: metal angle + deviation from subelement
        eq4 = beta2 - (self.beta_te + delta)

        # eq5 — radius constraint
        eq5 = (r2 - self.r_exit) / self.r_exit

        # eq6 — flow area (mode-dependent)
        if self.mode == "des":
            eq6 = 0.0   # Area floats; continuity (eq7) closes the system
        else:
            eq6 = (a_flow2 - (self.a_exit - self.a_blockage)) / self.a_exit

        # eq7 — continuity
        eq7 = (m2 - rho2 * a_flow2 * v_m2) / inlet.m

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    def execute(self, omega: float):
        inlet = self.Fl_I

        # Log incidence for diagnostics
        kine1     = self.velocity_triangle(omega, inlet.r, inlet.v_m,
                                           v_theta=inlet.v_theta)
        incidence = kine1["beta"] - self.beta_le
        print(f"[{self.name}] mode={self.mode} | ω={omega:.1f} rad/s "
              f"| incidence={np.degrees(incidence):.2f}°")

        # Initial guesses (ht2, pt2 are primary variables)
        guess_ht2 = inlet.ht + omega * inlet.r * 50.0 if omega > 0 else inlet.ht
        guess_pt2 = inlet.pt * 1.4                    if omega > 0 else inlet.pt * 0.98
        a_flow_0  = self.a_exit - self.a_blockage

        x0 = [inlet.m, guess_ht2, guess_pt2,
               self.beta_te, self.r_exit, a_flow_0, inlet.v_m]

        sol, _, ier, msg = fsolve(self.residuals, x0, args=(omega,), full_output=True)

        if ier != 1:
            print(f"  WARNING [{self.name}]: solver did not converge — {msg}")

        m2, ht2, pt2, beta2, r2, a_flow2, v_m2 = sol
        kine2 = self.velocity_triangle(omega, r2, v_m2, beta_rel=beta2)

        tt2 = ht2 / CP
        s2  = entropy_from_pt_tt(pt2, tt2)

        # In design mode, freeze the solved flow area as the blade geometry
        if self.mode == "des":
            self.a_exit = a_flow2 + self.a_blockage

        # Write output port — (ht, s) as primary anchors
        self.Fl_O.set_state_from_ht_s(
            m=m2, ht=ht2, s=s2, r=r2,
            v_theta=kine2["V_theta"], v_m=v_m2, beta=beta2
        )


# =============================================================================
# 6. INLET BOUNDARY
# =============================================================================
class InletBoundary(Element):
    """Injects boundary conditions. State pushed in execute(), not __init__."""

    def __init__(self, name: str, m: float, pt: float, tt: float,
                 r: float, v_theta: float, v_m: float, mode: str = "des"):
        super().__init__(name, mode)
        self._m       = m
        self._pt      = pt
        self._tt      = tt
        self._r       = r
        self._v_theta = v_theta
        self._v_m     = v_m

    def residuals(self, vars, **kwargs):
        return []

    def execute(self, **kwargs):
        self.Fl_O.set_state(self._m, self._pt, self._tt,
                            self._r, self._v_theta, self._v_m)
