import numpy as np

from thermo import *


# =============================================================================
# 2. FLUID PORT
# =============================================================================
class FlowStation:
    """
    Connection node between Elements.

    Primary thermodynamic anchors (matching OTAC FlowStation intent):
        ht  — specific total enthalpy  [J/kg]      (energy anchor)
        s   — specific entropy         [J/(kg·K)]  (loss anchor)

    Pt and Tt are stored for convenience and solver access, but are
    always consistent with ht and s:
        Tt = ht / Cp
        Pt = f(ht, s)  via pt_from_ht_s()

    Kinematic state:
        m       — mass flow rate          [kg/s]
        r       — radius                  [m]
        v_theta — absolute swirl velocity [m/s]
        v_m     — meridional velocity     [m/s]
        beta    — relative flow angle     [rad]
    """

    def __init__(self, name: str = "Port"):
        self.name    = name
        self.m       = 0.0
        self.ht      = 0.0
        self.s       = 0.0
        self.pt      = 0.0
        self.tt      = 0.0
        self.r       = 0.0
        self.v_theta = 0.0
        self.v_m     = 0.0
        self.beta    = 0.0

    def set_state(self, m: float, pt: float, tt: float,
                  r: float, v_theta: float, v_m: float, beta: float = 0.0):
        """Set state from (Pt, Tt) — derives ht and s internally."""
        self.m       = m
        self.pt      = pt
        self.tt      = tt
        self.ht      = CP * tt
        self.s       = entropy_from_pt_tt(pt, tt)
        self.r       = r
        self.v_theta = v_theta
        self.v_m     = v_m
        self.beta    = beta

    def set_state_from_ht_s(self, m: float, ht: float, s: float,
                             r: float, v_theta: float, v_m: float,
                             beta: float = 0.0):
        """Set state from (ht, s) — derives Pt and Tt internally."""
        self.m       = m
        self.ht      = ht
        self.s       = s
        self.tt      = tt_from_ht(ht)
        self.pt      = pt_from_ht_s(ht, s)
        self.r       = r
        self.v_theta = v_theta
        self.v_m     = v_m
        self.beta    = beta

    def copy_from(self, src: "FlowStation"):
        self.m       = src.m
        self.ht      = src.ht
        self.s       = src.s
        self.pt      = src.pt
        self.tt      = src.tt
        self.r       = src.r
        self.v_theta = src.v_theta
        self.v_m     = src.v_m
        self.beta    = src.beta

    def report(self):
        v_sq  = self.v_m**2 + self.v_theta**2
        h     = self.ht - 0.5 * v_sq
        t     = h / CP
        mach  = np.sqrt(v_sq / (GAMMA * R * t)) if t > 0 else 0.0
        print(f"--- {self.name} ---")
        print(f"  m     : {self.m:.4f} kg/s")
        print(f"  Pt    : {self.pt:.2f} Pa")
        print(f"  Tt    : {self.tt:.4f} K")
        print(f"  ht    : {self.ht:.2f} J/kg")
        print(f"  s     : {self.s:.6f} J/(kg·K)")
        print(f"  r     : {self.r:.4f} m")
        print(f"  Vm    : {self.v_m:.4f} m/s")
        print(f"  Vth   : {self.v_theta:.4f} m/s")
        print(f"  Mach  : {mach:.4f}")
        print(f"  beta  : {np.degrees(self.beta):.2f} deg")
        print("-" * 42)


def link_ports(port_out: FlowStation, port_in: FlowStation):
    """Mimics NPSS/OTAC linkPorts — propagates outlet state to the next inlet."""
    port_in.copy_from(port_out)