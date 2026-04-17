import numpy as np

from thermo import *

# =============================================================================
# 2. FLUID PORT
# =============================================================================
class FlowStation:

    def __init__(self, name: str = "Port"):
        self.name = name

        # Thermodynamics
        self.p_s = 0.0      # static pressure                    [Pa]
        self.t_s = 0.0      # static temperature                 [K]
        self.rho = 0.0      # static density                     [kg/m^3]
        self.h_s = 0.0      # static enthalpy                    [J/kg]
        self.s_s = 0.0      # static entropy                     [J/kgK]
        self.p_t = 0.0      # abs total pressure                 [Pa]
        self.t_t = 0.0      # abs total temperature              [K]
        self.h_t = 0.0      # abs total enthalpy                 [J/kg]
        self.p_tr = 0.0     # rel total pressure                 [Pa]
        self.t_tr = 0.0     # rel total temperature              [K]
        self.h_tr = 0.0     # rel total enthalpy                 [J/kg]

        # Properties
        self.gamma = GAMMA  # specific heat ratio                [-]
        self.R = R          # specific gas constant              [J/kgK]
        self.Cp = CP        # specific heat at constant pressure [J/kgK]
        self.Cv = CP/R      # specific heat at constant volume   [J/kgK]

        # Geometry
        self.A = 0.0        # area                               [m^2]
        self.r = 0.0        # radius                             [m]

        # Kinematics
        self.v = 0.0        # absolute velocity                  [m/s]
        self.v_a = 0.0      # axial velocity                     [m/s]
        self.v_u = 0.0      # circumferential velocity           [m/s]
        self.v_r = 0.0      # radial velocity                    [m/s]
        self.v_m =  0.0     # meridional velocity                [m/s]
        self.w = 0.0        # relative velocity                  [m/s]
        self.w_a = 0.0      # relative axial velocity            [m/s]
        self.w_u = 0.0      # relative circumferential velocity  [m/s]
        self.w_r = 0.0      # relative radial velocity           [m/s]
        self.alpha = 0.0    # absolute flow angle                [rad]
        self.beta = 0.0     # relative flow angle                [rad]
        self.eps = 0.0      # meridional flow angle              [rad]
        self.omega = 0.0    # angular velocity                   [rad/s]

        # Derived
        self.c = 0.0        # speed of sound                     [m/s]
        self.m = 0.0        # mass flow rate                     [kg/s]
        self.ma = 0.0       # mach number                        [-]
        self.ma_r = 0.0     # relative mach number               [-]

    def set_state(self, m: float, p_t: float, t_t: float, r: float,
                  v_u: float, v_m: float, beta: float = None):
        """Sets the port state from absolute total pressure and temperature."""
        self.m = m
        self.p_t = p_t
        self.t_t = max(t_t, 1.0)  # Avoid division by zero
        self.r = r
        self.v_u = v_u
        self.v_m = v_m

        # Total enthalpy
        self.h_t = self.Cp * self.t_t

        # Assuming entropy_from_pt_tt is imported from thermo.py
        from thermo import entropy_from_pt_tt
        self.s_s = entropy_from_pt_tt(self.p_t, self.t_t)

        self._resolve_state(beta)

    def set_state_from_h_t_s(self, m: float, h_t: float, s_s: float, r: float,
                             v_u: float, v_m: float, beta: float = None):
        """Sets the port state from total enthalpy and static entropy."""
        self.m = m
        self.h_t = h_t
        self.s_s = s_s
        self.r = r
        self.v_u = v_u
        self.v_m = v_m

        self.t_t = max(self.h_t / self.Cp, 1.0)

        # Calculate p_t from s_s and t_t using ideal gas relations
        # s = Cp * ln(T_t) - R * ln(P_t)  =>  P_t = exp((Cp * ln(T_t) - s) / R)
        # Note: Adjust this inversion if your thermo.py uses a different reference state.
        self.p_t = np.exp((self.Cp * np.log(self.t_t) - self.s_s) / self.R)

        self._resolve_state(beta)

    def _resolve_state(self, beta: float = None):
        """Calculates all derived static, relative, and kinematic properties."""

        # 1. Absolute Kinematics
        self.v = np.sqrt(self.v_u ** 2 + self.v_m ** 2)
        self.alpha = np.arctan2(self.v_u, self.v_m)

        # 2. Static Thermodynamics
        self.h_s = self.h_t - 0.5 * self.v ** 2
        self.t_s = max(self.h_s / self.Cp, 1.0)
        self.p_s = self.p_t * (self.t_s / self.t_t) ** (self.gamma / (self.gamma - 1.0))
        self.rho = max(self.p_s / (self.R * self.t_s), 1e-6)

        # 3. Derived flow properties
        self.c = np.sqrt(self.gamma * self.R * self.t_s)
        self.ma = self.v / self.c

        # 4. Relative Kinematics & Thermodynamics (if beta is provided)
        if beta is not None:
            self.beta = beta

            # Based on streamsegment.py convention: v_u = u - w_u => w_u = v_m * tan(beta)
            self.w_u = self.v_m * np.tan(self.beta)
            self.w = np.sqrt(self.w_u ** 2 + self.v_m ** 2)

            # Calculate angular velocity required to close the velocity triangle
            u = self.v_u + self.w_u
            self.omega = u / self.r if self.r > 1e-6 else 0.0

            # Relative totals
            self.h_tr = self.h_s + 0.5 * self.w ** 2
            self.t_tr = max(self.h_tr / self.Cp, 1.0)
            self.p_tr = self.p_s * (self.t_tr / self.t_s) ** (self.gamma / (self.gamma - 1.0))

            self.ma_r = self.w / self.c
        else:
            # Zero out relative properties if not in a rotating frame
            self.beta = 0.0
            self.w_u = 0.0
            self.w = 0.0
            self.omega = 0.0
            self.h_tr = self.h_t
            self.t_tr = self.t_t
            self.p_tr = self.p_t
            self.ma_r = self.ma

    def copy_from(self, src: "FlowStation"):
        for k, v in src.__dict__.items():
            if k != "name":
                setattr(self, k, v)

    def report(self):
        self.v = np.sqrt(self.v_m**2 + self.v_u**2)
        self.h_s = self.h_t - 0.5 * self.v**2
        self.t_s = self.h_s / self.Cp
        self.ma  = self.v / np.sqrt(self.gamma * self.R * self.t_s) if self.t_s > 0 else 0.0
        print(f"--- {self.name} ---")
        print(f"  m     : {self.m:.4f} kg/s")
        print(f"  Pt    : {self.p_t:.2f} Pa")
        print(f"  Tt    : {self.t_t:.4f} K")
        print(f"  ht    : {self.h_t:.2f} J/kg")
        print(f"  s     : {self.s_s:.6f} J/(kg·K)")
        print(f"  r     : {self.r:.4f} m")
        print(f"  Vm    : {self.v_m:.4f} m/s")
        print(f"  Vth   : {self.v_u:.4f} m/s")
        print(f"  Mach  : {self.ma:.4f}")
        print(f"  beta  : {np.degrees(self.beta):.2f} deg")
        print("-" * 42)


def link_ports(port_out: FlowStation, port_in: FlowStation):
    """Mimics NPSS/OTAC linkPorts — propagates outlet state to the next inlet."""
    port_in.copy_from(port_out)