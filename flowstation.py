import numpy as np
from thermo import CP, GAMMA, R, entropy_from_pt_tt, pt_from_ht_s


class FlowStation:
    """
    Fluid state at a calculation station.

    All attributes support both scalar (N=1, meanline) and 1-D array
    (N>1, streamline) values. numpy operations broadcast transparently
    over both cases — no conditional branching required.

    Primary setters
    ---------------
    set_state(m, p_t, t_t, r, v_u, v_m)
        Anchor: absolute total conditions (p_t, t_t).

    set_state_from_h_t_s(m, h_t, s_s, r, v_u, v_m)
        Anchor: total enthalpy and entropy (h_t, s_s).
        Used when the solver works in (h_t, s_s) space so entropy is
        the natural variable and p_t is derived — not prescribed.

    Both call _resolve_state() which populates every derived attribute.
    """

    def __init__(self, name: str = "Port"):
        self.name = name

        # --- Thermodynamics ---
        self.p_s  = 0.0   # static pressure              [Pa]
        self.t_s  = 0.0   # static temperature           [K]
        self.rho  = 0.0   # static density               [kg/m³]
        self.h_s  = 0.0   # static enthalpy              [J/kg]
        self.s_s  = 0.0   # specific entropy             [J/kg/K]
        self.p_t  = 0.0   # absolute total pressure      [Pa]
        self.t_t  = 0.0   # absolute total temperature   [K]
        self.h_t  = 0.0   # absolute total enthalpy      [J/kg]
        self.p_tr = 0.0   # relative total pressure      [Pa]
        self.t_tr = 0.0   # relative total temperature   [K]
        self.h_tr = 0.0   # relative total enthalpy      [J/kg]

        # --- Gas properties (uniform, perfect gas) ---
        self.gamma = GAMMA
        self.R     = R
        self.Cp    = CP
        self.Cv    = R / (GAMMA - 1.0)

        # --- Geometry ---
        self.A = 0.0   # flow area  [m²]
        self.r = 0.0   # radius     [m]

        # --- Absolute kinematics ---
        self.v   = 0.0   # velocity magnitude          [m/s]
        self.v_a = 0.0   # axial component             [m/s]
        self.v_u = 0.0   # circumferential component   [m/s]
        self.v_r = 0.0   # radial component            [m/s]
        self.v_m = 0.0   # meridional velocity         [m/s]

        # --- Relative kinematics ---
        self.w   = 0.0   # relative velocity magnitude [m/s]
        self.w_a = 0.0   # relative axial component    [m/s]
        self.w_u = 0.0   # relative circ. component    [m/s]
        self.w_r = 0.0   # relative radial component   [m/s]

        # --- Flow angles ---
        self.alpha = 0.0   # absolute flow angle   [rad]
        self.beta  = 0.0   # relative flow angle   [rad]
        self.eps   = 0.0   # meridional slope      [rad]

        # --- Derived ---
        self.c     = 0.0   # speed of sound            [m/s]
        self.m     = 0.0   # mass flow rate            [kg/s]
        self.ma    = 0.0   # absolute Mach number      [-]
        self.ma_r  = 0.0   # relative Mach number      [-]
        self.omega = 0.0   # angular velocity inferred [rad/s]

    # ------------------------------------------------------------------
    # N — number of streamtubes
    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        """
        Number of streamtubes represented by this FlowStation.
        Returns 1 for scalar (meanline) state, n for array (streamline).
        """
        return int(np.asarray(self.m).size)

    # ------------------------------------------------------------------
    # Primary setters
    # ------------------------------------------------------------------
    def set_state(self, m, p_t, t_t, r, v_u, v_m, beta=None):
        """Set state from absolute total conditions (p_t, t_t)."""
        self.m   = m
        self.p_t = np.asarray(p_t, dtype=float)
        self.t_t = np.maximum(np.asarray(t_t, dtype=float), 1.0)
        self.r   = np.asarray(r,   dtype=float)
        self.v_u = np.asarray(v_u, dtype=float)
        self.v_m = np.asarray(v_m, dtype=float)
        self.h_t = self.Cp * self.t_t
        self.s_s = entropy_from_pt_tt(self.p_t, self.t_t)
        self._resolve_state(beta)

    def set_state_from_h_t_s(self, m, h_t, s_s, r, v_u, v_m, beta=None):
        """Set state from total enthalpy and entropy (h_t, s_s)."""
        self.m   = m
        self.h_t = np.asarray(h_t, dtype=float)
        self.s_s = np.asarray(s_s, dtype=float)
        self.r   = np.asarray(r,   dtype=float)
        self.v_u = np.asarray(v_u, dtype=float)
        self.v_m = np.asarray(v_m, dtype=float)
        self.t_t = np.maximum(self.h_t / self.Cp, 1.0)
        self.p_t = pt_from_ht_s(self.h_t, self.s_s)   # correct inversion
        self._resolve_state(beta)

    # ------------------------------------------------------------------
    # Core derived-state resolver
    # ------------------------------------------------------------------
    def _resolve_state(self, beta=None):
        """
        Populate every derived attribute from the primary state.
        Works element-wise for scalar or array inputs.

        Velocity triangle convention
        ----------------------------
            W_u = U - V_u  (relative circ. = blade speed minus abs. circ.)
            W_u = V_m * tan(beta)
            beta = arctan2(W_u, W_m) with W_m = V_m

        If beta is None the station is treated as non-rotating
        (U = 0) and relative quantities equal their absolute counterparts.
        """
        # --- Absolute kinematics ---
        self.v_a = self.v_m          # meridional = axial (v_r = 0)
        self.v_r = 0.0
        self.v   = np.sqrt(self.v_u**2 + self.v_m**2)
        self.alpha = np.arctan2(self.v_u, self.v_m)

        # --- Static conditions ---
        self.h_s = self.h_t - 0.5 * self.v**2
        self.t_s = np.maximum(self.h_s / self.Cp, 1.0)
        self.p_s = (self.p_t
                    * (self.t_s / np.maximum(self.t_t, 1.0))
                    ** (self.gamma / (self.gamma - 1.0)))
        self.rho = np.maximum(self.p_s / (self.R * self.t_s), 1e-10)

        # --- Speed of sound and Mach ---
        self.c  = np.sqrt(self.gamma * self.R * np.maximum(self.t_s, 1.0))
        self.ma = self.v / np.maximum(self.c, 1e-10)

        # --- Flow area from continuity ---
        self.A = self.m / np.maximum(self.rho * self.v_m, 1e-10)

        # --- Relative frame ---
        if beta is not None:
            self.beta = np.asarray(beta, dtype=float)

            # W_u = V_m * tan(beta),  U = V_u + W_u
            self.w_u  = self.v_m * np.tan(self.beta)
            self.w_a  = self.v_a
            self.w_r  = 0.0
            self.w    = np.sqrt(self.w_u**2 + self.w_a**2)

            u = self.v_u + self.w_u
            self.omega = u / np.maximum(self.r, 1e-10)

            # Relative total enthalpy: h_tr = h_s + 0.5 * W^2
            self.h_tr = self.h_s + 0.5 * self.w**2
            self.t_tr = np.maximum(self.h_tr / self.Cp, 1.0)
            self.p_tr = (self.p_s
                         * (self.t_tr / np.maximum(self.t_s, 1.0))
                         ** (self.gamma / (self.gamma - 1.0)))
            self.ma_r = self.w / np.maximum(self.c, 1e-10)

        else:
            # Non-rotating station: relative = absolute
            self.beta  = 0.0
            self.w_u   = 0.0
            self.w_a   = self.v_a
            self.w_r   = 0.0
            self.w     = self.v
            self.omega = 0.0
            self.h_tr  = self.h_t
            self.t_tr  = self.t_t
            self.p_tr  = self.p_t
            self.ma_r  = self.ma

    # ------------------------------------------------------------------
    # Port linkage
    # ------------------------------------------------------------------
    def copy_from(self, src: "FlowStation"):
        """Copy all state from src. Works for scalar and array state."""
        for k, v in src.__dict__.items():
            if k != "name":
                setattr(self, k, v)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def report(self):
        """Print state. Iterates over streams when N > 1."""
        N = self.N

        def _get(attr, i):
            v = np.asarray(getattr(self, attr), dtype=float)
            return float(v.flat[i] if v.size > 1 else v)

        for i in range(N):
            suffix = f" [stream {i}]" if N > 1 else ""
            print(f"--- {self.name}{suffix} ---")
            print(f"  m     : {_get('m',   i):.4f} kg/s")
            print(f"  Pt    : {_get('p_t', i):.2f} Pa")
            print(f"  Tt    : {_get('t_t', i):.4f} K")
            print(f"  ht    : {_get('h_t', i):.2f} J/kg")
            print(f"  s     : {_get('s_s', i):.6f} J/(kg·K)")
            print(f"  r     : {_get('r',   i):.4f} m")
            print(f"  Vm    : {_get('v_m', i):.4f} m/s")
            print(f"  Vth   : {_get('v_u', i):.4f} m/s")
            print(f"  Mach  : {_get('ma',  i):.4f}")
            print(f"  beta  : {np.degrees(_get('beta', i)):.2f} deg")
            print("-" * 42)


def link_ports(port_out: FlowStation, port_in: FlowStation):
    """Propagate outlet state to the next element's inlet."""
    port_in.copy_from(port_out)