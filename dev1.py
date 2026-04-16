import numpy as np
from scipy.optimize import fsolve


# ==========================================
# 1. KINEMATICS MODULE
# ==========================================
def calc_velocity_triangle(omega, r, v_m, beta_rel, phi=0.0):
    """
    Calculates the complete velocity triangle at a given station.
    Handles axial, radial, and mixed flows via the meridional angle (phi).

    phi = 0.0      -> Purely Axial (Vm = Vz)
    phi = pi/2     -> Purely Radial (Vm = Vr)
    phi = (0, pi/2)-> Mixed Flow
    """
    u = omega * r

    # --- Meridional Plane Decomposition ---
    v_z = v_m * np.cos(phi)  # Axial velocity
    v_r = v_m * np.sin(phi)  # Radial velocity

    # --- Tangential Plane Calculations ---
    w_theta = v_m * np.tan(beta_rel)  # Relative tangential velocity
    v_theta = u - w_theta  # Absolute tangential velocity

    # Absolute flow angle (alpha) in the tangential plane
    alpha = np.arctan2(v_theta, v_m)

    # Velocity magnitudes squared
    v_sq = v_m ** 2 + v_theta ** 2
    w_sq = v_m ** 2 + w_theta ** 2

    return {
        'U': u,
        'Vm': v_m,
        'V_z': v_z,
        'V_r': v_r,
        'V_theta': v_theta,
        'W_theta': w_theta,
        'V_sq': v_sq,
        'W_sq': w_sq,
        'alpha': alpha,
        'beta': beta_rel,
        'phi': phi
    }


# ==========================================
# 2. HARD-CODED INPUTS (Compressor Design)
# ==========================================

gamma = 1.4
R = 287.0
Cp = gamma * R / (gamma - 1.0)

# State 1 (Inlet) conditions
pt1 = 101325.0
tt1 = 288.15
m1 = 10.0  # Mass flow [kg/s]
r1 = 0.3
v_theta1 = 0.0

# Machine / Blade Row Inputs
omega = 800.0  # Shaft speed [rad/s]
r_machine = 0.3
a_machine = 0.1
a_blockage = 0.005

beta_blade = np.radians(40.0)
delta = np.radians(2.0)
loss_dp = 2000.0

ht1 = Cp * tt1


# ==========================================
# 3. THE 7-EQUATION SYSTEM
# ==========================================
def otac_meanline_equations(vars):
    m2, ht2, pt2, beta2, r2, a_flow2, v_m2 = vars

    # --- Kinematics ---
    kine2 = calc_velocity_triangle(omega, r2, v_m2, beta2)

    # --- Thermodynamics ---
    h2 = ht2 - 0.5 * kine2['V_sq']
    t2 = h2 / Cp
    tt2 = ht2 / Cp

    p2 = pt2 * (t2 / tt2) ** (gamma / (gamma - 1.0))
    rho2 = p2 / (R * t2)
    pt2_ideal = pt1 * (tt2 / tt1) ** (gamma / (gamma - 1.0))

    # --- Residual Equations ---
    eq1 = m2 - m1
    eq2 = (ht2 - ht1) - omega * (r2 * kine2['V_theta'] - r1 * v_theta1)
    eq3 = pt2 - (pt2_ideal - loss_dp)
    eq4 = beta2 - (beta_blade + delta)
    eq5 = r2 - r_machine
    eq6 = a_flow2 - (a_machine - a_blockage)
    eq7 = m2 - (rho2 * a_flow2 * v_m2)

    # Normalize residuals
    return [
        eq1 / m1,
        eq2 / ht1,
        eq3 / pt1,
        eq4,
        eq5 / r_machine,
        eq6 / a_machine,
        eq7 / m1
    ]


# ==========================================
# 4. SOLVER SETUP & EXECUTION
# ==========================================

# Initial Guesses
initial_guesses = [
    m1,
    ht1 + 35000,
    pt1 + 40000,
    beta_blade + delta,
    r_machine,
    a_machine - a_blockage,
    85.0
]

solution = fsolve(otac_meanline_equations, initial_guesses)
m2_sol, ht2_sol, pt2_sol, beta2_sol, r2_sol, a_flow2_sol, v_m2_sol = solution

# Post-process final state using the standalone function
kine2_sol = calc_velocity_triangle(omega, r2_sol, v_m2_sol, beta2_sol)

tt2_sol = ht2_sol / Cp
pr_sol = pt2_sol / pt1
tr_sol = tt2_sol / tt1

# ==========================================
# 5. RESULTS OUTPUT
# ==========================================
print("--- OTAC 7-Equation Single Row Verification ---")
print(f"Exit Mass Flow (m2):         {m2_sol:.2f} kg/s")
print(f"Exit Total Enthalpy (ht2):   {ht2_sol:.2f} J/kg")
print(f"Exit Total Temp (Tt2):       {tt2_sol:.2f} K")
print(f"Exit Total Pressure (Pt2):   {pt2_sol:.2f} Pa")
print(f"Exit Meridional Vel (Vm2):   {kine2_sol['Vm']:.2f} m/s")
print(f"Exit Tangential Vel (Vth2):  {kine2_sol['V_theta']:.2f} m/s")
print(f"Stage Total Pressure Ratio:  {pr_sol:.4f}")
print(f"Stage Total Temp Ratio:      {tr_sol:.4f}")